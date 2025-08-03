"""
训练器模块，负责模型的训练和评估
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, mean_squared_error

from src.models import PredictionModel, WeightModel
from src.utils import Logger
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union, Callable


class BaseTrainer:
    """基础训练器，提供共用的训练功能"""
    
    def __init__(self, device: str = "cuda", logger: Logger = None):
        """
        初始化基础训练器
        
        Args:
            device: 计算设备
            logger: 日志记录器
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.logger = logger or Logger()
        
    def train_step(self, *args, **kwargs):
        """训练步骤，需要在子类中实现"""
        raise NotImplementedError("子类必须实现train_step方法")
        
    def evaluate(self, *args, **kwargs):
        """评估方法，需要在子类中实现"""
        raise NotImplementedError("子类必须实现evaluate方法")
        
    def save_model(self, model: nn.Module, save_path: str) -> None:
        """
        保存模型
        
        Args:
            model: 要保存的模型
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        self.logger.info(f"模型已保存至: {save_path}")
    
    def load_model(self, model: nn.Module, load_path: str) -> None:
        """
        加载模型
        
        Args:
            model: 要加载的模型
            load_path: 加载路径
        """
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path, map_location=self.device))
            self.logger.info(f"模型已从 {load_path} 加载")
        else:
            self.logger.warning(f"模型文件 {load_path} 不存在，无法加载")


class PredictionTrainer(BaseTrainer):
    """推荐模型训练器"""
    
    def __init__(self, 
                 prediction_model: PredictionModel,
                 weight_model: Optional[WeightModel] = None,
                 device: str = 'cuda',
                 lr: float = 0.001,
                 weight_decay: float = 1e-5,
                 clip_grad: Optional[float] = 1.0,
                 lambda_ctr: float = 1.0,
                 lambda_playtime: float = 0.5,
                 logger: Logger = None):
        """
        初始化推荐模型训练器
        
        Args:
            prediction_model: 预测模型
            weight_model: 权重模型，可选
            device: 训练设备
            lr: 学习率
            weight_decay: 权重衰减
            clip_grad: 梯度裁剪值，None表示不裁剪
            lambda_ctr: CTR损失权重
            lambda_playtime: 播放时长损失权重
            logger: 日志记录器
        """
        super().__init__(device, logger)
        
        self.prediction_model = prediction_model.to(self.device)
        self.weight_model = weight_model.to(self.device) if weight_model else None
        
        self.optimizer = optim.Adam(
            self.prediction_model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
        self.ctr_criterion = BCEWithLogitsLoss(reduction='none')
        self.playtime_criterion = MSELoss(reduction='none')
        
        self.lambda_ctr = lambda_ctr
        self.lambda_playtime = lambda_playtime
        self.clip_grad = clip_grad
        
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0
    
    def compute_weight(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算样本权重
        
        Args:
            features: 特征字典
            
        Returns:
            样本权重张量
        """
        if self.weight_model is None:
            # 确定批次大小，从user_features特征获取
            batch_size = features['user_features'].size(0) if 'user_features' in features else 1
            return torch.ones(batch_size, device=self.device)
        
        with torch.no_grad():
            self.weight_model.eval()
            # 从features字典中提取user_features张量，而不是直接传递字典
            user_features = features['user_features']
            weights = self.weight_model(user_features)
        
        return weights
    
    def compute_loss(self, 
                    batch: Dict[str, torch.Tensor], 
                    group_key: str = 'treatment',
                    weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            batch: 批次数据
            group_key: 分组键，'treatment'或'control'
            weights: 样本权重，可选
            
        Returns:
            总损失和损失字典
        """
        features = {k: v.to(self.device) for k, v in batch['features'].items()}
        group_mask = batch['group'] == (1 if group_key == 'treatment' else 0)
        
        if not group_mask.any():
            return torch.tensor(0.0, device=self.device), {
                'ctr_loss': 0.0,
                'playtime_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 获取组内数据
        group_features = {k: v[group_mask] for k, v in features.items()}
        ctr_target = batch['ctr'][group_mask].to(self.device)
        playtime_target = batch['playtime'][group_mask].to(self.device)
        
        # 从字典中提取特征张量用于模型输入
        if isinstance(group_features, dict):
            # 如果特征是字典，使用user_features或第一个可用的特征
            feature_tensor = group_features.get('user_features', next(iter(group_features.values())))
        else:
            # 否则直接使用特征
            feature_tensor = group_features
        
        # 前向传播
        ctr_pred, playtime_pred = self.prediction_model(feature_tensor)
        
        # 计算损失
        ctr_losses = self.ctr_criterion(ctr_pred, ctr_target)
        playtime_losses = self.playtime_criterion(playtime_pred, playtime_target)
        
        # 应用权重(如果有)
        if weights is not None:
            group_weights = weights[group_mask]
            ctr_losses = ctr_losses * group_weights
            playtime_losses = playtime_losses * group_weights
        
        # 计算平均损失
        ctr_loss = ctr_losses.mean()
        playtime_loss = playtime_losses.mean()
        
        # 组合损失
        total_loss = self.lambda_ctr * ctr_loss + self.lambda_playtime * playtime_loss
        
        loss_dict = {
            'ctr_loss': ctr_loss.item(),
            'playtime_loss': playtime_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor],
                   apply_weights: bool = True) -> Dict[str, Dict[str, float]]:
        """
        执行一步训练
        
        Args:
            batch: 批次数据
            apply_weights: 是否应用权重
            
        Returns:
            训练指标字典
        """
        self.prediction_model.train()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 获取特征
        features = batch['features']
        
        # 如果需要，计算样本权重
        weights = None
        if apply_weights and self.weight_model is not None:
            weights = self.compute_weight(features)
        
        # 分别对处理组和对照组计算损失
        t_loss, t_loss_dict = self.compute_loss(batch, 'treatment', weights)
        c_loss, c_loss_dict = self.compute_loss(batch, 'control', weights)
        
        # 总损失
        total_loss = t_loss + c_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.prediction_model.parameters(), self.clip_grad)
        
        # 参数更新
        self.optimizer.step()
        
        # 预测指标计算
        with torch.no_grad():
            metrics = self.compute_metrics(batch)
        
        # 合并损失和指标
        result = {
            'treatment': {**t_loss_dict, **metrics.get('treatment', {})},
            'control': {**c_loss_dict, **metrics.get('control', {})}
        }
        
        return {'train': result}
    
    def compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        计算评估指标
        
        Args:
            batch: 批次数据
            
        Returns:
            评估指标字典
        """
        self.prediction_model.eval()
        result = {'treatment': {}, 'control': {}}
        
        with torch.no_grad():
            features = {k: v.to(self.device) for k, v in batch['features'].items()}
            
            # 从字典中提取特征张量
            feature_tensor = features.get('user_features', next(iter(features.values())))
            
            # 获取预测结果
            ctr_pred, playtime_pred = self.prediction_model(feature_tensor)
            ctr_pred = torch.sigmoid(ctr_pred).cpu().numpy()
            playtime_pred = playtime_pred.cpu().numpy()
            
            # 目标值
            ctr_target = batch['ctr'].cpu().numpy()
            playtime_target = batch['playtime'].cpu().numpy()
            group = batch['group'].cpu().numpy()
            
            # 分组计算指标
            for group_id, group_name in [(1, 'treatment'), (0, 'control')]:
                mask = (group == group_id)
                if not mask.any():
                    continue
                    
                # CTR AUC
                try:
                    ctr_auc = roc_auc_score(ctr_target[mask], ctr_pred[mask])
                    result[group_name]['ctr_auc'] = ctr_auc
                except:
                    result[group_name]['ctr_auc'] = 0.5
                
                # 播放时间RMSE
                try:
                    playtime_rmse = np.sqrt(mean_squared_error(playtime_target[mask], playtime_pred[mask]))
                    result[group_name]['playtime_rmse'] = playtime_rmse
                except:
                    result[group_name]['playtime_rmse'] = 0.0
        
        return result
    
    def evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个批次
        
        Args:
            batch: 批次数据
            
        Returns:
            评估指标字典
        """
        self.prediction_model.eval()
        
        # 初始化指标收集器
        t_ctr_losses = []
        t_playtime_losses = []
        c_ctr_losses = []
        c_playtime_losses = []
        
        all_preds_t = []  # 处理组的CTR预测
        all_targets_t = []  # 处理组的CTR目标
        all_playtime_preds_t = []  # 处理组的播放时间预测
        all_playtime_targets_t = []  # 处理组的播放时间目标
        all_nonzero_playtime_t = []  # 处理组非零播放时间的相对误差
        
        all_preds_c = []  # 对照组的CTR预测
        all_targets_c = []  # 对照组的CTR目标
        all_playtime_preds_c = []  # 对照组的播放时间预测
        all_playtime_targets_c = []  # 对照组的播放时间目标
        all_nonzero_playtime_c = []  # 对照组非零播放时间的相对误差
        
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 计算损失
            t_loss, t_loss_dict = self.compute_loss(batch, 'treatment')
            c_loss, c_loss_dict = self.compute_loss(batch, 'control')
            
            # 收集损失
            if t_loss_dict['total_loss'] > 0:
                t_ctr_losses.append(t_loss_dict['ctr_loss'])
                t_playtime_losses.append(t_loss_dict['playtime_loss'])
            
            if c_loss_dict['total_loss'] > 0:
                c_ctr_losses.append(c_loss_dict['ctr_loss'])
                c_playtime_losses.append(c_loss_dict['playtime_loss'])
            
            # 预测和目标
            features = {k: v.to(self.device) for k, v in batch['features'].items()}
            
            # 从字典中提取特征张量
            feature_tensor = features.get('user_features', next(iter(features.values())))
            
            ctr_pred, playtime_pred = self.prediction_model(feature_tensor)
            ctr_pred = torch.sigmoid(ctr_pred).cpu().numpy()
            playtime_pred = playtime_pred.cpu().numpy()
            
            ctr_target = batch['ctr'].cpu().numpy()
            playtime_target = batch['playtime'].cpu().numpy()
            group = batch['group'].cpu().numpy()
            
            # 分组收集预测和目标
            t_mask = (group == 1)
            c_mask = (group == 0)
            
            if t_mask.any():
                all_preds_t.extend(ctr_pred[t_mask].tolist())
                all_targets_t.extend(ctr_target[t_mask].tolist())
                all_playtime_preds_t.extend(playtime_pred[t_mask].tolist())
                all_playtime_targets_t.extend(playtime_target[t_mask].tolist())
                
                # 计算非零播放时间的相对误差
                for i, (target, pred) in enumerate(zip(playtime_target[t_mask], playtime_pred[t_mask])):
                    if target > 0:
                        relative_error = abs(pred - target) / target
                        all_nonzero_playtime_t.append(relative_error)
            
            if c_mask.any():
                all_preds_c.extend(ctr_pred[c_mask].tolist())
                all_targets_c.extend(ctr_target[c_mask].tolist())
                all_playtime_preds_c.extend(playtime_pred[c_mask].tolist())
                all_playtime_targets_c.extend(playtime_target[c_mask].tolist())
                
                # 计算非零播放时间的相对误差
                for i, (target, pred) in enumerate(zip(playtime_target[c_mask], playtime_pred[c_mask])):
                    if target > 0:
                        relative_error = abs(pred - target) / target
                        all_nonzero_playtime_c.append(relative_error)
        
        # 处理评估结果，与evaluate方法相同
        t_metrics, c_metrics, is_best = self._process_evaluation_metrics(
            t_ctr_losses, t_playtime_losses, c_ctr_losses, c_playtime_losses,
            all_preds_t, all_targets_t, all_playtime_preds_t, all_playtime_targets_t,
            all_nonzero_playtime_t, all_preds_c, all_targets_c, all_playtime_preds_c, 
            all_playtime_targets_c, all_nonzero_playtime_c
        )
        
        return {'val': {'treatment': t_metrics, 'control': c_metrics, 'is_best': is_best}}

    def evaluate(self, 
                dataloader: DataLoader, 
                apply_weights: bool = True) -> Dict[str, Dict[str, float]]:
        """
        在验证集上评估模型
        
        Args:
            dataloader: 数据加载器
            apply_weights: 是否应用权重
            
        Returns:
            评估指标字典
        """
        self.prediction_model.eval()
        
        # 初始化指标收集器
        t_ctr_losses = []
        t_playtime_losses = []
        c_ctr_losses = []
        c_playtime_losses = []
        
        all_preds_t, all_targets_t = [], []  # 处理组CTR预测和目标
        all_preds_c, all_targets_c = [], []  # 对照组CTR预测和目标
        all_playtime_preds_t, all_playtime_targets_t = [], []  # 处理组播放时间预测和目标
        all_playtime_preds_c, all_playtime_targets_c = [], []  # 对照组播放时间预测和目标
        all_nonzero_playtime_t = []  # 处理组非零播放时间相对误差
        all_nonzero_playtime_c = []  # 对照组非零播放时间相对误差
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 如果需要，计算样本权重
                weights = None
                if apply_weights and self.weight_model is not None:
                    features = batch['features']
                    weights = self.compute_weight(features)
                
                # 分别对处理组和对照组计算损失
                _, t_loss_dict = self.compute_loss(batch, 'treatment', weights)
                _, c_loss_dict = self.compute_loss(batch, 'control', weights)
                
                # 收集损失
                if t_loss_dict['total_loss'] > 0:
                    t_ctr_losses.append(t_loss_dict['ctr_loss'])
                    t_playtime_losses.append(t_loss_dict['playtime_loss'])
                
                if c_loss_dict['total_loss'] > 0:
                    c_ctr_losses.append(c_loss_dict['ctr_loss'])
                    c_playtime_losses.append(c_loss_dict['playtime_loss'])
                
                # 预测和目标
                features = {k: v.to(self.device) for k, v in batch['features'].items()}
                
                # 从字典中提取特征张量
                feature_tensor = features.get('user_features', next(iter(features.values())))
                
                ctr_pred, playtime_pred = self.prediction_model(feature_tensor)
                ctr_pred = torch.sigmoid(ctr_pred).cpu().numpy()
                playtime_pred = playtime_pred.cpu().numpy()
                
                ctr_target = batch['ctr'].cpu().numpy()
                playtime_target = batch['playtime'].cpu().numpy()
                group = batch['group'].cpu().numpy()
                
                # 分组收集预测和目标
                t_mask = (group == 1)
                c_mask = (group == 0)
                
                if t_mask.any():
                    all_preds_t.extend(ctr_pred[t_mask].tolist())
                    all_targets_t.extend(ctr_target[t_mask].tolist())
                    all_playtime_preds_t.extend(playtime_pred[t_mask].tolist())
                    all_playtime_targets_t.extend(playtime_target[t_mask].tolist())
                    
                    # 计算非零播放时间的相对误差
                    for i, (target, pred) in enumerate(zip(playtime_target[t_mask], playtime_pred[t_mask])):
                        if target > 0:
                            relative_error = abs(pred - target) / target
                            all_nonzero_playtime_t.append(relative_error)
                
                if c_mask.any():
                    all_preds_c.extend(ctr_pred[c_mask].tolist())
                    all_targets_c.extend(ctr_target[c_mask].tolist())
                    all_playtime_preds_c.extend(playtime_pred[c_mask].tolist())
                    all_playtime_targets_c.extend(playtime_target[c_mask].tolist())
                    
                    # 计算非零播放时间的相对误差
                    for i, (target, pred) in enumerate(zip(playtime_target[c_mask], playtime_pred[c_mask])):
                        if target > 0:
                            relative_error = abs(pred - target) / target
                            all_nonzero_playtime_c.append(relative_error)
        
        # 使用辅助方法处理评估指标
        t_metrics, c_metrics, is_best = self._process_evaluation_metrics(
            t_ctr_losses, t_playtime_losses, c_ctr_losses, c_playtime_losses,
            all_preds_t, all_targets_t, all_playtime_preds_t, all_playtime_targets_t,
            all_nonzero_playtime_t, all_preds_c, all_targets_c, all_playtime_preds_c, 
            all_playtime_targets_c, all_nonzero_playtime_c
        )
        
        return {'val': {'treatment': t_metrics, 'control': c_metrics, 'is_best': is_best}}
        
        return {'val': {'treatment': t_metrics, 'control': c_metrics, 'is_best': is_best}}
        
    def _process_evaluation_metrics(self, 
                                t_ctr_losses, t_playtime_losses, 
                                c_ctr_losses, c_playtime_losses,
                                all_preds_t, all_targets_t, 
                                all_playtime_preds_t, all_playtime_targets_t,
                                all_nonzero_playtime_t,
                                all_preds_c, all_targets_c, 
                                all_playtime_preds_c, all_playtime_targets_c,
                                all_nonzero_playtime_c):
        """
        处理评估指标
        
        Args:
            t_ctr_losses: 处理组CTR损失列表
            t_playtime_losses: 处理组播放时间损失列表
            c_ctr_losses: 对照组CTR损失列表
            c_playtime_losses: 对照组播放时间损失列表
            all_preds_t: 处理组预测值
            all_targets_t: 处理组目标值
            all_playtime_preds_t: 处理组播放时间预测值
            all_playtime_targets_t: 处理组播放时间目标值
            all_nonzero_playtime_t: 处理组非零播放时间相对误差
            all_preds_c: 对照组预测值
            all_targets_c: 对照组目标值
            all_playtime_preds_c: 对照组播放时间预测值
            all_playtime_targets_c: 对照组播放时间目标值
            all_nonzero_playtime_c: 对照组非零播放时间相对误差
            
        Returns:
            处理组指标、对照组指标和是否为最佳模型的元组
        """
        # 计算处理组指标
        t_metrics = {'total_loss': 0.0, 'ctr_loss': 0.0, 'playtime_loss': 0.0}
        if t_ctr_losses:
            t_metrics['ctr_loss'] = np.mean(t_ctr_losses)
            t_metrics['playtime_loss'] = np.mean(t_playtime_losses)
            t_metrics['total_loss'] = (self.lambda_ctr * t_metrics['ctr_loss'] + 
                                      self.lambda_playtime * t_metrics['playtime_loss'])
        
        # 计算对照组指标
        c_metrics = {'total_loss': 0.0, 'ctr_loss': 0.0, 'playtime_loss': 0.0}
        if c_ctr_losses:
            c_metrics['ctr_loss'] = np.mean(c_ctr_losses)
            c_metrics['playtime_loss'] = np.mean(c_playtime_losses)
            c_metrics['total_loss'] = (self.lambda_ctr * c_metrics['ctr_loss'] + 
                                      self.lambda_playtime * c_metrics['playtime_loss'])
        
        # 计算AUC和RMSE
        if all_targets_t and len(set(all_targets_t)) > 1:
            t_metrics['ctr_auc'] = roc_auc_score(all_targets_t, all_preds_t)
        else:
            t_metrics['ctr_auc'] = 0.5
            
        if all_playtime_targets_t:
            t_metrics['playtime_rmse'] = np.sqrt(mean_squared_error(all_playtime_targets_t, all_playtime_preds_t))
            # 添加非零播放时间的平均相对误差
            if all_nonzero_playtime_t:
                t_metrics['playtime_rel_error'] = np.mean(all_nonzero_playtime_t) * 100  # 转为百分比
            else:
                t_metrics['playtime_rel_error'] = 0.0
        else:
            t_metrics['playtime_rmse'] = 0.0
            t_metrics['playtime_rel_error'] = 0.0
            
        if all_targets_c and len(set(all_targets_c)) > 1:
            c_metrics['ctr_auc'] = roc_auc_score(all_targets_c, all_preds_c)
        else:
            c_metrics['ctr_auc'] = 0.5
            
        if all_playtime_targets_c:
            c_metrics['playtime_rmse'] = np.sqrt(mean_squared_error(all_playtime_targets_c, all_playtime_preds_c))
            # 添加非零播放时间的平均相对误差
            if all_nonzero_playtime_c:
                c_metrics['playtime_rel_error'] = np.mean(all_nonzero_playtime_c) * 100  # 转为百分比
            else:
                c_metrics['playtime_rel_error'] = 0.0
        else:
            c_metrics['playtime_rmse'] = 0.0
            c_metrics['playtime_rel_error'] = 0.0
        
        # 更新学习率
        if t_metrics and 'total_loss' in t_metrics:
            self.scheduler.step(t_metrics['total_loss'])
        
        # 检查是否是最佳模型
        is_best = False
        if t_metrics and 'ctr_auc' in t_metrics and t_metrics['ctr_auc'] > self.best_val_auc:
            self.best_val_auc = t_metrics['ctr_auc']
            is_best = True
            self.patience_counter = 0
        elif t_metrics and 'total_loss' in t_metrics and t_metrics['total_loss'] < self.best_val_loss:
            self.best_val_loss = t_metrics['total_loss']
            is_best = True
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return t_metrics, c_metrics, is_best


class WeightTrainer(BaseTrainer):
    """权重模型训练器"""
    
    def __init__(self, 
                 weight_model: WeightModel,
                 device: str = 'cuda',
                 lr: float = 0.001,
                 weight_decay: float = 1e-5,
                 clip_grad: Optional[float] = 1.0,
                 logger: Logger = None):
        """
        初始化权重模型训练器
        
        Args:
            weight_model: 权重模型
            device: 训练设备
            lr: 学习率
            weight_decay: 权重衰减
            clip_grad: 梯度裁剪值，None表示不裁剪
            logger: 日志记录器
        """
        super().__init__(device, logger)
        
        self.weight_model = weight_model.to(self.device)
        self.optimizer = optim.Adam(
            self.weight_model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.criterion = BCEWithLogitsLoss()
        self.clip_grad = clip_grad
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        执行一步权重模型训练
        
        Args:
            batch: 批次数据
            
        Returns:
            训练指标字典
        """
        self.weight_model.train()
        
        # 将数据移动到设备
        features = {k: v.to(self.device) for k, v in batch['features'].items()}
        group = batch['group'].to(self.device).float()  # treatment=1, control=0
        
        # 从字典中提取特征张量
        if isinstance(features, dict):
            # 如果特征是字典，使用user_features
            feature_tensor = features.get('user_features', next(iter(features.values())))
        else:
            # 否则直接使用特征
            feature_tensor = features
            
        # 前向传播
        weight_logits = self.weight_model(feature_tensor)
        
        # 计算损失
        loss = self.criterion(weight_logits, group)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.weight_model.parameters(), self.clip_grad)
        
        # 参数更新
        self.optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            pred = (torch.sigmoid(weight_logits) > 0.5).float()
            accuracy = (pred == group).float().mean().item()
        
        return {'train': {'weight_model': {'loss': loss.item(), 'accuracy': accuracy}}}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        在验证集上评估权重模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            评估指标字典
        """
        self.weight_model.eval()
        
        all_losses = []
        all_accuracies = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移动到设备
                features = {k: v.to(self.device) for k, v in batch['features'].items()}
                group = batch['group'].to(self.device).float()
                
                # 从字典中提取特征张量
                if isinstance(features, dict):
                    # 如果特征是字典，使用user_features
                    feature_tensor = features.get('user_features', next(iter(features.values())))
                else:
                    # 否则直接使用特征
                    feature_tensor = features
                    
                # 前向传播
                weight_logits = self.weight_model(feature_tensor)
                
                # 计算损失
                loss = self.criterion(weight_logits, group)
                all_losses.append(loss.item())
                
                # 计算准确率
                pred = (torch.sigmoid(weight_logits) > 0.5).float()
                accuracy = (pred == group).float().mean().item()
                all_accuracies.append(accuracy)
                
                # 收集预测和目标
                all_preds.extend(torch.sigmoid(weight_logits).cpu().numpy().tolist())
                all_targets.extend(group.cpu().numpy().tolist())
        
        # 计算平均指标
        avg_loss = np.mean(all_losses)
        avg_accuracy = np.mean(all_accuracies)
        
        # 计算AUC
        if len(set(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
        
        # 检查是否是最佳模型
        is_best = False
        if avg_accuracy > self.best_val_acc:
            self.best_val_acc = avg_accuracy
            is_best = True
        elif avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            is_best = True
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'auc': auc,
            'is_best': is_best
        }
        
        return {'val': {'weight_model': metrics}}
