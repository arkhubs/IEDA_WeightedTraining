"""
多标签预测模型
管理多个独立的预测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Tuple, Any
from .mlp_model import MLPModel
from .loss_functions import get_loss_function

logger = logging.getLogger(__name__)

class MultiLabelModel:
    """多标签预测模型管理器"""
    
    def __init__(self, config: Dict, input_dim: int, device: torch.device):
        self.config = config
        self.input_dim = input_dim
        self.device = device
        self.labels = config['labels']
        
        # 为每个标签创建独立的模型
        self.models = {}
        self.optimizers = {}
        self.loss_functions = {}
        self.schedulers = {}
        
        self._build_models()
        
    def _build_models(self):
        """构建所有标签的模型"""
        logger.info("[模型构建] 开始构建多标签预测模型...")
        
        for label_config in self.labels:
            label_name = label_config['name']
            logger.info(f"[模型构建] 构建 {label_name} 模型...")
            
            # 创建模型并移动到指定设备
            model = MLPModel(
                input_dim=self.input_dim,
                hidden_layers=label_config['model_params']['hidden_layers'],
                output_dim=1,
                dropout=label_config['model_params']['dropout']
            ).to(self.device)
            
            # 创建优化器
            optimizer = optim.Adam(
                model.parameters(),
                lr=label_config['learning_rate'],
                weight_decay=label_config['weight_decay']
            )
            
            # 创建损失函数
            loss_fn = get_loss_function(label_config['loss_function'])
            
            # 创建学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            self.models[label_name] = model
            self.optimizers[label_name] = optimizer
            self.loss_functions[label_name] = loss_fn
            self.schedulers[label_name] = scheduler
            
            # 打印模型信息
            model_info = model.get_model_info()
            logger.info(f"[模型构建] {label_name} 模型: {model_info['total_params']} 参数")
        
        logger.info(f"[模型构建] 多标签模型构建完成，共 {len(self.models)} 个模型")
    
    def forward(self, x: torch.Tensor, label_name: str) -> torch.Tensor:
        """单个标签的前向传播"""
        if label_name not in self.models:
            raise ValueError(f"标签 {label_name} 的模型不存在")
        
        return self.models[label_name](x)

    def set_train_mode(self):
        """将所有模型设置为训练模式"""
        for model in self.models.values():
            model.train()

    def set_eval_mode(self):
        """将所有模型设置为评估模式"""
        for model in self.models.values():
            model.eval()
    
    def predict_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测所有标签"""
        self.set_eval_mode()
        predictions = {}
        
        with torch.no_grad():
            for label_name in self.models:
                pred = self.forward(x, label_name)
                
                # 根据标签类型处理输出
                label_config = next(lc for lc in self.labels if lc['name'] == label_name)
                if label_config['type'] == 'binary':
                    pred = torch.sigmoid(pred)
                elif label_config['type'] == 'numerical':
                    pred = torch.clamp(pred, min=0)  # 确保非负
                
                predictions[label_name] = pred
        
        return predictions
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测方法 - 与predict_all相同，保持接口兼容性"""
        return self.predict_all(x)
    
    def compute_losses(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算所有标签的损失"""
        losses = {}
        
        for label_name in self.models:
            if label_name in targets:
                pred = self.forward(x, label_name)
                target = targets[label_name]
                loss = self.loss_functions[label_name](pred, target)
                losses[label_name] = loss
        
        return losses
    
    def train_step(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        # 设置为训练模式
        self.set_train_mode()
        
        # 计算损失并更新模型
        losses = {}
        for label_name in self.models:
            if label_name in targets:
                # 清零梯度
                self.optimizers[label_name].zero_grad()
                
                # 前向传播
                pred = self.forward(x, label_name)
                target = targets[label_name]
                
                # 计算损失
                loss = self.loss_functions[label_name](pred, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.models[label_name].parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizers[label_name].step()
                
                losses[label_name] = loss.item()
        
        return losses
    
    def evaluate(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """评估模型，返回每个标签的损失值"""
        self.set_eval_mode()
        with torch.no_grad():
            losses = self.compute_losses(x, targets)
            return {name: loss.item() for name, loss in losses.items()}
    
    def get_combined_score(self, x: torch.Tensor, alpha_weights: Dict[str, float]) -> torch.Tensor:
        """根据alpha权重计算组合分数"""
        predictions = self.predict_all(x)
        
        combined_score = torch.zeros(x.size(0), 1, device=self.device)
        
        for label_name, alpha in alpha_weights.items():
            if label_name in predictions:
                pred = predictions[label_name]
                combined_score += alpha * pred
        
        return combined_score
    
    def save_models(self, save_dir: str, step_or_epoch_name):
        """保存所有模型，支持步骤数字或epoch名称"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'step_or_epoch': step_or_epoch_name,
            'config': self.config,
            'input_dim': self.input_dim
        }
        
        for label_name in self.models:
            checkpoint[f'{label_name}_model'] = self.models[label_name].state_dict()
            checkpoint[f'{label_name}_optimizer'] = self.optimizers[label_name].state_dict()
            checkpoint[f'{label_name}_scheduler'] = self.schedulers[label_name].state_dict()
        
        # 根据参数类型决定文件名
        if isinstance(step_or_epoch_name, int):
            save_path = os.path.join(save_dir, f'step_{step_or_epoch_name}.pt')
        else:
            save_path = os.path.join(save_dir, f'{step_or_epoch_name}.pt')
        
        torch.save(checkpoint, save_path)
        logger.info(f"[模型保存] 模型已保存到: {save_path}")
    
    def load_models(self, checkpoint_path: str):
        """加载所有模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        for label_name in self.models:
            if f'{label_name}_model' in checkpoint:
                self.models[label_name].load_state_dict(checkpoint[f'{label_name}_model'])
            if f'{label_name}_optimizer' in checkpoint:
                self.optimizers[label_name].load_state_dict(checkpoint[f'{label_name}_optimizer'])
            if f'{label_name}_scheduler' in checkpoint:
                self.schedulers[label_name].load_state_dict(checkpoint[f'{label_name}_scheduler'])
        
        logger.info(f"[模型加载] 模型已从 {checkpoint_path} 加载")
        return checkpoint.get('step_or_epoch', checkpoint.get('step', 0))
    
    def update_schedulers(self, metrics: Dict[str, float]):
        """更新学习率调度器"""
        for label_name, metric in metrics.items():
            if label_name in self.schedulers:
                self.schedulers[label_name].step(metric)