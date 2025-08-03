import os
import torch
import numpy as np
import pandas as pd
import logging
from torch.optim import Adam

class Trainer:
    """训练器类，负责模型的预训练和实验训练"""
    
    def __init__(self, config_manager, data_manager, models, device):
        """
        初始化训练器
        
        Args:
            config_manager: 配置管理器
            data_manager: 数据管理器
            models: 模型字典，键为标签名称
            device: 计算设备
        """
        self.config = config_manager.get_config()
        self.pretrain_config = config_manager.get_pretrain_config()
        self.global_config = config_manager.get_global_config()
        self.exp_dir = config_manager.get_exp_dir()
        self.checkpoints_dir = config_manager.get_checkpoints_dir()
        
        self.data_manager = data_manager
        self.models = models
        self.device = device
        
        self.logger = logging.getLogger('Trainer')
        
        # 获取标签配置
        self.label_configs = config_manager.get_label_info()
        
    def _get_label_specific_config(self, label_name):
        """获取特定标签的配置"""
        for label_config in self.label_configs:
            if label_config.get('name') == label_name:
                return label_config
                
        return {}
        
    def create_criterion(self, label_name=None):
        """
        创建损失函数
        
        Args:
            label_name: 标签名称，用于获取特定标签的损失函数
            
        Returns:
            损失函数实例
        """
        from libs.src.models import LogMAELoss
        
        # 获取标签类型
        if label_name:
            label_config = self._get_label_specific_config(label_name)
            label_type = label_config.get('type', 'numerical')
        else:
            # 兼容旧版本
            label_type = self.config.get('label', {}).get('type', 'numerical')
        
        # 根据标签类型选择合适的损失函数
        if label_type == 'binary':
            return torch.nn.BCEWithLogitsLoss()
        else:  # numerical
            return LogMAELoss()
    
    def create_optimizer(self, model, label_name=None):
        """
        创建优化器
        
        Args:
            model: 模型实例
            label_name: 标签名称，用于获取特定标签的优化器参数
            
        Returns:
            优化器实例
        """
        # 获取学习率和权重衰减
        lr = None
        weight_decay = None
        
        # 优先使用标签特定的配置
        if label_name:
            label_config = self._get_label_specific_config(label_name)
            lr = label_config.get('learning_rate')
            weight_decay = label_config.get('weight_decay')
        
        # 如果标签特定配置中没有设置，使用全局配置
        if lr is None:
            if self.config['mode'] == 'global':
                lr = self.global_config.get('learning_rate', 0.001)
            else:
                lr = self.pretrain_config.get('learning_rate', 0.001)
                
        if weight_decay is None:
            if self.config['mode'] == 'global':
                weight_decay = self.global_config.get('weight_decay', 0.0001)
            else:
                weight_decay = self.pretrain_config.get('weight_decay', 0.0001)
            
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def prepare_batch(self, interactions):
        """
        准备批次数据
        
        Args:
            interactions: 交互数据批次
            
        Returns:
            特征和标签
        """
        # 获取特征和标签
        numerical_features = []
        categorical_features = []
        labels = []
        
        for interaction in interactions:
            numerical_features.append(interaction['numerical_features'])
            categorical_features.append(interaction['categorical_features'])
            labels.append(interaction['label'])
        
        # 转换为张量
        numerical_tensor = torch.tensor(np.stack(numerical_features), device=self.device)
        categorical_tensor = torch.tensor(np.stack(categorical_features), device=self.device)
        label_tensor = torch.tensor(labels, device=self.device)
        
        return {
            'numerical_features': numerical_tensor,
            'categorical_features': categorical_tensor,
            'labels': label_tensor
        }
    
    def pretrain(self, criteria, optimizers):
        """
        预训练模型
        
        Args:
            criteria: 损失函数字典，键为标签名称
            optimizers: 优化器字典，键为标签名称
        """
        # 检查是否需要预训练
        if not self.pretrain_config.get('enabled', False):
            self.logger.info("[Pretrain] 预训练已禁用，跳过预训练阶段")
            return
        
        batch_size = self.pretrain_config['batch_size']
        epochs = self.pretrain_config['epochs']
        early_stopping = self.pretrain_config.get('early_stopping', 5)
        
        self.logger.info(f"[Pretrain] 开始预训练: epochs={epochs}, batch_size={batch_size}")
        
        # 为每个标签分别预训练模型
        for label_name, model in self.models.items():
            self.logger.info(f"[Pretrain] 预训练{label_name}模型")
            
            # 获取该标签的损失函数和优化器
            criterion = criteria.get(label_name)
            optimizer = optimizers.get(label_name)
            
            if criterion is None or optimizer is None:
                self.logger.warning(f"[Pretrain] {label_name}模型缺少损失函数或优化器，跳过预训练")
                continue
            
            # 记录最佳损失和无改善计数
            best_loss = float('inf')
            no_improve_count = 0
            
            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                batches = 0
                
                # 获取训练用户
                train_users = self.data_manager.get_train_users()
                
                # 按批次训练
                for i in range(0, len(train_users), batch_size):
                    batch_end = min(i + batch_size, len(train_users))
                    batch_users = train_users[i:batch_end]
                    
                    # 为每个用户收集训练样本
                    batch_user_ids = []
                    batch_video_ids = []
                    batch_labels = []
                    
                    for user_id in batch_users:
                        user_videos = self.data_manager.get_user_videos(user_id)
                        for video_data in user_videos:
                            batch_user_ids.append(user_id)
                            batch_video_ids.append(video_data['video_id'])
                            batch_labels.append(video_data.get(label_name, 0))
                    
                    if not batch_user_ids:
                        continue
                    
                    # 准备特征
                    feature_dict = self.data_manager.prepare_batch_features(batch_user_ids, batch_video_ids)
                    
                    if feature_dict is None:
                        continue
                    
                    # 获取数值特征和分类特征
                    numerical_features = feature_dict['numerical_features']
                    categorical_features = feature_dict['categorical_features']
                    
                    # 检查输入数据
                    if np.isnan(numerical_features).any() or np.isinf(numerical_features).any():
                        print(f"[Pretrain] 数值特征包含nan或inf，跳过此批次")
                        continue
                    
                    if np.isnan(categorical_features).any() or np.isinf(categorical_features).any():
                        print(f"[Pretrain] 分类特征包含nan或inf，跳过此批次") 
                        continue
                    
                    if np.isnan(batch_labels).any() or np.isinf(batch_labels).any():
                        print(f"[Pretrain] 标签包含nan或inf，跳过此批次")
                        continue
                    
                    # 转换为张量
                    X_num = torch.FloatTensor(numerical_features).to(self.device)
                    X_cat = torch.LongTensor(categorical_features).to(self.device)
                    y = torch.FloatTensor(batch_labels).to(self.device)
                    
                    # 检查张量
                    if torch.isnan(X_num).any() or torch.isinf(X_num).any():
                        print(f"[Pretrain] X_num张量包含nan或inf，跳过此批次")
                        continue
                        
                    if torch.isnan(X_cat.float()).any() or torch.isinf(X_cat.float()).any():
                        print(f"[Pretrain] X_cat张量包含nan或inf，跳过此批次")
                        continue
                        
                    if torch.isnan(y).any() or torch.isinf(y).any():
                        print(f"[Pretrain] y张量包含nan或inf，跳过此批次")
                        continue
                    
                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(X_num, X_cat).squeeze()
                    
                    # 检查模型输出是否包含nan
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"[Pretrain] 模型输出包含nan或inf，跳过此批次")
                        continue
                    
                    loss = criterion(outputs, y)
                    
                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[Pretrain] 损失为nan或inf，跳过此批次")
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batches += 1
                
                avg_loss = epoch_loss / max(1, batches)
                
                self.logger.info(f"[Pretrain] {label_name} epoch {epoch}/{epochs}, loss={avg_loss:.4f}")
                
                # 早停检查
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve_count = 0
                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'label_name': label_name
                    }, os.path.join(self.checkpoints_dir, f"pretrain_best_{label_name}.pt"))
                else:
                    no_improve_count += 1
                
                if no_improve_count >= early_stopping:
                    self.logger.info(f"[Pretrain] {label_name} 连续 {early_stopping} 个epoch没有改善，提前停止")
                    break
            
            self.logger.info(f"[Pretrain] {label_name}模型预训练完成，最佳损失: {best_loss:.4f}")
            
            # 加载最佳模型
            checkpoint_path = os.path.join(self.checkpoints_dir, f"pretrain_best_{label_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info(f"[Pretrain] 已加载{label_name}最佳预训练模型，epoch={checkpoint['epoch']}")
            
    def resume_training(self, recommender, optimizers, checkpoint=None):
        """
        恢复训练
        
        Args:
            recommender: 推荐器实例
            optimizers: 优化器字典，键为标签名称
            checkpoint: 检查点路径，默认为None表示寻找最新的检查点
            
        Returns:
            是否成功恢复
        """
        # 如果没有指定检查点目录，则尝试查找最新的检查点
        if checkpoint is None:
            checkpoints_by_label = {}
            
            # 对每个标签寻找最新的检查点
            for label_name in self.models.keys():
                label_checkpoints = [
                    f for f in os.listdir(self.checkpoints_dir) 
                    if f.startswith(f"{label_name}_step_") and f.endswith(".pt")
                ]
                
                if label_checkpoints:
                    # 按步数排序
                    label_checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
                    checkpoints_by_label[label_name] = os.path.join(self.checkpoints_dir, label_checkpoints[0])
            
            # 如果没有找到任何检查点，则检查是否有旧格式的检查点
            if not checkpoints_by_label:
                old_format_checkpoints = [
                    f for f in os.listdir(self.checkpoints_dir) 
                    if f.startswith("step_") and f.endswith(".pt")
                ]
                
                if old_format_checkpoints:
                    # 按步数排序
                    old_format_checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
                    checkpoint = os.path.join(self.checkpoints_dir, old_format_checkpoints[0])
                    self.logger.info(f"[Resume] 找到旧格式检查点: {checkpoint}")
                    
                    # 加载检查点
                    return recommender.load_checkpoint(optimizers, checkpoint)
                else:
                    self.logger.info("[Resume] 没有找到可恢复的检查点，将开始新训练")
                    return False
            
            # 对每个模型分别恢复
            success = True
            for label_name, checkpoint_path in checkpoints_by_label.items():
                self.logger.info(f"[Resume] 尝试恢复{label_name}模型: {checkpoint_path}")
                optimizer = optimizers.get(label_name)
                
                if optimizer:
                    if not recommender.load_checkpoint(optimizer, checkpoint_path, label_name):
                        success = False
                        self.logger.warning(f"[Resume] 恢复{label_name}模型失败")
                
            return success
        else:
            # 使用指定的检查点路径
            return recommender.load_checkpoint(optimizers, checkpoint)
