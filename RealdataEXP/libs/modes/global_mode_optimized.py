"""
Global模式优化版本
解决GPU利用率低和训练速度慢的问题
"""

import os
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from ..data import KuaiRandDataLoader, FeatureProcessor
from ..models import MultiLabelModel
from ..utils import MetricsTracker, save_results
import random
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    # 旧版本PyTorch兼容
    from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class OptimizedTensorDataset(TensorDataset):
    """优化的TensorDataset，支持GPU加速"""
    
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

class GlobalModeOptimized:
    """Global模式优化实验管理器"""
    
    def __init__(self, config: Dict, exp_dir: str):
        self.config = config
        self.exp_dir = exp_dir
        
        # 设备选择逻辑
        device_config = config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                logger.warning("[设备配置] CUDA不可用，退回到CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_config)
        
        # 混合精度训练
        self.use_amp = torch.cuda.is_available()
        # 修复AMP deprecation警告 - 兼容不同PyTorch版本
        if self.use_amp:
            try:
                # 新版本PyTorch (2.0+)
                self.scaler = GradScaler(device='cuda')
            except TypeError:
                # 旧版本PyTorch
                self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # 初始化组件
        self.data_loader = KuaiRandDataLoader(config)
        self.feature_processor = FeatureProcessor(config)
        self.multi_label_model = None
        
        # 数据存储
        self.merged_data = None
        self.user_video_lists = None
        self.train_users = None
        self.val_users = None
        self.processed_data = None
        
        # 预处理的GPU张量缓存
        self.cached_features = None
        self.cached_targets = None
        self.cached_indices = None
        
        # 数据预加载锁
        self.data_lock = threading.Lock()
        
        # 批处理池
        self.batch_pool_size = 10  # 预处理批次数
        self.feature_batch_pool = []
        self.target_batch_pool = []
        
        # 仿真状态
        self.used_videos = set()  # 记录已使用的视频
        
        # 指标跟踪
        self.metrics_tracker = MetricsTracker()
        
        # 仿真结果
        self.total_label_T = {label['name']: 0.0 for label in config['labels']}  # Treatment组累计收益
        self.total_label_C = {label['name']: 0.0 for label in config['labels']}  # Control组累计收益
        
        logger.info(f"[Global模式优化] 初始化完成，设备: {self.device}, AMP: {self.use_amp}")
    
    def preprocess_data_to_tensors(self, data: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """将数据预处理为GPU张量，避免重复转换"""
        logger.info("[数据预处理] 开始将数据转换为GPU张量...")
        
        feature_columns = self.feature_processor.get_feature_columns()
        
        # 特征张量
        feature_data = data[feature_columns].copy()
        # 处理缺失值和数据类型 - 确保所有数据都是数值类型
        for col in feature_columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
        
        # 验证数据类型
        for col in feature_columns:
            if feature_data[col].dtype == 'object':
                logger.warning(f"[数据预处理] 列 {col} 仍然是object类型，强制转换")
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
        
        # 转换为numpy数组，确保数据类型正确
        features_array = feature_data.values.astype(np.float32)
        features_tensor = torch.FloatTensor(features_array).to(self.device)
        
        # 标签张量
        targets = {}
        for label_config in self.config['labels']:
            label_name = label_config['name']
            target_col = label_config['target']
            
            # 确保标签数据也是数值类型
            target_values = pd.to_numeric(data[target_col], errors='coerce').fillna(0.0)
            target_array = target_values.values.astype(np.float32)
            targets[label_name] = torch.FloatTensor(target_array).to(self.device)
        
        logger.info(f"[数据预处理] 特征张量形状: {features_tensor.shape}")
        logger.info(f"[数据预处理] 张量已存储到设备: {self.device}")
        
        return features_tensor, targets
    
    def create_optimized_dataloader(self, data: pd.DataFrame, batch_size: int, shuffle: bool = True) -> DataLoader:
        """创建优化的DataLoader"""
        # 为了避免CUDA多进程问题，我们不在worker进程中使用CUDA张量
        # 而是在主进程中处理数据，然后在训练时转移到GPU
        
        feature_columns = self.feature_processor.get_feature_columns()
        
        # 特征数据 - 保持为numpy数组
        feature_data = data[feature_columns].copy()
        for col in feature_columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
        
        # 转换为numpy数组
        features_array = feature_data.values.astype(np.float32)
        
        # 标签数据 - 保持为numpy数组
        targets_list = []
        for label_config in self.config['labels']:
            label_name = label_config['name']
            target_col = label_config['target']
            
            target_values = pd.to_numeric(data[target_col], errors='coerce').fillna(0.0)
            target_array = target_values.values.astype(np.float32)
            targets_list.append(target_array)
        
        # 创建TensorDataset，使用CPU张量
        features_tensor = torch.FloatTensor(features_array)
        target_tensors = [torch.FloatTensor(target) for target in targets_list]
        
        dataset = OptimizedTensorDataset(features_tensor, *target_tensors)
        
        # 创建DataLoader，使用多线程但不使用CUDA在worker进程中
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,  # 使用4个工作线程
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,  # 保持工作进程
            prefetch_factor=2  # 预取因子
        )
        
        return dataloader
    
    def load_and_prepare_data(self):
        """加载和准备数据"""
        logger.info("[Global模式优化] 开始数据加载和准备...")
        
        # 加载原始数据
        self.merged_data, self.user_video_lists, self.train_users, self.val_users = \
            self.data_loader.load_and_prepare_data()
        
        # 打印数据集统计信息
        stats = self.data_loader.get_dataset_stats()
        logger.info(f"[数据统计] 总样本数: {stats['total_samples']}")
        logger.info(f"[数据统计] 唯一用户数: {stats['unique_users']}")
        logger.info(f"[数据统计] 唯一视频数: {stats['unique_videos']}")
        logger.info(f"[数据统计] 训练用户数: {stats['train_users']}")
        logger.info(f"[数据统计] 验证用户数: {stats['val_users']}")
        logger.info(f"[数据统计] 点击率: {stats['click_rate']:.4f}")
        logger.info(f"[数据统计] 平均播放时长: {stats['avg_play_time']:.2f}ms")
        
        # 特征处理
        logger.info("[特征处理] 开始特征预处理...")
        self.processed_data = self.feature_processor.fit_transform(self.merged_data)
        
        # 获取特征列
        feature_columns = self.feature_processor.get_feature_columns()
        input_dim = len(feature_columns)
        logger.info(f"[特征处理] 特征维度: {input_dim}")
        
        # 初始化多标签模型
        self.multi_label_model = MultiLabelModel(
            config=self.config,
            input_dim=input_dim,
            device=self.device
        )
        
        logger.info("[Global模式优化] 数据准备完成")
    
    def pretrain_models_optimized(self):
        """优化的预训练模型"""
        if not self.config['pretrain']['enabled']:
            logger.info("[预训练] 跳过预训练阶段")
            return
        
        logger.info("[预训练优化] 开始优化预训练阶段...")
        
        # 准备训练数据
        train_data = self.processed_data[self.processed_data['mask'] == 0].copy()
        logger.info(f"[预训练优化] 训练数据量: {len(train_data)}")
        
        # 创建优化的DataLoader
        batch_size = min(self.config['pretrain']['batch_size'] * 4, 512)  # 增大batch_size
        logger.info(f"[预训练优化] 使用批次大小: {batch_size}")
        
        train_dataloader = self.create_optimized_dataloader(train_data, batch_size, shuffle=True)
        
        for epoch in range(self.config['pretrain']['epochs']):
            logger.info(f"[预训练优化] Epoch {epoch+1}/{self.config['pretrain']['epochs']}")
            
            epoch_losses = {label['name']: [] for label in self.config['labels']}
            
            # 批次训练
            for batch_idx, batch in enumerate(train_dataloader):
                if len(batch) < 2:  # 确保有特征和至少一个标签
                    continue
                
                features = batch[0]  # 第一个是特征
                
                # 构建targets字典
                targets = {}
                for i, label_config in enumerate(self.config['labels']):
                    label_name = label_config['name']
                    targets[label_name] = batch[i + 1].unsqueeze(1)  # 从batch[1]开始是标签
                
                # 将数据转移到GPU
                features = features.to(self.device)
                targets = {name: tensor.to(self.device) for name, tensor in targets.items()}
                
                # 使用混合精度训练
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        losses = self.multi_label_model.train_step(features, targets)
                else:
                    losses = self.multi_label_model.train_step(features, targets)
                
                for label_name, loss in losses.items():
                    epoch_losses[label_name].append(loss)
                
                # 定期打印进度
                if batch_idx % 100 == 0:
                    logger.info(f"[预训练优化] Epoch {epoch+1}, Batch {batch_idx}, 样本数: {len(features)}")
            
            # 记录epoch结果
            avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items() if losses}
            loss_str = ", ".join([f"{name}: {loss:.6f}" for name, loss in avg_losses.items()])
            logger.info(f"[预训练优化] Epoch {epoch+1} 平均损失 - {loss_str}")
            
            # 在每个epoch结束时计算预测指标
            logger.info(f"[预训练优化] Epoch {epoch+1} 计算预测指标...")
            val_metrics = self.compute_pretrain_metrics()
            for metric_name, value in val_metrics.items():
                if 'accuracy' in metric_name or 'relative_error' in metric_name:
                    if 'accuracy' in metric_name:
                        logger.info(f"[预训练优化] {metric_name}: {value:.4f}")
                    elif 'relative_error' in metric_name:
                        logger.info(f"[预训练优化] {metric_name}: {value:.4f}")
            
            # GPU内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("[预训练优化] 优化预训练阶段完成")
    
    def compute_pretrain_metrics(self):
        """计算预训练阶段的预测指标"""
        # 使用训练数据的子集进行快速评估
        train_data = self.processed_data[self.processed_data['mask'] == 0].copy()
        
        # 采样少量数据用于指标计算
        sample_size = min(1000, len(train_data))
        if sample_size == 0:
            return {}
        
        sample_data = train_data.sample(sample_size, random_state=42)
        
        # 创建小批量DataLoader
        feature_columns = self.feature_processor.get_feature_columns()
        features_array = sample_data[feature_columns].values.astype(np.float32)
        
        targets_list = []
        for label_config in self.config['labels']:
            label_name = label_config['name']
            target_col = label_config['target']
            target_values = sample_data[target_col].values.astype(np.float32)
            targets_list.append(target_values)
        
        features_tensor = torch.FloatTensor(features_array).to(self.device)
        
        # 构建targets字典
        targets = {}
        for i, label_config in enumerate(self.config['labels']):
            label_name = label_config['name']
            targets[label_name] = torch.FloatTensor(targets_list[i]).unsqueeze(1).to(self.device)
        
        # 计算指标
        with torch.no_grad():
            metrics = self.multi_label_model.evaluate_with_metrics(features_tensor, targets)
        
        return metrics
    
    def sample_candidate_videos(self, user_id: int, n_candidate: int) -> List[int]:
        """为用户采样候选视频"""
        if user_id not in self.user_video_lists:
            return []
        
        # 获取用户的所有视频
        user_videos = self.user_video_lists[user_id]
        
        # 筛选可用视频（mask=0 且 used=0）
        available_videos = []
        for video_id in user_videos:
            video_data = self.processed_data[
                (self.processed_data['user_id'] == user_id) & 
                (self.processed_data['video_id'] == video_id)
            ]
            
            if len(video_data) > 0:
                mask = video_data.iloc[0]['mask']
                if mask == 0 and video_id not in self.used_videos:
                    available_videos.append(video_id)
        
        # 随机采样
        if len(available_videos) <= n_candidate:
            return available_videos
        else:
            return random.sample(available_videos, n_candidate)
    
    def get_user_video_features_optimized(self, user_id: int, video_ids: List[int]) -> torch.Tensor:
        """优化的获取用户-视频对特征"""
        if not video_ids:
            feature_columns = self.feature_processor.get_feature_columns()
            return torch.empty(0, len(feature_columns), device=self.device)
        
        # 批量查询
        query_condition = (
            (self.processed_data['user_id'] == user_id) & 
            (self.processed_data['video_id'].isin(video_ids))
        )
        batch_data = self.processed_data[query_condition]
        
        if len(batch_data) == 0:
            feature_columns = self.feature_processor.get_feature_columns()
            return torch.empty(0, len(feature_columns), device=self.device)
        
        # 预处理特征
        feature_columns = self.feature_processor.get_feature_columns()
        feature_data = batch_data[feature_columns].copy()
        
        # 调试信息：检查数据类型
        object_columns = []
        for col in feature_columns:
            if col in feature_data.columns and feature_data[col].dtype == 'object':
                object_columns.append(col)
        
        if object_columns:
            logger.warning(f"[特征处理] 发现object类型列: {object_columns}")
            for col in object_columns:
                logger.warning(f"[特征处理] 列 {col} 的数据类型: {feature_data[col].dtype}, 示例值: {feature_data[col].head().tolist()}")
        
        # 确保所有特征列都是数值类型 - 增强处理
        for col in feature_columns:
            if col in feature_data.columns:
                # 检查数据类型
                if feature_data[col].dtype == 'object':
                    logger.warning(f"[特征处理] 列 {col} 是object类型，强制转换")
                    # 尝试转换为数值
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                    # 如果仍有缺失值，填充0
                    feature_data[col] = feature_data[col].fillna(0.0)
                else:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
            else:
                logger.warning(f"[特征处理] 特征列 {col} 不在数据中，用0填充")
                feature_data[col] = 0.0
        
        # 确保特征数据与候选视频数一致
        if len(feature_data) != len(video_ids):
            logger.warning(f"[特征处理] 特征数据行数 {len(feature_data)} 与候选视频数 {len(video_ids)} 不匹配")
            # 为缺失的视频创建默认特征
            if len(feature_data) < len(video_ids):
                # 找出缺失的视频ID
                found_videos = set(batch_data['video_id'].unique())
                missing_videos = [vid for vid in video_ids if vid not in found_videos]
                
                # 为缺失的视频创建默认特征行
                default_features = {col: 0.0 for col in feature_columns}
                for missing_video in missing_videos:
                    new_row = default_features.copy()
                    new_row['user_id'] = user_id
                    new_row['video_id'] = missing_video
                    feature_data = pd.concat([feature_data, pd.DataFrame([new_row])], ignore_index=True)
                
                # 按照video_ids的顺序排序
                feature_data['video_id'] = pd.Categorical(feature_data['video_id'], categories=video_ids, ordered=True)
                feature_data = feature_data.sort_values('video_id').reset_index(drop=True)
                feature_data = feature_data.drop(columns=['video_id'])  # 移除临时排序列
        
        # 转换为numpy数组，确保数据类型正确 - 增强处理
        try:
            # 先检查每一列的数据类型
            for col in feature_columns:
                if col in feature_data.columns:
                    if feature_data[col].dtype == 'object':
                        logger.error(f"[特征处理] 列 {col} 仍然是object类型: {feature_data[col].head()}")
                        # 强制转换
                        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0)
            
            # 再次尝试转换
            features_array = feature_data.values.astype(np.float32)
            
        except (ValueError, TypeError) as e:
            logger.error(f"[特征处理] 数据类型转换失败: {e}")
            # 强制转换每一列
            features_list = []
            for col in feature_columns:
                if col in feature_data.columns:
                    col_data = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0).values.astype(np.float32)
                else:
                    col_data = np.zeros(len(feature_data), dtype=np.float32)
                features_list.append(col_data)
            features_array = np.column_stack(features_list)
        
        # 最终检查
        if features_array.dtype != np.float32:
            logger.error(f"[特征处理] 最终数组类型不是float32: {features_array.dtype}")
            features_array = features_array.astype(np.float32)
        
        features_tensor = torch.FloatTensor(features_array).to(self.device)
        return features_tensor
    
    def get_real_labels_optimized(self, user_id: int, video_ids: List[int]) -> Dict[str, torch.Tensor]:
        """优化的获取真实标签"""
        if not video_ids:
            return {}
        
        # 批量查询
        query_condition = (
            (self.merged_data['user_id'] == user_id) & 
            (self.merged_data['video_id'].isin(video_ids))
        )
        batch_data = self.merged_data[query_condition]
        
        if len(batch_data) == 0:
            return {}
        
        # 构建标签字典
        result = {}
        for label_config in self.config['labels']:
            label_name = label_config['name']
            target_col = label_config['target']
            if target_col in batch_data.columns:
                label_values = batch_data[target_col].values
                result[label_name] = torch.FloatTensor(label_values).unsqueeze(1).to(self.device)
        
        return result
    
    def run_batch_simulation_optimized(self, is_treatment: bool, step: int, batch_users: List[int]) -> Dict[str, float]:
        """优化的批量仿真处理"""
        prefix = "Treatment" if is_treatment else "Control"
        logger.info(f"[{prefix}仿真优化] Step {step}: 开始处理 {len(batch_users)} 个用户")
        
        step_rewards = {label['name']: 0.0 for label in self.config['labels']}
        processed_users = 0
        
        # 批量处理数据
        batch_features = []
        batch_targets = []
        batch_valid_users = []
        
        for user_id in batch_users:
            # 1. 候选视频生成
            candidates = self.sample_candidate_videos(user_id, self.config['global']['n_candidate'])
            
            if len(candidates) == 0:
                continue  # 该用户没有可用视频
            
            # 2. 获取特征
            X = self.get_user_video_features_optimized(user_id, candidates)
            
            if X.size(0) == 0:
                continue  # 没有有效特征
            
            # 3. 模型预测与加权排序  
            alpha_weights = {}
            for label_config in self.config['labels']:
                label_name = label_config['name']
                alpha_key = 'alpha_T' if is_treatment else 'alpha_C'
                alpha_weights[label_name] = label_config[alpha_key]
            
            combined_scores = self.multi_label_model.get_combined_score(X, alpha_weights)
            
            # 4. 选出胜出视频
            # 确保combined_scores是正确的形状并获取有效索引
            scores_squeezed = combined_scores.squeeze()
            if scores_squeezed.dim() == 0:
                # 如果是标量，转换为1D张量
                scores_squeezed = scores_squeezed.unsqueeze(0)
            elif scores_squeezed.dim() > 1:
                # 如果是多维，展平为1D
                scores_squeezed = scores_squeezed.flatten()
            
            # 确保索引在有效范围内
            if len(scores_squeezed) != len(candidates):
                logger.warning(f"[{prefix}仿真优化] 分数张量长度 {len(scores_squeezed)} 与候选视频数 {len(candidates)} 不匹配，调整索引范围")
                # 取最小的长度作为安全范围
                safe_length = min(len(scores_squeezed), len(candidates))
                if safe_length == 0:
                    continue  # 跳过这个用户
                winner_idx = torch.argmax(scores_squeezed[:safe_length]).item()
            else:
                winner_idx = torch.argmax(scores_squeezed).item()
            
            winner_video = candidates[winner_idx]
            
            # 5. 获取真实反馈
            real_labels = self.get_real_labels_optimized(user_id, [winner_video])
            winner_features = X[winner_idx:winner_idx+1]
            
            if real_labels:
                batch_features.append(winner_features)
                batch_targets.append(real_labels)
                batch_valid_users.append(user_id)
                
                # 累加收益
                for label_name, label_tensor in real_labels.items():
                    # 确保张量是标量，然后提取值
                    if label_tensor.numel() == 1:
                        reward_value = label_tensor.item()
                    else:
                        # 如果张量有多个元素，取第一个元素或求和
                        reward_value = label_tensor.sum().item()
                    step_rewards[label_name] += reward_value
                
                # 7. 更新used状态
                self.used_videos.add(winner_video)
                processed_users += 1
        
        # 6. 批量模型训练
        if batch_features:
            # 合并所有特征和标签
            all_features = torch.cat(batch_features, dim=0)
            
            # 合并标签
            combined_targets = {}
            for label_name in batch_targets[0].keys():
                label_tensors = [targets[label_name] for targets in batch_targets]
                combined_targets[label_name] = torch.cat(label_tensors, dim=0)
            
            # 验证特征和标签的样本数量是否一致
            n_features = all_features.size(0)
            n_targets = combined_targets[list(combined_targets.keys())[0]].size(0)
            
            if n_features != n_targets:
                logger.warning(f"[{prefix}仿真优化] 特征样本数 {n_features} 与标签样本数 {n_targets} 不匹配，调整批量训练")
                # 使用较小的样本数以确保一致性
                min_samples = min(n_features, n_targets)
                if min_samples == 0:
                    logger.warning(f"[{prefix}仿真优化] 没有有效样本用于训练，跳过批量训练")
                else:
                    # 调整张量大小
                    all_features = all_features[:min_samples]
                    for label_name in combined_targets:
                        combined_targets[label_name] = combined_targets[label_name][:min_samples]
                    
                    # 批量训练
                    if self.use_amp:
                        with autocast(device_type='cuda'):
                            _ = self.multi_label_model.train_step(all_features, combined_targets)
                    else:
                        _ = self.multi_label_model.train_step(all_features, combined_targets)
            else:
                # 批量训练
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        _ = self.multi_label_model.train_step(all_features, combined_targets)
                else:
                    _ = self.multi_label_model.train_step(all_features, combined_targets)
        
        logger.info(f"[{prefix}仿真优化] Step {step}: 批量处理了 {processed_users} 个用户")
        return step_rewards
    
    def validate_models(self, step: int):
        """验证模型性能"""
        logger.info(f"[验证] Step {step}: 开始验证")
        
        # 使用验证集用户
        val_sample_size = min(100, len(self.val_users))  # 限制验证样本数量
        val_users_sample = random.sample(self.val_users, val_sample_size)
        
        all_metrics = {label['name']: [] for label in self.config['labels']}
        
        for user_id in val_users_sample:
            # 获取该用户的所有视频（不考虑used和mask）
            user_videos = self.user_video_lists.get(user_id, [])
            
            if len(user_videos) == 0:
                continue
            
            # 随机选择几个视频进行验证  
            sample_videos = random.sample(user_videos, min(5, len(user_videos)))
            
            X = self.get_user_video_features_optimized(user_id, sample_videos)
            real_labels = self.get_real_labels_optimized(user_id, sample_videos)
            
            if X.size(0) > 0 and real_labels:
                metrics = self.multi_label_model.evaluate_with_metrics(X, real_labels)
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # 计算平均验证指标
        avg_val_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                avg_val_metrics[f'val_{metric_name}'] = np.mean(values)
        
        # 记录指标
        self.metrics_tracker.update(avg_val_metrics, step)
        self.metrics_tracker.log_current("验证")
        
        # 打印预测指标
        for metric_name, value in avg_val_metrics.items():
            if 'accuracy' in metric_name or 'relative_error' in metric_name:
                if 'accuracy' in metric_name:
                    logger.info(f"[验证] {metric_name}: {value:.4f}")
                elif 'relative_error' in metric_name:
                    logger.info(f"[验证] {metric_name}: {value:.4f}")
        
        # 更新学习率调度器（只使用损失）
        loss_metrics = {k: v for k, v in avg_val_metrics.items() if 'loss' in k}
        self.multi_label_model.update_schedulers(loss_metrics)
    
    def run_global_simulation(self):
        """运行Global仿真"""
        logger.info("[Global仿真优化] 开始全局仿真流程...")
        
        n_steps = self.config['global']['n_steps']
        batch_size = self.config['global']['batch_size']
        validate_every = self.config['global']['validate_every']
        save_every = self.config['global']['save_every']
        
        for step in range(1, n_steps + 1):
            logger.info(f"[Global仿真优化] ===== Step {step}/{n_steps} =====")
            
            # 1. 用户批次抽样（GT和GC使用相同的用户批次）
            batch_users = random.sample(self.train_users, min(batch_size, len(self.train_users)))
            
            # 2. Treatment仿真（GT）
            logger.info("[GT流程优化] 开始Treatment组仿真...")
            step_rewards_T = self.run_batch_simulation_optimized(True, step, batch_users)
            
            # 累加到总收益
            for label_name, reward in step_rewards_T.items():
                self.total_label_T[label_name] += reward
            
            # 3. Control仿真（GC）
            logger.info("[GC流程优化] 开始Control组仿真...")
            step_rewards_C = self.run_batch_simulation_optimized(False, step, batch_users)
            
            # 累加到总收益
            for label_name, reward in step_rewards_C.items():
                self.total_label_C[label_name] += reward
            
            # 4. 记录步骤指标
            step_metrics = {}
            for label_name in step_rewards_T:
                step_metrics[f'step_{label_name}_T'] = step_rewards_T[label_name]
                step_metrics[f'step_{label_name}_C'] = step_rewards_C[label_name]  
                step_metrics[f'total_{label_name}_T'] = self.total_label_T[label_name]
                step_metrics[f'total_{label_name}_C'] = self.total_label_C[label_name]
            
            self.metrics_tracker.update(step_metrics, step)
            self.metrics_tracker.log_current(f"训练 Step {step}")
            
            # 5. 验证
            if step % validate_every == 0:
                self.validate_models(step)
            
            # 6. 保存模型
            if step % save_every == 0:
                checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
                self.multi_label_model.save_models(checkpoint_dir, step)
                
                # 保存特征处理器
                self.feature_processor.save_processors(checkpoint_dir)
            
            # GPU内存清理
            if torch.cuda.is_available() and step % 10 == 0:
                torch.cuda.empty_cache()
        
        logger.info("[Global仿真优化] 全局仿真完成")
    
    def compute_gte(self) -> Dict[str, float]:
        """计算GTE（Global Treatment Effect）"""
        logger.info("[GTE计算] 开始计算全局处理效应...")
        
        gte_results = {}
        
        for label_name in self.total_label_T:
            gt_total = self.total_label_T[label_name]
            gc_total = self.total_label_C[label_name]
            
            # 计算GTE
            gte = gt_total - gc_total
            gte_relative = (gte / gc_total * 100) if gc_total != 0 else 0
            
            gte_results[f'GTE_{label_name}'] = gte
            gte_results[f'GTE_{label_name}_relative'] = gte_relative
            gte_results[f'GT_{label_name}'] = gt_total
            gte_results[f'GC_{label_name}'] = gc_total
            
            logger.info(f"[GTE计算] {label_name}: GT={gt_total:.4f}, GC={gc_total:.4f}, GTE={gte:.4f} ({gte_relative:+.2f}%)")
        
        return gte_results
    
    def run(self):
        """运行完整的Global模式实验"""
        logger.info("[Global模式优化] 开始运行完整实验...")
        
        try:
            # 1. 数据加载和准备
            self.load_and_prepare_data()
            
            # 2. 优化预训练
            self.pretrain_models_optimized()
            
            # 3. 全局仿真
            self.run_global_simulation()
            
            # 4. 计算GTE
            gte_results = self.compute_gte()
            
            # 5. 保存最终结果
            final_results = {
                'config': self.config,
                'gte_results': gte_results,
                'metrics_summary': self.metrics_tracker.get_summary(),
                'dataset_stats': self.data_loader.get_dataset_stats()
            }
            
            results_path = os.path.join(self.exp_dir, 'result.json')
            save_results(final_results, results_path)
            
            logger.info("[Global模式优化] 实验完成！")
            
            # 打印最终结果
            logger.info("========== 最终实验结果 ==========")
            for key, value in gte_results.items():
                logger.info(f"{key}: {value}")
                
        except Exception as e:
            logger.error(f"[Global模式优化] 实验执行失败: {e}")
            raise