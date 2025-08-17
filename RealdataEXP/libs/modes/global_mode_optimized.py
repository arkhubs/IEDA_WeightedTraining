"""
Global模式优化实现 - 解决GPU利用率低下问题
主要优化：
1. 使用PyTorch DataLoader进行多进程数据加载
2. 增加GPU状态监控和诊断
3. 优化批处理和内存使用
4. 添加详细的性能分析
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import random
import time

# 使用新的设备管理工具替代旧的autocast导入

from ..data import KuaiRandDataLoader, FeatureProcessor
from ..models import MultiLabelModel
from ..utils import MetricsTracker, save_results, get_device_and_amp_helpers
from ..utils.gpu_utils import log_gpu_info, log_gpu_memory_usage, test_gpu_training_speed, setup_gpu_monitoring

logger = logging.getLogger(__name__)

class TabularDataset(Dataset):
    """优化的表格数据Dataset，支持GPU加速"""
    
    def __init__(self, features: np.ndarray, labels: Dict[str, np.ndarray], device='cpu'):
        """
        Args:
            features: 特征数组 (N, D)
            labels: 标签字典 {label_name: array(N,)}
            device: 目标设备
        """
        self.device = device
        # 预转换为tensor以减少运行时开销
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = {}
        for name, label_array in labels.items():
            self.labels[name] = torch.tensor(label_array, dtype=torch.float32).unsqueeze(1)
        
        logger.info(f"[数据集] 创建TabularDataset，样本数: {len(self.features)}, 特征维度: {self.features.shape[1]}")
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_vector = self.features[idx]
        target_dict = {name: label_tensor[idx] for name, label_tensor in self.labels.items()}
        return feature_vector, target_dict

class GlobalModeOptimized:
    """优化的Global模式实验管理器"""
    
    def __init__(self, config: Dict, exp_dir: str, device_choice: str = 'auto'):
        self.config = config
        self.exp_dir = exp_dir
        
        # 使用新的设备选择辅助函数
        self.device, self.autocast, GradScalerClass = get_device_and_amp_helpers(device_choice)

        # 初始化混合精度训练
        self.use_amp = self.device.type != 'cpu' and config.get('use_amp', True)
        self.scaler = GradScalerClass(enabled=self.use_amp)
        
        logger.info(f"[Global模式优化] 初始化完成，设备: {self.device}, AMP: {self.use_amp}")
        
        self.data_loader_wrapper = KuaiRandDataLoader(config)
        self.feature_processor = FeatureProcessor(config)
        self.multi_label_model = None
        self.merged_data = None
        self.user_video_lists = None
        self.train_users = None
        self.val_users = None
        self.processed_data = None
        
        # 独立的used视频集合
        self.used_videos_T = set()  # Treatment组
        self.used_videos_C = set()  # Control组
        
        self.metrics_tracker = MetricsTracker()
        self.total_label_T = {label['name']: 0.0 for label in config['labels']}
        self.total_label_C = {label['name']: 0.0 for label in config['labels']}
        
        # GPU监控器
        self.gpu_monitor = None
        
    def start_gpu_monitoring(self):
        """启动GPU监控"""
        if torch.cuda.is_available():
            self.gpu_monitor = setup_gpu_monitoring(log_interval=60)  # 每分钟记录一次
            
    def stop_gpu_monitoring(self):
        """停止GPU监控"""
        if self.gpu_monitor:
            self.gpu_monitor.stop_monitoring()
            
    def create_optimized_dataloader(self, data: pd.DataFrame, batch_size: int, shuffle: bool = True) -> DataLoader:
        """创建优化的DataLoader"""
        feature_columns = self.feature_processor.get_feature_columns()
        
        # 准备特征数据
        features = data[feature_columns].values.astype(np.float32)
        
        # 准备标签数据
        labels = {}
        for label_config in self.config['labels']:
            target_col = label_config['target']
            labels[label_config['name']] = data[target_col].values.astype(np.float32)
        
        # 创建Dataset
        dataset = TabularDataset(features, labels, self.device)
        
        # DataLoader参数
        num_workers = self.config['dataset'].get('num_workers', 4)
        pin_memory = self.config['dataset'].get('pin_memory', True) and torch.cuda.is_available()
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0  # 保持worker进程
        )
        
        logger.info(f"[DataLoader] 创建完成 - batch_size: {batch_size}, num_workers: {num_workers}, pin_memory: {pin_memory}")
        return dataloader
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        logger.info("[Global模式优化] 开始数据加载和准备...")
        
        self.merged_data, self.user_video_lists, self.train_users, self.val_users = \
            self.data_loader_wrapper.load_and_prepare_data()
        
        stats = self.data_loader_wrapper.get_dataset_stats()
        for key, value in stats.items():
            logger.info(f"[数据统计] {key}: {value}")
        
        logger.info("[特征处理] 开始特征预处理...")
        self.processed_data = self.feature_processor.fit_transform(self.merged_data)
        
        feature_columns = self.feature_processor.get_feature_columns()
        input_dim = len(feature_columns)
        logger.info(f"[特征处理] 特征维度: {input_dim}")
        logger.info(f"[特征处理] 特征列: {feature_columns}")
        
        self.multi_label_model = MultiLabelModel(
            config=self.config, input_dim=input_dim, device=self.device
        )
        logger.info("[Global模式优化] 数据准备完成")

    def pretrain_models_optimized(self):
        """优化的预训练过程"""
        if not self.config['pretrain']['enabled']:
            logger.info("[预训练] 跳过预训练阶段")
            return
        
        logger.info("[预训练优化] 开始预训练阶段...")
        log_gpu_memory_usage(" - 预训练开始前")
        
        # 准备训练数据
        train_data = self.processed_data[self.processed_data['mask'] == 0].copy()
        logger.info(f"[预训练优化] 训练数据量: {len(train_data)}")
        
        # 创建优化的DataLoader
        batch_size = self.config['pretrain']['batch_size']
        train_loader = self.create_optimized_dataloader(train_data, batch_size, shuffle=True)
        
        epochs = self.config['pretrain']['epochs']
        
        for epoch in range(epochs):
            logger.info(f"[预训练优化] Epoch {epoch+1}/{epochs}")
            epoch_losses = {label['name']: [] for label in self.config['labels']}
            
            # 记录epoch开始时间
            epoch_start_time = time.time()
            
            # 使用tqdm显示进度
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            batch_count = 0
            
            for X_batch, targets_batch in pbar:
                batch_start_time = time.time()
                
                # 移动数据到GPU
                X_batch = X_batch.to(self.device, non_blocking=True)
                targets_batch = {name: tensor.to(self.device, non_blocking=True) 
                               for name, tensor in targets_batch.items()}
                
                # 使用混合精度训练
                if self.use_amp:
                    with self.autocast(device_type=self.device.type, enabled=self.use_amp):
                        losses = self.multi_label_model.train_step(X_batch, targets_batch)
                else:
                    losses = self.multi_label_model.train_step(X_batch, targets_batch)
                
                # 记录损失
                for label_name, loss in losses.items():
                    epoch_losses[label_name].append(loss)
                
                batch_time = time.time() - batch_start_time
                batch_count += 1
                
                # 更新进度条
                loss_info = {f"{k}": f"{v:.4f}" for k, v in losses.items()}
                loss_info['batch_time'] = f"{batch_time:.3f}s"
                pbar.set_postfix(loss_info)
                
                # 每100个batch记录一次GPU状态
                if batch_count % 100 == 0:
                    log_gpu_memory_usage(f" - Epoch {epoch+1} Batch {batch_count}")

            epoch_time = time.time() - epoch_start_time
            
            # 计算平均损失
            avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items() if losses}
            loss_str = ", ".join([f"{name}: {loss:.6f}" for name, loss in avg_losses.items()])
            
            logger.info(f"[预训练优化] Epoch {epoch+1} 完成，用时: {epoch_time:.2f}秒")
            logger.info(f"[预训练优化] Epoch {epoch+1} 平均损失 - {loss_str}")
            logger.info(f"[预训练优化] Epoch {epoch+1} 吞吐量: {len(train_data)/epoch_time:.0f} 样本/秒")
        
        log_gpu_memory_usage(" - 预训练完成后")
        logger.info("[预训练优化] 预训练阶段完成")

    def run_single_simulation_step_optimized(self, step: int, is_treatment: bool):
        """优化的单步仿真"""
        prefix = "Treatment" if is_treatment else "Control"
        used_videos = self.used_videos_T if is_treatment else self.used_videos_C
        
        batch_size = self.config['global']['batch_size']
        n_candidate = self.config['global']['n_candidate']
        
        # 抽样用户
        batch_users = random.sample(self.train_users, min(batch_size, len(self.train_users)))
        
        step_rewards = {label['name']: [] for label in self.config['labels']}
        processed_users = 0
        
        for user_id in batch_users:
            user_videos = self.user_video_lists.get(user_id, [])
            available_videos = [v for v in user_videos if v not in used_videos]
            
            if len(available_videos) < n_candidate:
                continue
                
            # 随机选择候选视频
            candidates = random.sample(available_videos, n_candidate)
            
            # 获取候选视频的特征 - 添加数据类型安全转换
            candidate_mask = self.processed_data['video_id'].isin(candidates)
            candidate_data = self.processed_data[candidate_mask & 
                                              (self.processed_data['user_id'] == user_id)]
            
            if len(candidate_data) == 0:
                continue
                
            feature_columns = self.feature_processor.get_feature_columns()
            
            # 安全的数据类型转换
            try:
                candidate_features = candidate_data[feature_columns].copy()
                # 确保所有列都是数值类型
                for col in feature_columns:
                    candidate_features[col] = pd.to_numeric(candidate_features[col], errors='coerce').fillna(0.0)
                
                X_candidates = torch.tensor(
                    candidate_features.values.astype(np.float32), 
                    dtype=torch.float32, 
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"[{prefix}仿真优化] 特征转换失败: {e}")
                continue
            
            # 预测每个候选视频的分数
            with torch.no_grad():
                if self.use_amp:
                    with self.autocast(device_type=self.device.type, enabled=self.use_amp):
                        predictions = self.multi_label_model.predict(X_candidates)
                else:
                    predictions = self.multi_label_model.predict(X_candidates)
            
            # 计算加权分数
            combined_scores = torch.zeros(len(candidates), device=self.device)
            for label_config in self.config['labels']:
                label_name = label_config['name']
                if label_name in predictions:
                    alpha = label_config.get('alpha_T' if is_treatment else 'alpha_C', 1.0)
                    pred_scores = predictions[label_name].squeeze()
                    if pred_scores.dim() == 0:
                        pred_scores = pred_scores.unsqueeze(0)
                    combined_scores += alpha * pred_scores
            
            # 确保combined_scores是正确的形状并获取有效索引
            scores_squeezed = combined_scores.squeeze()
            if scores_squeezed.dim() == 0:
                scores_squeezed = scores_squeezed.unsqueeze(0)
            elif scores_squeezed.dim() > 1:
                scores_squeezed = scores_squeezed.flatten()
            
            # 确保索引在有效范围内
            if len(scores_squeezed) != len(candidates):
                logger.warning(f"[{prefix}仿真优化] 分数张量长度 {len(scores_squeezed)} 与候选视频数 {len(candidates)} 不匹配")
                safe_length = min(len(scores_squeezed), len(candidates))
                if safe_length == 0:
                    continue  # 跳过这个用户
                winner_idx = torch.argmax(scores_squeezed[:safe_length]).item()
            else:
                winner_idx = torch.argmax(scores_squeezed).item()
            
            # 安全索引检查
            if winner_idx >= len(candidates):
                logger.warning(f"[{prefix}仿真优化] 获胜索引 {winner_idx} 超出候选范围 {len(candidates)}")
                continue
                
            winner_video = candidates[winner_idx]
            used_videos.add(winner_video)
            
            # 获取真实反馈
            winner_mask = (self.processed_data['video_id'] == winner_video) & \
                         (self.processed_data['user_id'] == user_id)
            winner_data = self.processed_data[winner_mask]
            
            if len(winner_data) == 0:
                continue
                
            # 记录奖励并准备训练数据
            for label_config in self.config['labels']:
                label_name = label_config['name']
                target_col = label_config['target']
                if target_col in winner_data.columns:
                    label_tensor = torch.tensor(
                        winner_data[target_col].values, 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    
                    # 确保张量是标量，然后提取值
                    if label_tensor.numel() == 1:
                        reward_value = label_tensor.item()
                    else:
                        # 如果张量有多个元素，取第一个元素或求和
                        reward_value = label_tensor.sum().item()
                    
                    step_rewards[label_name].append(reward_value)
            
            processed_users += 1
        
        # 批量训练（如果有数据）
        if processed_users > 0:
            self.batch_training_optimized(batch_users, used_videos, prefix)
        
        # 累加总奖励
        if is_treatment:
            total_rewards = self.total_label_T
        else:
            total_rewards = self.total_label_C
            
        for label_name, rewards in step_rewards.items():
            if rewards:
                total_rewards[label_name] += sum(rewards)
        
        logger.info(f"[{prefix}仿真优化] Step {step}: 处理用户数 {processed_users}, "
                   f"使用视频数 {len(used_videos)}")
        
        return processed_users

    def batch_training_optimized(self, batch_users: List[int], used_videos: set, prefix: str):
        """优化的批量训练"""
        try:
            # 获取这些用户使用过的视频的数据
            user_mask = self.processed_data['user_id'].isin(batch_users)
            video_mask = self.processed_data['video_id'].isin(used_videos)
            training_data = self.processed_data[user_mask & video_mask]
            
            if len(training_data) == 0:
                logger.warning(f"[{prefix}仿真优化] 没有训练数据，跳过批量训练")
                return
            
            feature_columns = self.feature_processor.get_feature_columns()
            
            # 安全的特征数据转换
            try:
                features_df = training_data[feature_columns].copy()
                # 确保所有列都是数值类型
                for col in feature_columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
                
                all_features = torch.tensor(
                    features_df.values.astype(np.float32), 
                    dtype=torch.float32, 
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"[{prefix}仿真优化] 特征转换失败: {e}")
                return
            
            # 准备标签
            combined_targets = {}
            for label_config in self.config['labels']:
                target_col = label_config['target']
                if target_col in training_data.columns:
                    combined_targets[label_config['name']] = torch.tensor(
                        training_data[target_col].values, 
                        dtype=torch.float32, 
                        device=self.device
                    ).unsqueeze(1)
            
            # 验证特征和标签的样本数量是否一致
            n_features = all_features.size(0)
            n_targets = combined_targets[list(combined_targets.keys())[0]].size(0)
            
            if n_features != n_targets:
                logger.warning(f"[{prefix}仿真优化] 特征样本数 {n_features} 与标签样本数 {n_targets} 不匹配，调整批量训练")
                min_samples = min(n_features, n_targets)
                if min_samples == 0:
                    logger.warning(f"[{prefix}仿真优化] 没有有效样本用于训练，跳过批量训练")
                    return
                
                # 调整张量大小
                all_features = all_features[:min_samples]
                for label_name in combined_targets:
                    combined_targets[label_name] = combined_targets[label_name][:min_samples]
            
            # 执行训练步骤
            if self.use_amp:
                with self.autocast(device_type=self.device.type, enabled=self.use_amp):
                    losses = self.multi_label_model.train_step(all_features, combined_targets)
            else:
                losses = self.multi_label_model.train_step(all_features, combined_targets)
            
            # 记录损失（可选）
            loss_str = ", ".join([f"{name}: {loss:.6f}" for name, loss in losses.items()])
            logger.debug(f"[{prefix}仿真优化] 批量训练损失 - {loss_str}")
            
        except Exception as e:
            logger.error(f"[{prefix}仿真优化] 批量训练失败: {e}")

    def run_simulation_for_group_optimized(self, is_treatment: bool):
        """为单个组运行完整的优化仿真"""
        prefix = "Treatment" if is_treatment else "Control"
        logger.info(f"========== 开始 {prefix} 组仿真（优化版） ==========")
        
        n_steps = self.config['global']['n_steps']
        validate_every = self.config['global']['validate_every']
        
        start_time = time.time()
        
        for step in range(1, n_steps + 1):
            step_start_time = time.time()
            
            processed_users = self.run_single_simulation_step_optimized(step, is_treatment)
            
            step_time = time.time() - step_start_time
            
            if step % 10 == 0:  # 每10步报告一次
                logger.info(f"[{prefix}仿真优化] Step {step}/{n_steps}, "
                           f"处理用户: {processed_users}, 用时: {step_time:.2f}秒")
            
            # 验证模型（如果需要）
            if step % validate_every == 0:
                self.validate_models_optimized(step, prefix)
        
        total_time = time.time() - start_time
        logger.info(f"========== {prefix} 组仿真完成，总用时: {total_time:.2f}秒 ==========")

    def validate_models_optimized(self, step: int, prefix: str):
        """优化的模型验证"""
        logger.info(f"[{prefix}验证优化] Step {step} 模型验证...")
        
        # 简化的验证逻辑，避免耗时的验证过程
        val_data = self.processed_data[self.processed_data['mask'] == 1].sample(
            min(1000, len(self.processed_data[self.processed_data['mask'] == 1]))
        )
        
        if len(val_data) == 0:
            return
        
        feature_columns = self.feature_processor.get_feature_columns()
        
        # 安全的数据转换
        try:
            val_features = val_data[feature_columns].copy()
            # 确保所有列都是数值类型
            for col in feature_columns:
                val_features[col] = pd.to_numeric(val_features[col], errors='coerce').fillna(0.0)
            
            X_val = torch.tensor(
                val_features.values.astype(np.float32), 
                dtype=torch.float32, 
                device=self.device
            )
        except Exception as e:
            logger.warning(f"[{prefix}验证优化] 特征转换失败: {e}")
            return
        
        with torch.no_grad():
            if self.use_amp:
                with self.autocast(device_type=self.device.type, enabled=self.use_amp):
                    predictions = self.multi_label_model.predict(X_val)
            else:
                predictions = self.multi_label_model.predict(X_val)
        
        # 计算验证指标
        for label_config in self.config['labels']:
            label_name = label_config['name']
            target_col = label_config['target']
            
            if label_name in predictions and target_col in val_data.columns:
                pred = predictions[label_name].cpu().numpy().flatten()
                true = val_data[target_col].values
                
                # 计算相对误差
                non_zero_mask = true != 0
                if np.any(non_zero_mask):
                    relative_errors = np.abs((pred[non_zero_mask] - true[non_zero_mask]) / true[non_zero_mask])
                    mean_relative_error = np.mean(relative_errors) * 100
                    logger.info(f"[{prefix}验证优化] Step {step} {label_name} 平均相对误差: {mean_relative_error:.2f}%")

    def run_global_simulation_optimized(self):
        """运行优化的Global仿真"""
        logger.info("[Global仿真优化] 开始完整实验...")
        
        # 启动GPU监控
        self.start_gpu_monitoring()
        
        try:
            # Treatment组仿真
            self.run_simulation_for_group_optimized(is_treatment=True)
            
            logger.info("[Global仿真优化] Treatment组完成，开始Control组...")
            
            # Control组仿真
            self.run_simulation_for_group_optimized(is_treatment=False)
            
        finally:
            # 停止GPU监控
            self.stop_gpu_monitoring()

    def compute_gte_optimized(self) -> Dict[str, float]:
        """计算优化的GTE"""
        logger.info("[GTE计算优化] 开始计算全局处理效应...")
        gte_results = {}
        
        for label_name in self.total_label_T:
            gt_total = self.total_label_T[label_name]
            gc_total = self.total_label_C[label_name]
            gte = gt_total - gc_total
            gte_relative = (gte / gc_total * 100) if gc_total != 0 else 0
            
            gte_results[f'GTE_{label_name}'] = gte
            gte_results[f'GTE_{label_name}_relative'] = gte_relative
            
            logger.info(f"[GTE计算优化] {label_name}: GT={gt_total:.4f}, GC={gc_total:.4f}, "
                       f"GTE={gte:.4f} ({gte_relative:+.2f}%)")
        
        return gte_results

    def run(self):
        """运行完整的优化Global模式实验"""
        logger.info("[Global模式优化] 开始运行完整实验...")
        
        try:
            # GPU诊断
            log_gpu_info()
            test_gpu_training_speed()
            
            # 数据准备
            self.load_and_prepare_data()
            
            # 优化预训练
            self.pretrain_models_optimized()
            
            # 优化仿真
            self.run_global_simulation_optimized()
            
            # 计算GTE
            gte_results = self.compute_gte_optimized()
            
            # 保存结果
            final_results = {
                'config': self.config,
                'gte_results': gte_results,
                'metrics_summary': self.metrics_tracker.get_summary(),
                'dataset_stats': self.data_loader_wrapper.get_dataset_stats()
            }
            
            results_path = os.path.join(self.exp_dir, 'result.json')
            save_results(final_results, results_path)
            
            logger.info("[Global模式优化] 实验完成！")
            logger.info("========== 最终GTE结果 ==========")
            for key, value in gte_results.items():
                logger.info(f"{key}: {value}")
                
        except Exception as e:
            logger.error(f"[Global模式优化] 实验执行失败: {e}", exc_info=True)
            raise
