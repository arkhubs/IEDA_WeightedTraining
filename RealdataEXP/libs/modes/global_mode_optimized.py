"""
Global模式优化实现 - 解决GPU利用率低下问题
主要优化：
1. 使用PyTorch DataLoader进行多进程数据加载
2. 增加GPU状态监控和诊断
3. 优化批处理和内存使用
4. 添加详细的性能分析
5. (新增) 基于迭代次数的验证、高级指标（准确率/AUC）和最佳模型检查点保存
6. (修复) 添加训练损失跟踪和修复绘图字体问题
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
# 新增的库
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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
        
        # --- 关键修复：为autocast准备兼容性参数 ---
        self.autocast_kwargs = {'enabled': self.use_amp}
        if self.device.type in ['cuda', 'xpu']:
            self.autocast_kwargs['device_type'] = self.device.type
        # For IPEX, we don't add 'device_type'
        # --- 修复结束 ---
        
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
        
        # 新增：用于存储预训练过程中的指标，以便绘图
        self.pretrain_metrics = []
        
        # --- 新增：用于基于迭代次数的验证和绘图 ---
        self.global_iteration_step = 0
        
        # --- 新增：用于最佳模型检查点保存 ---
        self.best_metrics = {} 
        self.primary_metric = self.config.get('validation', {}).get('primary_metric', 'val_play_time_loss')
        logger.info(f"[检查点] 用于保存'最佳整体'模型的主要指标是: {self.primary_metric}")
        
        # GPU监控器
        self.gpu_monitor = None
        
    def _perform_training_step(self, X_batch, targets_batch):
        """执行一个优化的训练步骤，支持AMP"""
        self.multi_label_model.set_train_mode()
        
        # 清零所有梯度以备下次迭代
        for optimizer in self.multi_label_model.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        # --- 关键修复：使用准备好的兼容性参数 ---
        with self.autocast(**self.autocast_kwargs):
            losses = self.multi_label_model.compute_losses(X_batch, targets_batch)
            total_loss = sum(losses.values())
        # --- 修复结束 ---

        if self.scaler.is_enabled():
            self.scaler.scale(total_loss).backward()
            for optimizer in self.multi_label_model.optimizers.values():
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            for optimizer in self.multi_label_model.optimizers.values():
                optimizer.step()

        return {name: loss.item() for name, loss in losses.items()}
        
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
        
        # --- 新增：加载预训练权重 ---
        checkpoint_path = self.config['pretrain'].get('load_checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"[模型加载] 发现预训练权重配置，正在从 {checkpoint_path} 加载...")
            try:
                self.multi_label_model.load_models(checkpoint_path)
                logger.info(f"[模型加载] 成功加载预训练权重")
            except Exception as e:
                logger.error(f"[模型加载] 加载预训练权重失败: {e}")
        elif checkpoint_path:
            logger.warning(f"[模型加载] 配置文件中指定的权重文件不存在: {checkpoint_path}")
        # --- 结束新增部分 ---
        
        # --- 关键修复：根据use_amp决定IPEX优化策略 ---
        if self.device.type == 'xpu':
            logger.info("[IPEX] Applying ipex.optimize() to all models and optimizers...")
            import intel_extension_for_pytorch as ipex
            
            for label_name in self.multi_label_model.models:
                model = self.multi_label_model.models[label_name]
                optimizer = self.multi_label_model.optimizers[label_name]
                
                if self.use_amp:
                    logger.info(f"[IPEX] Optimizing '{label_name}' with bfloat16 for AMP.")
                    # 当AMP启用时，使用bfloat16进行混合精度优化
                    optimized_model, optimized_optimizer = ipex.optimize(
                        model, optimizer=optimizer, dtype=torch.bfloat16
                    )
                else:
                    logger.info(f"[IPEX] Optimizing '{label_name}' in float32 (AMP disabled).")
                    # 当AMP禁用时，不传递dtype参数，模型保持float32
                    optimized_model, optimized_optimizer = ipex.optimize(
                        model, optimizer=optimizer
                    )
                
                # 将原始模型和优化器替换为优化后的版本
                self.multi_label_model.models[label_name] = optimized_model
                self.multi_label_model.optimizers[label_name] = optimized_optimizer
            logger.info("[IPEX] ipex.optimize() applied successfully.")
        # --- 修复结束 ---
        
        logger.info("[Global模式优化] 数据准备完成")

    def _pretrain_validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """在预训练期间进行验证，并计算包括准确率和AUC在内的高级指标"""
        self.multi_label_model.set_eval_mode()
        
        all_preds = {label['name']: [] for label in self.config['labels']}
        all_targets = {label['name']: [] for label in self.config['labels']}
        
        # 新增：为验证添加tqdm进度条
        pbar = tqdm(val_loader, desc="验证中", leave=False, total=len(val_loader))
        for X_batch, targets_batch in pbar:
            X_batch = X_batch.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                with self.autocast(**self.autocast_kwargs):
                    preds_batch = self.multi_label_model.predict(X_batch)
            
            for label_name, pred_tensor in preds_batch.items():
                target_tensor = targets_batch[label_name]
                all_preds[label_name].append(pred_tensor.cpu())
                all_targets[label_name].append(target_tensor.cpu())

        val_metrics = {}
        for label_config in self.config['labels']:
            label_name = label_config['name']
            
            preds = torch.cat(all_preds[label_name]).numpy().flatten()
            targets = torch.cat(all_targets[label_name]).numpy().flatten()

            # --- 1. 损失计算（处理inf/nan值） ---
            # 在CPU上重新计算损失以确保所有指标在同一数据集上
            loss_fn = self.multi_label_model.loss_functions[label_name]
            loss_tensor = loss_fn(torch.tensor(preds), torch.tensor(targets))
            
            valid_losses = loss_tensor[~torch.isinf(loss_tensor) & ~torch.isnan(loss_tensor)]
            inf_count = torch.isinf(loss_tensor).sum().item()

            if inf_count > 0:
                logger.warning(f"[验证] 在 '{label_name}' 损失计算中发现 {inf_count} 个无穷值")
            
            avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0
            val_metrics[f'val_{label_name}_loss'] = avg_loss
            val_metrics[f'val_{label_name}_inf_count'] = inf_count

            # --- 2. 二元分类的准确率和AUC ---
            if label_config['type'] == 'binary':
                # 预测是logits，应用sigmoid
                probs = 1 / (1 + np.exp(-preds))
                # 获取二元预测
                binary_preds = (probs >= 0.5).astype(int)
                
                try:
                    accuracy = accuracy_score(targets, binary_preds)
                    val_metrics[f'val_{label_name}_accuracy'] = accuracy
                except Exception as e:
                    logger.warning(f"无法计算 {label_name} 的准确率: {e}")

                try:
                    # 检查目标中是否有多个类别
                    if len(np.unique(targets)) > 1:
                        auc = roc_auc_score(targets, probs)
                        val_metrics[f'val_{label_name}_auc'] = auc
                    else:
                        logger.debug(f"跳过 {label_name} 的AUC计算，因为此验证批次中只存在一个类别")

                except Exception as e:
                    logger.warning(f"无法计算 {label_name} 的AUC: {e}")

        return val_metrics

    def _plot_pretrain_metrics(self):

        """绘制并保存预训练期间的所有指标（损失、准确率、AUC），不包含 inf_count"""
        if not self.pretrain_metrics:
            return
        try:
            metrics_df = pd.DataFrame(self.pretrain_metrics)
            iterations = metrics_df['iteration'].values
            train_cols = [c for c in metrics_df.columns if 'train_' in c]
            val_cols = [c for c in metrics_df.columns if 'val_' in c]
            # 排除 inf_count 指标
            base_metrics = sorted(list(set(c.replace('train_', '').replace('val_', '') for c in train_cols + val_cols if 'inf_count' not in c)))

            num_plots = len(base_metrics)
            if num_plots == 0:
                return
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True, squeeze=False)
            for i, base_name in enumerate(base_metrics):
                ax = axes.flatten()[i]
                train_key = f'train_{base_name}'
                val_key = f'val_{base_name}'
                if train_key in metrics_df.columns:
                    ax.plot(iterations, metrics_df[train_key], 'o-', label=f'Train {base_name.replace("_", " ").title()}', markersize=3, alpha=0.7)
                if val_key in metrics_df.columns:
                    ax.plot(iterations, metrics_df[val_key], 's-', label=f'Validation {base_name.replace("_", " ").title()}', markersize=3, alpha=0.7)
                ax.set_title(f'Pre-training {base_name.replace("_", " ").title()} vs. Iterations')
                ax.set_ylabel(base_name.split('_')[-1].capitalize())
                ax.legend()
                ax.grid(True, alpha=0.3)
            axes.flatten()[-1].set_xlabel('Iteration')
            plt.tight_layout()
            plot_path = os.path.join(self.exp_dir, 'pretrain_metrics_curves.png')
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"[绘图] 预训练指标曲线图已保存: {plot_path}")
        except Exception as e:
            logger.error(f"[绘图] 保存指标曲线图失败: {e}")
            if 'fig' in locals():
                plt.close(fig)
            
    def _update_best_checkpoints(self, val_metrics: Dict[str, float]):
        """根据验证指标，更新并保存最佳模型检查点"""
        checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        
        # --- 为每个跟踪的值更新最佳指标 ---
        for key, value in val_metrics.items():
            # 对于损失，越低越好。对于准确率/AUC，越高越好。
            is_loss = 'loss' in key or 'inf' in key
            current_best = self.best_metrics.get(key, float('inf') if is_loss else float('-inf'))

            if (is_loss and value < current_best) or (not is_loss and value > current_best):
                self.best_metrics[key] = value
                logger.info(f"[检查点] '{key}' 新纪录: {value:.6f}。保存模型...")
                self.multi_label_model.save_models(checkpoint_dir, f"pretrain_best_{key}")

        # --- 基于主要指标更新最佳整体模型 ---
        primary_value = val_metrics.get(self.primary_metric)
        if primary_value is not None:
            is_primary_loss = 'loss' in self.primary_metric or 'inf' in self.primary_metric
            current_best_overall = self.best_metrics.get('overall', float('inf') if is_primary_loss else float('-inf'))
            
            if (is_primary_loss and primary_value < current_best_overall) or \
               (not is_primary_loss and primary_value > current_best_overall):
                self.best_metrics['overall'] = primary_value
                logger.info(f"[检查点] 新的最佳整体模型 (基于 {self.primary_metric}): {primary_value:.6f}。保存模型...")
                self.multi_label_model.save_models(checkpoint_dir, "pretrain_best_overall")

    def pretrain_models_optimized(self):
        """优化的预训练过程，基于迭代次数进行验证和保存，包含训练损失跟踪"""
        if not self.config['pretrain']['enabled']:
            logger.info("[预训练] 跳过预训练阶段")
            return
        
        logger.info("[预训练优化] 开始预训练阶段...")
        log_gpu_memory_usage(" - 预训练开始前")
        
        # --- 1. 准备并划分数据 ---
        full_train_data = self.processed_data[self.processed_data['mask'] == 0].copy()
        val_split_ratio = self.config['pretrain'].get('val_split_ratio', 0.5)
        
        pretrain_train_df, pretrain_val_df = train_test_split(
            full_train_data, test_size=val_split_ratio, random_state=42
        )
        logger.info(f"[预训练优化] 数据划分完成 - 训练集: {len(pretrain_train_df)}, 验证集: {len(pretrain_val_df)}")
        
        # --- 2. 创建DataLoaders ---
        batch_size = self.config['pretrain']['batch_size']
        train_loader = self.create_optimized_dataloader(pretrain_train_df, batch_size, shuffle=True)
        val_loader = self.create_optimized_dataloader(pretrain_val_df, batch_size, shuffle=False)
        
        epochs = self.config['pretrain']['epochs']
        validate_every_iters = self.config['pretrain'].get('validate_every_iters', 100)
        
        self.global_iteration_step = 0
        
        # --- 新增：用于累积训练损失的列表 ---
        train_loss_accumulator = {name: [] for name in self.multi_label_model.models.keys()}
        
        for epoch in range(1, epochs + 1):
            logger.info(f"[预训练优化] Epoch {epoch}/{epochs}")
            
            self.multi_label_model.set_train_mode()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
            
            for X_batch, targets_batch in pbar:
                self.global_iteration_step += 1
                
                X_batch = X_batch.to(self.device, non_blocking=True)
                targets_batch = {name: tensor.to(self.device, non_blocking=True) for name, tensor in targets_batch.items()}
                
                losses = self._perform_training_step(X_batch, targets_batch)
                
                # 累积训练损失
                for name, loss_val in losses.items():
                    train_loss_accumulator[name].append(loss_val)
                
                pbar.set_postfix({f"{k}_loss": f"{v:.4f}" for k, v in losses.items()})

                # --- 基于迭代次数的验证 ---
                if self.global_iteration_step % validate_every_iters == 0:
                    # 计算区间内的平均训练损失
                    avg_train_losses = {}
                    for name, loss_list in train_loss_accumulator.items():
                        if loss_list:
                            avg_train_losses[f'train_{name}_loss'] = np.mean(loss_list)
                            loss_list.clear()  # 为下一个区间重置

                    # 运行验证
                    logger.info(f"\n--- 迭代 {self.global_iteration_step}: 运行验证 ---")
                    val_metrics = self._pretrain_validate_epoch(val_loader)
                    
                    # 合并并记录所有指标
                    all_metrics = {**avg_train_losses, **val_metrics}
                    log_str = ", ".join([f"{k}: {v:.6f}" for k, v in all_metrics.items()])
                    logger.info(f"[验证] 迭代 {self.global_iteration_step} - 指标: {log_str}")

                    current_iter_metrics = {'iteration': self.global_iteration_step, **all_metrics}
                    self.pretrain_metrics.append(current_iter_metrics)

                    # 更新最佳检查点
                    self._update_best_checkpoints(val_metrics)

                    # 绘制指标
                    if self.config['pretrain'].get('plot_loss_curves', True):
                        self._plot_pretrain_metrics()

                    # 保存最新模型以便恢复
                    checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
                    self.multi_label_model.save_models(checkpoint_dir, "pretrain_latest")
                    
                    # 验证后重新设置为训练模式
                    self.multi_label_model.set_train_mode()

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
                with self.autocast(**self.autocast_kwargs):
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
            losses = self._perform_training_step(all_features, combined_targets)
            
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
            with self.autocast(**self.autocast_kwargs):
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
