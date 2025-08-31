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
import itertools  # 新增：用于限制验证批次
# 新增的库
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 使用新的设备管理工具替代旧的autocast导入

from ..data import KuaiRandDataLoader, FeatureProcessor, CacheManager
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
        self.cache_manager = CacheManager(config['dataset']['cache_path'])
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
        self.best_overall_score_config = self.config.get('validation', {}).get('best_overall_score', {})
        
        # GPU监控器
        self.gpu_monitor = None
        
    def _get_cache_key(self, base_key: str) -> str:
        """Generates a dataset-specific cache key."""
        dataset_name = self.config['dataset']['name']
        return f"{dataset_name}_{base_key}"
        
    def _apply_model_optimizations(self):
        """Applies advanced model optimizations like torch.compile or IPEX."""
        # --- New: Configurable Optimization Strategy ---
        opt_config = self.config.get('optimization', {})
        compile_config = opt_config.get('torch_compile', {})
        ipex_strategy = opt_config.get('ipex_strategy', 'optimize') # Default to 'optimize' for safety

        # Apply optimizations based on device and strategy
        if self.device.type == 'xpu':
            logger.info(f"[Optimization] IPEX device detected. Selected strategy: '{ipex_strategy}'")
            
            # --- Strategy 1: Use torch.compile with IPEX backend (Experimental) ---
            if ipex_strategy == 'compile' and compile_config.get('enabled', False):
                try:
                    if hasattr(torch, 'compile'):
                        logger.info("[Optimization] Applying torch.compile(backend='ipex')...")
                        for label_name, model in self.multi_label_model.models.items():
                            self.multi_label_model.models[label_name] = torch.compile(model, backend="ipex")
                        logger.info("[Optimization] torch.compile with IPEX backend applied successfully.")
                    else:
                        logger.warning("[Optimization] torch.compile not found. Please ensure PyTorch version >= 2.0.")
                except Exception as e:
                    logger.error(f"[Optimization] Failed to apply torch.compile with IPEX backend: {e}. Consider switching ipex_strategy to 'optimize'.")

            # --- Strategy 2: Use ipex.optimize (Default and Recommended) ---
            else:
                if ipex_strategy != 'optimize':
                    logger.warning(f"Invalid ipex_strategy '{ipex_strategy}' or torch.compile disabled. Defaulting to 'optimize'.")
                
                logger.info("[Optimization] Applying ipex.optimize()...")
                try:
                    import intel_extension_for_pytorch as ipex
                    for i, label_name in enumerate(self.multi_label_model.models):
                        model = self.multi_label_model.models[label_name]
                        optimizer = self.multi_label_model.optimizers[label_name]
                        opt_name = self.config['labels'][i]['optimizer']['name']

                        if self.use_amp:
                            if 'sgd' in opt_name.lower():
                                logger.info(f"[IPEX] Optimizing '{label_name}' with bfloat16 for AMP (SGD detected).")
                                optimized_model, optimized_optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
                            else:
                                logger.warning(f"[IPEX] Optimizer '{opt_name}' for label '{label_name}' is not SGD. Skipping bfloat16 optimization to prevent instability. Training will use float32.")
                                optimized_model, optimized_optimizer = ipex.optimize(model, optimizer=optimizer)
                        else:
                            logger.info(f"[IPEX] Optimizing '{label_name}' in float32 (AMP disabled).")
                            optimized_model, optimized_optimizer = ipex.optimize(model, optimizer=optimizer)
                        
                        self.multi_label_model.models[label_name] = optimized_model
                        self.multi_label_model.optimizers[label_name] = optimized_optimizer
                    logger.info("[Optimization] ipex.optimize() applied successfully.")
                except Exception as e:
                    logger.error(f"[Optimization] Failed to apply ipex.optimize(): {e}")

        # --- Strategy 3: Use torch.compile with CUDA backend ---
        elif self.device.type == 'cuda' and compile_config.get('enabled', False):
            try:
                if hasattr(torch, 'compile'):
                    logger.info(f"[Optimization] Enabling torch.compile on {self.device.type} backend...")
                    backend = compile_config.get('backend', 'default')
                    for label_name, model in self.multi_label_model.models.items():
                        self.multi_label_model.models[label_name] = torch.compile(model, backend=backend)
                    logger.info(f"[Optimization] torch.compile enabled successfully with backend '{backend}'.")
                else:
                    logger.warning("[Optimization] torch.compile not found. Please ensure PyTorch version >= 2.0.")
            except Exception as e:
                logger.error(f"[Optimization] Failed to apply torch.compile: {e}")
        
        else:
            if compile_config.get('enabled', False):
                 logger.info(f"[Optimization] No advanced optimizations applicable for the current device ('{self.device.type}') and configuration.")
        # --- End of Optimization Strategy ---
        
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
            pin_memory=pin_memory
        )
        
        logger.info(f"[DataLoader] 创建完成 - batch_size: {batch_size}, num_workers: {num_workers}, pin_memory: {pin_memory}")
        return dataloader
        
    def _chunk_and_cache_data(self):
        """
        Fits feature processor, then chunks and caches the data.
        Assumes self.merged_data and self.train_users/val_users are already loaded.
        """
        logger.info("Starting data chunking and caching process...")
        chunk_config = self.config['dataset'].get('chunking', {})
        num_chunks = chunk_config.get('num_chunks', 1)
        fit_sample_ratio = chunk_config.get('fit_sample_ratio', 0.2)

        # 1. Fit FeatureProcessor on a sample of the data
        logger.info(f"Fitting feature processor on {fit_sample_ratio*100}% of the data...")
        fit_sample_data = self.merged_data.sample(frac=fit_sample_ratio, random_state=42)
        self.feature_processor.fit_transform(fit_sample_data)
        del fit_sample_data
        
        # Save the fitted processors (scaler, mappings, etc.)
        self.feature_processor.save_processors(self.config['dataset']['cache_path'])

        # 2. Chunk users
        # Note: self.train_users and self.val_users are used here. We need all users for processing.
        all_users = np.concatenate([self.train_users, self.val_users])
        np.random.shuffle(all_users)
        user_chunks = np.array_split(all_users, num_chunks)
        
        # Clear old chunk files before creating new ones
        for i in range(num_chunks):
            self.cache_manager.clear(self._get_cache_key(f"processed_features_chunk_{i}"))
            self.cache_manager.clear(self._get_cache_key(f"processed_labels_chunk_{i}"))

        # 3. Process and cache data chunk by chunk
        for i, user_chunk in enumerate(user_chunks):
            logger.info(f"--- Processing chunk {i+1}/{num_chunks} ---")
            chunk_data = self.merged_data[self.merged_data['user_id'].isin(user_chunk)]
            
            logger.info(f"Chunk {i+1} has {len(chunk_data)} rows. Applying feature transformation...")
            processed_chunk_df = self.feature_processor.transform(chunk_data)
            
            feature_columns = self.feature_processor.get_feature_columns()
            features_array = processed_chunk_df[feature_columns].values.astype(np.float32)
            
            labels_dict = {}
            for label_config in self.config['labels']:
                target_col = label_config['target']
                labels_dict[label_config['name']] = processed_chunk_df[target_col].values.astype(np.float32)
            
            # Cache the processed numpy arrays using dataset-specific keys
            self.cache_manager.save(features_array, self._get_cache_key(f"processed_features_chunk_{i}"))
            self.cache_manager.save(labels_dict, self._get_cache_key(f"processed_labels_chunk_{i}"))
            logger.info(f"Chunk {i+1} processed and cached successfully.")
        
    def load_and_prepare_data(self):
        """
        Loads and prepares data, with support for chunking and caching to handle large datasets.
        """
        logger.info("[Global模式优化] 开始数据加载和准备...")
        chunk_config = self.config['dataset'].get('chunking', {'enabled': False})

        if chunk_config.get('enabled', False):
            num_chunks = chunk_config.get('num_chunks', 1)
            # 检查所有块是否都已缓存
            all_chunks_cached = True
            for i in range(num_chunks):
                if not self.cache_manager.exists(self._get_cache_key(f"processed_features_chunk_{i}")):
                    all_chunks_cached = False
                    break
            
            if all_chunks_cached:
                logger.info(f"所有 {num_chunks} 个数据块都已在缓存中找到。跳过数据处理。")
                # 即使跳过处理，仍然需要初始化模型
                logger.info("[特征处理] 从缓存加载特征维度信息...")
                # 假设第一个块的特征处理器信息是通用的
                self.feature_processor.load_processors(self.config['dataset']['cache_path'])
                input_dim = self.feature_processor.total_numerical_dim + self.feature_processor.total_categorical_dim

                self.multi_label_model = MultiLabelModel(config=self.config, input_dim=input_dim, device=self.device)
                self._apply_model_optimizations() # 应用模型优化
                # 加载用户列表以便进行仿真
                self.train_users, self.val_users = self.cache_manager.load(self._get_cache_key("user_split"))
                logger.info(f"从缓存加载用户划分: {len(self.train_users)} 训练用户, {len(self.val_users)} 验证用户")

                return # 提前退出

        logger.info("缓存未找到或未启用分块。开始完整的数据处理流程...")
        self.merged_data, self.user_video_lists, self.train_users, self.val_users = \
            self.data_loader_wrapper.load_and_prepare_data()
        
        # 缓存用户划分信息
        self.cache_manager.save((self.train_users, self.val_users), self._get_cache_key("user_split"))

        stats = self.data_loader_wrapper.get_dataset_stats()
        for key, value in stats.items():
            logger.info(f"[数据统计] {key}: {value}")
        
        logger.info("[特征处理] 开始特征预处理...")
        
        # --- 分块处理 ---
        if chunk_config.get('enabled', False):
            logger.info("启用数据分块处理...")
            num_chunks = chunk_config.get('num_chunks', 1)
            fit_sample_ratio = chunk_config.get('fit_sample_ratio', 0.2)

            # 1. 在数据样本上拟合 FeatureProcessor
            logger.info(f"在 {fit_sample_ratio*100}% 的数据样本上拟合特征处理器...")
            fit_sample_data = self.merged_data.sample(frac=fit_sample_ratio, random_state=42)
            self.feature_processor.fit_transform(fit_sample_data)
            del fit_sample_data
            
            # 保存处理器（scaler, mappings）
            self.feature_processor.save_processors(self.config['dataset']['cache_path'])

            # 2. 将用户分块
            all_users = self.merged_data['user_id'].unique()
            np.random.shuffle(all_users)
            user_chunks = np.array_split(all_users, num_chunks)

            # 3. 逐块处理和缓存数据
            for i, user_chunk in enumerate(user_chunks):
                logger.info(f"--- 正在处理块 {i+1}/{num_chunks} ---")
                chunk_data = self.merged_data[self.merged_data['user_id'].isin(user_chunk)]
                
                logger.info(f"块 {i+1} 包含 {len(chunk_data)} 行数据。应用特征转换...")
                processed_chunk_df = self.feature_processor.transform(chunk_data)
                
                feature_columns = self.feature_processor.get_feature_columns()
                features_array = processed_chunk_df[feature_columns].values.astype(np.float32)
                
                labels_dict = {}
                for label_config in self.config['labels']:
                    target_col = label_config['target']
                    labels_dict[label_config['name']] = processed_chunk_df[target_col].values.astype(np.float32)
                
                # 缓存处理好的 NumPy 数组
                self.cache_manager.save(features_array, self._get_cache_key(f"processed_features_chunk_{i}"))
                self.cache_manager.save(labels_dict, self._get_cache_key(f"processed_labels_chunk_{i}"))
                logger.info(f"块 {i+1} 已处理并缓存。")

            # 4. 释放内存
            del self.merged_data
            del self.processed_data
            self.merged_data = None
            self.processed_data = None
            import gc
            gc.collect()
            logger.info("原始数据已从内存中释放。")

            input_dim = self.feature_processor.total_numerical_dim + self.feature_processor.total_categorical_dim
        else: # 非分块模式
            self.processed_data = self.feature_processor.fit_transform(self.merged_data)
            input_dim = self.feature_processor.total_numerical_dim + self.feature_processor.total_categorical_dim

        # 5. 初始化模型
        self.multi_label_model = MultiLabelModel(
            config=self.config, input_dim=input_dim, device=self.device
        )
        self._apply_model_optimizations()
        
        # Ensure user_video_lists is loaded for the global simulation phase if not loaded already
        if self.user_video_lists is None:
            logger.info("为 Global 仿真阶段加载 user_video_lists...")
            self.user_video_lists = self.cache_manager.load(self._get_cache_key("user_video_lists"))
            if self.user_video_lists is None:
                 logger.error("无法加载 user_video_lists，Global 仿真阶段可能会失败！")
        
        logger.info("[Global模式优化] 数据准备完成。")

    def _pretrain_validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """在预训练期间进行验证，并计算包括准确率和AUC在内的高级指标"""
        self.multi_label_model.set_eval_mode()
        
        all_preds = {label['name']: [] for label in self.config['labels']}
        all_targets = {label['name']: [] for label in self.config['labels']}
        
        # 新增：限制验证批次数量（如果在配置中指定）
        validation_batches = self.config.get('validation', {}).get('validation_batches')
        
        pbar_total = validation_batches if validation_batches is not None else len(val_loader)
        pbar = tqdm(itertools.islice(val_loader, validation_batches), desc="验证中", leave=False, total=pbar_total)
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
                # --- 新增：数值型标签如 play_time 的 MAPE 计算 ---
                if label_config['type'] == 'numerical':
                    non_zero_mask = targets != 0
                    if np.any(non_zero_mask):
                        mape = np.mean(np.abs((targets[non_zero_mask] - preds[non_zero_mask]) / targets[non_zero_mask])) * 100
                        val_metrics[f'val_{label_name}_mape'] = mape

        return val_metrics

    def _plot_pretrain_metrics(self):
        """绘制并保存预训练期间的所有指标（损失、准确率、AUC），不包含 inf_count，使用英文标题和标签解决字体渲染问题"""
        if not self.pretrain_metrics: 
            return
        try:
            metrics_df = pd.DataFrame(self.pretrain_metrics)
            
            # 获取所有基础指标名称，排除 'inf_count'
            train_cols = [c for c in metrics_df.columns if 'train_' in c]
            val_cols = [c for c in metrics_df.columns if 'val_' in c]
            all_base_metrics = set(c.replace('train_', '').replace('val_', '') for c in train_cols + val_cols)
            base_metrics_to_plot = sorted([m for m in all_base_metrics if 'inf_count' not in m])

            num_plots = len(base_metrics_to_plot)
            if num_plots == 0: 
                return
            
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True, squeeze=False)
            for i, base_name in enumerate(base_metrics_to_plot):
                ax = axes.flatten()[i]
                train_key = f'train_{base_name}'
                val_key = f'val_{base_name}'
                
                if train_key in metrics_df.columns:
                    ax.plot(metrics_df['iteration'], metrics_df[train_key], 'o-', label=f'Train {base_name}', markersize=3, alpha=0.7)
                if val_key in metrics_df.columns:
                    ax.plot(metrics_df['iteration'], metrics_df[val_key], 's-', label=f'Validation {base_name}', markersize=3, alpha=0.7)
                
                # 使用英文标题和标签
                ax.set_title(f'Pre-training {base_name.replace("_", " ").title()} vs. Iterations')
                ax.set_ylabel(base_name.split('_')[-1].capitalize())
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes.flatten()[-1].set_xlabel('Iteration')
            plt.tight_layout()
            plt.savefig(os.path.join(self.exp_dir, 'pretrain_metrics_curves.png'), dpi=150)
            plt.close(fig)
            logger.info(f"[Plotting] Pre-training metrics chart saved.")
        except Exception as e:
            logger.error(f"Failed to plot metrics: {e}")
            if 'fig' in locals():
                plt.close(fig)
            
    def _update_best_checkpoints(self, val_metrics: Dict[str, float], iteration: int):
        """根据验证指标，更新并保存最佳模型检查点"""
        checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        
        # --- Update best metric for each tracked value ---
        for key, value in val_metrics.items():
            # --- New: Skip saving useless inf_count checkpoints ---
            if 'inf_count' in key:
                continue

            is_loss = 'loss' in key
            current_best = self.best_metrics.get(key, float('inf') if is_loss else float('-inf'))

            if (is_loss and value < current_best) or (not is_loss and value > current_best):
                self.best_metrics[key] = value
                logger.info(f"[Checkpoint] New best for '{key}': {value:.6f}. Saving model...")
                self.multi_label_model.save_models(checkpoint_dir, f"pretrain_best_{key.replace('val_','')}", iteration)
        
        # --- New: Calculate and check weighted overall score ---
        if self.best_overall_score_config:
            current_overall_score = 0
            is_score_valid = True
            for metric_name, weight in self.best_overall_score_config.items():
                if metric_name in val_metrics:
                    current_overall_score += weight * val_metrics[metric_name]
                else:
                    logger.warning(f"[Checkpoint] Metric '{metric_name}' for overall score not found in validation results. Skipping overall score check.")
                    is_score_valid = False
                    break
            
            if is_score_valid:
                # For the overall score, higher is always better
                current_best_overall = self.best_metrics.get('overall_score', float('-inf'))
                if current_overall_score > current_best_overall:
                    self.best_metrics['overall_score'] = current_overall_score
                    logger.info(f"[Checkpoint] New best OVERALL weighted score: {current_overall_score:.6f}. Saving model...")
                    self.multi_label_model.save_models(checkpoint_dir, "pretrain_best_overall", iteration)

    def pretrain_models_optimized(self):
        """
        Optimized pre-training process that loads and trains on one data chunk per epoch
        and supports re-chunking the data periodically.
        """
        if not self.config['pretrain']['enabled']:
            logger.info("[预训练] 跳过预训练阶段")
            return
        
        logger.info("[预训练优化] 开始预训练阶段...")
        log_gpu_memory_usage(" - 预训练开始前")
        
        chunk_config = self.config['dataset'].get('chunking', {'enabled': False})
        num_chunks = chunk_config.get('num_chunks', 1) if chunk_config.get('enabled') else 1
        rechunk_interval = chunk_config.get('rechunk_every_n_epochs') # Can be None or 0
        
        epochs = self.config['pretrain']['epochs']
        validate_every_iters = self.config['pretrain'].get('validate_every_iters', 100)
        self.global_iteration_step = 0
        train_loss_accumulator = {name: [] for name in self.multi_label_model.models.keys()}

        for epoch in range(1, epochs + 1):
            # --- Re-chunking Logic ---
            if chunk_config.get('enabled') and rechunk_interval and epoch > 1 and (epoch - 1) % rechunk_interval == 0:
                logger.info(f"--- Epoch {epoch}: 达到重新分块阈值 ({rechunk_interval} epochs)。开始重新处理数据... ---")
                try:
                    # 1. Reload raw data
                    self.merged_data, self.user_video_lists, self.train_users, self.val_users = \
                        self.data_loader_wrapper.load_and_prepare_data()
                    # 2. Re-cache the new user split
                    self.cache_manager.save((self.train_users, self.val_users), self._get_cache_key("user_split"))
                    # 3. Re-chunk and re-cache data based on the new split
                    self._chunk_and_cache_data()
                    # 4. Clean up memory
                    del self.merged_data
                    self.merged_data = None
                    import gc
                    gc.collect()
                    logger.info("重新分块完成，原始数据已从内存释放。")
                except Exception as e:
                    logger.error(f"重新分块失败: {e}. 继续使用旧的数据块。")
            
            logger.info(f"[预训练优化] Epoch {epoch}/{epochs}")

            current_chunk_index = (epoch - 1) % num_chunks
            logger.info(f"Epoch {epoch}: 正在加载数据块 {current_chunk_index + 1}/{num_chunks}")
            
            try:
                if chunk_config.get('enabled'):
                    features_array = self.cache_manager.load(self._get_cache_key(f"processed_features_chunk_{current_chunk_index}"))
                    labels_dict = self.cache_manager.load(self._get_cache_key(f"processed_labels_chunk_{current_chunk_index}"))
                    if features_array is None or labels_dict is None:
                        logger.error(f"无法从缓存加载块 {current_chunk_index}。")
                        continue
                else:
                    if self.processed_data is None:
                        logger.error("非分块模式下，processed_data 未加载。")
                        return
                    feature_columns = self.feature_processor.get_feature_columns()
                    features_array = self.processed_data[feature_columns].values.astype(np.float32)
                    labels_dict = {}
                    for label_config in self.config['labels']:
                        target_col = label_config['target']
                        labels_dict[label_config['name']] = self.processed_data[target_col].values.astype(np.float32)

            except Exception as e:
                logger.error(f"加载数据块 {current_chunk_index} 时出错: {e}")
                continue
            
            # --- Data Loader Creation (no changes here) ---
            val_split_ratio = self.config['pretrain'].get('val_split_ratio', 0.2)
            indices = np.arange(len(features_array))
            train_indices, val_indices = train_test_split(indices, test_size=val_split_ratio, random_state=42)
            train_features = features_array[train_indices]
            val_features = features_array[val_indices]
            train_labels = {name: arr[train_indices] for name, arr in labels_dict.items()}
            val_labels = {name: arr[val_indices] for name, arr in labels_dict.items()}
            batch_size = self.config['pretrain']['batch_size']
            train_dataset = TabularDataset(train_features, train_labels, self.device)
            val_dataset = TabularDataset(val_features, val_labels, self.device)
            num_workers = self.config['dataset'].get('num_workers', 4)
            pin_memory = self.config['dataset'].get('pin_memory', True) and torch.cuda.is_available()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            
            # --- Training and Validation Loop (no changes here) ---
            self.multi_label_model.set_train_mode()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training on Chunk {current_chunk_index+1}")
            
            for X_batch, targets_batch in pbar:
                self.global_iteration_step += 1
                X_batch = X_batch.to(self.device, non_blocking=True)
                targets_batch = {name: tensor.to(self.device, non_blocking=True) for name, tensor in targets_batch.items()}
                
                losses = self._perform_training_step(X_batch, targets_batch)
                for name, loss_val in losses.items():
                    train_loss_accumulator[name].append(loss_val)
                pbar.set_postfix({f"{k}_loss": f"{v:.4f}" for k, v in losses.items()})

                if self.global_iteration_step % validate_every_iters == 0:
                    avg_train_losses = {}
                    for name, loss_list in train_loss_accumulator.items():
                        if loss_list:
                            avg_train_losses[f'train_{name}_loss'] = np.mean(loss_list)
                            loss_list.clear()

                    logger.info(f"\n--- 迭代 {self.global_iteration_step}: 运行验证 ---")
                    val_metrics = self._pretrain_validate_epoch(val_loader)
                    all_metrics = {**avg_train_losses, **val_metrics}
                    log_str = ", ".join([f"{k}: {v:.6f}" for k, v in all_metrics.items()])
                    logger.info(f"[验证] 迭代 {self.global_iteration_step} - 指标: {log_str}")

                    current_iter_metrics = {'iteration': self.global_iteration_step, **all_metrics}
                    self.pretrain_metrics.append(current_iter_metrics)
                    self._update_best_checkpoints(val_metrics, self.global_iteration_step)
                    if self.config['pretrain'].get('plot_loss_curves', True):
                        self._plot_pretrain_metrics()

                    checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
                    self.multi_label_model.save_models(checkpoint_dir, "pretrain_latest", self.global_iteration_step)
                    
                    # --- 新增：每10000次迭代无条件保存一次 ---
                    if self.global_iteration_step > 0 and self.global_iteration_step % 10000 == 0:
                        logger.info(f"[Checkpoint] 达到 {self.global_iteration_step} 次迭代，执行无条件保存...")
                        self.multi_label_model.save_models(checkpoint_dir, "unconditional_save", self.global_iteration_step)

                    self.multi_label_model.set_train_mode()

            del features_array, labels_dict, train_dataset, val_dataset, train_loader, val_loader
            import gc
            gc.collect()

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
        # TODO: Global 仿真阶段尚未更新以处理数据块。
        # 当前它假设 self.processed_data 和 self.user_video_lists 已加载到内存中。
        # 未来的优化需要修改此处的逻辑，以按需加载用户数据。
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
