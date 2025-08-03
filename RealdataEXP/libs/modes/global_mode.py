"""
Global模式实现
计算真实GTE的核心模块（GT与GC对称运行）
"""

import os
import torch
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from ..data import KuaiRandDataLoader, FeatureProcessor
from ..models import MultiLabelModel
from ..utils import MetricsTracker, save_results
import random

logger = logging.getLogger(__name__)

class GlobalMode:
    """Global模式实验管理器"""
    
    def __init__(self, config: Dict, exp_dir: str):
        self.config = config
        self.exp_dir = exp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # 仿真状态
        self.used_videos = set()  # 记录已使用的视频
        
        # 指标跟踪
        self.metrics_tracker = MetricsTracker()
        
        # 仿真结果
        self.total_label_T = {label['name']: 0.0 for label in config['labels']}  # Treatment组累计收益
        self.total_label_C = {label['name']: 0.0 for label in config['labels']}  # Control组累计收益
        
        logger.info(f"[Global模式] 初始化完成，设备: {self.device}")
    
    def ensure_float_data(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """确保数据为float类型并转换为numpy数组"""
        try:
            # 提取指定列
            subset = data[columns].copy()
            
            # 逐列强制转换为数值类型
            for col in columns:
                if col in subset.columns:
                    # 先尝试转换为数值类型
                    subset[col] = pd.to_numeric(subset[col], errors='coerce')
                    # 填充NaN值
                    subset[col] = subset[col].fillna(0.0)
                    # 确保是float类型
                    subset[col] = subset[col].astype(np.float32)
            
            # 转换为numpy数组
            array = subset.values.astype(np.float32)
            
            # 检查是否还有非数值类型
            if array.dtype == np.object_:
                logger.error(f"[数据转换] 数据中仍有非数值类型，列: {columns}")
                # 强制转换每个元素
                array = np.array([[float(x) if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else 0.0 
                                 for x in row] for row in array], dtype=np.float32)
            
            return array
            
        except Exception as e:
            logger.error(f"[数据转换] 转换失败: {e}")
            # 创建零矩阵作为备选
            return np.zeros((len(data), len(columns)), dtype=np.float32)
    
    def load_and_prepare_data(self):
        """加载和准备数据"""
        logger.info("[Global模式] 开始数据加载和准备...")
        
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
        logger.info(f"[特征处理] 特征列: {feature_columns}")
        
        # 初始化多标签模型
        self.multi_label_model = MultiLabelModel(
            config=self.config,
            input_dim=input_dim,
            device=self.device
        )
        
        logger.info("[Global模式] 数据准备完成")
    
    def pretrain_models(self):
        """预训练模型"""
        if not self.config['pretrain']['enabled']:
            logger.info("[预训练] 跳过预训练阶段")
            return
        
        logger.info("[预训练] 开始预训练阶段...")
        
        # 准备训练数据
        train_data = self.processed_data[self.processed_data['mask'] == 0].copy()
        logger.info(f"[预训练] 训练数据量: {len(train_data)}")
        
        # 创建数据加载器
        batch_size = self.config['pretrain']['batch_size']
        
        for epoch in range(self.config['pretrain']['epochs']):
            logger.info(f"[预训练] Epoch {epoch+1}/{self.config['pretrain']['epochs']}")
            
            # 随机打乱数据
            train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)
            
            epoch_losses = {label['name']: [] for label in self.config['labels']}
            
            # 批次训练
            for i in range(0, len(train_data_shuffled), batch_size):
                batch_data = train_data_shuffled[i:i+batch_size]
                
                if len(batch_data) == 0:
                    continue
                
                # 准备特征和标签
                feature_columns = self.feature_processor.get_feature_columns()
                
                # 使用新的数据转换函数
                feature_array = self.ensure_float_data(batch_data, feature_columns)
                X = torch.FloatTensor(feature_array).to(self.device)
                
                targets = {}
                for label_config in self.config['labels']:
                    label_name = label_config['name']
                    target_col = label_config['target']
                    y = torch.FloatTensor(batch_data[target_col].values).unsqueeze(1).to(self.device)
                    targets[label_name] = y
                
                # 训练步骤
                losses = self.multi_label_model.train_step(X, targets)
                
                for label_name, loss in losses.items():
                    epoch_losses[label_name].append(loss)
            
            # 记录epoch结果
            avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items() if losses}
            loss_str = ", ".join([f"{name}: {loss:.6f}" for name, loss in avg_losses.items()])
            logger.info(f"[预训练] Epoch {epoch+1} 平均损失 - {loss_str}")
        
        logger.info("[预训练] 预训练阶段完成")
    
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
    
    def get_user_video_features(self, user_id: int, video_ids: List[int]) -> torch.Tensor:
        """获取用户-视频对的特征"""
        features_list = []
        feature_columns = self.feature_processor.get_feature_columns()
        
        for video_id in video_ids:
            # 获取该用户-视频对的特征
            row = self.processed_data[
                (self.processed_data['user_id'] == user_id) & 
                (self.processed_data['video_id'] == video_id)
            ]
            
            if len(row) > 0:
                # 使用新的数据转换函数
                feature_array = self.ensure_float_data(row, feature_columns)
                if len(feature_array) > 0:
                    features_list.append(feature_array[0])
        
        if features_list:
            feature_array = np.array(features_list, dtype=np.float32)
            return torch.FloatTensor(feature_array).to(self.device)
        else:
            return torch.empty(0, len(feature_columns)).to(self.device)
    
    def get_real_labels(self, user_id: int, video_ids: List[int]) -> Dict[str, torch.Tensor]:
        """获取真实标签"""
        labels = {label['name']: [] for label in self.config['labels']}
        
        for video_id in video_ids:
            # 获取真实标签
            row = self.merged_data[
                (self.merged_data['user_id'] == user_id) & 
                (self.merged_data['video_id'] == video_id)
            ]
            
            if len(row) > 0:
                for label_config in self.config['labels']:
                    label_name = label_config['name']
                    target_col = label_config['target']
                    label_value = row[target_col].values[0]
                    labels[label_name].append(label_value)
        
        # 转换为tensor
        result = {}
        for label_name, values in labels.items():
            if values:
                result[label_name] = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        return result
    
    def run_single_simulation(self, is_treatment: bool, step: int, batch_users: List[int]) -> Dict[str, float]:
        """运行单次仿真步骤"""
        prefix = "Treatment" if is_treatment else "Control"
        logger.info(f"[{prefix}仿真] Step {step}: 开始处理 {len(batch_users)} 个用户")
        
        step_rewards = {label['name']: 0.0 for label in self.config['labels']}
        processed_users = 0
        
        for user_id in batch_users:
            # 1. 候选视频生成
            candidates = self.sample_candidate_videos(user_id, self.config['global']['n_candidate'])
            
            if len(candidates) == 0:
                continue  # 该用户没有可用视频
            
            # 2. 获取特征
            X = self.get_user_video_features(user_id, candidates)
            
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
            winner_idx = torch.argmax(combined_scores.squeeze()).item()
            winner_video = candidates[winner_idx]
            
            # 5. 获取真实反馈
            real_labels = self.get_real_labels(user_id, [winner_video])
            winner_features = X[winner_idx:winner_idx+1]
            
            # 6. 模型训练
            if real_labels:
                _ = self.multi_label_model.train_step(winner_features, real_labels)
                
                # 累加收益
                for label_name, label_tensor in real_labels.items():
                    reward_value = label_tensor.item()
                    step_rewards[label_name] += reward_value
            
            # 7. 更新used状态
            self.used_videos.add(winner_video)
            
            processed_users += 1
        
        logger.info(f"[{prefix}仿真] Step {step}: 处理了 {processed_users} 个用户")
        return step_rewards
    
    def validate_models(self, step: int):
        """验证模型性能"""
        logger.info(f"[验证] Step {step}: 开始验证")
        
        # 使用验证集用户
        val_sample_size = min(100, len(self.val_users))  # 限制验证样本数量
        val_users_sample = random.sample(self.val_users, val_sample_size)
        
        total_losses = {label['name']: [] for label in self.config['labels']}
        
        for user_id in val_users_sample:
            # 获取该用户的所有视频（不考虑used和mask）
            user_videos = self.user_video_lists.get(user_id, [])
            
            if len(user_videos) == 0:
                continue
            
            # 随机选择几个视频进行验证
            sample_videos = random.sample(user_videos, min(5, len(user_videos)))
            
            X = self.get_user_video_features(user_id, sample_videos)
            real_labels = self.get_real_labels(user_id, sample_videos)
            
            if X.size(0) > 0 and real_labels:
                losses = self.multi_label_model.evaluate(X, real_labels)
                for label_name, loss in losses.items():
                    total_losses[label_name].append(loss)
        
        # 计算平均验证损失
        avg_val_losses = {}
        for label_name, losses in total_losses.items():
            if losses:
                avg_val_losses[f'val_{label_name}_loss'] = np.mean(losses)
        
        self.metrics_tracker.update(avg_val_losses, step)
        self.metrics_tracker.log_current("验证")
        
        # 更新学习率调度器
        self.multi_label_model.update_schedulers(avg_val_losses)
    
    def run_global_simulation(self):
        """运行Global仿真"""
        logger.info("[Global仿真] 开始全局仿真流程...")
        
        n_steps = self.config['global']['n_steps']
        batch_size = self.config['global']['batch_size']
        validate_every = self.config['global']['validate_every']
        save_every = self.config['global']['save_every']
        
        for step in range(1, n_steps + 1):
            logger.info(f"[Global仿真] ===== Step {step}/{n_steps} =====")
            
            # 1. 用户批次抽样（GT和GC使用相同的用户批次）
            batch_users = random.sample(self.train_users, min(batch_size, len(self.train_users)))
            
            # 2. Treatment仿真（GT）
            logger.info("[GT流程] 开始Treatment组仿真...")
            step_rewards_T = self.run_single_simulation(True, step, batch_users)
            
            # 累加到总收益
            for label_name, reward in step_rewards_T.items():
                self.total_label_T[label_name] += reward
            
            # 3. Control仿真（GC）
            logger.info("[GC流程] 开始Control组仿真...")
            step_rewards_C = self.run_single_simulation(False, step, batch_users)
            
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
        
        logger.info("[Global仿真] 全局仿真完成")
    
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
        logger.info("[Global模式] 开始运行完整实验...")
        
        try:
            # 1. 数据加载和准备
            self.load_and_prepare_data()
            
            # 2. 预训练
            self.pretrain_models()
            
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
            
            logger.info("[Global模式] 实验完成！")
            
            # 打印最终结果
            logger.info("========== 最终实验结果 ==========")
            for key, value in gte_results.items():
                logger.info(f"{key}: {value}")
                
        except Exception as e:
            logger.error(f"[Global模式] 实验执行失败: {e}")
            raise