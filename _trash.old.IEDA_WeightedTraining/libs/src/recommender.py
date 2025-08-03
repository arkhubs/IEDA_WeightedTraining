import torch
import numpy as np
import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

class Recommender:
    """推荐器类，负责实施推荐逻辑和GTE计算"""
    
    def __init__(self, config_manager, data_manager_methods, models, device):
        """
        初始化推荐器
        
        Args:
            config_manager: 配置管理器
            data_manager_methods: 数据管理方法实例
            models: 预测模型字典，格式为 {label_name: model}
            device: 计算设备
        """
        self.config = config_manager.get_config()
        self.global_config = config_manager.get_global_config()
        base_dir = config_manager.get_config().get('base_dir', os.getcwd())
        self.exp_dir = os.path.join(base_dir, config_manager.get_exp_dir())
        self.checkpoints_dir = os.path.join(base_dir, config_manager.get_checkpoints_dir())
        
        self.data_methods = data_manager_methods
        self.models = models  # 多个模型，用于多标签预测
        self.device = device
        
        self.logger = logging.getLogger('Recommender')
        
        # 获取alpha参数 - 针对每个标签
        self.label_names = data_manager_methods.data_manager.get_label_names()
        self.alpha_T = {}
        self.alpha_C = {}
        
        # 初始化每个标签的alpha权重
        for label_name in self.label_names:
            label_specific_config = self._get_label_specific_config(label_name)
            alpha_T = label_specific_config.get('alpha_T', 1.0)
            alpha_C = label_specific_config.get('alpha_C', 0.5)
            
            # 确保alpha值是数值类型
            if isinstance(alpha_T, str):
                alpha_T = float(alpha_T)
            if isinstance(alpha_C, str):
                alpha_C = float(alpha_C)
            
            self.alpha_T[label_name] = torch.tensor(alpha_T, device=self.device)
            self.alpha_C[label_name] = torch.tensor(alpha_C, device=self.device)
        
        # 记录主标签名称（用于兼容旧的指标计算）
        self.primary_label = self.label_names[0] if self.label_names else 'label'
        
        # 初始化仿真状态
        self.step = 0
        
        # 为每个标签维护单独的指标
        self.total_labels_T = {label_name: 0.0 for label_name in self.label_names}
        self.total_labels_C = {label_name: 0.0 for label_name in self.label_names}
        self.training_losses = {label_name: [] for label_name in self.label_names}
        self.validation_metrics = {
            label_name: {'steps': [], 'avg_rewards': [], 'loss': [], 'rel_error': [], 'auc': [], 'accuracy': []}
            for label_name in self.label_names
        }
        
        # 存储最新的结果
        self._latest_results = None
        
        # 最佳性能记录
        self.best_val_rewards = {label_name: 0.0 for label_name in self.label_names}
        self.best_steps = {label_name: 0 for label_name in self.label_names}
        
        # 累计指标
        self.metrics_history = {
            'steps': []
        }
        
        # 为每个标签添加指标
        for label_name in self.label_names:
            self.metrics_history[f'total_label_T_{label_name}'] = []
            self.metrics_history[f'total_label_C_{label_name}'] = []
            self.metrics_history[f'GTE_{label_name}'] = []
    
    def _get_label_specific_config(self, label_name):
        """获取特定标签的配置"""
        if 'labels' not in self.config:
            return {}
        
        for label_config in self.config['labels']:
            if label_config.get('name') == label_name:
                return label_config
                
        return {}
        
    def recommend(self, user_candidates, alphas, mode='train'):
        """
        为用户推荐视频
        
        Args:
            user_candidates: 用户候选视频字典 {user_id: [video_data, ...]}
            alphas: 权重参数字典 {label_name: alpha}
            mode: 训练模式或验证模式
            
        Returns:
            推荐结果列表 [(user_id, video_id, score), ...]
        """
        if not user_candidates:
            return []
        
        # 准备特征
        features, user_video_indices = self.data_methods.prepare_features_for_candidates(user_candidates)
        if features is None:
            return []
        
        # 将特征转换为张量
        numerical_features = torch.tensor(features['numerical_features'], device=self.device)
        categorical_features = torch.tensor(features['categorical_features'], device=self.device)
        
        # 对每个标签进行预测并加权
        weighted_sum_predictions = None
        
        for label_name, model in self.models.items():
            # 获取该标签对应的权重
            if label_name in alphas:
                # 如果传入了特定标签的alpha
                alpha = alphas[label_name]
            else:
                # 使用默认的alpha
                self.logger.warning(f"标签 {label_name} 没有提供alpha权重，使用默认值1.0")
                alpha = torch.tensor(1.0, device=self.device)
            
            # 预测
            model.eval()
            with torch.no_grad():
                predictions = model(numerical_features, categorical_features)
            
            # 对预测结果加权
            weighted_predictions = predictions * alpha
            
            # 累加加权后的预测结果
            if weighted_sum_predictions is None:
                weighted_sum_predictions = weighted_predictions
            else:
                weighted_sum_predictions += weighted_predictions
        
        # 如果没有任何模型预测
        if weighted_sum_predictions is None:
            return []
        
        recommendations = []
        
        # 对每个用户选择最佳候选视频
        for user_id, indices in user_video_indices.items():
            if not indices:
                continue
                
            # 获取该用户的所有候选视频预测分数
            user_scores = weighted_sum_predictions[indices].cpu().numpy()
            
            # 找到最高分的候选视频
            best_idx = np.argmax(user_scores)
            best_candidate_idx = indices[best_idx]
            best_score = user_scores[best_idx]
            
            # 获取候选视频信息
            candidate_idx_in_user_list = best_idx
            video_data = user_candidates[user_id][candidate_idx_in_user_list]
            video_id = video_data['video_id']
            
            # 添加推荐
            recommendations.append((user_id, video_id, best_score))
            
            # 如果是训练模式，标记视频为已使用
            if mode == 'train':
                self.data_methods.mark_videos_used([(user_id, video_id)])
        
        return recommendations
    
    def train_step(self, optimizers, criteria, user_ids, video_ids, labels_dict):
        """
        单步训练模型
        
        Args:
            optimizers: 优化器字典 {label_name: optimizer}
            criteria: 损失函数字典 {label_name: criterion}
            user_ids: 用户ID列表
            video_ids: 视频ID列表
            labels_dict: 标签字典 {label_name: labels_array}
            
        Returns:
            训练指标字典
        """
        from libs.src.evaluation_metrics import calculate_non_zero_relative_error, calculate_binary_metrics
        
        if not user_ids or not video_ids:
            return {}
        
        # 准备特征
        features = self.data_methods.data_manager.prepare_batch_features(user_ids, video_ids)
        if not features or 'numerical_features' not in features:
            return {}
        
        # 将特征转换为张量
        numerical_features = torch.tensor(features['numerical_features'], device=self.device)
        categorical_features = torch.tensor(features['categorical_features'], device=self.device)
        
        metrics = {}
        
        # 为每个标签训练对应的模型
        for label_name, labels in labels_dict.items():
            # 跳过没有对应模型的标签
            if label_name not in self.models:
                continue
                
            # 获取该标签对应的模型、优化器和损失函数
            model = self.models[label_name]
            optimizer = optimizers.get(label_name)
            criterion = criteria.get(label_name)
            
            if optimizer is None or criterion is None:
                continue
            
            # 将标签转换为张量
            labels_tensor = torch.tensor(labels, device=self.device)
            
            # 训练模式
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(numerical_features, categorical_features)
            
            # 计算损失
            loss = criterion(predictions, labels_tensor)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 初始化指标
            metrics[label_name] = {'loss': loss.item()}
            
            # 根据标签类型计算额外的指标
            label_type = self._get_label_specific_config(label_name).get('type', 'numerical')
            
            if label_type == 'binary':
                # 二分类指标
                binary_metrics = calculate_binary_metrics(predictions, labels)
                metrics[label_name].update(binary_metrics)
            else:  # numerical
                # 非零值相对误差
                rel_error = calculate_non_zero_relative_error(predictions, labels)
                metrics[label_name]['rel_error'] = float(rel_error)
            
            # 记录训练指标历史
            if 'train_metrics' not in self.__dict__:
                self.train_metrics = {}
            
            if label_name not in self.train_metrics:
                self.train_metrics[label_name] = {
                    'steps': [], 'loss': [], 'rel_error': [], 'auc': [], 'accuracy': []
                }
            
            # 只在特定步骤记录训练指标，以避免过多的存储
            if self.step % 10 == 0:
                self.train_metrics[label_name]['steps'].append(self.step)
                self.train_metrics[label_name]['loss'].append(metrics[label_name]['loss'])
                
                if 'auc' in metrics[label_name]:
                    self.train_metrics[label_name]['auc'].append(metrics[label_name]['auc'])
                if 'accuracy' in metrics[label_name]:
                    self.train_metrics[label_name]['accuracy'].append(metrics[label_name]['accuracy'])
                if 'rel_error' in metrics[label_name]:
                    self.train_metrics[label_name]['rel_error'].append(metrics[label_name]['rel_error'])
        
        return metrics
    
    def validate(self, alphas, optimizers=None, criteria=None):
        """
        在验证集上评估模型
        
        Args:
            alphas: 权重参数字典 {label_name: alpha}
            optimizers: 优化器字典 {label_name: optimizer}，可选
            criteria: 损失函数字典 {label_name: criterion}，可选
            
        Returns:
            验证指标
        """
        from libs.src.evaluation_metrics import calculate_non_zero_relative_error, calculate_binary_metrics
        
        self.logger.info(f"[validate] step {self.step}")
        
        # 获取验证集用户
        batch_size = self.global_config['batch_size']
        n_candidate = self.global_config['n_candidate']
        val_users = self.data_methods.sample_users(batch_size, mode='val')
        
        if not val_users:
            return {}
        
        # 为验证用户生成候选视频
        user_candidates = {}
        for user_id in val_users:
            # 验证时不考虑used和masked状态，获取所有交互过的视频
            videos = self.data_methods.data_manager.get_user_videos(user_id)
            if videos:
                # 随机选择n_candidate个视频
                if len(videos) > n_candidate:
                    selected_videos = np.random.choice(videos, n_candidate, replace=False).tolist()
                else:
                    selected_videos = videos
                user_candidates[user_id] = selected_videos
        
        # 使用Treatment策略进行推荐
        recommendations_T = self.recommend(user_candidates, self.alpha_T, mode='val')
        
        # 收集真实反馈
        user_ids_T = [rec[0] for rec in recommendations_T]
        video_ids_T = [rec[1] for rec in recommendations_T]
        true_labels_dict_T = self.data_methods.get_true_labels_dict(user_ids_T, video_ids_T)
        
        # 为验证准备特征，用于获取模型预测
        features, _ = self.data_methods.prepare_features_for_candidates({user_id: [{'video_id': vid}] for user_id, vid in zip(user_ids_T, video_ids_T)})
        if features is None:
            return {}
            
        numerical_features = torch.tensor(features['numerical_features'], device=self.device)
        categorical_features = torch.tensor(features['categorical_features'], device=self.device)
        
        # 为每个标签计算评估指标
        results = {}
        
        for label_name, true_labels_T in true_labels_dict_T.items():
            if true_labels_T.size > 0 and label_name in self.models:
                # 获取模型预测
                model = self.models[label_name]
                model.eval()
                with torch.no_grad():
                    predictions = model(numerical_features, categorical_features)
                
                # 初始化标签的指标存储
                if label_name not in self.validation_metrics:
                    self.validation_metrics[label_name] = {
                        'steps': [], 'avg_rewards': [], 'loss': [],
                        'rel_error': [], 'auc': [], 'accuracy': []
                    }
                
                # 根据标签类型计算不同的指标
                label_type = self._get_label_specific_config(label_name).get('type', 'numerical')
                
                # 通用指标
                avg_reward_T = true_labels_T.mean()
                true_labels_tensor = torch.tensor(true_labels_T, device=self.device)
                loss = criteria[label_name](predictions, true_labels_tensor).item() if label_name in criteria else 0.0
                
                # 记录基础指标
                self.validation_metrics[label_name]['steps'].append(self.step)
                self.validation_metrics[label_name]['avg_rewards'].append(float(avg_reward_T))
                self.validation_metrics[label_name]['loss'].append(float(loss))
                
                # 特定标签类型的指标
                if label_type == 'binary':
                    # 二分类指标
                    binary_metrics = calculate_binary_metrics(predictions, true_labels_T)
                    auc = binary_metrics['auc']
                    accuracy = binary_metrics['accuracy']
                    
                    self.validation_metrics[label_name]['auc'].append(float(auc))
                    self.validation_metrics[label_name]['accuracy'].append(float(accuracy))
                    
                    # 记录日志
                    self.logger.info(f"[validate] {label_name} 平均奖励: {avg_reward_T:.4f}, "
                                    f"AUC: {auc:.4f}, 准确率: {accuracy:.4f}, Loss: {loss:.4f}")
                    
                    # 检查是否是最佳性能 (使用AUC作为指标)
                    if auc > self.best_val_rewards.get(label_name, 0.0):
                        self.best_val_rewards[label_name] = auc
                        self.best_steps[label_name] = self.step
                        if optimizers and label_name in optimizers:
                            self._save_model(optimizers[label_name], f"best_{label_name}", label_name)
                        
                    results[label_name] = {
                        'reward': float(avg_reward_T),  # 添加reward键
                        'avg_reward': float(avg_reward_T),
                        'auc': float(auc),
                        'accuracy': float(accuracy),
                        'loss': float(loss)
                    }
                    
                else:  # numerical
                    # 计算非零值的相对误差
                    rel_error = calculate_non_zero_relative_error(predictions, true_labels_T)
                    self.validation_metrics[label_name]['rel_error'].append(float(rel_error))
                    
                    # 记录日志
                    self.logger.info(f"[validate] {label_name} 平均奖励: {avg_reward_T:.4f}, "
                                    f"非零相对误差: {rel_error:.2f}%, Loss: {loss:.4f}")
                    
                    # 检查是否是最佳性能 (使用相对误差作为指标，越小越好)
                    # 初始化为一个非常大的值
                    if 'best_rel_error' not in self.best_val_rewards:
                        self.best_val_rewards['best_rel_error'] = {}
                    
                    if label_name not in self.best_val_rewards['best_rel_error'] or rel_error < self.best_val_rewards['best_rel_error'].get(label_name, float('inf')):
                        self.best_val_rewards['best_rel_error'][label_name] = rel_error
                        self.best_steps[label_name] = self.step
                        if optimizers and label_name in optimizers:
                            self._save_model(optimizers[label_name], f"best_{label_name}", label_name)
                        
                    results[label_name] = {
                        'reward': float(avg_reward_T),  # 添加reward键
                        'avg_reward': float(avg_reward_T),
                        'rel_error': float(rel_error),
                        'loss': float(loss)
                    }
        
        # 如果没有标签的验证结果，返回空字典
        return results
    
    def run_simulation(self, optimizers, criteria):
        """
        运行仿真实验 - 已废弃，请使用libs.exp_modes.global_mode.GlobalExperiment
        
        Args:
            optimizers: 优化器字典 {label_name: optimizer}
            criteria: 损失函数字典 {label_name: criterion}
        """
        self.logger.warning("run_simulation方法已被废弃。请使用libs.exp_modes.global_mode.GlobalExperiment来进行全局实验。")
        raise NotImplementedError("run_simulation方法已被废弃，请使用GlobalExperiment")
    
    def run_simulation_old(self, optimizers, criteria):
        """
        旧版仿真实验实现（已废弃）
        
        Args:
            optimizers: 优化器字典 {label_name: optimizer}
            criteria: 损失函数字典 {label_name: criterion}
        """
        batch_size = self.global_config['batch_size']
        n_candidate = self.global_config['n_candidate']
        n_steps = self.global_config['n_steps']
        validate_every = self.global_config.get('validate_every', 50)
        save_every = self.global_config.get('save_every', 100)
        
        self.logger.info(f"[Sim] 开始仿真实验: steps={n_steps}, batch_size={batch_size}, n_candidate={n_candidate}")
        
        # 打印每个标签的权重
        for label_name in self.label_names:
            self.logger.info(f"[Sim] {label_name} Treatment组权重: {self.alpha_T.get(label_name, 0.0)}")
            self.logger.info(f"[Sim] {label_name} Control组权重: {self.alpha_C.get(label_name, 0.0)}")
        
        for step in range(self.step + 1, n_steps + 1):
            self.step = step
            
            # 抽样用户
            user_batch = self.data_methods.sample_users(batch_size, mode='train')
            
            # 为用户生成候选视频
            user_candidates = self.data_methods.generate_candidates(user_batch, n_candidate)
            
            if not user_candidates:
                self.logger.warning(f"[train] step {step}/{n_steps}: 没有足够的候选视频，跳过此步")
                continue
            
            # Treatment组推荐
            recommendations_T = self.recommend(user_candidates.copy(), self.alpha_T)
            
            # Control组推荐
            recommendations_C = self.recommend(user_candidates.copy(), self.alpha_C)
            
            # 收集真实反馈
            user_ids_T = [rec[0] for rec in recommendations_T]
            video_ids_T = [rec[1] for rec in recommendations_T]
            true_labels_dict_T = self.data_methods.get_true_labels_dict(user_ids_T, video_ids_T)
            
            user_ids_C = [rec[0] for rec in recommendations_C]
            video_ids_C = [rec[1] for rec in recommendations_C]
            true_labels_dict_C = self.data_methods.get_true_labels_dict(user_ids_C, video_ids_C)
            
            # 注意：以下代码违背了实验设计初衷
            # 全局模式应该是独立运行两次实验，而不是混合训练
            # 下面的代码仅作为历史参考
            
            # 训练模型（使用两组数据）
            all_user_ids = user_ids_T + user_ids_C
            all_video_ids = video_ids_T + video_ids_C
            
            # 合并标签
            all_labels_dict = {}
            for label_name in self.label_names:
                # 确保两组数据都有该标签
                if label_name in true_labels_dict_T and label_name in true_labels_dict_C:
                    all_labels_dict[label_name] = np.concatenate([
                        true_labels_dict_T[label_name], 
                        true_labels_dict_C[label_name]
                    ])
            
            # 训练每个模型
            losses = self.train_step(optimizers, criteria, all_user_ids, all_video_ids, all_labels_dict)
            
            # 记录损失
            for label_name, loss in losses.items():
                self.training_losses[label_name].append(loss)
            
            # 累积标签
            for label_name in self.label_names:
                if label_name in true_labels_dict_T:
                    self.total_labels_T[label_name] += true_labels_dict_T[label_name].sum()
                if label_name in true_labels_dict_C:
                    self.total_labels_C[label_name] += true_labels_dict_C[label_name].sum()
            
            # 记录当前步骤的指标
            self.metrics_history['steps'].append(step)
            
            # 为每个标签记录指标
            for label_name in self.label_names:
                total_label_T = self.total_labels_T.get(label_name, 0.0)
                total_label_C = self.total_labels_C.get(label_name, 0.0)
                gte = total_label_T - total_label_C
                
                self.metrics_history[f'total_label_T_{label_name}'].append(float(total_label_T))
                self.metrics_history[f'total_label_C_{label_name}'].append(float(total_label_C))
                self.metrics_history[f'GTE_{label_name}'].append(float(gte))
            
            # 每隔一定步数输出日志
            if step % 10 == 0 or step == 1:
                log_message = f"[train] step {step}/{n_steps}"
                
                # 添加每个标签的损失和GTE
                for label_name in self.label_names:
                    if label_name in losses:
                        loss = losses[label_name]
                        total_label_T = self.total_labels_T.get(label_name, 0.0)
                        total_label_C = self.total_labels_C.get(label_name, 0.0)
                        gte = total_label_T - total_label_C
                        
                        log_message += f"\n  {label_name}: loss={loss:.4f}, "
                        log_message += f"累计T={total_label_T:.2f}, 累计C={total_label_C:.2f}, "
                        log_message += f"GTE={gte:.2f}"
                
                self.logger.info(log_message)
            
            # 定期验证
            if step % validate_every == 0:
                self.validate(optimizers, criteria)
                
            # 定期保存模型
            if step % save_every == 0:
                for label_name, optimizer in optimizers.items():
                    self._save_model(optimizer, f"step_{step}", label_name)
                self._save_metrics()
                self._plot_metrics()
        
        # 实验结束，保存最终模型和指标
        self._save_model(optimizer, "final")
        self._save_metrics()
        self._plot_metrics()
        
        # 输出最终结果
        gte = self.total_label_T - self.total_label_C
        self.logger.info(f"实验完成: 累计收益T={self.total_label_T:.2f}, 累计收益C={self.total_label_C:.2f}, GTE={gte:.2f}")
        
    def _save_model(self, optimizer, tag, label_name=None):
        """
        保存模型检查点
        
        Args:
            optimizer: 优化器
            tag: 标签
            label_name: 标签名称，如果为None则使用主标签
        """
        if label_name is None:
            label_name = self.primary_label
            
        # 确保标签名称是有效的
        if label_name not in self.models:
            return
            
        model = self.models[label_name]
        
        # 构建文件名，包含标签名称
        file_name = f"{label_name}_{tag}.pt" if label_name != self.primary_label else f"{tag}.pt"
        checkpoint_path = os.path.join(self.checkpoints_dir, file_name)
        
        # 保存模型和相关状态
        torch.save({
            'step': self.step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'total_label_T': self.total_labels_T.get(label_name, 0.0),
            'total_label_C': self.total_labels_C.get(label_name, 0.0),
            'best_val_reward': self.best_val_rewards.get(label_name, 0.0),
            'best_step': self.best_steps.get(label_name, 0),
            'label_name': label_name
        }, checkpoint_path)
        
        self.logger.info(f"[Save] 已保存{label_name}模型检查点: {checkpoint_path}")
    
    def _save_metrics(self):
        """保存实验指标"""
        # 创建总体指标字典
        metrics = {
            'steps': self.metrics_history['steps'],
            'alpha_T': {name: float(value.cpu().numpy()) if isinstance(value, torch.Tensor) else value 
                       for name, value in self.alpha_T.items()},
            'alpha_C': {name: float(value.cpu().numpy()) if isinstance(value, torch.Tensor) else value 
                       for name, value in self.alpha_C.items()},
            'validation': {},
            'training': {},
            'labels': {}
        }
        
        # 添加训练指标
        if hasattr(self, 'train_metrics'):
            metrics['training'] = self.train_metrics
        
        # 为每个标签添加指标
        for label_name in self.label_names:
            label_metrics = {}
            
            # 添加累积奖励和GTE
            if f'total_label_T_{label_name}' in self.metrics_history:
                label_metrics['total_label_T'] = self.metrics_history[f'total_label_T_{label_name}']
            if f'total_label_C_{label_name}' in self.metrics_history:
                label_metrics['total_label_C'] = self.metrics_history[f'total_label_C_{label_name}']
            if f'GTE_{label_name}' in self.metrics_history:
                label_metrics['GTE'] = self.metrics_history[f'GTE_{label_name}']
            
            # 添加最佳验证指标
            if isinstance(self.best_val_rewards.get(label_name), float):
                label_metrics['best_val_reward'] = float(self.best_val_rewards.get(label_name, 0.0))
            elif isinstance(self.best_val_rewards.get('best_rel_error', {}), dict) and label_name in self.best_val_rewards['best_rel_error']:
                label_metrics['best_rel_error'] = float(self.best_val_rewards['best_rel_error'][label_name])
                
            label_metrics['best_step'] = self.best_steps.get(label_name, 0)
            
            # 计算最终GTE
            final_gte = self.total_labels_T.get(label_name, 0.0) - self.total_labels_C.get(label_name, 0.0)
            label_metrics['final_GTE'] = float(final_gte)
            
            # 添加验证指标
            if label_name in self.validation_metrics:
                val_metrics = {}
                for metric_name, metric_values in self.validation_metrics[label_name].items():
                    if metric_values:  # 确保有值
                        val_metrics[metric_name] = metric_values
                metrics['validation'][label_name] = val_metrics
            
            # 添加到总指标
            metrics['labels'][label_name] = label_metrics
        
        metrics_path = os.path.join(self.exp_dir, 'result.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info(f"[Save] 已保存实验指标: {metrics_path}")
        
        # 存储最新的结果供后续使用
        self._latest_results = metrics
        try:
            from libs.src.plot_results import plot_advanced_metrics
            plot_advanced_metrics(metrics, self.exp_dir)
            self.logger.info(f"[Plot] 已生成高级指标图表")
        except ImportError:
            self.logger.info(f"[Plot] 高级绘图功能未找到，使用内置简单绘图")
            self._plot_simple_metrics()
    
    def _plot_simple_metrics(self):
        """绘制简单的实验指标图表"""
        # 为每个标签创建单独的图表
        for label_name in self.label_names:
            plt.figure(figsize=(15, 12))
            fig_title = f"指标图表 - {label_name}"
            plt.suptitle(fig_title, fontsize=16)
            
            # 绘制GTE
            plt.subplot(3, 2, 1)
            if f'GTE_{label_name}' in self.metrics_history:
                plt.plot(self.metrics_history['steps'], self.metrics_history[f'GTE_{label_name}'])
            plt.title(f'Global Treatment Effect ({label_name})')
            plt.xlabel('Step')
            plt.ylabel('GTE')
            plt.grid(True)
            
            # 绘制累积奖励
            plt.subplot(3, 2, 2)
            if f'total_label_T_{label_name}' in self.metrics_history:
                plt.plot(self.metrics_history['steps'], self.metrics_history[f'total_label_T_{label_name}'], 
                         label='Treatment')
            if f'total_label_C_{label_name}' in self.metrics_history:
                plt.plot(self.metrics_history['steps'], self.metrics_history[f'total_label_C_{label_name}'], 
                         label='Control')
            plt.title(f'Cumulative Reward ({label_name})')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.legend()
            
            # 绘制训练损失
            plt.subplot(3, 2, 3)
            if hasattr(self, 'train_metrics') and label_name in self.train_metrics and self.train_metrics[label_name]['loss']:
                plt.plot(self.train_metrics[label_name]['steps'], self.train_metrics[label_name]['loss'], label='Train')
            if label_name in self.validation_metrics and 'loss' in self.validation_metrics[label_name]:
                plt.plot(self.validation_metrics[label_name]['steps'], self.validation_metrics[label_name]['loss'], label='Validation')
            plt.title(f'Loss ({label_name})')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # 根据标签类型绘制不同的指标
            label_type = self._get_label_specific_config(label_name).get('type', 'numerical')
            
            if label_type == 'binary':
                # 绘制AUC
                plt.subplot(3, 2, 4)
                if hasattr(self, 'train_metrics') and label_name in self.train_metrics and 'auc' in self.train_metrics[label_name]:
                    plt.plot(self.train_metrics[label_name]['steps'], self.train_metrics[label_name]['auc'], label='Train')
                if label_name in self.validation_metrics and 'auc' in self.validation_metrics[label_name]:
                    plt.plot(self.validation_metrics[label_name]['steps'], self.validation_metrics[label_name]['auc'], label='Validation')
                plt.title(f'AUC ({label_name})')
                plt.xlabel('Step')
                plt.ylabel('AUC')
                plt.grid(True)
                plt.legend()
                
                # 绘制准确率
                plt.subplot(3, 2, 5)
                if hasattr(self, 'train_metrics') and label_name in self.train_metrics and 'accuracy' in self.train_metrics[label_name]:
                    plt.plot(self.train_metrics[label_name]['steps'], self.train_metrics[label_name]['accuracy'], label='Train')
                if label_name in self.validation_metrics and 'accuracy' in self.validation_metrics[label_name]:
                    plt.plot(self.validation_metrics[label_name]['steps'], self.validation_metrics[label_name]['accuracy'], label='Validation')
                plt.title(f'Accuracy ({label_name})')
                plt.xlabel('Step')
                plt.ylabel('Accuracy')
                plt.grid(True)
                plt.legend()
                
            else:  # numerical
                # 绘制相对误差
                plt.subplot(3, 2, 4)
                if hasattr(self, 'train_metrics') and label_name in self.train_metrics and 'rel_error' in self.train_metrics[label_name]:
                    plt.plot(self.train_metrics[label_name]['steps'], self.train_metrics[label_name]['rel_error'], label='Train')
                if label_name in self.validation_metrics and 'rel_error' in self.validation_metrics[label_name]:
                    plt.plot(self.validation_metrics[label_name]['steps'], self.validation_metrics[label_name]['rel_error'], label='Validation')
                plt.title(f'非零值相对误差 % ({label_name})')
                plt.xlabel('Step')
                plt.ylabel('相对误差 %')
                plt.grid(True)
                plt.legend()
                
                # 绘制平均奖励
                plt.subplot(3, 2, 5)
                if label_name in self.validation_metrics and 'avg_rewards' in self.validation_metrics[label_name]:
                    plt.plot(self.validation_metrics[label_name]['steps'], self.validation_metrics[label_name]['avg_rewards'], label='Validation')
                plt.title(f'Average Reward ({label_name})')
                plt.xlabel('Step')
                plt.ylabel('Average Reward')
                plt.grid(True)
                plt.legend()
            
            # 保存图表
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(self.exp_dir, f'metrics_{label_name}.png'))
            plt.close()
            
        # 绘制整体指标
        plt.figure(figsize=(15, 10))
        plt.suptitle('整体指标', fontsize=16)
        
        # 为每个标签绘制GTE曲线
        plt.subplot(2, 2, 1)
        for label_name in self.label_names:
            if f'GTE_{label_name}' in self.metrics_history:
                plt.plot(self.metrics_history['steps'], self.metrics_history[f'GTE_{label_name}'], label=label_name)
        plt.title('Global Treatment Effects')
        plt.xlabel('Step')
        plt.ylabel('GTE')
        plt.grid(True)
        plt.legend()
        
        # 保存整体图表
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.exp_dir, 'metrics_overall.png'))
        plt.close()
        
    def _plot_metrics(self):
        """绘制实验指标图表（现在调用高级版本）"""
        self._save_metrics()  # 保存指标会调用绘图函数
        plt.grid(True)
        
        # 绘制训练损失
        if self.training_losses:
            plt.subplot(2, 2, 3)
            plt.plot(range(1, len(self.training_losses) + 1), self.training_losses)
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True)
        
        # 绘制验证奖励
        if self.validation_metrics['steps']:
            plt.subplot(2, 2, 4)
            plt.plot(self.validation_metrics['steps'], self.validation_metrics['avg_rewards'])
            plt.title('Validation Average Reward')
            plt.xlabel('Step')
            plt.ylabel('Average Reward')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'plot.svg'))
        plt.savefig(os.path.join(self.exp_dir, 'plot.png'))
        plt.close()
        
        self.logger.info(f"已保存指标图表: {os.path.join(self.exp_dir, 'plot.svg')}")
    
    def load_checkpoint(self, optimizer, checkpoint_path):
        """
        加载检查点
        
        Args:
            optimizer: 优化器
            checkpoint_path: 检查点路径
            
        Returns:
            是否成功加载
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.total_label_T = checkpoint['total_label_T']
            self.total_label_C = checkpoint['total_label_C']
            self.best_val_reward = checkpoint.get('best_val_reward', 0.0)
            self.best_step = checkpoint.get('best_step', 0)
            
            self.logger.info(f"已加载检查点: {checkpoint_path}")
            self.logger.info(f"恢复状态: step={self.step}, total_label_T={self.total_label_T:.2f}, "
                          f"total_label_C={self.total_label_C:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
            return False
            
    def get_results(self):
        """
        获取最新的实验结果
        
        Returns:
            结果数据字典
        """
        # 如果没有缓存的结果，先保存一次
        if self._latest_results is None:
            self._save_metrics()
            
        return self._latest_results
