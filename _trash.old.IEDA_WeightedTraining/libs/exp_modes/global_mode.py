"""
Global 实验模式

该模式独立运行两次实验：一次全员使用处理组推荐参数，一次全员使用对照组参数。
两次仿真完全独立，不共享训练数据或使用记录。
"""

import numpy as np
import torch
import logging
import os
import time

class GlobalExperiment:
    """全局实验模式实现"""
    
    def __init__(self, config_manager, data_manager_methods, models, device):
        """
        初始化全局实验模式
        
        Args:
            config_manager: 配置管理器
            data_manager_methods: 数据管理方法
            models: 模型字典 {标签名: 模型}
            device: 计算设备
        """
        self.config = config_manager.get_config()
        self.global_config = config_manager.get_global_config()
        self.exp_dir = config_manager.get_exp_dir()
        
        self.data_methods = data_manager_methods
        self.models = models
        self.device = device
        
        self.logger = logging.getLogger('GlobalExperiment')
        
        # 获取标签名称
        self.label_names = data_manager_methods.data_manager.get_label_names()
        
        # 获取每个标签的alpha参数
        self.alpha_T = {}
        self.alpha_C = {}
        
        for label_name in self.label_names:
            label_specific_config = self._get_label_specific_config(label_name, config_manager)
            alpha_T = label_specific_config.get('alpha_T', 1.0)
            alpha_C = label_specific_config.get('alpha_C', 0.5)
            
            # 确保alpha值是数值类型
            if isinstance(alpha_T, str):
                alpha_T = float(alpha_T)
            if isinstance(alpha_C, str):
                alpha_C = float(alpha_C)
            
            self.alpha_T[label_name] = torch.tensor(alpha_T, device=self.device)
            self.alpha_C[label_name] = torch.tensor(alpha_C, device=self.device)
        
        # 初始化记录指标
        self.step = 0
        self.metrics_T = {
            'steps': [],
            'total_rewards': {label_name: 0.0 for label_name in self.label_names},
            'total_samples': {label_name: 0 for label_name in self.label_names},  # 添加样本计数
            'cumulative_rewards': {label_name: [] for label_name in self.label_names},
            'training_losses': {label_name: [] for label_name in self.label_names},
            'validation': {
                label_name: {'steps': [], 'rewards': [], 'metrics': {}} 
                for label_name in self.label_names
            }
        }
        
        self.metrics_C = {
            'steps': [],
            'total_rewards': {label_name: 0.0 for label_name in self.label_names},
            'total_samples': {label_name: 0 for label_name in self.label_names},  # 添加样本计数
            'cumulative_rewards': {label_name: [] for label_name in self.label_names},
            'training_losses': {label_name: [] for label_name in self.label_names},
            'validation': {
                label_name: {'steps': [], 'rewards': [], 'metrics': {}} 
                for label_name in self.label_names
            }
        }
        
        # 最佳模型记录
        self.best_models_T = {label_name: None for label_name in self.label_names}
        self.best_rewards_T = {label_name: 0.0 for label_name in self.label_names}
        self.best_steps_T = {label_name: 0 for label_name in self.label_names}
        
        self.best_models_C = {label_name: None for label_name in self.label_names}
        self.best_rewards_C = {label_name: 0.0 for label_name in self.label_names}
        self.best_steps_C = {label_name: 0 for label_name in self.label_names}
        
        # 最终结果
        self.results = {
            'treatment': {label_name: 0.0 for label_name in self.label_names},
            'control': {label_name: 0.0 for label_name in self.label_names},
            'GTE': {label_name: 0.0 for label_name in self.label_names}
        }
    
    def _get_label_specific_config(self, label_name, config_manager):
        """获取特定标签的配置"""
        config = config_manager.get_config()
        if 'labels' not in config:
            return {}
        
        for label_config in config['labels']:
            if label_config.get('name') == label_name:
                return label_config
                
        return {}
    
    def run_experiment(self, recommender, optimizers, criteria):
        """
        运行全局实验
        
        Args:
            recommender: 推荐器实例
            optimizers: 优化器字典 {标签名: 优化器}
            criteria: 损失函数字典 {标签名: 损失函数}
        
        Returns:
            实验结果字典
        """
        # 首先运行处理组实验
        self.logger.info("[Global] 开始运行处理组(Treatment)实验")
        self._run_single_experiment(
            recommender, optimizers, criteria, 
            self.alpha_T, self.metrics_T, 
            self.best_models_T, self.best_rewards_T, self.best_steps_T,
            "Treatment"
        )
        
        # 保存处理组模型
        self._save_models(recommender, "treatment")
        
        # 重置模型状态
        self._reset_models(recommender)
        
        # 运行对照组实验
        self.logger.info("[Global] 开始运行对照组(Control)实验")
        self._run_single_experiment(
            recommender, optimizers, criteria, 
            self.alpha_C, self.metrics_C, 
            self.best_models_C, self.best_rewards_C, self.best_steps_C,
            "Control"
        )
        
        # 保存对照组模型
        self._save_models(recommender, "control")
        
        # 计算最终的GTE（使用平均值）
        for label_name in self.label_names:
            # 计算平均奖励
            avg_reward_T = self.metrics_T['total_rewards'][label_name] / max(1, self.metrics_T['total_samples'][label_name])
            avg_reward_C = self.metrics_C['total_rewards'][label_name] / max(1, self.metrics_C['total_samples'][label_name])
            
            self.results['treatment'][label_name] = avg_reward_T
            self.results['control'][label_name] = avg_reward_C
            self.results['GTE'][label_name] = avg_reward_T - avg_reward_C
            
            self.logger.info(f"[Global] 标签 {label_name} 的GTE: {self.results['GTE'][label_name]:.4f}")
            self.logger.info(f"  - Treatment组平均奖励: {self.results['treatment'][label_name]:.4f} (总样本: {self.metrics_T['total_samples'][label_name]})")
            self.logger.info(f"  - Control组平均奖励: {self.results['control'][label_name]:.4f} (总样本: {self.metrics_C['total_samples'][label_name]})")
        
        # 保存最终结果
        self._save_results()
        
        return self.results
    
    def _run_single_experiment(self, recommender, optimizers, criteria, alphas, metrics, best_models, best_rewards, best_steps, exp_type):
        """运行单次实验（处理组或对照组）"""
        n_steps = self.global_config.get('n_steps', 1000)
        val_interval = self.global_config.get('validate_every', 50)
        user_batch_size = self.global_config.get('batch_size', 64)
        
        start_time = time.time()
        
        for step in range(n_steps):
            # 更新步数
            self.step = step + 1
            metrics['steps'].append(self.step)
            
            # 抽取用户批次
            user_batch = self.data_methods.sample_users(batch_size=user_batch_size)
            
            # 为每个用户生成候选视频
            user_candidates = {}
            for user_id in user_batch:
                candidates = self.data_methods.get_user_candidates(user_id)
                if candidates:
                    user_candidates[user_id] = candidates
            
            # 生成推荐
            recommendations = recommender.recommend(user_candidates.copy(), alphas)
            
            # 收集真实反馈
            user_ids = [rec[0] for rec in recommendations]
            video_ids = [rec[1] for rec in recommendations]
            true_labels_dict = self.data_methods.get_true_labels_dict(user_ids, video_ids)
            
            # 训练模型
            losses = recommender.train_step(optimizers, criteria, user_ids, video_ids, true_labels_dict)
            
            # 记录损失
            for label_name, loss in losses.items():
                metrics['training_losses'][label_name].append(loss)
            
            # 累积标签值
            for label_name in self.label_names:
                if label_name in true_labels_dict:
                    metrics['total_rewards'][label_name] += true_labels_dict[label_name].sum()
                    metrics['total_samples'][label_name] += len(true_labels_dict[label_name])  # 记录样本数
                    metrics['cumulative_rewards'][label_name].append(metrics['total_rewards'][label_name])
            
            # 定期验证
            if (step + 1) % val_interval == 0:
                self.logger.info(f"[{exp_type}] 步骤 {step+1}/{n_steps} 完成, 耗时 {time.time() - start_time:.2f}s")
                
                # 运行验证
                validation_results = recommender.validate(alphas, criteria=criteria)
                
                # 记录验证结果
                for label_name, val_metrics in validation_results.items():
                    metrics['validation'][label_name]['steps'].append(step + 1)
                    metrics['validation'][label_name]['rewards'].append(val_metrics['reward'])
                    
                    # 更新其他指标
                    for metric_name, metric_value in val_metrics.items():
                        if metric_name != 'reward':
                            if metric_name not in metrics['validation'][label_name]['metrics']:
                                metrics['validation'][label_name]['metrics'][metric_name] = []
                            metrics['validation'][label_name]['metrics'][metric_name].append(metric_value)
                    
                    # 检查是否是最佳模型
                    if val_metrics['reward'] > best_rewards[label_name]:
                        best_rewards[label_name] = val_metrics['reward']
                        best_steps[label_name] = step + 1
                        
                        # 保存模型
                        best_models[label_name] = recommender.models[label_name].state_dict().copy()
                        
                        self.logger.info(f"[{exp_type}] 标签 {label_name} 的新最佳模型 @ 步骤 {step+1}: 奖励 = {val_metrics['reward']:.4f}")
                
                # 重置计时器
                start_time = time.time()
    
    def _save_models(self, recommender, experiment_type):
        """保存模型"""
        checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints', experiment_type)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        for label_name, model in recommender.models.items():
            # 保存最佳模型
            best_model_path = os.path.join(checkpoints_dir, f"best_{label_name}.pt")
            if experiment_type == "treatment" and self.best_models_T[label_name] is not None:
                torch.save({
                    'model_state_dict': self.best_models_T[label_name],
                    'step': self.best_steps_T[label_name],
                    'reward': self.best_rewards_T[label_name]
                }, best_model_path)
            elif experiment_type == "control" and self.best_models_C[label_name] is not None:
                torch.save({
                    'model_state_dict': self.best_models_C[label_name],
                    'step': self.best_steps_C[label_name],
                    'reward': self.best_rewards_C[label_name]
                }, best_model_path)
            
            # 保存最终模型
            final_model_path = os.path.join(checkpoints_dir, f"final_{label_name}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': self.step,
                'experiment_type': experiment_type
            }, final_model_path)
    
    def _reset_models(self, recommender):
        """重置模型到初始状态"""
        for label_name, model in recommender.models.items():
            # 重新初始化模型权重
            def weight_reset(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            model.apply(weight_reset)
            
            # 重置优化器
            # 注意：这应该在调用此方法之后在外部完成
    
    def _save_results(self):
        """保存实验结果"""
        import json
        import numpy as np
        
        def convert_to_serializable(obj):
            """将numpy类型转换为JSON可序列化的类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        # 创建结果字典
        results = {
            'global_experiment': {
                'treatment': {
                    'steps': convert_to_serializable(self.metrics_T['steps']),
                    'labels': {}
                },
                'control': {
                    'steps': convert_to_serializable(self.metrics_C['steps']),
                    'labels': {}
                },
                'GTE': {}
            }
        }
        
        # 添加每个标签的指标
        for label_name in self.label_names:
            # 处理组指标
            results['global_experiment']['treatment']['labels'][label_name] = {
                'cumulative_rewards': convert_to_serializable(self.metrics_T['cumulative_rewards'][label_name]),
                'total_reward': convert_to_serializable(self.metrics_T['total_rewards'][label_name]),
                'training_losses': convert_to_serializable(self.metrics_T['training_losses'][label_name])
            }
            
            # 添加验证指标
            if 'validation' in self.metrics_T and label_name in self.metrics_T['validation']:
                validation_T = self.metrics_T['validation'][label_name]
                results['global_experiment']['treatment']['labels'][label_name]['validation'] = {
                    'steps': convert_to_serializable(validation_T['steps']),
                    'rewards': convert_to_serializable(validation_T['rewards'])
                }
                
                # 添加其他验证指标
                if 'metrics' in validation_T:
                    for metric_name, metric_values in validation_T['metrics'].items():
                        results['global_experiment']['treatment']['labels'][label_name]['validation'][metric_name] = convert_to_serializable(metric_values)
            
            # 对照组指标
            results['global_experiment']['control']['labels'][label_name] = {
                'cumulative_rewards': convert_to_serializable(self.metrics_C['cumulative_rewards'][label_name]),
                'total_reward': convert_to_serializable(self.metrics_C['total_rewards'][label_name]),
                'training_losses': convert_to_serializable(self.metrics_C['training_losses'][label_name])
            }
            
            # 添加验证指标
            if 'validation' in self.metrics_C and label_name in self.metrics_C['validation']:
                validation_C = self.metrics_C['validation'][label_name]
                results['global_experiment']['control']['labels'][label_name]['validation'] = {
                    'steps': convert_to_serializable(validation_C['steps']),
                    'rewards': convert_to_serializable(validation_C['rewards'])
                }
                
                # 添加其他验证指标
                if 'metrics' in validation_C:
                    for metric_name, metric_values in validation_C['metrics'].items():
                        results['global_experiment']['control']['labels'][label_name]['validation'][metric_name] = convert_to_serializable(metric_values)
            
            # GTE 结果
            results['global_experiment']['GTE'][label_name] = convert_to_serializable(self.results['GTE'][label_name])
        
        # 保存结果
        result_path = os.path.join(self.exp_dir, 'global_result.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"[Global] 已保存全局实验结果: {result_path}")
        
        return results
