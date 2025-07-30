import sys
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np
 

from src.data_manager import DataManager, HistoryBuffer
from src.models import PredictionModel, WeightingModel
from src.trainers import WeightedTrainer, PoolingTrainer, SplittingTrainer, SnapshotTrainer
from src.recommender import Recommender
from src.utils import setup_logging, save_model, save_results, compute_metrics, compute_roc_auc, compute_log_mae
from torch.utils.data import DataLoader


import datetime
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../configs/experiment_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
# 强制修正 data_path 为实际数据集绝对路径
config['data_path'] = '/home/zhixuanhu/IEDA_WeightedTraining/KuaiRand/Pure/data'
# 设置实验结果保存目录为 /home/zhixuanhu/IEDA_WeightedTraining/results/时间戳/
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
base_results_dir = '/home/zhixuanhu/IEDA_WeightedTraining/results'
result_dir = os.path.join(base_results_dir, timestamp)
os.makedirs(result_dir, exist_ok=True)
config['results_path'] = result_dir

def pretrain_models(config, logger, data_manager, device):
    """预训练阶段：从数据集纯随机选择5000个交互进行预训练"""
    pretrain_size = config.get('initial_dataset_size', 5000)
    logger.info(f"[Pretrain] 开始预训练，随机选择 {pretrain_size} 个交互")
    
    # 从数据集中纯随机选择样本
    all_interactions = list(data_manager.get_all_interactions())
    if len(all_interactions) < pretrain_size:
        logger.warning(f"数据集只有 {len(all_interactions)} 个交互，小于预期的 {pretrain_size}")
        pretrain_size = len(all_interactions)
    
    # 纯随机选择
    import random
    random.shuffle(all_interactions)
    pretrain_interactions = all_interactions[:pretrain_size]
    
    # 构建预训练数据
    pretrain_buffer = HistoryBuffer()
    for interaction in pretrain_interactions:
        user_id = interaction['user_id']
        video_id = interaction['video_id']
        # 随机分配treatment（纯随机，不通过推荐流程）
        Z = torch.bernoulli(torch.tensor(config['p_treatment'])).item()
        
        # 获取特征和标签
        interaction_features = data_manager.create_interaction_features(user_id, [video_id])
        X_observed = interaction_features[0]
        Y_observed = data_manager.get_ground_truth_label(user_id, video_id)
        
        pretrain_buffer.add(X_observed, Y_observed, torch.tensor(Z))
    
    # 创建预训练模型（用于预训练的临时模型）
    input_dim = config['prediction_model']['input_dim']
    shared_dims = config['prediction_model']['shared_dims']
    ctr_head_dims = config['prediction_model']['ctr_head_dims']
    playtime_head_dims = config['prediction_model']['playtime_head_dims']
    
    pretrain_model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
    
    # 使用Pooling方式进行预训练（简单直接）
    pretrain_trainer = PoolingTrainer(config, pretrain_model, device=device)
    
    # 创建DataLoader
    pretrain_loader = DataLoader(pretrain_buffer, batch_size=config.get('batch_size', 128), shuffle=True)
    
    # 进行预训练
    logger.info(f"[Pretrain] 开始训练，数据量: {len(pretrain_buffer)}")
    pretrain_trainer.train_on_pooling(pretrain_loader)
    
    # 保存预训练权重
    pretrain_weights_path = os.path.join(config['results_path'], 'pretrain_weights.pth')
    torch.save(pretrain_model.state_dict(), pretrain_weights_path)
    logger.info(f"[Pretrain] 预训练完成，权重已保存到: {pretrain_weights_path}")
    
    return pretrain_weights_path

def run_experiment(config):
    logger = setup_logging(config)
    data_manager = DataManager(config)
    history_buffer = HistoryBuffer()
    # 验证集比例
    val_ratio = config.get('val_ratio', 0.2)
    val_buffer = HistoryBuffer()
    input_dim = config['prediction_model']['input_dim']
    shared_dims = config['prediction_model']['shared_dims']
    ctr_head_dims = config['prediction_model']['ctr_head_dims']
    playtime_head_dims = config['prediction_model']['playtime_head_dims']
    weight_hidden_dims = config.get('weighting_model', {}).get('hidden_dims', [64, 32])
    method = config.get('experiment_method', 'weighted')

    # 设备选择
    device_cfg = config.get('device', 'auto')
    available_devices = []
    if torch.cuda.is_available():
        available_devices.append('cuda')
    available_devices.append('cpu')
    if device_cfg == 'auto':
        device = 'cuda' if 'cuda' in available_devices else 'cpu'
    else:
        device = device_cfg if device_cfg in available_devices else 'cpu'
    print(f"[Device] Available devices: {available_devices}, selected: {device}")
    logger.info(f"Using device: {device}")

    # 预训练阶段
    pretrain_weights_path = pretrain_models(config, logger, data_manager, device)

    # 初始化模型和训练器
    if method == 'weighted':
        model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model_T.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
        model_C.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
        weight_model = WeightingModel(input_dim, weight_hidden_dims).to(device)
        trainer = WeightedTrainer(config, model_T, model_C, weight_model, device=device)
        recommender_T = Recommender(model_T, alpha=config['recommender']['treatment_alpha'])
        recommender_C = Recommender(model_C, alpha=config['recommender']['control_alpha'])
    elif method == 'pooling':
        model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
        trainer = PoolingTrainer(config, model, device=device)
        recommender_T = recommender_C = Recommender(model)
    elif method == 'splitting':
        model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model_T.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
        model_C.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
        trainer = SplittingTrainer(config, model_T, model_C, device=device)
        recommender_T = Recommender(model_T)
        recommender_C = Recommender(model_C)
    elif method == 'snapshot':
        model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
        trainer = SnapshotTrainer(config, model, device=device)
        recommender_T = recommender_C = Recommender(model)
    else:
        raise ValueError(f'Unknown experiment_method: {method}')

    # 主循环
    import platform
    import sys as _sys
    import time as _time
    results = {
        'meta': {
            'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'os': platform.platform(),
            'python_version': _sys.version,
            'device': device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'env': dict(os.environ),
        },
        'step': [],
        'metrics': []
    }
    for step, interaction in enumerate(data_manager.get_simulation_stream(config['total_simulation_steps'])):
        user_id = interaction['user_id']
        Z = torch.bernoulli(torch.tensor(config['p_treatment'])).item()
        candidate_video_ids = data_manager.get_candidate_videos(user_id, config['candidate_pool_size'])
        interaction_features = data_manager.create_interaction_features(user_id, candidate_video_ids)
        if Z == 1:
            recommended_idx = recommender_T.recommend(interaction_features)
        else:
            recommended_idx = recommender_C.recommend(interaction_features)
        recommended_video_id = candidate_video_ids[recommended_idx]
        X_observed = interaction_features[recommended_idx]
        Y_observed = data_manager.get_ground_truth_label(user_id, recommended_video_id)
        # 按比例分配到训练/验证集
        if np.random.rand() < val_ratio:
            val_buffer.add(X_observed, Y_observed, torch.tensor(Z))
        else:
            history_buffer.add(X_observed, Y_observed, torch.tensor(Z))
        # 定期训练和评估
        if (step + 1) % config['train_every_n_steps'] == 0:
            logger.info(f"Step {step+1}: Retraining models on {len(history_buffer)} train, {len(val_buffer)} val samples...")
            history_loader = DataLoader(history_buffer, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_buffer, batch_size=config['batch_size'], shuffle=False) if len(val_buffer) > 0 else None
            
            # 所有方法都使用train_on_history，因为已经从预训练权重开始
            trainer.train_on_history(history_loader)
            if hasattr(trainer, 'save_models'):
                trainer.save_models(step)
            elif hasattr(trainer, 'save_model'):
                trainer.save_model(step)
            # 评估当前模型（训练集）
            def eval_loader(loader):
                y_true, y_pred = [], []
                # 自动切换到eval模式，避免BN报错
                model_was_training = recommender_T.model.training
                recommender_T.model.eval()
                for X, Y, _ in loader:
                    with torch.no_grad():
                        pred_click, pred_time = recommender_T.model(X)
                        y_true.append(Y.numpy())
                        y_pred.append(torch.stack([pred_click, pred_time], dim=1).numpy())
                if model_was_training:
                    recommender_T.model.train()
                y_true = np.concatenate(y_true, axis=0)
                y_pred = np.concatenate(y_pred, axis=0)
                return y_true, y_pred
            y_true_tr, y_pred_tr = eval_loader(history_loader)
            metrics_tr = compute_metrics(y_true_tr, y_pred_tr)
            # 新增AUC/Log-MAE
            ctr_auc_tr = None
            logmae_tr = None
            try:
                ctr_auc_tr = compute_roc_auc(y_true_tr[:,0], torch.sigmoid(torch.tensor(y_pred_tr[:,0])).numpy())[2]
                logmae_tr = compute_log_mae(y_true_tr[:,1], y_pred_tr[:,1])
            except Exception as e:
                pass
            metrics_tr['CTR_AUC'] = ctr_auc_tr
            metrics_tr['LogMAE_play_time'] = logmae_tr
            # 验证集
            if val_loader is not None and len(val_buffer) > 0:
                y_true_val, y_pred_val = eval_loader(val_loader)
                metrics_val = compute_metrics(y_true_val, y_pred_val)
                ctr_auc_val = None
                logmae_val = None
                try:
                    ctr_auc_val = compute_roc_auc(y_true_val[:,0], torch.sigmoid(torch.tensor(y_pred_val[:,0])).numpy())[2]
                    logmae_val = compute_log_mae(y_true_val[:,1], y_pred_val[:,1])
                except Exception as e:
                    pass
                metrics_val['CTR_AUC'] = ctr_auc_val
                metrics_val['LogMAE_play_time'] = logmae_val
            else:
                metrics_val = None
            results['step'].append(step+1)
            # 确保所有指标为Python float类型，避免float32无法序列化
            def to_pyfloat(d):
                if d is None:
                    return None
                return {k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v for k, v in d.items()}
            results['metrics'].append({'train': to_pyfloat(metrics_tr), 'val': to_pyfloat(metrics_val)})
            logger.info(f"Step {step+1} metrics: train={metrics_tr}, val={metrics_val}")
    # 保存最终结果
    save_results(results, os.path.join(config['results_path'], 'exp_results.json'))

if __name__ == '__main__':
    try:
        run_experiment(config)
        # 自动运行画图脚本
        import subprocess
        plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/plot_exp_results.py')
        if os.path.exists(plot_script):
            print("[INFO] 自动运行画图脚本...")
            # 传递结果目录作为参数
            subprocess.run(['python', plot_script, result_dir], check=True)
        else:
            print(f"[WARN] 未找到画图脚本: {plot_script}")
    except Exception as e:
        import traceback
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
