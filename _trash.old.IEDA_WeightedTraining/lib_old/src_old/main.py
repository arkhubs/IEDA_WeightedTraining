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
import os

# 确保当前工作目录为项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)

config_path = os.path.join('IEDA_WeightedTraining', 'configs', 'experiment_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 处理所有路径为绝对路径，基于项目根目录
base_dir = config.get('base_dir', '.')
config['data_path'] = os.path.abspath(os.path.join(base_dir, config['data_path']))
config['results_path'] = os.path.abspath(os.path.join(base_dir, config['results_path']))
config['cache_dir'] = os.path.abspath(os.path.join(base_dir, config.get('cache_dir', 'KuaiRand/cache/')))

# 设置实验结果保存目录
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = os.path.join(config['results_path'], timestamp)
os.makedirs(result_dir, exist_ok=True)
config['results_path'] = result_dir

# 确保缓存目录存在
os.makedirs(config['cache_dir'], exist_ok=True)

def pretrain_models(config, logger, data_manager, device):
    """预训练阶段：分别为实验组和对照组预训练模型"""
    pretrain_size = config.get('initial_dataset_size', 50)  # 大幅减小到50进行快速测试
    logger.info(f"[Pretrain] 开始预训练，每组随机选择 {pretrain_size} 个交互")

    logger.info("[Pretrain] Step 1: 获取所有交互样本 ...")
    all_interactions = list(data_manager.get_all_interactions())
    logger.info(f"[Pretrain] Step 1: 获取所有交互样本完成，数量: {len(all_interactions)}")
    
    # 确保有足够的数据进行分组预训练
    total_needed = pretrain_size * 2
    if len(all_interactions) < total_needed:
        logger.warning(f"数据集只有 {len(all_interactions)} 个交互，小于预期的 {total_needed}")
        pretrain_size = len(all_interactions) // 2

    logger.info("[Pretrain] Step 2: 随机采样交互 ...")
    import random
    random.shuffle(all_interactions)
    
    # 分别为实验组和对照组采样
    pretrain_interactions_T = all_interactions[:pretrain_size]
    pretrain_interactions_C = all_interactions[pretrain_size:pretrain_size*2]
    logger.info(f"[Pretrain] Step 2: 随机采样完成，实验组: {len(pretrain_interactions_T)}, 对照组: {len(pretrain_interactions_C)}")

    logger.info("[Pretrain] Step 3: 批量构建预训练数据 ...")
    
    # 批量获取user_ids和video_ids
    user_ids_T = [inter['user_id'] for inter in pretrain_interactions_T]
    video_ids_T = [inter['video_id'] for inter in pretrain_interactions_T]
    user_ids_C = [inter['user_id'] for inter in pretrain_interactions_C]
    video_ids_C = [inter['video_id'] for inter in pretrain_interactions_C]
    
    # 批量生成特征（实验组）
    logger.info("[Pretrain] Step 3: 批量生成实验组特征 ...")
    pretrain_buffer_T = HistoryBuffer()
    for i in range(len(user_ids_T)):
        if i % 50 == 0:
            logger.info(f"[Pretrain] 实验组特征生成进度 {i}/{len(user_ids_T)} ...")
        
        interaction_features = data_manager.create_interaction_features(user_ids_T[i], [video_ids_T[i]])
        X_observed = interaction_features[0]
        Y_observed = data_manager.get_ground_truth_label(user_ids_T[i], video_ids_T[i])
        Z = torch.tensor(1.0)  # 实验组标记为1
        
        pretrain_buffer_T.add(X_observed, Y_observed, Z)
    
    # 批量生成特征（对照组）
    logger.info("[Pretrain] Step 3: 批量生成对照组特征 ...")
    pretrain_buffer_C = HistoryBuffer()
    for i in range(len(user_ids_C)):
        if i % 50 == 0:
            logger.info(f"[Pretrain] 对照组特征生成进度 {i}/{len(user_ids_C)} ...")
        
        interaction_features = data_manager.create_interaction_features(user_ids_C[i], [video_ids_C[i]])
        X_observed = interaction_features[0]
        Y_observed = data_manager.get_ground_truth_label(user_ids_C[i], video_ids_C[i])
        Z = torch.tensor(0.0)  # 对照组标记为0
        
        pretrain_buffer_C.add(X_observed, Y_observed, Z)
    
    logger.info(f"[Pretrain] Step 3: 构建预训练数据完成，实验组: {len(pretrain_buffer_T)}, 对照组: {len(pretrain_buffer_C)}")

    logger.info("[Pretrain] Step 4: 创建预训练模型 ...")
    input_dim = config['prediction_model']['input_dim']
    shared_dims = config['prediction_model']['shared_dims']
    ctr_head_dims = config['prediction_model']['ctr_head_dims']
    playtime_head_dims = config['prediction_model']['playtime_head_dims']

    # 分别创建实验组和对照组模型
    pretrain_model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
    pretrain_model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
    logger.info("[Pretrain] Step 4: 创建预训练模型完成")

    logger.info("[Pretrain] Step 5: 创建PoolingTrainer ...")
    pretrain_trainer_T = PoolingTrainer(config, pretrain_model_T, device=device)
    pretrain_trainer_C = PoolingTrainer(config, pretrain_model_C, device=device)
    logger.info("[Pretrain] Step 5: 创建PoolingTrainer完成")

    logger.info("[Pretrain] Step 6: 创建DataLoader ...")
    pretrain_loader_T = DataLoader(pretrain_buffer_T, batch_size=config.get('batch_size', 128), shuffle=True)
    pretrain_loader_C = DataLoader(pretrain_buffer_C, batch_size=config.get('batch_size', 128), shuffle=True)
    logger.info("[Pretrain] Step 6: 创建DataLoader完成")

    logger.info(f"[Pretrain] Step 7: 开始训练实验组模型，数据量: {len(pretrain_buffer_T)}")
    pretrain_trainer_T.train_on_pooling(pretrain_loader_T)
    
    logger.info(f"[Pretrain] Step 8: 开始训练对照组模型，数据量: {len(pretrain_buffer_C)}")
    pretrain_trainer_C.train_on_pooling(pretrain_loader_C)
    
    # 保存预训练权重
    pretrain_weights_path_T = os.path.join(config['results_path'], 'pretrain_weights_T.pth')
    pretrain_weights_path_C = os.path.join(config['results_path'], 'pretrain_weights_C.pth')
    torch.save(pretrain_model_T.state_dict(), pretrain_weights_path_T)
    torch.save(pretrain_model_C.state_dict(), pretrain_weights_path_C)
    logger.info(f"[Pretrain] 预训练完成，实验组权重已保存到: {pretrain_weights_path_T}")
    logger.info(f"[Pretrain] 预训练完成，对照组权重已保存到: {pretrain_weights_path_C}")

    return pretrain_weights_path_T, pretrain_weights_path_C

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
    pretrain_weights_path_T, pretrain_weights_path_C = pretrain_models(config, logger, data_manager, device)

    # 初始化模型和训练器
    if method == 'weighted':
        model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model_T.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        model_C.load_state_dict(torch.load(pretrain_weights_path_C, map_location=device))
        weight_model = WeightingModel(input_dim, weight_hidden_dims).to(device)
        trainer = WeightedTrainer(config, model_T, model_C, weight_model, device=device)
        recommender_T = Recommender(model_T, alpha=config['recommender']['treatment_alpha'])
        recommender_C = Recommender(model_C, alpha=config['recommender']['control_alpha'])
    elif method == 'pooling':
        model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重（默认使用实验组权重）
        model.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        trainer = PoolingTrainer(config, model, device=device)
        recommender_T = recommender_C = Recommender(model)
    elif method == 'splitting':
        model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model_T.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        model_C.load_state_dict(torch.load(pretrain_weights_path_C, map_location=device))
        trainer = SplittingTrainer(config, model_T, model_C, device=device)
        recommender_T = Recommender(model_T)
        recommender_C = Recommender(model_C)
    elif method == 'snapshot':
        model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重（默认使用实验组权重）
        model.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        trainer = SnapshotTrainer(config, model, device=device)
        recommender_T = recommender_C = Recommender(model)
    else:
        raise ValueError(f'Unknown experiment_method: {method}')

    # 主循环
    return run_main_loop(config, logger, data_manager, device, pretrain_weights_path_T, pretrain_weights_path_C)


def run_main_loop(config, logger, data_manager, device, pretrain_weights_path_T, pretrain_weights_path_C):
    """实验主循环"""
    import platform
    import sys as _sys
    import time as _time
    
    # 定义epoch和评估的步数
    steps_per_epoch = 1000
    eval_every_n_steps = 20
    current_epoch = 0
    
    # 获取模型配置参数
    input_dim = config['prediction_model']['input_dim']
    shared_dims = config['prediction_model']['shared_dims']
    ctr_head_dims = config['prediction_model']['ctr_head_dims']
    playtime_head_dims = config['prediction_model']['playtime_head_dims']
    weight_hidden_dims = config.get('weighting_model', {}).get('hidden_dims', [64, 32])
    
    # 验证集比例
    val_ratio = config.get('val_ratio', 0.2)
    method = config.get('experiment_method', 'weighted')
    
    # 初始化交互历史缓冲区
    history_buffer = HistoryBuffer()
    val_buffer = HistoryBuffer()
    
    # 初始化模型和推荐器
    logger.info(f"初始化{method}方法的模型...")
    
    # 根据实验方法初始化不同的模型和推荐器
    if method == 'weighted':
        model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model_T.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        model_C.load_state_dict(torch.load(pretrain_weights_path_C, map_location=device))
        # 初始化权重模型
        weight_model = WeightingModel(input_dim, weight_hidden_dims).to(device)
        # 创建训练器和推荐器
        trainer = WeightedTrainer(config, model_T, model_C, weight_model, device=device)
        recommender_T = Recommender(model_T, alpha=config['recommender']['treatment_alpha'])
        recommender_C = Recommender(model_C, alpha=config['recommender']['control_alpha'])
    elif method == 'pooling':
        model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        trainer = PoolingTrainer(config, model, device=device)
        recommender_T = recommender_C = Recommender(model)
    elif method == 'splitting':
        model_T = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        model_C = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重
        model_T.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        model_C.load_state_dict(torch.load(pretrain_weights_path_C, map_location=device))
        trainer = SplittingTrainer(config, model_T, model_C, device=device)
        recommender_T = Recommender(model_T)
        recommender_C = Recommender(model_C)
    elif method == 'snapshot':
        model = PredictionModel(input_dim, shared_dims, ctr_head_dims, playtime_head_dims).to(device)
        # 加载预训练权重（默认使用实验组权重）
        model.load_state_dict(torch.load(pretrain_weights_path_T, map_location=device))
        trainer = SnapshotTrainer(config, model, device=device)
        recommender_T = recommender_C = Recommender(model)
    else:
        raise ValueError(f'Unknown experiment_method: {method}')
    
    results = {
        'meta': {
            'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'os': platform.platform(),
            'python_version': _sys.version,
            'device': device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'env': dict(os.environ),
            'steps_per_epoch': steps_per_epoch,
            'eval_every_n_steps': eval_every_n_steps,
        },
        'step': [],
        'epoch': [],
        'step_in_epoch': [],
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
        
        # 计算当前的epoch和step_in_epoch
        step_in_epoch = (step + 1) % steps_per_epoch
        if step_in_epoch == 0:
            step_in_epoch = steps_per_epoch
            # epoch完成时递增
            if step > 0:  # 忽略第一步
                logger.info(f"Epoch {current_epoch} completed (Step {step+1})")
                current_epoch += 1
                
        # 定期训练和评估
        if (step + 1) % eval_every_n_steps == 0:
            logger.info(f"Epoch {current_epoch}, Step {step_in_epoch}/{steps_per_epoch} (Global Step {step+1}): Retraining models on {len(history_buffer)} train, {len(val_buffer)} val samples...")
            history_loader = DataLoader(history_buffer, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_buffer, batch_size=config['batch_size'], shuffle=False) if len(val_buffer) > 0 else None
            
            # 所有方法都使用train_on_history，因为已经从预训练权重开始
            trainer.train_on_history(history_loader)
            if hasattr(trainer, 'save_models'):
                trainer.save_models(step)
            elif hasattr(trainer, 'save_model'):
                trainer.save_model(step) 
            # 分离评估实验组和对照组模型
            def eval_loader_separated(loader, model_T, model_C):
                """分别评估实验组和对照组模型"""
                y_true_T, y_pred_T = [], []
                y_true_C, y_pred_C = [], []
                
                model_T_was_training = model_T.training
                model_C_was_training = model_C.training
                model_T.eval()
                model_C.eval()
                
                for X, Y, Z in loader:
                    with torch.no_grad():
                        # 实验组数据
                        mask_T = (Z == 1)
                        if mask_T.any():
                            X_T, Y_T = X[mask_T], Y[mask_T]
                            pred_click_T, pred_time_T = model_T(X_T)
                            y_true_T.append(Y_T.numpy())
                            y_pred_T.append(torch.stack([pred_click_T, pred_time_T], dim=1).numpy())
                        
                        # 对照组数据
                        mask_C = (Z == 0)
                        if mask_C.any():
                            X_C, Y_C = X[mask_C], Y[mask_C]
                            pred_click_C, pred_time_C = model_C(X_C)
                            y_true_C.append(Y_C.numpy())
                            y_pred_C.append(torch.stack([pred_click_C, pred_time_C], dim=1).numpy())
                
                if model_T_was_training:
                    model_T.train()
                if model_C_was_training:
                    model_C.train()
                
                # 合并结果
                results = {}
                if y_true_T:
                    y_true_T = np.concatenate(y_true_T, axis=0)
                    y_pred_T = np.concatenate(y_pred_T, axis=0)
                    results['treatment'] = (y_true_T, y_pred_T)
                if y_true_C:
                    y_true_C = np.concatenate(y_true_C, axis=0)
                    y_pred_C = np.concatenate(y_pred_C, axis=0)
                    results['control'] = (y_true_C, y_pred_C)
                
                return results
            
            # 根据方法选择评估策略
            if method == 'weighted' or method == 'splitting':
                # 分离评估
                eval_results_tr = eval_loader_separated(history_loader, recommender_T.model, recommender_C.model)
                metrics_tr = {}
                
                if 'treatment' in eval_results_tr:
                    y_true_T, y_pred_T = eval_results_tr['treatment']
                    metrics_T = compute_metrics(y_true_T, y_pred_T)
                    try:
                        ctr_auc_T = compute_roc_auc(y_true_T[:,0], torch.sigmoid(torch.tensor(y_pred_T[:,0])).numpy())[2]
                        logmae_T = compute_log_mae(y_true_T[:,1], y_pred_T[:,1])
                        metrics_T['CTR_AUC'] = ctr_auc_T
                        metrics_T['LogMAE_play_time'] = logmae_T
                    except:
                        metrics_T['CTR_AUC'] = None
                        metrics_T['LogMAE_play_time'] = None
                    metrics_tr['treatment'] = metrics_T
                
                if 'control' in eval_results_tr:
                    y_true_C, y_pred_C = eval_results_tr['control']
                    metrics_C = compute_metrics(y_true_C, y_pred_C)
                    try:
                        ctr_auc_C = compute_roc_auc(y_true_C[:,0], torch.sigmoid(torch.tensor(y_pred_C[:,0])).numpy())[2]
                        logmae_C = compute_log_mae(y_true_C[:,1], y_pred_C[:,1])
                        metrics_C['CTR_AUC'] = ctr_auc_C
                        metrics_C['LogMAE_play_time'] = logmae_C
                    except:
                        metrics_C['CTR_AUC'] = None
                        metrics_C['LogMAE_play_time'] = None
                    metrics_tr['control'] = metrics_C
                
                # 如果是weighted方法，额外评估weight_model
                if method == 'weighted':
                    # 评估权重模型的表现
                    weight_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
                    try:
                        # 采样训练数据评估权重模型
                        weight_model = trainer.weight_model
                        weight_model_was_training = weight_model.training
                        weight_model.eval()
                        
                        # 收集所有预测结果
                        all_pred_w = []
                        all_true_z = []
                        
                        for X, _, Z in history_loader:
                            X, Z = X.to(device), Z.to(device)
                            with torch.no_grad():
                                pred_w = weight_model(X)
                                pred_z = (pred_w > 0.5).float()  # 二分类阈值
                                all_pred_w.append(pred_w.cpu().numpy())
                                all_true_z.append(Z.cpu().numpy())
                                
                        if weight_model_was_training:
                            weight_model.train()
                            
                        if all_pred_w and all_true_z:
                            all_pred_w = np.concatenate([p.flatten() for p in all_pred_w])
                            all_true_z = np.concatenate([z.flatten() for z in all_true_z])
                            
                            # 计算分类指标
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                            
                            pred_z_binary = (all_pred_w > 0.5).astype(int)
                            
                            if len(np.unique(all_true_z)) > 1:  # 确保有正负样本
                                weight_metrics['accuracy'] = accuracy_score(all_true_z, pred_z_binary)
                                weight_metrics['precision'] = precision_score(all_true_z, pred_z_binary)
                                weight_metrics['recall'] = recall_score(all_true_z, pred_z_binary)
                                weight_metrics['f1'] = f1_score(all_true_z, pred_z_binary)
                                weight_metrics['auc'] = roc_auc_score(all_true_z, all_pred_w)
                            
                    except Exception as e:
                        logger.warning(f"评估权重模型时出错: {e}")
                    
                    # 将权重模型指标添加到结果中
                    metrics_tr['weight_model'] = weight_metrics
            else:
                # pooling和snapshot方法的统一评估
                def eval_loader_unified(loader, model):
                    y_true, y_pred = [], []
                    model_was_training = model.training
                    model.eval()
                    for X, Y, _ in loader:
                        with torch.no_grad():
                            pred_click, pred_time = model(X)
                            y_true.append(Y.numpy())
                            y_pred.append(torch.stack([pred_click, pred_time], dim=1).numpy())
                    if model_was_training:
                        model.train()
                    y_true = np.concatenate(y_true, axis=0)
                    y_pred = np.concatenate(y_pred, axis=0)
                    return y_true, y_pred
                
                y_true_tr, y_pred_tr = eval_loader_unified(history_loader, recommender_T.model)
                metrics_tr = compute_metrics(y_true_tr, y_pred_tr)
                try:
                    ctr_auc_tr = compute_roc_auc(y_true_tr[:,0], torch.sigmoid(torch.tensor(y_pred_tr[:,0])).numpy())[2]
                    logmae_tr = compute_log_mae(y_true_tr[:,1], y_pred_tr[:,1])
                    metrics_tr['CTR_AUC'] = ctr_auc_tr
                    metrics_tr['LogMAE_play_time'] = logmae_tr
                except:
                    metrics_tr['CTR_AUC'] = None
                    metrics_tr['LogMAE_play_time'] = None
            # 验证集评估
            if val_loader is not None and len(val_buffer) > 0:
                if method == 'weighted' or method == 'splitting':
                    # 分离评估验证集
                    eval_results_val = eval_loader_separated(val_loader, recommender_T.model, recommender_C.model)
                    metrics_val = {}
                    
                    if 'treatment' in eval_results_val:
                        y_true_T_val, y_pred_T_val = eval_results_val['treatment']
                        metrics_T_val = compute_metrics(y_true_T_val, y_pred_T_val)
                        try:
                            ctr_auc_T_val = compute_roc_auc(y_true_T_val[:,0], torch.sigmoid(torch.tensor(y_pred_T_val[:,0])).numpy())[2]
                            logmae_T_val = compute_log_mae(y_true_T_val[:,1], y_pred_T_val[:,1])
                            metrics_T_val['CTR_AUC'] = ctr_auc_T_val
                            metrics_T_val['LogMAE_play_time'] = logmae_T_val
                        except:
                            metrics_T_val['CTR_AUC'] = None
                            metrics_T_val['LogMAE_play_time'] = None
                        metrics_val['treatment'] = metrics_T_val
                    
                    if 'control' in eval_results_val:
                        y_true_C_val, y_pred_C_val = eval_results_val['control']
                        metrics_C_val = compute_metrics(y_true_C_val, y_pred_C_val)
                        try:
                            ctr_auc_C_val = compute_roc_auc(y_true_C_val[:,0], torch.sigmoid(torch.tensor(y_pred_C_val[:,0])).numpy())[2]
                            logmae_C_val = compute_log_mae(y_true_C_val[:,1], y_pred_C_val[:,1])
                            metrics_C_val['CTR_AUC'] = ctr_auc_C_val
                            metrics_C_val['LogMAE_play_time'] = logmae_C_val
                        except:
                            metrics_C_val['CTR_AUC'] = None
                            metrics_C_val['LogMAE_play_time'] = None
                        metrics_val['control'] = metrics_C_val
                else:
                    # 统一评估验证集
                    y_true_val, y_pred_val = eval_loader_unified(val_loader, recommender_T.model)
                    metrics_val = compute_metrics(y_true_val, y_pred_val)
                    try:
                        ctr_auc_val = compute_roc_auc(y_true_val[:,0], torch.sigmoid(torch.tensor(y_pred_val[:,0])).numpy())[2]
                        logmae_val = compute_log_mae(y_true_val[:,1], y_pred_val[:,1])
                        metrics_val['CTR_AUC'] = ctr_auc_val
                        metrics_val['LogMAE_play_time'] = logmae_val
                    except:
                        metrics_val['CTR_AUC'] = None
                        metrics_val['LogMAE_play_time'] = None
            else:
                metrics_val = None
            
            results['step'].append(step+1)
            results['epoch'].append(current_epoch)
            results['step_in_epoch'].append(step_in_epoch)
            
            # 确保所有指标为Python float类型，避免float32无法序列化
            def to_pyfloat(d):
                if d is None:
                    return None
                if isinstance(d, dict):
                    return {k: to_pyfloat(v) for k, v in d.items()}
                elif isinstance(d, (np.floating, np.float32, np.float64)):
                    return float(d)
                else:
                    return d
            
            results['metrics'].append({'train': to_pyfloat(metrics_tr), 'val': to_pyfloat(metrics_val)})
            logger.info(f"Epoch {current_epoch}, Step {step_in_epoch}/{steps_per_epoch} (Global Step {step+1}) metrics: train={metrics_tr}, val={metrics_val}")
            
            # 在每个 epoch 结束时更新图表
            if step_in_epoch >= steps_per_epoch - 1:
                # 保存当前结果
                result_json_path = os.path.join(config['results_path'], 'exp_results.json')
                save_results(results, result_json_path)
                
                # 非阻塞式调用绘图脚本
                try:
                    # 导入绘图模块
                    sys.path.append('IEDA_WeightedTraining/results')
                    import plot_exp_results
                    logger.info(f"Epoch {current_epoch} 完成，非阻塞更新图表...")
                    plot_exp_results.plot_async(config['results_path'])
                except Exception as e:
                    logger.warning(f"更新图表时出错: {e}")
                    
    # 保存最终结果
    save_results(results, os.path.join(config['results_path'], 'exp_results.json'))

def run_experiment(config):
    """执行实验的主函数"""
    # 设置日志
    logger = setup_logging('experiment', config['results_path'], 'experiment.log')
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[Device] Available devices: {['cuda' if torch.cuda.is_available() else 'cpu']}, selected: {device}")
    
    # 初始化数据管理器
    data_manager = DataManager(config)
    
    # 预训练模型
    pretrain_weights_path_T, pretrain_weights_path_C = pretrain_models(config, logger, data_manager, device)
    
    # 实验主循环
    run_main_loop(config, logger, data_manager, device, pretrain_weights_path_T, pretrain_weights_path_C)


if __name__ == '__main__':
    try:
        run_experiment(config)
        # 自动运行画图脚本，最终结果
        import subprocess
        plot_script = os.path.join('IEDA_WeightedTraining/results', 'plot_exp_results.py')
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
