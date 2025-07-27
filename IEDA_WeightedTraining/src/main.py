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
from src.utils import setup_logging, save_model, save_results, compute_metrics
from torch.utils.data import DataLoader

with open('configs/experiment_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def run_experiment(config):
    logger = setup_logging(config)
    data_manager = DataManager(config)
    history_buffer = HistoryBuffer()
    input_dim = config['prediction_model']['input_dim']
    pred_hidden_dims = config['prediction_model']['hidden_dims']
    output_dim = config['prediction_model']['output_dim']
    weight_hidden_dims = config.get('weighting_model', {}).get('hidden_dims', [64, 32])
    method = config.get('experiment_method', 'weighted')

    # 初始化模型和训练器
    if method == 'weighted':
        model_T = PredictionModel(input_dim, pred_hidden_dims, output_dim)
        model_C = PredictionModel(input_dim, pred_hidden_dims, output_dim)
        weight_model = WeightingModel(input_dim, weight_hidden_dims)
        trainer = WeightedTrainer(config, model_T, model_C, weight_model)
        recommender_T = Recommender(model_T, alpha=config['recommender']['treatment_alpha'])
        recommender_C = Recommender(model_C, alpha=config['recommender']['control_alpha'])
    elif method == 'pooling':
        model = PredictionModel(input_dim, pred_hidden_dims, output_dim)
        trainer = PoolingTrainer(config, model)
        recommender_T = recommender_C = Recommender(model)
    elif method == 'splitting':
        model_T = PredictionModel(input_dim, pred_hidden_dims, output_dim)
        model_C = PredictionModel(input_dim, pred_hidden_dims, output_dim)
        trainer = SplittingTrainer(config, model_T, model_C)
        recommender_T = Recommender(model_T)
        recommender_C = Recommender(model_C)
    elif method == 'snapshot':
        model = PredictionModel(input_dim, pred_hidden_dims, output_dim)
        trainer = SnapshotTrainer(config, model)
        recommender_T = recommender_C = Recommender(model)
    else:
        raise ValueError(f'Unknown experiment_method: {method}')

    # 主循环
    results = {'step': [], 'metrics': []}
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
        history_buffer.add(X_observed, Y_observed, torch.tensor(Z))
        # 定期训练和评估
        if (step + 1) % config['train_every_n_steps'] == 0:
            logger.info(f"Step {step+1}: Retraining models on {len(history_buffer)} samples...")
            history_loader = DataLoader(history_buffer, batch_size=config['batch_size'], shuffle=True)
            if method == 'snapshot' and step == config.get('initial_dataset_size', 5000):
                trainer.train_on_snapshot(history_loader)
                trainer.save_model(step)
            else:
                trainer.train_on_history(history_loader)
                if hasattr(trainer, 'save_models'):
                    trainer.save_models(step)
                elif hasattr(trainer, 'save_model'):
                    trainer.save_model(step)
            # 评估当前模型
            y_true = []
            y_pred = []
            for X, Y, _ in history_loader:
                with torch.no_grad():
                    if method in ['weighted', 'splitting']:
                        pred_click, pred_time = recommender_T.model(X)
                    else:
                        pred_click, pred_time = recommender_T.model(X)
                    y_true.append(Y.numpy())
                    y_pred.append(torch.stack([pred_click, pred_time], dim=1).numpy())
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            metrics = compute_metrics(y_true, y_pred)
            results['step'].append(step+1)
            results['metrics'].append(metrics)
            logger.info(f"Step {step+1} metrics: {metrics}")
    # 保存最终结果
    save_results(results, os.path.join(config['results_path'], 'exp_results.json'))

if __name__ == '__main__':
    run_experiment(config)
