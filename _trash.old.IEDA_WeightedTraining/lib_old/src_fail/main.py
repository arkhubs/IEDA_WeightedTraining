"""
主程序模块，实现实验的主要逻辑
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.data_manager import DataManager
from src.models import PredictionModel, WeightModel
from src.trainers import PredictionTrainer, WeightTrainer
from src.recommender import Recommender
from src.utils import Logger, setup_seed
from src.config_manager import ConfigManager


def create_experiment_dir(config: Dict[str, Any]) -> str:
    """
    创建实验目录
    
    Args:
        config: 配置字典
    
    Returns:
        实验目录路径
    """
    # 获取基础保存路径
    base_save_dir = config['output']['save_dir']
    
    # 创建带时间戳的实验目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_name = config['experiment']['name']
    exp_dir = os.path.join(base_save_dir, f"{experiment_name}_{timestamp}")
    
    # 创建目录
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    # 保存配置
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return exp_dir


def pretrain_weight_model(data_manager: DataManager, 
                          weight_model: WeightModel, 
                          config: Dict[str, Any],
                          logger: Logger,
                          device: str) -> None:
    """
    预训练权重模型
    
    Args:
        data_manager: 数据管理器
        weight_model: 权重模型
        config: 配置字典
        logger: 日志记录器
        device: 设备
    """
    logger.info("开始预训练权重模型")
    
    # 创建权重模型训练器
    trainer = WeightTrainer(
        weight_model=weight_model,
        device=device,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        clip_grad=config['training'].get('clip_grad', 1.0),
        logger=logger
    )
    
    # 获取预训练数据加载器
    train_dataloader, val_dataloader = data_manager.get_weight_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # 预训练步骤
    steps = config['training'].get('weight_pretrain_steps', 1000)
    eval_steps = config['training'].get('eval_steps', 100)
    
    logger.info(f"开始训练，总步骤数: {steps}，评估间隔: {eval_steps}")
    
    # 训练循环
    best_accuracy = 0.0
    best_model_path = None
    
    for step in range(1, steps + 1):
        # 获取批次数据
        try:
            batch = next(iter(train_dataloader))
        except StopIteration:
            train_dataloader = iter(data_manager.get_weight_dataloaders(
                batch_size=config['data']['batch_size'],
                num_workers=config['data'].get('num_workers', 4)
            )[0])
            batch = next(train_dataloader)
        
        # 训练步骤
        train_metrics = trainer.train_step(batch)
        
        # 定期评估
        if step % eval_steps == 0:
            logger.info(f"步骤 {step}/{steps}: 训练损失 = {train_metrics['train']['weight_model']['loss']:.4f}, "
                       f"训练准确率 = {train_metrics['train']['weight_model']['accuracy']:.4f}")
            
            # 在验证集上评估
            eval_metrics = trainer.evaluate(val_dataloader)
            val_accuracy = eval_metrics['val']['weight_model']['accuracy']
            val_loss = eval_metrics['val']['weight_model']['loss']
            logger.info(f"验证损失 = {val_loss:.4f}, 验证准确率 = {val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = os.path.join(
                    config['output']['save_dir'], 
                    'models', 
                    f"weight_model_best.pth"
                )
                trainer.save_model(weight_model, best_model_path)
                logger.info(f"发现更好的模型，准确率: {val_accuracy:.4f}，已保存")
    
    logger.info(f"权重模型训练完成，最佳验证准确率: {best_accuracy:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(
        config['output']['save_dir'], 
        'models', 
        f"weight_model_final.pth"
    )
    trainer.save_model(weight_model, final_model_path)
    logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 如果需要，加载最佳模型
    if best_model_path is not None:
        weight_model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"已加载最佳权重模型，准确率: {best_accuracy:.4f}")
    
    return


def run_weighted_training(config: Dict[str, Any], exp_dir: str):
    """
    运行加权训练实验
    
    Args:
        config: 配置字典
        exp_dir: 实验目录
    """
    # 创建日志记录器
    logger = Logger(
        log_file=os.path.join(exp_dir, 'logs', 'experiment.log'),
        log_level=config['output'].get('log_level', 'INFO')
    )
    
    # 记录配置信息
    logger.info(f"实验名称: {config['experiment']['name']}")
    logger.info(f"实验结果将保存至: {exp_dir}")
    
    # 设置随机种子
    seed = config['experiment']['seed']
    setup_seed(seed)
    logger.info(f"设置随机种子: {seed}")
    
    # 设备选择
    device = config['experiment']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA不可用，切换到CPU")
    logger.info(f"使用设备: {device}")
    
    # 初始化数据管理器
    logger.info("初始化数据管理器")
    # 配置数据路径
    config['data_path'] = config['data'].get('data_path', './KuaiRand')
    config['cache_dir'] = config['data'].get('cache_dir', './KuaiRand/cache')
    # 添加默认的数据集类型
    config['dataset_type'] = config['data'].get('dataset_type', '1K')
    
    data_manager = DataManager(config=config)
    
    # 创建模型
    logger.info("创建模型")
    
    # 创建权重模型
    weight_model_config = config['model']['weight']
    weight_model = WeightModel(weight_model_config).to(device)
    logger.info(f"创建权重模型: {weight_model_config}")
    
    # 创建预测模型
    prediction_model_config = config['model']['prediction']
    prediction_model = PredictionModel(prediction_model_config).to(device)
    logger.info(f"创建预测模型: {prediction_model_config}")
    
    # 预训练权重模型
    if config.get('training', {}).get('pretrain_weight_model', True):
        logger.info("预训练权重模型")
        pretrain_weight_model(
            data_manager=data_manager,
            weight_model=weight_model,
            config=config,
            logger=logger,
            device=device
        )
    
    # 创建预测模型训练器
    prediction_trainer = PredictionTrainer(
        prediction_model=prediction_model,
        weight_model=weight_model,
        device=device,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        clip_grad=config['training'].get('clip_grad', 1.0),
        lambda_ctr=config['training']['lambda_ctr'],
        lambda_playtime=config['training']['lambda_playtime'],
        logger=logger
    )
    
    # 运行主训练循环
    train_main_loop(
        data_manager=data_manager,
        prediction_trainer=prediction_trainer,
        config=config,
        logger=logger,
        exp_dir=exp_dir,
        device=device
    )
    
    # 最终评估
    logger.info("训练完成，开始最终评估")
    evaluate_model(
        data_manager=data_manager,
        prediction_trainer=prediction_trainer,
        config=config,
        logger=logger,
        exp_dir=exp_dir
    )
    
    logger.info("实验完成")


def train_main_loop(
    data_manager: DataManager,
    prediction_trainer: PredictionTrainer,
    config: Dict[str, Any],
    logger: Logger,
    exp_dir: str,
    device: str
) -> None:
    """
    主训练循环，按照算法1的流程进行训练
    
    Args:
        data_manager: 数据管理器
        prediction_trainer: 预测模型训练器
        config: 配置字典
        logger: 日志记录器
        exp_dir: 实验目录
        device: 设备
    """
    # 获取训练参数
    total_simulation_steps = config['training']['steps']  # 重命名为total_simulation_steps
    eval_steps = config['training']['eval_steps']
    save_best = config['training'].get('save_best', True)
    patience = config['training'].get('patience', 5)
    
    # 从配置中获取新的参数
    n_user_T = config['training'].get('n_user_T', 100)  # 每步处理组用户数
    n_user_C = config['training'].get('n_user_C', 100)  # 每步对照组用户数
    candidate_pool_size_train = config['training'].get('candidate_pool_size_train', 10)  # 训练时候选池大小
    
    logger.info(f"开始训练，总模拟步骤数: {total_simulation_steps}，评估间隔: {eval_steps}")
    logger.info(f"每步处理组用户数: {n_user_T}, 对照组用户数: {n_user_C}, 候选池大小: {candidate_pool_size_train}")
    
    # 初始化结果字典
    results = {
        'step': [],
        'metrics': []
    }
    
    # 初始化早停计数器和最佳验证指标
    early_stop_counter = 0
    best_val_metric = float('inf')  # 用于记录最佳验证损失
    
    # 训练循环
    for step in range(1, total_simulation_steps + 1):
        # 1. 获取当前步骤的数据，包括推荐和收集交互
        step_data = data_manager.get_step_data(
            n_users_t=n_user_T, 
            n_users_c=n_user_C, 
            candidate_pool_size=candidate_pool_size_train,
            treatment_model=prediction_trainer.prediction_model,
            control_model=prediction_trainer.prediction_model  # 在此阶段两个模型相同
        )
        
        # 2. 使用收集到的数据创建批次
        # 训练数据
        train_batch = {
            'features': {'user_features': torch.stack(step_data['train']['features']) if step_data['train']['features'] else torch.zeros((0, data_manager.feature_dim))},
            'treatment': torch.stack(step_data['train']['treatments']) if step_data['train']['treatments'] else torch.zeros(0),
            'label': torch.stack(step_data['train']['labels']) if step_data['train']['labels'] else torch.zeros((0, 2)),
            'group': torch.stack(step_data['train']['treatments']) if step_data['train']['treatments'] else torch.zeros(0),
            'ctr': torch.tensor([label[0] for label in step_data['train']['labels']]) if step_data['train']['labels'] else torch.zeros(0),
            'playtime': torch.tensor([label[1] for label in step_data['train']['labels']]) if step_data['train']['labels'] else torch.zeros(0)
        }
        
        # 测试/验证数据
        t_test_features = step_data['test']['treatment']['features']
        t_test_treatments = step_data['test']['treatment']['treatments']
        t_test_labels = step_data['test']['treatment']['labels']
        
        c_test_features = step_data['test']['control']['features']
        c_test_treatments = step_data['test']['control']['treatments']
        c_test_labels = step_data['test']['control']['labels']
        
        all_test_features = t_test_features + c_test_features
        all_test_treatments = t_test_treatments + c_test_treatments
        all_test_labels = t_test_labels + c_test_labels
        
        test_batch = {
            'features': {'user_features': torch.stack(all_test_features) if all_test_features else torch.zeros((0, data_manager.feature_dim))},
            'treatment': torch.stack(all_test_treatments) if all_test_treatments else torch.zeros(0),
            'label': torch.stack(all_test_labels) if all_test_labels else torch.zeros((0, 2)),
            'group': torch.stack(all_test_treatments) if all_test_treatments else torch.zeros(0),
            'ctr': torch.tensor([label[0] for label in all_test_labels]) if all_test_labels else torch.zeros(0),
            'playtime': torch.tensor([label[1] for label in all_test_labels]) if all_test_labels else torch.zeros(0)
        }
        
        # 3. 训练步骤
        # 如果数据不为空，则训练模型
        train_metrics = {'train': {'treatment': {'total_loss': 0.0}, 'control': {'total_loss': 0.0}}}
        if step_data['train']['treatment']['features'] or step_data['train']['control']['features']:
            train_metrics = prediction_trainer.train_step(train_batch)
        
        # 记录结果
        results['step'].append(step)
        
        # 4. 定期评估
        if step % eval_steps == 0 or step == total_simulation_steps:
            # 在测试批次上评估
            val_metrics = {'val': {'treatment': {'total_loss': 0.0}, 'control': {'total_loss': 0.0}}}
            if all_test_features:  # 确保有测试数据
                val_metrics = prediction_trainer.evaluate_batch(test_batch)
            
            # 合并训练和验证指标
            step_metrics = {
                'train': train_metrics['train'],
                'val': val_metrics['val'],
                'train_users': {
                    'treatment': step_data['train']['treatment']['n_users'],
                    'control': step_data['train']['control']['n_users']
                },
                'test_users': {
                    'treatment': step_data['test']['treatment']['n_users'],
                    'control': step_data['test']['control']['n_users']
                }
            }
            
            results['metrics'].append(step_metrics)
            
            # 打印主要指标
            logger.info(f"步骤 {step}/{total_simulation_steps}:")
            logger.info(f"  训练损失 - 处理组: {train_metrics['train']['treatment']['total_loss']:.4f}, "
                       f"对照组: {train_metrics['train']['control']['total_loss']:.4f}")
            logger.info(f"  训练用户数 - 处理组: {step_data['train']['treatment']['n_users']}, "
                       f"对照组: {step_data['train']['control']['n_users']}")
            
            val_t_metrics = val_metrics['val']['treatment']
            val_c_metrics = val_metrics['val']['control']
            
            logger.info(f"  验证损失 - 处理组: {val_t_metrics['total_loss']:.4f}, "
                       f"对照组: {val_c_metrics['total_loss']:.4f}")
            logger.info(f"  测试用户数 - 处理组: {step_data['test']['treatment']['n_users']}, "
                       f"对照组: {step_data['test']['control']['n_users']}")
            
            if 'ctr_auc' in val_t_metrics:
                logger.info(f"  验证AUC - 处理组: {val_t_metrics['ctr_auc']:.4f}, "
                           f"对照组: {val_c_metrics['ctr_auc']:.4f}")
            
            # 保存结果
            save_path = os.path.join(exp_dir, 'exp_results.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            # 如果是最佳模型，保存
            # 使用总损失作为指标，也可以使用其他指标如AUC
            current_val_loss = (val_t_metrics['total_loss'] + val_c_metrics['total_loss']) / 2
            is_best = current_val_loss < best_val_metric
            
            if save_best and is_best:
                best_val_metric = current_val_loss
                logger.info("发现最佳模型，保存中...")
                best_model_path = os.path.join(exp_dir, 'models', 'prediction_model_best.pth')
                prediction_trainer.save_model(prediction_trainer.prediction_model, best_model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # 保存当前模型检查点
            if step % (eval_steps * 10) == 0 or step == total_simulation_steps:
                checkpoint_path = os.path.join(exp_dir, 'models', f'prediction_model_step{step}.pth')
                prediction_trainer.save_model(prediction_trainer.prediction_model, checkpoint_path)
                logger.info(f"当前模型已保存至 {checkpoint_path}")
            
            # 早停
            if patience > 0 and early_stop_counter >= patience:
                logger.info(f"验证性能已连续{patience}次未改善，提前停止训练")
                break
    
    # 保存最终模型
    final_model_path = os.path.join(exp_dir, 'models', 'prediction_model_final.pth')
    prediction_trainer.save_model(prediction_trainer.prediction_model, final_model_path)
    logger.info(f"最终模型已保存至 {final_model_path}")
    
    logger.info("训练完成")
    return


def evaluate_model(
    data_manager: DataManager,
    prediction_trainer: PredictionTrainer,
    config: Dict[str, Any],
    logger: Logger,
    exp_dir: str
) -> None:
    """
    评估模型
    
    Args:
        data_manager: 数据管理器
        prediction_trainer: 预测模型训练器
        config: 配置字典
        logger: 日志记录器
        exp_dir: 实验目录
    """
    # 加载测试数据加载器
    _, test_dataloader = data_manager.get_prediction_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        is_test=True
    )
    
    # 评估模型
    logger.info("在测试集上评估模型")
    test_metrics = prediction_trainer.evaluate(test_dataloader)
    
    # 打印测试集指标
    val_t_metrics = test_metrics['val']['treatment']
    val_c_metrics = test_metrics['val']['control']
    
    logger.info("测试集结果:")
    logger.info(f"  处理组 - 总损失: {val_t_metrics['total_loss']:.4f}, "
               f"CTR AUC: {val_t_metrics.get('ctr_auc', 0.0):.4f}, "
               f"播放时间 RMSE: {val_t_metrics.get('playtime_rmse', 0.0):.4f}")
    logger.info(f"  对照组 - 总损失: {val_c_metrics['total_loss']:.4f}, "
               f"CTR AUC: {val_c_metrics.get('ctr_auc', 0.0):.4f}, "
               f"播放时间 RMSE: {val_c_metrics.get('playtime_rmse', 0.0):.4f}")
    
    # 保存测试结果
    test_results_path = os.path.join(exp_dir, 'test_results.json')
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"测试结果已保存至 {test_results_path}")
    return


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='IEDA加权训练实验')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    config = config_manager.get_all()
    
    # 创建实验目录
    exp_dir = create_experiment_dir(config)
    
    # 运行实验
    run_weighted_training(config, exp_dir)


if __name__ == "__main__":
    main()
