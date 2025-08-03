import os
import sys
import argparse
import logging
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 引入项目模块
from libs.src.config_manager import ConfigManager
from libs.src.data_manager import DataManager
from libs.src.data_manager_methods import DataManagerMethods
from libs.src.models import create_model
from libs.src.recommender import Recommender
from libs.src.trainers import Trainer
from libs.src.utils import setup_seed, get_device

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="IEDA全局GTE计算")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, default=None, help="实验模式，会覆盖配置文件中的mode")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--no_pretrain", action="store_true", help="跳过预训练")
    return parser.parse_args()

def main():
    # 设置随机种子
    setup_seed(42)
    
    # 解析参数
    args = parse_args()
    
    # 初始化配置管理器
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # 如果指定了模式，覆盖配置文件中的模式
    if args.mode is not None:
        config['mode'] = args.mode
        
    # 获取实验目录
    # 统一实验目录为 base_dir 拼接
    base_dir = config.get('base_dir', os.getcwd())
    exp_dir = os.path.join(base_dir, config_manager.get_exp_dir())
    
    # 确保实验目录存在
    os.makedirs(exp_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, 'run.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("main")
    logger.info(f"实验配置已加载: {args.config}")
    logger.info(f"实验模式: {config['mode']}")
    logger.info(f"实验目录: {exp_dir}")
    
    # 获取计算设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 初始化数据管理器
    logger.info("初始化数据管理器...")
    data_manager = DataManager(config_manager)
    data_manager_methods = DataManagerMethods(data_manager)
    
    # 获取特征信息
    feature_info = data_manager.get_feature_info()
    
    # 初始化模型
    logger.info("[Init] 初始化模型...")
    models = {}
    optimizers = {}
    criteria = {}
    label_configs = config_manager.get_label_info()
    
    # 为每个标签创建单独的模型
    for label_config in label_configs:
        label_name = label_config['name']
        model_name = label_config.get('model', 'MLP')  # 默认使用MLP模型
        logger.info(f"[Init] 为标签 {label_name} 创建 {model_name} 模型")
        
        # 获取模型配置
        if 'model_params' in label_config:
            # 使用标签特定的模型参数
            model_config = label_config['model_params']
            model_config['type'] = model_name
        else:
            # 使用全局模型参数
            model_config = config_manager.get_model_config()
        
        # 创建模型
        model = create_model(model_config, feature_info).to(device)
        models[label_name] = model
    
    # 根据实验模式选择相应的流程
    if config['mode'] == 'global':
        # 初始化推荐器
        logger.info("[Init] 初始化推荐器...")
        recommender = Recommender(config_manager, data_manager_methods, models, device)
        
        # 初始化训练器
        logger.info("[Init] 初始化训练器...")
        trainer = Trainer(config_manager, data_manager, models, device)
        
        # 为每个模型创建损失函数和优化器
        for label_name, model in models.items():
            logger.info(f"[Init] 为标签 {label_name} 创建损失函数和优化器")
            criterion = trainer.create_criterion(label_name)
            optimizer = trainer.create_optimizer(model, label_name)
            criteria[label_name] = criterion
            optimizers[label_name] = optimizer
        
        # 检查是否恢复训练
        if args.resume:
            logger.info(f"[Resume] 尝试从检查点恢复训练: {args.resume}")
            if trainer.resume_training(recommender, optimizers, args.resume):
                logger.info("[Resume] 成功恢复训练")
            else:
                logger.warning("[Resume] 恢复训练失败，将开始新训练")
        
        # 预训练（如果需要）
        if not args.no_pretrain and not args.resume:
            logger.info("[Train] 开始预训练...")
            trainer.pretrain(criteria, optimizers)
        
        # 初始化全局实验模式
        logger.info("[Init] 初始化全局实验模式...")
        from libs.exp_modes.global_mode import GlobalExperiment
        global_exp = GlobalExperiment(config_manager, data_manager_methods, models, device)
        
        # 运行全局实验
        logger.info("[Global] 开始全局实验...")
        results = global_exp.run_experiment(recommender, optimizers, criteria)
        
        # 输出实验结果
        logger.info("[Result] 全局实验完成")
        for label_name in results['GTE']:
            gte = results['GTE'][label_name]
            treatment = results['treatment'][label_name]
            control = results['control'][label_name]
            
            logger.info(f"[Result] 标签 {label_name}:")
            logger.info(f"  - 最终GTE: {gte:.4f}")
            logger.info(f"  - Treatment组累积奖励: {treatment:.4f}")
            logger.info(f"  - Control组累积奖励: {control:.4f}")
        
        # 绘制结果可视化
        logger.info("[Viz] 生成实验结果可视化")
        try:
            from libs.src.plot_results import plot_advanced_metrics, load_results, load_training_logs
            from libs.src.plot_training_metrics import plot_training_metrics
            
            # 加载结果文件
            result_file = os.path.join(exp_dir, 'global_result.json')
            with open(result_file, 'r') as f:
                import json
                results = json.load(f)
            
            # 绘制高级指标图表
            plot_advanced_metrics(results, None, exp_dir)
            
            # 绘制训练指标
            plot_training_metrics(exp_dir)
            
            logger.info(f"[Viz] 实验结果可视化已保存至: {exp_dir}")
        except Exception as e:
            logger.error(f"[Viz] 生成可视化时出错: {str(e)}")
    else:
        logger.error(f"不支持的实验模式: {config['mode']}")
        
if __name__ == "__main__":
    main()
