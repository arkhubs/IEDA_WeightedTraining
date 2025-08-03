import os
import sys
import logging

# 添加项目根目录到路径，确保能导入到libs模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from libs.src.utils_logger import LoggerManager

def init_all(config_path):
    """
    初始化所有模块
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器、数据管理器、数据管理方法、模型、推荐器、训练器、设备
    """
    # 设置随机种子
    setup_seed(42)
    
    # 初始化配置管理器
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    exp_dir = config_manager.get_exp_dir()
    
    # 初始化日志管理器
    logger_manager = LoggerManager(exp_dir)
    logger = logger_manager.get_logger("Initialization")
    logger_manager.log_experiment_start(config)
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 初始化数据管理器
    logger.info("初始化数据管理器")
    data_manager = DataManager(config_manager)
    
    # 初始化数据管理方法
    logger.info("初始化数据管理方法")
    data_manager_methods = DataManagerMethods(data_manager)
    
    # 获取特征信息
    feature_info = data_manager.get_feature_info()
    
    # 初始化模型 - 多标签支持
    logger.info("初始化模型")
    models = {}
    
    # 获取标签配置
    label_configs = config_manager.get_label_info()
    
    # 为每个标签创建单独的模型
    for label_config in label_configs:
        label_name = label_config['name']
        model_name = label_config.get('model', 'MLP')  # 默认使用MLP模型
        logger.info(f"为标签 {label_name} 创建 {model_name} 模型")
        
        # 获取模型配置
        if 'model_params' in label_config:
            # 使用标签特定的模型参数
            model_config = label_config['model_params']
            model_config['type'] = model_name
        else:
            # 使用全局模型参数
            model_config = config_manager.get_model_config(label_config.get('model_name'))
        
        # 创建模型
        model = create_model(model_config, feature_info).to(device)
        models[label_name] = model
    
    # 如果没有标签配置，使用单一模型（向后兼容）
    if not models:
        model_config = config_manager.get_model_config()
        model = create_model(model_config, feature_info).to(device)
        models['label'] = model
    
    # 初始化推荐器
    logger.info("初始化推荐器")
    recommender = Recommender(config_manager, data_manager_methods, models, device)
    
    # 初始化训练器
    logger.info("初始化训练器")
    trainer = Trainer(config_manager, data_manager, models, device)
    
    return config_manager, data_manager, data_manager_methods, model, recommender, trainer, device
