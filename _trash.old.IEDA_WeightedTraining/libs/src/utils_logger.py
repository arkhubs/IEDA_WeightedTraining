import logging
import os
import time
from datetime import datetime
import json

class LoggerManager:
    """日志管理类，负责设置和管理各模块的日志"""
    
    def __init__(self, exp_dir, log_level=logging.INFO):
        """
        初始化日志管理器
        
        Args:
            exp_dir: 实验目录
            log_level: 日志级别
        """
        self.exp_dir = exp_dir
        self.log_level = log_level
        self.log_file = os.path.join(exp_dir, "run.log")
        self.loggers = {}
        
        # 设置根日志记录器
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """设置根日志记录器"""
        # 创建目录
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # 设置格式化器 - 修改格式，在消息前添加模块标识
        formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def get_module_logger(self, module_name, phase=None):
        """
        获取带有阶段标识的模块日志记录器
        
        Args:
            module_name: 模块名称
            phase: 当前执行阶段，如'init', 'train', 'validate'等
            
        Returns:
            日志记录器实例
        """
        name = module_name
        if phase:
            name = f"{module_name}.{phase}"
            
        return self.get_logger(name)
        
    def get_logger(self, name):
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器实例
        """
        if name in self.loggers:
            return self.loggers[name]
        
        # 创建新的日志记录器
        logger = logging.getLogger(name)
        self.loggers[name] = logger
        
        return logger
    
    def log_config(self, config):
        """
        记录配置信息
        
        Args:
            config: 配置字典
        """
        logger = self.get_logger("Config")
        logger.info("实验配置:")
        
        # 记录主要配置
        logger.info(f"实验模式: {config['mode']}")
        
        if 'dataset' in config:
            dataset_config = config['dataset']
            logger.info(f"数据集: {dataset_config.get('name', 'unknown')}")
        
        if 'feature' in config:
            feature_config = config['feature']
            num_features = len(feature_config.get('numerical', []))
            cat_features = len(feature_config.get('categorical', []))
            logger.info(f"特征: {num_features} 数值特征, {cat_features} 分类特征")
        
        if 'label' in config:
            label_config = config['label']
            logger.info(f"标签: {label_config.get('target', 'unknown')} ({label_config.get('type', 'unknown')})")
        
        if config['mode'] == 'global' and 'global' in config:
            global_config = config['global']
            logger.info(f"全局配置: batch_size={global_config.get('batch_size', 'unknown')}, "
                       f"n_steps={global_config.get('n_steps', 'unknown')}, "
                       f"n_candidate={global_config.get('n_candidate', 'unknown')}")
        
        if 'recommender' in config:
            rec_config = config['recommender']
            logger.info(f"推荐器配置: alpha_T={rec_config.get('alpha_T', 'unknown')}, "
                       f"alpha_C={rec_config.get('alpha_C', 'unknown')}")
        
        # 保存完整配置到实验目录
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_experiment_start(self, config):
        """
        记录实验开始
        
        Args:
            config: 配置字典
        """
        logger = self.get_logger("Experiment")
        
        # 记录时间
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"实验开始时间: {start_time}")
        
        # 记录配置
        self.log_config(config)
        
        # 记录实验信息
        logger.info("=" * 50)
        logger.info(f"启动新实验: 模式={config['mode']}")
        logger.info("=" * 50)
    
    def log_experiment_end(self):
        """记录实验结束"""
        logger = self.get_logger("Experiment")
        
        # 记录时间
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"实验结束时间: {end_time}")
        
        logger.info("=" * 50)
        logger.info("实验完成")
        logger.info("=" * 50)
