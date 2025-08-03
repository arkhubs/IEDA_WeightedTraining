import os
import yaml
import logging
from datetime import datetime

class ConfigManager:
    """配置管理类，负责读取和处理配置文件"""
    
    def __init__(self, config_path):
        """
        初始化配置管理器
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.exp_dir = self._create_exp_dir()
        self._setup_logging()
        
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {str(e)}")
    
    def _create_exp_dir(self):
        """创建实验目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        exp_dir = os.path.join(self.config['logging']['log_dir'], timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        
        # 保存当前配置到实验目录
        with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        return exp_dir
    
    def _setup_logging(self):
        """设置日志记录"""
        log_level = getattr(logging, self.config['logging']['level'])
        log_file = os.path.join(self.exp_dir, 'run.log')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def get_config(self):
        """获取完整配置"""
        return self.config
    
    def get_mode(self):
        """获取实验模式"""
        return self.config['mode']
    
    def get_features(self):
        """获取特征配置"""
        return self.config['feature']
    
    def get_label_info(self):
        """获取标签配置"""
        return self.config.get('labels', [])
    
    def get_dataset_config(self):
        """获取数据集配置"""
        return self.config['dataset']
    
    def get_model_config(self, model_name=None):
        """
        获取模型配置
        
        Args:
            model_name: 模型名称，如果为None则返回第一个模型配置
        
        Returns:
            模型配置字典
        """
        if 'models' in self.config:
            if model_name and model_name in self.config['models']:
                return self.config['models'][model_name]
            elif model_name:
                self.logger.warning(f"模型 {model_name} 不存在，返回第一个可用模型配置")
                return list(self.config['models'].values())[0] if self.config['models'] else {}
            else:
                return list(self.config['models'].values())[0] if self.config['models'] else {}
        elif 'model' in self.config:
            # 向后兼容旧配置
            return self.config['model']
        else:
            return {}
    
    def get_pretrain_config(self):
        """获取预训练配置"""
        return self.config['pretrain']
    
    def get_global_config(self):
        """获取全局仿真配置"""
        return self.config['global']
    
    def get_exp_dir(self):
        """获取实验目录"""
        return self.exp_dir
    
    def get_checkpoints_dir(self):
        """获取检查点目录"""
        return os.path.join(self.exp_dir, 'checkpoints')
