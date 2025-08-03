"""
配置文件管理模块，用于处理训练实验的各种配置项
"""
import os
import json
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器，用于加载和管理实验配置"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，支持.json或.yaml格式
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {str(e)}")
            
        return self.config
    
    def save_config(self, save_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        file_ext = os.path.splitext(save_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            elif file_ext in ['.yml', '.yaml']:
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持用.分隔的嵌套键
            default: 默认值，如果键不存在则返回此值
            
        Returns:
            配置值
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键，支持用.分隔的嵌套键
            value: 配置值
        """
        if '.' not in key:
            self.config[key] = value
            return
        
        parts = key.split('.')
        config = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            elif not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
            
        config[parts[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_dict: 要更新的配置字典
        """
        self._update_recursive(self.config, config_dict)
    
    def _update_recursive(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        递归更新字典
        
        Args:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_recursive(target[key], value)
            else:
                target[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            完整配置字典的副本
        """
        return self.config.copy()


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        "experiment": {
            "name": "weighted_training_experiment",
            "seed": 42,
            "device": "cuda",
            "verbose": True
        },
        "data": {
            "train_path": "data/train.csv",
            "val_path": "data/val.csv",
            "test_path": "data/test.csv",
            "batch_size": 128,
            "num_workers": 4,
            "shuffle": True,
            "features": {
                "categorical": ["user_id", "item_id", "category"],
                "numerical": ["user_age", "item_popularity", "position"]
            }
        },
        "training": {
            "steps": 10000,
            "eval_steps": 500,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "save_best": True,
            "patience": 5,
            "clip_grad": 1.0,
            "lambda_ctr": 1.0,
            "lambda_playtime": 0.5
        },
        "model": {
            "prediction": {
                "type": "MLP",
                "hidden_dims": [128, 64, 32],
                "dropout": 0.2,
                "use_batch_norm": True
            },
            "weight": {
                "type": "MLP",
                "hidden_dims": [64, 32],
                "dropout": 0.1,
                "use_batch_norm": True
            },
            "embedding_dim": 16
        },
        "output": {
            "save_dir": "results/",
            "model_dir": "models/",
            "log_level": "INFO"
        }
    }


# 预定义配置模板示例
DEFAULT_CONFIG_TEMPLATE = {
    "small": {
        "data": {"batch_size": 64},
        "model": {
            "prediction": {"hidden_dims": [64, 32, 16]},
            "weight": {"hidden_dims": [32, 16]},
            "embedding_dim": 8
        }
    },
    "medium": {
        "data": {"batch_size": 128},
        "model": {
            "prediction": {"hidden_dims": [128, 64, 32]},
            "weight": {"hidden_dims": [64, 32]},
            "embedding_dim": 16
        }
    },
    "large": {
        "data": {"batch_size": 256},
        "model": {
            "prediction": {"hidden_dims": [256, 128, 64]},
            "weight": {"hidden_dims": [128, 64]},
            "embedding_dim": 32
        }
    }
}


if __name__ == "__main__":
    # 示例：创建默认配置文件
    default_config = create_default_config()
    os.makedirs("config", exist_ok=True)
    
    config_manager = ConfigManager()
    config_manager.config = default_config
    
    # 保存默认配置
    config_manager.save_config("config/default_config.json")
    
    # 生成不同规模的配置
    for size, template in DEFAULT_CONFIG_TEMPLATE.items():
        config_manager.config = default_config.copy()
        config_manager.update(template)
        config_manager.save_config(f"config/{size}_config.json")
        
    print("配置文件已生成完毕")
