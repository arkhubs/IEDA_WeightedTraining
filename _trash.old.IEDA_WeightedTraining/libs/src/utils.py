import os
import logging
import json
import numpy as np
import random
import torch
from datetime import datetime

def setup_seed(seed=42):
    """
    设置随机种子，确保实验可重复
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def setup_logger(log_dir, name="main"):
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        name: 日志名称
        
    Returns:
        日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_json(data, filepath):
    """
    保存JSON数据
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """
    加载JSON数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def create_experiment_dir(base_dir="results"):
    """
    创建实验目录
    
    Args:
        base_dir: 基础目录
        
    Returns:
        实验目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    
    return exp_dir

def get_device():
    """
    获取计算设备
    
    Returns:
        计算设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """提前停止训练的工具类"""
    
    def __init__(self, patience=5, min_delta=0):
        """
        初始化
        
        Args:
            patience: 耐心值，连续多少次没有改善后停止
            min_delta: 最小变化量，小于此值视为没有改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        检查是否应该提前停止
        
        Args:
            score: 当前分数
            
        Returns:
            是否是新的最佳分数
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
