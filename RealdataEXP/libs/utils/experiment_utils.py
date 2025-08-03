"""
实验工具函数
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_experiment_dir(base_dir: str) -> str:
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = os.path.join(base_dir, "results", timestamp)
    
    # 创建目录结构
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    logger.info(f"[实验目录] 创建实验目录: {exp_dir}")
    return exp_dir

def save_results(results: Dict[str, Any], save_path: str):
    """保存实验结果到JSON文件"""
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"[结果保存] 结果已保存到: {save_path}")
    except Exception as e:
        logger.error(f"[结果保存] 保存失败: {e}")

def load_results(file_path: str) -> Dict[str, Any]:
    """从JSON文件加载实验结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"[结果加载] 结果已从 {file_path} 加载")
        return results
    except Exception as e:
        logger.error(f"[结果加载] 加载失败: {e}")
        return {}