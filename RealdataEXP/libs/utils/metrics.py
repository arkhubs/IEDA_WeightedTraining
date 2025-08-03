"""
指标跟踪器
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_metrics = {}
        
    def update(self, metrics: Dict[str, float], step: int = None):
        """更新指标"""
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        self.current_metrics = metrics.copy()
        if step is not None:
            self.current_metrics['step'] = step
    
    def get_latest(self) -> Dict[str, float]:
        """获取最新指标"""
        return self.current_metrics.copy()
    
    def get_history(self, key: str) -> List[float]:
        """获取指标历史"""
        return self.metrics[key].copy()
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取指标摘要"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': float(values[-1])
                }
        return summary
    
    def log_current(self, prefix: str = ""):
        """记录当前指标"""
        if self.current_metrics:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in self.current_metrics.items()])
            logger.info(f"[{prefix}] {metrics_str}")
    
    def reset(self):
        """重置指标"""
        self.metrics.clear()
        self.current_metrics.clear()