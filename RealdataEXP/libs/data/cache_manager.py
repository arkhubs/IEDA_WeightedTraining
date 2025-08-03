"""
缓存管理器
提供数据的持久化缓存功能，避免重复计算
"""

import os
import pickle
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"[缓存] 创建缓存目录: {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def save(self, data: Any, key: str) -> None:
        """保存数据到缓存"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"[缓存] 数据已保存: {key}")
        except Exception as e:
            logger.error(f"[缓存] 保存失败 {key}: {e}")
            
    def load(self, key: str) -> Optional[Any]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"[缓存] 数据已加载: {key}")
            return data
        except Exception as e:
            logger.error(f"[缓存] 加载失败 {key}: {e}")
            return None
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        cache_path = self._get_cache_path(key)
        return os.path.exists(cache_path)
    
    def clear(self, key: str) -> None:
        """清除指定缓存"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"[缓存] 缓存已清除: {key}")
    
    def clear_all(self) -> None:
        """清除所有缓存"""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("[缓存] 所有缓存已清除")