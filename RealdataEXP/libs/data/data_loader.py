"""
KuaiRand数据集加载器
负责从原始数据文件中加载用户行为日志、用户特征和视频特征
支持分片文件自动合并和优化的PyTorch Dataset接口
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)

class TabularDataset(Dataset):
    """用于表格数据的PyTorch Dataset"""
    
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame, label_configs: List[Dict]):
        """
        Args:
            features (pd.DataFrame): 包含所有输入特征的DataFrame
            labels (pd.DataFrame): 包含所有目标标签的DataFrame
            label_configs (List[Dict]): 标签的配置信息
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = {}
        self.label_configs = label_configs
        
        for config in self.label_configs:
            target_col = config['target']
            if target_col in labels.columns:
                self.labels[config['name']] = torch.tensor(labels[target_col].values, dtype=torch.float32).unsqueeze(1)
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_vector = self.features[idx]
        target_dict = {name: label_tensor[idx] for name, label_tensor in self.labels.items()}
        return feature_vector, target_dict

class KuaiRandDataLoader:
    """KuaiRand数据集加载器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_path = config['dataset']['path']
        self.cache_manager = CacheManager(config['dataset']['cache_path'])
        
        # 根据数据集名称选择数据文件映射
        dataset_name = config['dataset']['name']
        if dataset_name == "KuaiRand-Pure":
            self.data_files = {
                'log_random': 'data/log_random_4_22_to_5_08_pure.csv',
                'log_standard_early': 'data/log_standard_4_08_to_4_21_pure.csv', 
                'log_standard_late': 'data/log_standard_4_22_to_5_08_pure.csv',
                'user_features': 'data/user_features_pure.csv',
                'video_basic': 'data/video_features_basic_pure.csv',
                'video_statistic': 'data/video_features_statistic_pure.csv'
            }
        elif dataset_name == "KuaiRand-27K":
            self.data_files = {
                'log_random': 'data/log_random_4_22_to_5_08_27k.csv',
                'log_standard_early': 'data/log_standard_4_08_to_4_21_27k.csv', 
                'log_standard_late': 'data/log_standard_4_22_to_5_08_27k.csv',
                'user_features': 'data/user_features_27k.csv',
                'video_basic': 'data/video_features_basic_27k.csv',
                'video_statistic': 'data/video_features_statistic_27k.csv'
            }
        elif dataset_name == "KuaiRand-1K":
            self.data_files = {
                'log_random': 'data/log_random_4_22_to_5_08_1k.csv',
                'log_standard_early': 'data/log_standard_4_08_to_4_21_1k.csv', 
                'log_standard_late': 'data/log_standard_4_22_to_5_08_1k.csv',
                'user_features': 'data/user_features_1k.csv',
                'video_basic': 'data/video_features_basic_1k.csv',
                'video_statistic': 'data/video_features_statistic_1k.csv'
            }
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 内存中的数据
        self.user_video_lists = {}  # user_id -> list of video_ids
        self.merged_data = None
        self.train_users = None
        self.val_users = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据文件，支持分片文件自动合并"""
        logger.info("[数据加载] 开始加载所有数据文件...")
        
        data = {}
        total_files = len(self.data_files)
        
        for i, (key, file_path) in enumerate(self.data_files.items(), 1):
            full_path = os.path.join(self.dataset_path, file_path)
            logger.info(f"[数据加载] ({i}/{total_files}) 正在加载 {key}: {file_path}")
            
            # 检查是否存在分片文件
            base_name = os.path.splitext(file_path)[0]
            base_path = os.path.join(self.dataset_path, base_name)
            
            # 查找分片文件
            part_files = []
            for part_num in range(1, 10):  # 最多支持9个分片
                part_file = f"{base_path}_part{part_num}.csv"
                if os.path.exists(part_file):
                    part_files.append(part_file)
                else:
                    break
            
            if part_files:
                # 存在分片文件，进行合并
                logger.info(f"[数据加载] 发现 {key} 的分片文件 {len(part_files)} 个，开始合并...")
                dfs = []
                for part_file in part_files:
                    logger.info(f"[数据加载] 正在加载分片: {os.path.basename(part_file)}")
                    df_part = pd.read_csv(part_file)
                    logger.info(f"[数据加载] 分片形状: {df_part.shape}")
                    dfs.append(df_part)
                
                # 合并所有分片
                data[key] = pd.concat(dfs, ignore_index=True)
                logger.info(f"[数据加载] {key} 分片合并完成，总形状: {data[key].shape}")
            else:
                # 没有分片文件，直接加载
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"数据文件不存在: {full_path}")
                    
                data[key] = pd.read_csv(full_path)
                logger.info(f"[数据加载] {key} 加载完成，形状: {data[key].shape}")
            
        logger.info("[数据加载] 所有数据文件加载完成")
        return data
    
    def merge_features(self, log_data: pd.DataFrame, user_features: pd.DataFrame,
                      video_basic: pd.DataFrame, video_statistic: pd.DataFrame) -> pd.DataFrame:
        """合并所有特征数据"""
        logger.info("[特征合并] 开始合并用户和视频特征...")
        
        # 合并用户特征
        merged = log_data.merge(user_features, on='user_id', how='left')
        logger.info(f"[特征合并] 合并用户特征后形状: {merged.shape}")
        
        # 合并视频基础特征
        merged = merged.merge(video_basic, on='video_id', how='left')
        logger.info(f"[特征合并] 合并视频基础特征后形状: {merged.shape}")
        
        # 合并视频统计特征
        merged = merged.merge(video_statistic, on='video_id', how='left')
        logger.info(f"[特征合并] 合并视频统计特征后形状: {merged.shape}")
        
        logger.info("[特征合并] 特征合并完成")
        return merged
    
    def create_user_video_lists(self, merged_data: pd.DataFrame) -> Dict[int, List[int]]:
        """创建用户-视频交互列表（缓存机制）"""
        cache_key = "user_video_lists"
        
        # 尝试从缓存加载
        cached_data = self.cache_manager.load(cache_key)
        if cached_data is not None:
            logger.info("[缓存] 从缓存加载用户-视频交互列表")
            return cached_data
            
        logger.info("[用户视频列表] 开始创建用户-视频交互列表...")
        
        user_video_lists = {}
        for user_id in merged_data['user_id'].unique():
            video_list = merged_data[merged_data['user_id'] == user_id]['video_id'].tolist()
            user_video_lists[user_id] = video_list
            
        logger.info(f"[用户视频列表] 创建完成，共 {len(user_video_lists)} 个用户")
        
        # 保存到缓存
        self.cache_manager.save(user_video_lists, cache_key)
        logger.info("[缓存] 用户-视频交互列表已保存到缓存")
        
        return user_video_lists
    
    def split_users(self, user_list: List[int], val_ratio: float) -> Tuple[List[int], List[int]]:
        """将用户划分为训练集和验证集"""
        logger.info(f"[用户划分] 开始划分用户，验证集比例: {val_ratio}")
        
        np.random.shuffle(user_list)
        split_idx = int(len(user_list) * (1 - val_ratio))
        
        train_users = user_list[:split_idx]
        val_users = user_list[split_idx:]
        
        logger.info(f"[用户划分] 训练用户数: {len(train_users)}, 验证用户数: {len(val_users)}")
        return train_users, val_users
    
    def add_mask_and_used_flags(self, merged_data: pd.DataFrame, val_users: List[int]) -> pd.DataFrame:
        """添加mask和used标记位"""
        logger.info("[标记位] 添加mask和used标记位...")
        
        # 添加mask标记：验证集用户的视频标记为1
        merged_data['mask'] = merged_data['user_id'].isin(val_users).astype(int)
        
        # 添加used标记：初始化为0
        merged_data['used'] = 0
        
        mask_count = merged_data['mask'].sum()
        total_count = len(merged_data)
        
        logger.info(f"[标记位] mask=1的样本数: {mask_count}/{total_count} ({mask_count/total_count:.2%})")
        logger.info("[标记位] 标记位添加完成")
        
        return merged_data
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict[int, List[int]], List[int], List[int]]:
        """加载并准备所有数据"""
        logger.info("[数据准备] 开始数据加载和准备流程...")
        
        # 加载原始数据
        raw_data = self.load_all_data()
        
        # 合并日志数据
        logger.info("[数据合并] 合并多个日志文件...")
        log_combined = pd.concat([
            raw_data['log_random'],
            raw_data['log_standard_early'], 
            raw_data['log_standard_late']
        ], ignore_index=True)
        logger.info(f"[数据合并] 合并后日志数据形状: {log_combined.shape}")
        
        # 合并特征
        merged_data = self.merge_features(
            log_combined, 
            raw_data['user_features'],
            raw_data['video_basic'],
            raw_data['video_statistic']
        )
        
        # 创建用户-视频交互列表
        user_video_lists = self.create_user_video_lists(merged_data)
        
        # 用户划分
        all_users = list(merged_data['user_id'].unique())
        train_users, val_users = self.split_users(all_users, self.config['global']['user_p_val'])
        
        # 添加标记位
        merged_data = self.add_mask_and_used_flags(merged_data, val_users)
        
        logger.info("[数据准备] 数据准备流程完成")
        
        # 存储到实例变量
        self.merged_data = merged_data
        self.user_video_lists = user_video_lists
        self.train_users = train_users
        self.val_users = val_users
        
        return merged_data, user_video_lists, train_users, val_users
    
    def get_dataset_stats(self) -> Dict:
        """获取数据集统计信息"""
        if self.merged_data is None:
            raise ValueError("数据尚未加载，请先调用 load_and_prepare_data()")
            
        stats = {
            'total_samples': len(self.merged_data),
            'unique_users': self.merged_data['user_id'].nunique(),
            'unique_videos': self.merged_data['video_id'].nunique(),
            'train_users': len(self.train_users),
            'val_users': len(self.val_users),
            'click_rate': self.merged_data['is_click'].mean(),
            'avg_play_time': self.merged_data['play_time_ms'].mean(),
            'features_used': {
                'numerical': self.config['feature']['numerical'],
                'categorical': self.config['feature']['categorical']
            }
        }
        
        return stats