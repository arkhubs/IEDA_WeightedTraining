#!/usr/bin/env python3
import sys
import os
sys.path.append('IEDA_WeightedTraining/src')

import yaml
import pandas as pd
import numpy as np

# 加载配置和数据
config_path = 'IEDA_WeightedTraining/configs/experiment_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

from data_manager import DataManager

print("加载数据管理器...")
data_manager = DataManager(config)

print(f"数据集大小: {len(data_manager.master_df)}")
print(f"数据集列: {list(data_manager.master_df.columns)}")

# 检查标签分布
print("\n=== 标签分布 ===")
if 'is_click' in data_manager.master_df.columns:
    click_dist = data_manager.master_df['is_click'].value_counts()
    print(f"is_click分布: {click_dist}")
    
if 'play_time_ms' in data_manager.master_df.columns:
    play_time = data_manager.master_df['play_time_ms']
    print(f"play_time_ms统计: mean={play_time.mean():.2f}, std={play_time.std():.2f}, min={play_time.min()}, max={play_time.max()}")

# 检查特征维度
print("\n=== 特征检查 ===")
sample_user_id = data_manager.master_df['user_id'].iloc[0]
sample_video_ids = data_manager.master_df['video_id'].iloc[:5].tolist()

print(f"样本用户ID: {sample_user_id}")
print(f"样本视频IDs: {sample_video_ids}")

# 生成特征
features = data_manager.create_interaction_features(sample_user_id, sample_video_ids)
print(f"特征矩阵形状: {features.shape}")
print(f"特征范围: min={features.min():.4f}, max={features.max():.4f}")
print(f"特征是否有NaN: {np.isnan(features).any()}")
print(f"特征是否有Inf: {np.isinf(features).any()}")

# 检查用户和视频特征
print(f"\n用户特征数量: {len(data_manager.user_features) if data_manager.user_features is not None else 0}")
print(f"视频特征数量: {len(data_manager.video_features) if data_manager.video_features is not None else 0}")

if data_manager.user_features is not None:
    print(f"用户特征列: {list(data_manager.user_features.columns)}")
if data_manager.video_features is not None:
    print(f"视频特征列: {list(data_manager.video_features.columns)}")
