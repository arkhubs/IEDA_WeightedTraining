import pandas as pd
import torch
from torch.utils.data import Dataset


import os
import numpy as np

class DataManager:
    def __init__(self, config):
        """
        加载所有CSV文件并合并成 master_df。
        预处理特征，按时间排序。
        """
        dataset_name = config.get('dataset_name', 'Pure')
        data_path = config.get('data_path', './data/')
        # 日志文件列表
        log_files = [
            f'log_standard_4_08_to_4_21_{dataset_name.lower()}.csv',
            f'log_standard_4_22_to_5_08_{dataset_name.lower()}.csv',
            f'log_random_4_22_to_5_08_{dataset_name.lower()}.csv'
        ]
        # 加载日志数据
        logs = []
        for fname in log_files:
            fpath = os.path.join(data_path, fname)
            print(f"[DEBUG] Checking log file: {fpath} exists={os.path.exists(fpath)}", flush=True)
            if os.path.exists(fpath):
                logs.append(pd.read_csv(fpath))
            else:
                print(f"[WARNING] Log file not found: {fpath}", flush=True)
        print(f"[DEBUG] logs length: {len(logs)}", flush=True)
        self.master_df = pd.concat(logs, ignore_index=True)
        self.master_df.sort_values('time_ms', inplace=True)
        # 加载用户特征
        user_feat_file = f'user_features_{dataset_name.lower()}.csv'
        user_feat_path = os.path.join(data_path, user_feat_file)
        self.user_features = pd.read_csv(user_feat_path)
        # 加载视频特征（基础+统计）
        video_basic_file = f'video_features_basic_{dataset_name.lower()}.csv'
        video_stat_file = f'video_features_statistic_{dataset_name.lower()}.csv'
        video_basic_path = os.path.join(data_path, video_basic_file)
        video_stat_path = os.path.join(data_path, video_stat_file)
        video_basic = pd.read_csv(video_basic_path)
        video_stat = pd.read_csv(video_stat_path)
        self.video_features = pd.merge(video_basic, video_stat, on='video_id', how='left')
        # 建立索引，便于快速查找
        self.user_feat_dict = self.user_features.set_index('user_id').to_dict(orient='index')
        self.video_feat_dict = self.video_features.set_index('video_id').to_dict(orient='index')
        self.master_df.reset_index(drop=True, inplace=True)

    def get_simulation_stream(self, num_steps):
        """返回一个生成器，模拟 num_steps 次用户-视频交互（按时间流）"""
        for _, interaction in self.master_df.head(num_steps).iterrows():
            yield interaction

    def get_candidate_videos(self, user_id, size):
        """为指定用户随机选择候选视频池（不考虑历史，直接采样）"""
        all_video_ids = self.video_features['video_id'].values
        candidate_ids = np.random.choice(all_video_ids, size=size, replace=False)
        return candidate_ids

    def create_interaction_features(self, user_id, video_ids):
        """
        为用户和一组视频创建交互特征矩阵 X (numpy array)
        1. 用户特征 user_active_degree、视频特征 video_type、tag 做 one-hot
        2. 剔除所有id、区间、日期等无用特征
        3. 只保留数值和哑变量特征
        4. 数值特征缺失用均值填充，分类变量缺失视为 Missing 类别
        """
        # 计算用户和视频数值特征均值（用于填充）
        user_num_keys = [k for k in self.user_features.columns if k not in ['user_id','user_active_degree','follow_user_num_range','fans_user_num_range','friend_user_num_range','register_days_range'] and self.user_features[k].dtype != object]
        user_num_means = self.user_features[user_num_keys].mean()
        video_num_keys = [k for k in self.video_features.columns if k not in ['video_id','author_id','music_id','video_type','upload_dt','upload_type','tag'] and self.video_features[k].dtype != object]
        video_num_means = self.video_features[video_num_keys].mean()

        # 用户特征处理
        user_feat_row = self.user_features[self.user_features['user_id'] == user_id]
        if user_feat_row.empty:
            user_feat_row = self.user_features.sample(1)
        user_feat = user_feat_row.iloc[0].to_dict()
        # 用户 one-hot，缺失视为 Missing
        user_active_degree = user_feat.get('user_active_degree', 'Missing')
        if pd.isna(user_active_degree):
            user_active_degree = 'Missing'
        user_active_degree_oh = pd.get_dummies([user_active_degree], prefix='user_active_degree')
        # 选取用户数值特征，缺失用均值填充
        user_num = []
        for k in user_num_keys:
            v = user_feat.get(k, np.nan)
            if pd.isna(v):
                v = user_num_means[k]
            user_num.append(v)
        # 用户 onehot 区域
        user_onehot_keys = [k for k in user_feat.keys() if k.startswith('onehot_feat')]
        user_onehot = [user_feat[k] if not pd.isna(user_feat[k]) else 0 for k in user_onehot_keys]
        # 合并用户特征
        user_vec = np.concatenate([user_active_degree_oh.values.flatten(), np.array(user_num), np.array(user_onehot)])

        feats = []
        for vid in video_ids:
            video_feat_row = self.video_features[self.video_features['video_id'] == vid]
            if video_feat_row.empty:
                video_feat_row = self.video_features.sample(1)
            video_feat = video_feat_row.iloc[0].to_dict()
            # video_type one-hot，缺失视为 Missing
            video_type = video_feat.get('video_type', 'Missing')
            if pd.isna(video_type):
                video_type = 'Missing'
            video_type_oh = pd.get_dummies([video_type], prefix='video_type')
            # tag one-hot，缺失视为 Missing
            tag = video_feat.get('tag', 'Missing')
            if pd.isna(tag):
                tag = 'Missing'
            tag_oh = pd.get_dummies([str(tag)], prefix='tag')
            # 数值特征，缺失用均值填充
            video_num = []
            for k in video_num_keys:
                v = video_feat.get(k, np.nan)
                if pd.isna(v):
                    v = video_num_means[k]
                video_num.append(v)
            # 合并
            video_vec = np.concatenate([video_type_oh.values.flatten(), tag_oh.values.flatten(), np.array(video_num)])
            feat = np.concatenate([user_vec, video_vec])
            feats.append(feat)
        # 对齐长度（one-hot后不同样本可能长度不同，需补零）
        maxlen = max(len(f) for f in feats)
        feats_pad = np.array([np.pad(f, (0, maxlen-len(f)), 'constant') for f in feats])
        print(f"[调试] 当前特征维度: {feats_pad.shape[1]}")
        # 检查是否还有 NaN
        if np.isnan(feats_pad).any():
            print("[警告] 特征矩阵仍含有 NaN！")
        return torch.tensor(feats_pad, dtype=torch.float)

    def get_ground_truth_label(self, user_id, video_id):
        """查找特定交互的真实标签 Y (is_click, play_time_ms)"""
        row = self.master_df[(self.master_df['user_id'] == user_id) & (self.master_df['video_id'] == video_id)]
        if len(row) == 0:
            # 若无真实交互，返回默认值
            return torch.tensor([0.0, 0.0], dtype=torch.float)
        is_click = float(row.iloc[0]['is_click'])
        play_time_ms = float(row.iloc[0]['play_time_ms'])
        return torch.tensor([is_click, play_time_ms], dtype=torch.float)

class HistoryBuffer(Dataset):
    """动态数据集，存储已观察到的交互历史 (X, Y, Z)"""
    def __init__(self):
        self.buffer = []

    def add(self, x, y, z):
        self.buffer.append((x, y, z))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
