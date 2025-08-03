import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import glob
from pathlib import Path

class DataManager:
    def __init__(self, config):
        """
        加载所有CSV文件并合并成 master_df。
        支持缓存和特征选择优化。
        """
        self.config = config
        dataset_name = config.get('dataset_name', 'Pure')
        data_path = config.get('data_path', './data/')
        cache_dir = config.get('cache_dir', 'KuaiRand/cache/')
        enable_cache = config.get('enable_cache', True)
        
        # 缓存文件路径
        cache_file = os.path.join(cache_dir, f'master_df_{dataset_name.lower()}.pkl')
        
        if enable_cache and os.path.exists(cache_file):
            print(f"[INFO] Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.master_df = cache_data['master_df']
                self.user_features = cache_data['user_features']
                self.video_features = cache_data['video_features']
            print(f"[INFO] Loaded {len(self.master_df)} interactions from cache")
        else:
            print(f"[INFO] Loading data from CSV files...")
            self._load_and_cache_data(dataset_name, data_path, cache_file, enable_cache)
    
    def _load_and_cache_data(self, dataset_name, data_path, cache_file, enable_cache):
        """加载原始数据并缓存"""
        # 加载日志文件
        logs = self._load_log_files(dataset_name, data_path)
        
        if not logs:
            raise RuntimeError("No log files found for dataset. Please check data_path and file names.")
        
        print(f"[INFO] Concatenating {len(logs)} log files...")
        self.master_df = pd.concat(logs, ignore_index=True)
        self.master_df.sort_values('time_ms', inplace=True)
        
        # 加载特征文件
        print(f"[INFO] Loading feature files...")
        self._load_feature_files(dataset_name, data_path)
        
        # 缓存数据
        if enable_cache:
            print(f"[INFO] Caching data to {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data = {
                'master_df': self.master_df,
                'user_features': self.user_features,
                'video_features': self.video_features
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[INFO] Data cached successfully")
    
    def _load_log_files(self, dataset_name, data_path):
        """分块加载日志文件，解决内存问题"""
        logs = []
        log_types = [
            'log_standard_4_08_to_4_21',
            'log_standard_4_22_to_5_08',
            'log_random_4_22_to_5_08'
        ]
        
        chunk_size = self.config.get('chunk_size', 10000)
        
        for log_type in log_types:
            part_pattern = os.path.join(data_path, f'{log_type}_{dataset_name.lower()}_part*.csv')
            print(f"[DEBUG] Pattern: {part_pattern}", flush=True)
            part_files = sorted(glob.glob(part_pattern))
            print(f"[DEBUG] Matched files: {part_files}", flush=True)
            
            if part_files:
                for fpath in part_files:
                    print(f"[DEBUG] Loading log file: {fpath}", flush=True)
                    try:
                        # 分块读取大文件
                        chunks = []
                        for chunk in pd.read_csv(fpath, chunksize=chunk_size):
                            chunks.append(chunk)
                        if chunks:
                            df = pd.concat(chunks, ignore_index=True)
                            logs.append(df)
                    except Exception as e:
                        print(f"[ERROR] Failed to load {fpath}: {e}", flush=True)
            else:
                single_file = os.path.join(data_path, f'{log_type}_{dataset_name.lower()}.csv')
                print(f"[DEBUG] Check single file: {single_file}, exists={os.path.exists(single_file)}", flush=True)
                if os.path.exists(single_file):
                    print(f"[DEBUG] Loading log file: {single_file}", flush=True)
                    try:
                        # 分块读取大文件
                        chunks = []
                        for chunk in pd.read_csv(single_file, chunksize=chunk_size):
                            chunks.append(chunk)
                        if chunks:
                            df = pd.concat(chunks, ignore_index=True)
                            logs.append(df)
                    except Exception as e:
                        print(f"[ERROR] Failed to load {single_file}: {e}", flush=True)
                else:
                    print(f"[WARNING] No log file found for {log_type}", flush=True)
        
        print(f"[DEBUG] logs length: {len(logs)}", flush=True)
        return logs
    
    def _load_feature_files(self, dataset_name, data_path):
        """根据配置选择性加载特征文件"""
        load_features = self.config.get('load_features', ["user", "video_basic", "video_statistic"])
        
        self.user_features = None
        self.video_features = None
        
        # 加载用户特征
        if "user" in load_features:
            user_file = os.path.join(data_path, f'user_features_{dataset_name.lower()}.csv')
            if os.path.exists(user_file):
                print(f"[INFO] Loading user features: {user_file}")
                self.user_features = pd.read_csv(user_file)
            else:
                print(f"[WARNING] User features file not found: {user_file}")
        
        # 加载视频特征
        video_dfs = []
        
        if "video_basic" in load_features:
            video_basic_file = os.path.join(data_path, f'video_features_basic_{dataset_name.lower()}.csv')
            if os.path.exists(video_basic_file):
                print(f"[INFO] Loading video basic features: {video_basic_file}")
                video_dfs.append(pd.read_csv(video_basic_file))
        
        if "video_statistic" in load_features:
            # 视频统计特征可能有分片
            video_stat_pattern = os.path.join(data_path, f'video_features_statistic_{dataset_name.lower()}_part*.csv')
            video_stat_files = sorted(glob.glob(video_stat_pattern))
            
            if video_stat_files:
                for fpath in video_stat_files:
                    print(f"[INFO] Loading video statistic features: {fpath}")
                    video_dfs.append(pd.read_csv(fpath))
            else:
                video_stat_file = os.path.join(data_path, f'video_features_statistic_{dataset_name.lower()}.csv')
                if os.path.exists(video_stat_file):
                    print(f"[INFO] Loading video statistic features: {video_stat_file}")
                    video_dfs.append(pd.read_csv(video_stat_file))
        
        # 合并视频特征
        if video_dfs:
            if len(video_dfs) == 1:
                self.video_features = video_dfs[0]
            else:
                # 按video_id合并所有视频特征
                self.video_features = video_dfs[0]
                for df in video_dfs[1:]:
                    self.video_features = pd.merge(self.video_features, df, on='video_id', how='outer')
        
        # 建立索引，便于快速查找
        if self.user_features is not None:
            self.user_feat_dict = self.user_features.set_index('user_id').to_dict(orient='index')
        else:
            self.user_feat_dict = {}
            
        if self.video_features is not None:
            self.video_feat_dict = self.video_features.set_index('video_id').to_dict(orient='index')
        else:
            self.video_feat_dict = {}
        self.master_df.reset_index(drop=True, inplace=True)

    def get_all_interactions(self):
        """返回所有交互数据，用于预训练时随机采样"""
        # 避免 iterrows() 的 Series._name 问题，改用 to_dict()
        for idx in range(len(self.master_df)):
            interaction = self.master_df.iloc[idx].to_dict()
            yield interaction

    def get_simulation_stream(self, num_steps):
        """返回一个生成器，模拟 num_steps 次用户-视频交互（按时间流）"""
        subset_df = self.master_df.head(num_steps)
        for idx in range(len(subset_df)):
            interaction = subset_df.iloc[idx].to_dict()
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
        5. 对数值特征进行标准化
        """
        # 计算用户和视频数值特征均值和标准差（用于标准化）
        user_num_keys = [k for k in self.user_features.columns if k not in ['user_id','user_active_degree','follow_user_num_range','fans_user_num_range','friend_user_num_range','register_days_range'] and self.user_features[k].dtype != object]
        user_num_means = self.user_features[user_num_keys].mean()
        user_num_stds = self.user_features[user_num_keys].std()
        user_num_stds = user_num_stds.replace(0, 1)  # 防止除0
        
        video_num_keys = [k for k in self.video_features.columns if k not in ['video_id','author_id','music_id','video_type','upload_dt','upload_type','tag'] and self.video_features[k].dtype != object]
        video_num_means = self.video_features[video_num_keys].mean()
        video_num_stds = self.video_features[video_num_keys].std()
        video_num_stds = video_num_stds.replace(0, 1)  # 防止除0

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
        # 选取用户数值特征，缺失用均值填充，然后标准化
        user_num = []
        for k in user_num_keys:
            v = user_feat.get(k, np.nan)
            if pd.isna(v):
                v = user_num_means[k]
            # 标准化
            v_norm = (v - user_num_means[k]) / user_num_stds[k]
            user_num.append(v_norm)
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
            # 数值特征，缺失用均值填充，然后标准化
            video_num = []
            for k in video_num_keys:
                v = video_feat.get(k, np.nan)
                if pd.isna(v):
                    v = video_num_means[k]
                # 标准化
                v_norm = (v - video_num_means[k]) / video_num_stds[k]
                video_num.append(v_norm)
            # 合并
            video_vec = np.concatenate([video_type_oh.values.flatten(), tag_oh.values.flatten(), np.array(video_num)])
            feat = np.concatenate([user_vec, video_vec])
            feats.append(feat)
        # 对齐长度（one-hot后不同样本可能长度不同，需补零）
        maxlen = max(len(f) for f in feats)
        feats_pad = np.array([np.pad(f, (0, maxlen-len(f)), 'constant') for f in feats])
        # print(f"[调试] 当前特征维度: {feats_pad.shape[1]}")
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
