import os
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader

class UserVideoDataset(Dataset):
    """用户-视频交互数据集类"""
    
    def __init__(self, users, videos, interactions, features, label_name, user_mask=None):
        """
        初始化数据集
        
        Args:
            users (pd.DataFrame): 用户特征数据
            videos (pd.DataFrame): 视频特征数据
            interactions (pd.DataFrame): 用户-视频交互数据
            features (dict): 特征配置
            label_name (str): 标签名称
            user_mask (dict): 用户掩码字典，标记验证集用户
        """
        self.users = users
        self.videos = videos
        self.interactions = interactions
        self.numerical_features = features['numerical']
        self.categorical_features = features['categorical']
        self.label_name = label_name
        self.user_mask = user_mask or {}

    def get_default_label_name(self):
        """
        获取默认标签名称
        
        Returns:
            默认标签名称
        """
        if hasattr(self, 'labels_config') and self.labels_config and len(self.labels_config) > 0:
            return self.labels_config[0]['name']
        return 'label'

    def get_video_label(self, user_id, video_id):
        """
        获取视频的真实标签（已弃用，仅为兼容性保留）
        请使用 DataManagerMethods.get_video_label() 方法代替
        """
        videos = self.user_video_map.get(user_id, [])
        for video in videos:
            if video['video_id'] == video_id:
                # 优先使用新格式
                if 'labels' in video and video['labels']:
                    # 返回第一个标签的值
                    return next(iter(video['labels'].values()))
                # 兼容旧格式
                return video.get('label', None)
        return None

    def get_label_names(self):
        """
        获取所有标签名称
        
        Returns:
            标签名称列表
        """
        return [label_config['name'] for label_config in self.labels_config]
    
    def __len__(self):
        """获取数据集长度"""
        return len(self.interactions)
    
    def __getitem__(self, idx):
        """获取单个数据项"""
        interaction = self.interactions.iloc[idx]
        user_id = interaction['user_id']
        video_id = interaction['video_id']
        
        # 获取用户和视频特征
        user_data = self.users[self.users['user_id'] == user_id].iloc[0]
        video_data = self.videos[self.videos['video_id'] == video_id].iloc[0]
        
        # 提取数值特征
        numerical_features = []
        for feat in self.numerical_features:
            if feat.startswith('user_'):
                numerical_features.append(user_data.get(feat, 0))
            elif feat.startswith('video_'):
                numerical_features.append(video_data.get(feat, 0))
            else:
                numerical_features.append(interaction.get(feat, 0))
                
        # 提取分类特征
        categorical_features = []
        for feat in self.categorical_features:
            if feat.startswith('user_'):
                categorical_features.append(user_data.get(feat, 'unknown'))
            elif feat.startswith('video_'):
                categorical_features.append(video_data.get(feat, 'unknown'))
            else:
                categorical_features.append(interaction.get(feat, 'unknown'))
        
        # 获取标签
        label = interaction[self.label_name]
        
        # 检查用户是否在验证集中
        mask = 1 if user_id in self.user_mask else 0
        used = 0  # 初始化为未使用状态
        
        return {
            'user_id': user_id,
            'video_id': video_id,
            'numerical_features': np.array(numerical_features, dtype=np.float32),
            'categorical_features': np.array(categorical_features, dtype=object),
            'label': label,
            'mask': mask,
            'used': used
        }


class DataManager:
    """数据管理类，负责数据加载、预处理和管理"""
    
    def __init__(self, config_manager):
        """
        初始化数据管理器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager.get_config()
        self.dataset_config = config_manager.get_dataset_config()
        self.feature_config = config_manager.get_features()
        self.labels_config = config_manager.get_label_info()
        self.exp_dir = config_manager.get_exp_dir()
        
        self.logger = logging.getLogger('DataManager')
        self.cache_path = os.path.join(self.dataset_config['cache_path'])
        os.makedirs(self.cache_path, exist_ok=True)
        
        self.logger.info("[Init] 初始化数据管理器")
        
        # 初始化数据加载和预处理
        self.logger.info("[Init] 开始加载数据集")
        self.users, self.videos, self.interactions, self.video_stats = self._load_data()
        
        # 初始化特征处理
        self.logger.info("[Init] 初始化特征处理")
        self._init_feature_processors()
        
        # 生成用户-视频交互映射
        self.logger.info("[Init] 创建用户-视频交互映射")
        self.user_video_map = self._create_user_video_map()
        
        # 数据集分割
        self.logger.info("[Init] 分割数据集")
        self.user_train, self.user_val = self._split_users()
        
        self._log_data_stats()
    
    def _load_data(self):
        """加载数据集"""
        # 统一路径拼接，所有路径基于 base_dir
        base_dir = self.config.get('base_dir', os.getcwd())
        dataset_path = os.path.join(base_dir, self.dataset_config['path'])
        self.logger.info(f"[Load] 加载数据集: {self.dataset_config['name']}")
        
        try:
            # 检查是否存在预处理缓存
            cache_file = os.path.join(base_dir, self.cache_path, f"{self.dataset_config['name']}_data.pkl")
            if os.path.exists(cache_file):
                self.logger.info(f"[Load] 从缓存加载预处理数据: {cache_file}")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if len(data) == 4:  # 检查是否包含video_stats
                        users, videos, interactions, video_stats = data
                    else:
                        users, videos, interactions = data
                        video_stats = None
                        
                if video_stats is None:
                    self.logger.warning(f"[Load] 缓存中没有视频统计特征，尝试加载")
                    try:
                        video_stats_path = os.path.join(dataset_path, "video_features_statistic_pure.csv")
                        video_stats = pd.read_csv(video_stats_path)
                        # 缓存更新的数据
                        with open(cache_file, 'wb') as f:
                            pickle.dump((users, videos, interactions, video_stats), f)
                    except Exception as e:
                        self.logger.warning(f"[Load] 加载视频统计特征失败: {str(e)}")
                        video_stats = None
                
                return users, videos, interactions, video_stats
            
            # 加载原始数据
            self.logger.info(f"[Load] 加载原始数据文件")
            
            # 自动匹配 Pure/data 目录下的相关 CSV 文件
            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                raise RuntimeError(f"数据目录不存在: {data_dir}")

            def find_csv(pattern):
                for fname in os.listdir(data_dir):
                    if fname.endswith('.csv') and pattern in fname:
                        return os.path.join(data_dir, fname)
                return None

            user_features_path = find_csv('user_features')
            video_basic_path = find_csv('video_features_basic')
            video_stats_path = find_csv('video_features_statistic')
            # 匹配并合并所有 log_*.csv 作为交互数据
            import re
            log_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if re.match(r'log_.*\.csv', fname)]
            if not log_files:
                raise RuntimeError(f"未找到交互日志文件: log_*.csv in {data_dir}")

            users = pd.read_csv(user_features_path)
            videos = pd.read_csv(video_basic_path)
            video_stats = pd.read_csv(video_stats_path) if video_stats_path and os.path.exists(video_stats_path) else None
            interactions_list = [pd.read_csv(f) for f in log_files]
            interactions = pd.concat(interactions_list, ignore_index=True)
            interactions = interactions.drop_duplicates(subset=['user_id', 'video_id'], keep='first')
            
            # 重命名列名，使之符合代码中使用的命名规范
            if 'fans_user_num' in users.columns and 'user_fans_count' not in users.columns:
                users = users.rename(columns={'fans_user_num': 'user_fans_count'})
            
            if 'follow_user_num' in users.columns and 'user_follow_count' not in users.columns:
                users = users.rename(columns={'follow_user_num': 'user_follow_count'})
                
            if 'friend_user_num' in users.columns and 'user_friend_count' not in users.columns:
                users = users.rename(columns={'friend_user_num': 'user_friend_count'})
                
            # 重命名视频特征
            if video_stats is not None:
                rename_dict = {
                    'like_cnt': 'video_like_count',
                    'comment_cnt': 'video_comment_count',
                    'share_cnt': 'video_forward_count'
                }
                for old_name, new_name in rename_dict.items():
                    if old_name in video_stats.columns and new_name not in video_stats.columns:
                        video_stats = video_stats.rename(columns={old_name: new_name})
            
            # 合并视频基础特征和统计特征
            if video_stats is not None:
                videos = pd.merge(videos, video_stats, on='video_id', how='left')
            
            # 进行基本数据预处理
            self.logger.info(f"[Load] 处理数据缺失值")
            self._handle_missing_values(users, videos, interactions)
            
            # 缓存预处理后的数据
            self.logger.info(f"[Load] 缓存预处理数据")
            with open(cache_file, 'wb') as f:
                pickle.dump((users, videos, interactions, video_stats), f)
            
            self.logger.info(f"[Load] 数据加载完成")
            return users, videos, interactions, video_stats
        
        except Exception as e:
            self.logger.error(f"[Load] 加载数据失败: {str(e)}")
            raise RuntimeError(f"数据加载失败: {str(e)}")
    
    def _handle_missing_values(self, users, videos, interactions):
        """处理缺失值"""
        self.logger.info(f"[Process] 处理数据缺失值")
        
        # 数值特征用0填充
        for feat in self.feature_config['numerical']:
            if feat.startswith('user_') and feat in users.columns:
                missing_count = users[feat].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 用户特征 {feat} 有 {missing_count} 个缺失值，用0填充")
                users[feat] = users[feat].fillna(0)
            elif feat.startswith('video_') and feat in videos.columns:
                missing_count = videos[feat].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 视频特征 {feat} 有 {missing_count} 个缺失值，用0填充")
                videos[feat] = videos[feat].fillna(0)
            elif feat in interactions.columns:
                missing_count = interactions[feat].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 交互特征 {feat} 有 {missing_count} 个缺失值，用0填充")
                interactions[feat] = interactions[feat].fillna(0)
        
        # 分类特征用"unknown"填充
        for feat in self.feature_config['categorical']:
            if feat.startswith('user_') and feat in users.columns:
                missing_count = users[feat].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 用户分类特征 {feat} 有 {missing_count} 个缺失值，用'unknown'填充")
                users[feat] = users[feat].fillna('unknown')
            elif feat.startswith('video_') and feat in videos.columns:
                missing_count = videos[feat].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 视频分类特征 {feat} 有 {missing_count} 个缺失值，用'unknown'填充")
                videos[feat] = videos[feat].fillna('unknown')
            elif feat in interactions.columns:
                missing_count = interactions[feat].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 交互分类特征 {feat} 有 {missing_count} 个缺失值，用'unknown'填充")
                interactions[feat] = interactions[feat].fillna('unknown')
                
        # 确保所有标签数据都有值
        for label_config in self.labels_config:
            target = label_config['target']
            if target in interactions.columns:
                missing_count = interactions[target].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"[Process] 标签 {target} 有 {missing_count} 个缺失值，用0填充")
                    interactions[target] = interactions[target].fillna(0)
    
    def _init_feature_processors(self):
        """初始化特征处理器"""
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        
        # 处理数值特征 - 基于样本数据初始化标准化器
        sample_features = []
        sample_size = min(1000, len(self.interactions))  # 使用样本初始化
        sample_interactions = self.interactions.sample(n=sample_size, random_state=42)
        
        for _, interaction in sample_interactions.iterrows():
            user_id = interaction['user_id']
            video_id = interaction['video_id']
            
            user_data = self.get_user_features(user_id)
            video_data = self.get_video_features(video_id)
            
            if user_data is not None and video_data is not None:
                # 提取数值特征（和prepare_batch_features中的逻辑一致）
                num_features = []
                for feat in self.feature_config['numerical']:
                    if feat.startswith('user_'):
                        num_features.append(user_data.get(feat, 0))
                    elif feat.startswith('video_'):
                        num_features.append(video_data.get(feat, 0))
                    elif feat in video_data:
                        num_features.append(video_data.get(feat, 0))
                    elif feat in user_data:
                        num_features.append(user_data.get(feat, 0))
                    else:
                        num_features.append(0)
                
                sample_features.append(num_features)
        
        if sample_features:
            sample_features = np.array(sample_features, dtype=np.float32)
            self.numerical_scaler.fit(sample_features)
        
        # 处理分类特征
        for feat in self.feature_config['categorical']:
            encoder = LabelEncoder()
            if feat.startswith('user_') and feat in self.users.columns:
                encoder.fit(self.users[feat].astype(str))
            elif feat.startswith('video_') and feat in self.videos.columns:
                encoder.fit(self.videos[feat].astype(str))
            elif feat in self.videos.columns:
                # 处理没有前缀但在videos表中的特征，如tag
                encoder.fit(self.videos[feat].astype(str))
            elif feat in self.interactions.columns:
                # 处理在interactions表中的特征
                encoder.fit(self.interactions[feat].astype(str))
            else:
                # 如果特征不存在，创建一个默认编码器
                encoder.fit(['unknown'])
            self.categorical_encoders[feat] = encoder
    
    def _create_user_video_map(self):
        """创建用户-视频交互映射"""
        self.logger.info("[Map] 创建用户-视频交互映射")
        
        # 检查是否存在缓存
        cache_file = os.path.join(self.cache_path, f"{self.dataset_config['name']}_user_video_map.pkl")
        if os.path.exists(cache_file):
            self.logger.info(f"[Map] 从缓存加载用户-视频映射: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 创建映射
        user_video_map = {}
        for _, interaction in self.interactions.iterrows():
            user_id = interaction['user_id']
            video_id = interaction['video_id']
            
            if user_id not in user_video_map:
                user_video_map[user_id] = []
            
            # 收集所有标签的值
            labels = {}
            for label_config in self.labels_config:
                target = label_config['target']
                if target in interaction:
                    labels[label_config['name']] = interaction[target]
            
            user_video_map[user_id].append({
                'video_id': video_id,
                'labels': labels,  # 存储所有标签
                'mask': 0,         # 初始化为未掩码
                'used': 0          # 初始化为未使用
            })
        
        # 缓存映射
        with open(cache_file, 'wb') as f:
            pickle.dump(user_video_map, f)
        
        # 打印统计信息
        total_users = len(user_video_map)
        total_interactions = sum(len(videos) for videos in user_video_map.values())
        avg_videos_per_user = total_interactions / total_users if total_users > 0 else 0
        
        self.logger.info(f"[Map] 用户总数: {total_users}")
        self.logger.info(f"[Map] 交互总数: {total_interactions}")
        self.logger.info(f"[Map] 平均每用户交互视频数: {avg_videos_per_user:.2f}")
        
        return user_video_map
    
    def _split_users(self):
        """分割用户为训练集和验证集"""
        self.logger.info("分割用户为训练集和验证集")
        
        # 获取所有用户ID
        user_ids = list(self.user_video_map.keys())
        np.random.shuffle(user_ids)
        
        # 根据配置的比例分割
        val_ratio = self.config['global']['user_p_val']
        split_idx = int(len(user_ids) * (1 - val_ratio))
        
        user_train = user_ids[:split_idx]
        user_val = user_ids[split_idx:]
        
        # 设置验证集用户的视频掩码
        user_mask = {}
        for user_id in user_val:
            user_mask[user_id] = True
            for video_data in self.user_video_map[user_id]:
                video_data['mask'] = 1
        
        self.logger.info(f"训练集用户数: {len(user_train)}")
        self.logger.info(f"验证集用户数: {len(user_val)}")
        
        return user_train, user_val
    
    def _log_data_stats(self):
        """记录数据统计信息"""
        self.logger.info(f"[Stats] ========= 数据集统计 =========")
        self.logger.info(f"[Stats] 用户总数: {len(self.users)}")
        self.logger.info(f"[Stats] 视频总数: {len(self.videos)}")
        self.logger.info(f"[Stats] 交互总数: {len(self.interactions)}")
        
        # 记录数值特征
        self.logger.info(f"[Stats] --- 数值特征: {len(self.feature_config['numerical'])} ---")
        for feat in self.feature_config['numerical']:
            if feat.startswith('user_') and feat in self.users.columns:
                data = self.users[feat]
                self.logger.info(f"[Stats] 用户特征 {feat}: 均值={data.mean():.2f}, 标准差={data.std():.2f}, 最小值={data.min():.2f}, 最大值={data.max():.2f}")
            elif feat.startswith('video_') and feat in self.videos.columns:
                data = self.videos[feat]
                self.logger.info(f"[Stats] 视频特征 {feat}: 均值={data.mean():.2f}, 标准差={data.std():.2f}, 最小值={data.min():.2f}, 最大值={data.max():.2f}")
            elif feat in self.interactions.columns:
                data = self.interactions[feat]
                self.logger.info(f"[Stats] 交互特征 {feat}: 均值={data.mean():.2f}, 标准差={data.std():.2f}, 最小值={data.min():.2f}, 最大值={data.max():.2f}")
                
        # 记录分类特征
        self.logger.info(f"[Stats] --- 分类特征: {len(self.feature_config['categorical'])} ---")
        for feat in self.feature_config['categorical']:
            if feat.startswith('user_') and feat in self.users.columns:
                n_unique = self.users[feat].nunique()
                self.logger.info(f"[Stats] 用户分类特征 {feat}: 唯一值数量={n_unique}")
            elif feat.startswith('video_') and feat in self.videos.columns:
                n_unique = self.videos[feat].nunique()
                self.logger.info(f"[Stats] 视频分类特征 {feat}: 唯一值数量={n_unique}")
            elif feat in self.videos.columns:
                # 处理没有前缀但在videos表中的特征，如tag
                n_unique = self.videos[feat].nunique()
                self.logger.info(f"[Stats] 视频分类特征 {feat}: 唯一值数量={n_unique}")
            elif feat in self.interactions.columns:
                n_unique = self.interactions[feat].nunique()
                self.logger.info(f"[Stats] 交互分类特征 {feat}: 唯一值数量={n_unique}")
            else:
                self.logger.info(f"[Stats] 分类特征 {feat}: 特征不存在或未处理")
        
        # 记录标签信息
        self.logger.info(f"[Stats] --- 标签: {len(self.labels_config)} ---")
        for label_config in self.labels_config:
            target = label_config['target']
            if target in self.interactions.columns:
                data = self.interactions[target]
                if label_config['type'] == 'binary':
                    pos_rate = data.mean() * 100
                    self.logger.info(f"[Stats] 标签 {label_config['name']} ({target}): 类型={label_config['type']}, 正例率={pos_rate:.2f}%")
                else:
                    self.logger.info(f"[Stats] 标签 {label_config['name']} ({target}): 类型={label_config['type']}, " +
                                   f"均值={data.mean():.2f}, 标准差={data.std():.2f}, 最小值={data.min():.2f}, 最大值={data.max():.2f}")
        
        self.logger.info(f"[Stats] ===============================")
    
    def get_label_names(self):
        """
        获取所有标签名称
        
        Returns:
            标签名称列表
        """
        return [label_config['name'] for label_config in self.labels_config]
    
    def get_train_users(self):
        """获取训练集用户"""
        return self.user_train
    
    def get_val_users(self):
        """获取验证集用户"""
        return self.user_val
    
    def get_user_videos(self, user_id):
        """获取用户交互过的视频列表"""
        return self.user_video_map.get(user_id, [])
    
    def get_candidate_videos(self, user_id, n_candidate=None):
        """
        获取用户的候选视频
        
        Args:
            user_id: 用户ID
            n_candidate: 候选视频数量，默认为None表示获取所有
            
        Returns:
            候选视频列表
        """
        videos = self.user_video_map.get(user_id, [])
        
        # 筛选未掩码且未使用的视频
        available_videos = [v for v in videos if v['mask'] == 0 and v['used'] == 0]
        
        if not available_videos:
            return []
        
        # 随机选择指定数量的候选视频
        if n_candidate is not None and n_candidate < len(available_videos):
            return np.random.choice(available_videos, n_candidate, replace=False).tolist()
        else:
            return available_videos
    
    def mark_video_used(self, user_id, video_id):
        """标记视频为已使用"""
        videos = self.user_video_map.get(user_id, [])
        for video in videos:
            if video['video_id'] == video_id:
                video['used'] = 1
                return True
        return False
    
    def get_feature_info(self):
        """获取特征信息"""
        return {
            'numerical': self.feature_config['numerical'],
            'categorical': self.feature_config['categorical'],
            'numerical_scaler': self.numerical_scaler,
            'categorical_encoders': self.categorical_encoders
        }
    
    def get_video_features(self, video_id):
        """获取视频特征"""
        video_data = self.videos[self.videos['video_id'] == video_id]
        if video_data.empty:
            return None
        return video_data.iloc[0]
    
    def get_user_features(self, user_id):
        """获取用户特征"""
        user_data = self.users[self.users['user_id'] == user_id]
        if user_data.empty:
            return None
        return user_data.iloc[0]

    def prepare_batch_features(self, user_ids, video_ids):
        """
        为一批用户-视频对准备特征
        
        Args:
            user_ids: 用户ID列表
            video_ids: 视频ID列表
            
        Returns:
            特征字典
        """
        batch_size = len(user_ids)
        numerical_features = []
        categorical_features = []
        
        for i in range(batch_size):
            user_id = user_ids[i]
            video_id = video_ids[i]
            
            user_data = self.get_user_features(user_id)
            video_data = self.get_video_features(video_id)
            
            if user_data is None or video_data is None:
                continue
            
            # 提取数值特征
            num_features = []
            for feat in self.feature_config['numerical']:
                if feat.startswith('user_'):
                    num_features.append(user_data.get(feat, 0))
                elif feat.startswith('video_'):
                    num_features.append(video_data.get(feat, 0))
                elif feat in video_data:
                    # 处理没有前缀但在视频数据中的特征
                    num_features.append(video_data.get(feat, 0))
                elif feat in user_data:
                    # 处理没有前缀但在用户数据中的特征
                    num_features.append(user_data.get(feat, 0))
                else:
                    # 特征不存在，使用默认值
                    num_features.append(0)
            
            # 提取分类特征
            cat_features = []
            for feat in self.feature_config['categorical']:
                if feat.startswith('user_'):
                    value = str(user_data.get(feat, 'unknown'))
                    encoder = self.categorical_encoders[feat]
                    try:
                        cat_features.append(encoder.transform([value])[0])
                    except:
                        cat_features.append(0)  # 未知类别用0表示
                        
                elif feat.startswith('video_'):
                    value = str(video_data.get(feat, 'unknown'))
                    encoder = self.categorical_encoders[feat]
                    try:
                        cat_features.append(encoder.transform([value])[0])
                    except:
                        cat_features.append(0)  # 未知类别用0表示
                        
                elif feat in video_data:
                    # 处理没有前缀但在视频数据中的特征，如tag
                    value = str(video_data.get(feat, 'unknown'))
                    encoder = self.categorical_encoders[feat]
                    try:
                        cat_features.append(encoder.transform([value])[0])
                    except:
                        cat_features.append(0)  # 未知类别用0表示
                        
                elif feat in user_data:
                    # 处理没有前缀但在用户数据中的特征
                    value = str(user_data.get(feat, 'unknown'))
                    encoder = self.categorical_encoders[feat]
                    try:
                        cat_features.append(encoder.transform([value])[0])
                    except:
                        cat_features.append(0)  # 未知类别用0表示
                else:
                    # 特征不存在，使用默认值
                    cat_features.append(0)
            
            numerical_features.append(np.array(num_features, dtype=np.float32))
            categorical_features.append(np.array(cat_features, dtype=np.int64))
        
        # 转换为批次数据
        if numerical_features:
            numerical_features = np.stack(numerical_features)
            # 标准化数值特征
            numerical_features = self.numerical_scaler.transform(numerical_features)
        else:
            numerical_features = np.zeros((batch_size, len(self.feature_config['numerical'])), dtype=np.float32)
        
        if categorical_features:
            categorical_features = np.stack(categorical_features)
        else:
            categorical_features = np.zeros((batch_size, len(self.feature_config['categorical'])), dtype=np.int64)
        
        return {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }
    
    def get_video_label(self, user_id, video_id):
        """获取视频的真实标签"""
        videos = self.user_video_map.get(user_id, [])
        for video in videos:
            if video['video_id'] == video_id:
                return video['label']
        return None
    
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        return {
            'num_users': len(self.users),
            'num_videos': len(self.videos),
            'num_interactions': len(self.interactions),
            'numerical_features': self.feature_config['numerical'],
            'categorical_features': self.feature_config['categorical'],
            'label': self.label_config['target']
        }
