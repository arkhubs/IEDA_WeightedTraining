"""
数据管理模块，负责加载数据、处理数据、生成特征等。
"""
import os
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset


def weight_collate_fn(batch):
    """
    自定义的收集函数，将单个样本合并成一个批次
    
    Args:
        batch: 样本列表，每个样本是一个字典
    
    Returns:
        批次字典
    """
    # 初始化批次字典
    collated_batch = {
        'features': {'user_features': []},
        'treatment': [],
        'label': [],
        'group': [],
        'ctr': [],
        'playtime': []
    }
    
    # 遍历所有样本
    for sample in batch:
        collated_batch['features']['user_features'].append(sample['features']['user_features'])
        collated_batch['treatment'].append(sample['treatment'])
        collated_batch['label'].append(sample['label'])
        collated_batch['group'].append(sample['group'])
        collated_batch['ctr'].append(sample['ctr'])
        collated_batch['playtime'].append(sample['playtime'])
    
    # 将列表转换为张量
    collated_batch['features']['user_features'] = torch.stack(collated_batch['features']['user_features'])
    collated_batch['treatment'] = torch.stack(collated_batch['treatment'])
    collated_batch['label'] = torch.stack(collated_batch['label'])
    collated_batch['group'] = torch.stack(collated_batch['group'])
    collated_batch['ctr'] = torch.stack(collated_batch['ctr'])
    collated_batch['playtime'] = torch.stack(collated_batch['playtime'])
    
    return collated_batch


class WeightDataset(Dataset):
    """权重模型的自定义数据集类"""
    
    def __init__(self, features, treatments, labels):
        """
        初始化数据集
        
        Args:
            features: 特征列表
            treatments: 处理标志列表
            labels: 标签列表
        """
        self.features = features
        self.treatments = treatments
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 构造字典形式的批次，确保features是字典结构
        # 这里将features张量直接包装在features字典中，作为user_features
        feature_tensor = self.features[idx]
        
        # 为了兼容trainers.py中的处理逻辑，确保features是字典格式
        return {
            'features': {'user_features': feature_tensor},
            'treatment': self.treatments[idx],
            'label': self.labels[idx],
            'group': self.treatments[idx],  # group与treatment相同，1为处理组，0为对照组
            'ctr': self.labels[idx][0] if len(self.labels[idx]) >= 1 else torch.tensor(0.0),
            'playtime': self.labels[idx][1] if len(self.labels[idx]) >= 2 else torch.tensor(0.0)
        }


class DataManager:
    """数据管理类，负责数据的加载、预处理和特征生成。"""

    def __init__(self, config):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典，包含数据路径等信息
        """
        self.config = config
        self.data_path = config['data_path']
        self.cache_dir = config['cache_dir']
        
        # 加载数据
        self.master_df = self._load_data()
        
        # 特征维度 - 固定为102维，与src_old保持一致
        self.feature_dim = 102  # 使用原始的102个特征
        
        # 划分训练集和测试集（视频层面）
        self._split_train_test_videos()
        
        # 划分处理组和对照组（用户层面）
        self._split_treatment_control_users()
        
    def _load_data(self) -> pd.DataFrame:
        """
        加载数据，优先从缓存加载，如果缓存不存在则从原始数据加载并创建缓存
        
        Returns:
            包含所有交互数据的DataFrame
        """
        cache_path = os.path.join(self.cache_dir, 'master_df.pkl')
        if os.path.exists(cache_path) and not self.config.get('force_reload_data', False):
            print(f"从缓存加载数据: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"从原始数据加载，将创建缓存: {cache_path}")
        
        # 这里应根据实际数据集结构调整
        dataset_type = self.config.get('dataset_type', '1K')
        if dataset_type == '1K':
            from KuaiRand.load_data_1k import load_data
        elif dataset_type == '27K':
            from KuaiRand.load_data_27k import load_data
        elif dataset_type == 'Pure':
            from KuaiRand.load_data_pure import load_data
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")
        
        master_df = load_data(self.data_path)
        
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # 保存到缓存
        with open(cache_path, 'wb') as f:
            pickle.dump(master_df, f)
        
        return master_df
    
    def _split_train_test_videos(self):
        """
        在视频层面划分训练集和测试集
        按照每个视频的ID划分，确保同一个视频只出现在训练集或测试集中
        """
        unique_videos = self.master_df['video_id'].unique()
        test_size = self.config.get('test_video_ratio', 0.2)
        
        # 划分视频ID
        train_videos, test_videos = train_test_split(
            unique_videos, 
            test_size=test_size, 
            random_state=self.config.get('random_seed', 42)
        )
        
        self.train_videos = set(train_videos)
        self.test_videos = set(test_videos)
        
        # 创建训练集和测试集的掩码
        self.train_mask = self.master_df['video_id'].isin(self.train_videos)
        self.test_mask = self.master_df['video_id'].isin(self.test_videos)
        
        # 分离训练集和测试集
        self.train_df = self.master_df[self.train_mask]
        self.test_df = self.master_df[self.test_mask]
        
        print(f"训练集视频数: {len(self.train_videos)}, 测试集视频数: {len(self.test_videos)}")
        print(f"训练集交互数: {len(self.train_df)}, 测试集交互数: {len(self.test_df)}")
    
    def _split_treatment_control_users(self):
        """
        在用户层面划分处理组和对照组
        处理组用户比例由p_treatment参数决定
        验证集用户比例由p_val参数决定
        """
        unique_users = self.master_df['user_id'].unique()
        p_treatment = self.config.get('p_treatment', 0.5)
        p_val = self.config.get('p_val', 0.2)
        
        # 首先将用户分为训练和验证
        train_users, val_users = train_test_split(
            unique_users,
            test_size=p_val,
            random_state=self.config.get('random_seed', 42)
        )
        
        # 然后将训练用户分为处理组和对照组
        treatment_users, control_users = train_test_split(
            train_users,
            test_size=1 - p_treatment,  # 处理组比例
            random_state=self.config.get('random_seed', 42)
        )
        
        # 保存划分结果
        self.treatment_users = set(treatment_users)
        self.control_users = set(control_users)
        self.val_users = set(val_users)
        
        # 用户组映射
        self.user_group_map = {}
        for user in treatment_users:
            self.user_group_map[user] = 'treatment'
        for user in control_users:
            self.user_group_map[user] = 'control'
        for user in val_users:
            self.user_group_map[user] = 'validation'
        
        print(f"处理组用户数: {len(treatment_users)}, 对照组用户数: {len(control_users)}, 验证集用户数: {len(val_users)}")
    
    def get_initial_dataset(self, size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        获取初始预训练数据集
        
        Args:
            size: 每组样本数量
        
        Returns:
            处理组和对照组的初始数据
        """
        # 从训练集中随机抽样
        treatment_data = self.train_df[self.train_df['user_id'].isin(self.treatment_users)].sample(
            min(size, len(self.train_df[self.train_df['user_id'].isin(self.treatment_users)])), 
            random_state=self.config.get('random_seed', 42)
        ).to_dict('records')
        
        control_data = self.train_df[self.train_df['user_id'].isin(self.control_users)].sample(
            min(size, len(self.train_df[self.train_df['user_id'].isin(self.control_users)])),
            random_state=self.config.get('random_seed', 42)
        ).to_dict('records')
        
        return treatment_data, control_data
    
    def get_user_train_videos(self, user_id: str, size: int) -> List[str]:
        """
        获取用户在训练集中有交互的视频，作为推荐候选
        
        Args:
            user_id: 用户ID
            size: 候选池大小
        
        Returns:
            视频ID列表
        """
        user_videos = self.train_df[self.train_df['user_id'] == user_id]['video_id'].unique()
        if len(user_videos) <= size:
            return user_videos.tolist()
        else:
            # 随机抽取size个视频
            indices = np.random.choice(len(user_videos), size, replace=False)
            return user_videos[indices].tolist()
    
    def get_user_test_videos(self, user_id: str, size: Optional[int] = None) -> List[str]:
        """
        获取用户在测试集中有交互的视频，用于评估
        
        Args:
            user_id: 用户ID
            size: 可选，最大返回数量
        
        Returns:
            视频ID列表
        """
        user_videos = self.test_df[self.test_df['user_id'] == user_id]['video_id'].unique()
        if size is None or len(user_videos) <= size:
            return user_videos.tolist()
        else:
            # 随机抽取size个视频
            indices = np.random.choice(len(user_videos), size, replace=False)
            return user_videos[indices].tolist()
            
    def get_weight_dataloaders(self, batch_size=128, num_workers=4):
        """
        获取权重模型的训练和验证数据加载器
        
        Args:
            batch_size: 批次大小
            num_workers: 数据加载线程数
        
        Returns:
            训练和验证数据加载器的元组
        """
        # 生成训练数据
        train_features = []
        train_treatments = []
        train_labels = []
        
        # 从处理组采样
        n_treatment_samples = 1000  # 可以根据需要调整
        treatment_users = self.get_treatment_users(n_treatment_samples)
        for user_id in treatment_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(1.0)  # 处理组标志
                label = self.get_ground_truth(user_id, video_id)
                
                train_features.append(features)
                train_treatments.append(treatment)
                train_labels.append(label)
        
        # 从对照组采样
        n_control_samples = 1000  # 可以根据需要调整
        control_users = self.get_control_users(n_control_samples)
        for user_id in control_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(0.0)  # 对照组标志
                label = self.get_ground_truth(user_id, video_id)
                
                train_features.append(features)
                train_treatments.append(treatment)
                train_labels.append(label)
        
        # 生成验证数据
        val_features = []
        val_treatments = []
        val_labels = []
        
        # 从验证集采样
        n_val_samples = 500  # 可以根据需要调整
        val_users = self.get_validation_users(n_val_samples)
        for user_id in val_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                # 随机分配处理/对照标志，因为验证集用户可能在任一组
                treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
                label = self.get_ground_truth(user_id, video_id)
                
                val_features.append(features)
                val_treatments.append(treatment)
                val_labels.append(label)
        
        # 创建数据集
        train_dataset = WeightDataset(
            features=train_features,
            treatments=train_treatments,
            labels=train_labels
        )
        
        val_dataset = WeightDataset(
            features=val_features,
            treatments=val_treatments,
            labels=val_labels
        )
        
        # 创建数据加载器
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=weight_collate_fn
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=weight_collate_fn
        )
        
        return train_dataloader, val_dataloader
    
    def get_treatment_users(self, n_users: int) -> List[str]:
        """
        获取处理组用户
        
        Args:
            n_users: 需要的用户数量
        
        Returns:
            用户ID列表
        """
        treatment_users = list(self.treatment_users)
        if len(treatment_users) <= n_users:
            return treatment_users
        else:
            # 随机抽取n_users个用户
            indices = np.random.choice(len(treatment_users), n_users, replace=False)
            return [treatment_users[i] for i in indices]
    
    def get_control_users(self, n_users: int) -> List[str]:
        """
        获取对照组用户
        
        Args:
            n_users: 需要的用户数量
        
        Returns:
            用户ID列表
        """
        control_users = list(self.control_users)
        if len(control_users) <= n_users:
            return control_users
        else:
            # 随机抽取n_users个用户
            indices = np.random.choice(len(control_users), n_users, replace=False)
            return [control_users[i] for i in indices]
    
    def get_validation_users(self, n_users: int) -> List[str]:
        """
        获取验证集用户
        
        Args:
            n_users: 需要的用户数量
        
        Returns:
            用户ID列表
        """
        val_users = list(self.val_users)
        if len(val_users) <= n_users:
            return val_users
        else:
            # 随机抽取n_users个用户
            indices = np.random.choice(len(val_users), n_users, replace=False)
            return [val_users[i] for i in indices]
    
    def create_features(self, user_id: str, video_id: str) -> torch.Tensor:
        """
        为用户和视频创建交互特征
        
        Args:
            user_id: 用户ID
            video_id: 视频ID
        
        Returns:
            特征张量，固定为102维，与src_old保持一致
        """
        try:
            # 查找该用户-视频对的行
            row = self.master_df[(self.master_df['user_id'] == user_id) & 
                                 (self.master_df['video_id'] == video_id)]
            
            if len(row) == 0:
                # 如果找不到匹配的交互，返回随机特征
                return torch.randn(self.feature_dim)
            
            # 创建一个安全的特征向量，避免numpy.object_类型的问题
            # 保证使用102个特征
            features = []
            
            # 尝试从行中提取所有列作为特征
            # 先将DataFrame行转换为字典，保留所有列
            row_dict = row.iloc[0].to_dict()
            
            # 提取所有可能的特征
            for col_name, val in row_dict.items():
                try:
                    # 检查是否可以转换为浮点数
                    if isinstance(val, (int, float)):
                        features.append(float(val))
                    else:
                        # 尝试转换非数值类型
                        try:
                            features.append(float(val))
                        except:
                            # 如果无法转换，使用0.0作为占位符
                            features.append(0.0)
                except:
                    features.append(0.0)
            
            # 确保特征数量为102，截断或填充
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]  # 截断多余特征
            
            # 如果特征不足，用0填充
            while len(features) < self.feature_dim:
                features.append(0.0)
            
            # 转换为张量
            return torch.tensor(features, dtype=torch.float32)
        
        except Exception as e:
            print(f"创建特征时出错: {e}")
            # 发生错误时返回全零特征
            return torch.zeros(self.feature_dim)
    
    def get_prediction_dataloaders(self, batch_size=128, num_workers=4, is_test=False):
        """
        获取预测模型的训练和验证/测试数据加载器
        
        Args:
            batch_size: 批次大小
            num_workers: 数据加载线程数
            is_test: 是否返回测试集（而非验证集）
        
        Returns:
            训练和验证/测试数据加载器的元组
        """
        # 生成训练数据
        train_features = []
        train_treatments = []
        train_labels = []
        
        # 从处理组和对照组采样
        n_train_samples = 2000  # 可以根据需要调整
        train_users = self.get_treatment_users(n_train_samples // 2) + self.get_control_users(n_train_samples // 2)
        for user_id in train_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
                label = self.get_ground_truth(user_id, video_id)
                
                train_features.append(features)
                train_treatments.append(treatment)
                train_labels.append(label)
        
        # 创建训练数据集
        train_dataset = WeightDataset(
            features=train_features,
            treatments=train_treatments,
            labels=train_labels
        )
        
        # 创建训练数据加载器
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=weight_collate_fn
        )
        
        # 选择验证集或测试集
        eval_users = self.get_validation_users(1000) if not is_test else []
        # 如果是测试集，从处理组和对照组选择未在训练中使用的用户
        if is_test:
            test_treatment_users = [uid for uid in list(self.treatment_users) if uid not in train_users][:500]
            test_control_users = [uid for uid in list(self.control_users) if uid not in train_users][:500]
            eval_users = test_treatment_users + test_control_users
        
        # 生成验证/测试数据
        eval_features = []
        eval_treatments = []
        eval_labels = []
        
        for user_id in eval_users:
            # 对每个用户生成几个样本
            videos = self.get_user_test_videos(user_id, 5) if is_test else self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
                label = self.get_ground_truth(user_id, video_id)
                
                eval_features.append(features)
                eval_treatments.append(treatment)
                eval_labels.append(label)
        
        # 创建验证/测试数据集
        eval_dataset = WeightDataset(
            features=eval_features,
            treatments=eval_treatments,
            labels=eval_labels
        )
        
        # 创建验证/测试数据加载器
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=weight_collate_fn
        )
        
        return train_dataloader, eval_dataloader
    
    def get_ground_truth(self, user_id: str, video_id: str) -> torch.Tensor:
        """
        获取用户-视频对的真实标签（CTR和播放时间）
        
        Args:
            user_id: 用户ID
            video_id: 视频ID
        
        Returns:
            包含CTR和播放时间的标签张量
        """
        try:
            # 查找该用户-视频对的行
            row = self.master_df[(self.master_df['user_id'] == user_id) & 
                                 (self.master_df['video_id'] == video_id)]
            
            if len(row) == 0:
                # 如果找不到匹配的交互，返回零向量
                return torch.zeros(2)
            
            # 获取点击和播放时间
            # 假设数据框中有'is_click'和'play_time'列
            ctr = row['is_click'].values[0]
            play_time = row['play_time_ms'].values[0] / 1000.0  # 转换为秒
            
            return torch.tensor([float(ctr), float(play_time)], dtype=torch.float32)
        
        except Exception as e:
            print(f"获取真实标签时出错: {e}")
            # 出错时返回零向量
            return torch.zeros(2)


    def recommend_for_users(self, user_ids: List[str], candidate_pool_size: int, model, treatment_flag: bool) -> Dict[str, str]:
        """
        为一组用户推荐视频，每个用户推荐1个得分最高的视频
        
        Args:
            user_ids: 用户ID列表
            candidate_pool_size: 每个用户的候选视频池大小
            model: 用于推荐的模型（处理组或对照组模型）
            treatment_flag: 是否是处理组
        
        Returns:
            用户ID到推荐视频ID的映射字典
        """
        recommendations = {}
        
        for user_id in user_ids:
            # 获取用户在训练集中有交互的视频作为候选池
            candidate_videos = self.get_user_train_videos(user_id, candidate_pool_size)
            
            if not candidate_videos:
                continue
                
            # 计算所有候选视频的得分
            scores = []
            features_list = []
            
            for video_id in candidate_videos:
                features = self.create_features(user_id, video_id)
                features_list.append(features)
            
            # 将特征转换为批次
            if features_list:
                features_batch = torch.stack(features_list)
                
                # 使用模型预测得分（CTR、播放时间等）
                with torch.no_grad():
                    ctr_scores, playtime_scores = model(features_batch)
                    
                    # 这里可以使用自定义的排序策略，例如CTR和播放时间的加权和
                    # 为简化，这里使用CTR作为得分
                    scores = ctr_scores.cpu().numpy()
                
                # 找到得分最高的视频
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    recommendations[user_id] = candidate_videos[best_idx]
            
        return recommendations
    
    def collect_interactions(self, user_video_pairs: Dict[str, str]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        收集一组用户-视频对的交互数据
        
        Args:
            user_video_pairs: 用户ID到视频ID的映射字典
        
        Returns:
            特征、处理标志和标签的元组
        """
        features = []
        treatments = []
        labels = []
        
        for user_id, video_id in user_video_pairs.items():
            # 生成特征
            feature = self.create_features(user_id, video_id)
            
            # 确定处理标志
            treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
            
            # 获取真实标签
            label = self.get_ground_truth(user_id, video_id)
            
            features.append(feature)
            treatments.append(treatment)
            labels.append(label)
            
        return features, treatments, labels
    
    def get_step_data(self, n_users_t: int, n_users_c: int, candidate_pool_size: int, 
                    treatment_model, control_model) -> Dict[str, Any]:
        """
        获取一个训练步骤的数据，包括推荐、收集交互和评估
        
        Args:
            n_users_t: 处理组用户数量
            n_users_c: 对照组用户数量
            candidate_pool_size: 候选池大小
            treatment_model: 处理组模型
            control_model: 对照组模型
            
        Returns:
            包含训练和评估数据的字典
        """
        # 获取用户
        treatment_users = self.get_treatment_users(n_users_t)
        control_users = self.get_control_users(n_users_c)
        
        # 为处理组用户推荐视频
        t_recommendations = self.recommend_for_users(
            treatment_users, 
            candidate_pool_size, 
            treatment_model, 
            True
        )
        
        # 为对照组用户推荐视频
        c_recommendations = self.recommend_for_users(
            control_users, 
            candidate_pool_size, 
            control_model, 
            False
        )
        
        # 收集交互数据
        t_features, t_treatments, t_labels = self.collect_interactions(t_recommendations)
        c_features, c_treatments, c_labels = self.collect_interactions(c_recommendations)
        
        # 将处理组和对照组数据合并
        all_features = t_features + c_features
        all_treatments = t_treatments + c_treatments
        all_labels = t_labels + c_labels
        
        # 为评估收集测试集数据
        # 这里使用相同的用户，但是从测试集中获取视频
        t_test_recommendations = {}
        for user_id in treatment_users:
            test_videos = self.get_user_test_videos(user_id, 5)
            if test_videos:
                t_test_recommendations[user_id] = test_videos[0]  # 为简化，只选择第一个测试视频
                
        c_test_recommendations = {}
        for user_id in control_users:
            test_videos = self.get_user_test_videos(user_id, 5)
            if test_videos:
                c_test_recommendations[user_id] = test_videos[0]  # 为简化，只选择第一个测试视频
                
        # 收集测试集交互数据
        t_test_features, t_test_treatments, t_test_labels = self.collect_interactions(t_test_recommendations)
        c_test_features, c_test_treatments, c_test_labels = self.collect_interactions(c_test_recommendations)
        
        # 返回所有数据
        return {
            'train': {
                'features': all_features,
                'treatments': all_treatments,
                'labels': all_labels,
                'treatment': {
                    'features': t_features,
                    'treatments': t_treatments,
                    'labels': t_labels,
                    'n_users': len(set(t_recommendations.keys()))  # 实际参与的用户数
                },
                'control': {
                    'features': c_features,
                    'treatments': c_treatments,
                    'labels': c_labels,
                    'n_users': len(set(c_recommendations.keys()))  # 实际参与的用户数
                }
            },
            'test': {
                'treatment': {
                    'features': t_test_features,
                    'treatments': t_test_treatments,
                    'labels': t_test_labels,
                    'n_users': len(set(t_test_recommendations.keys()))  # 实际参与的用户数
                },
                'control': {
                    'features': c_test_features,
                    'treatments': c_test_treatments,
                    'labels': c_test_labels,
                    'n_users': len(set(c_test_recommendations.keys()))  # 实际参与的用户数
                }
            }
        }


class HistoryBuffer:
    """历史数据缓冲区，用于存储训练过程中收集的交互数据"""
    
    def __init__(self):
        self.features = []  # 特征
        self.labels = []    # 标签
        self.treatments = []  # 处理指示器 (1表示处理组，0表示对照组)
        
    def add(self, features: torch.Tensor, label: torch.Tensor, treatment: torch.Tensor):
        """
        添加一条交互记录
        
        Args:
            features: 特征张量
            label: 标签张量
            treatment: 处理指示器
        """
        self.features.append(features)
        self.labels.append(label)
        self.treatments.append(treatment)
    
    def add_batch(self, features_batch: List[torch.Tensor], labels_batch: List[torch.Tensor], treatments_batch: List[torch.Tensor]):
        """
        批量添加交互记录
        
        Args:
            features_batch: 特征张量列表
            labels_batch: 标签张量列表
            treatments_batch: 处理指示器列表
        """
        self.features.extend(features_batch)
        self.labels.extend(labels_batch)
        self.treatments.extend(treatments_batch)
    
    def get_treatment_data(self):
        """获取处理组数据"""
        treatment_indices = [i for i, t in enumerate(self.treatments) if t == 1]
        if not treatment_indices:
            return None, None
        
        X_t = torch.stack([self.features[i] for i in treatment_indices])
        Y_t = torch.stack([self.labels[i] for i in treatment_indices])
        
        return X_t, Y_t
    
    def get_control_data(self):
        """获取对照组数据"""
        control_indices = [i for i, t in enumerate(self.treatments) if t == 0]
        if not control_indices:
            return None, None
        
        X_c = torch.stack([self.features[i] for i in control_indices])
        Y_c = torch.stack([self.labels[i] for i in control_indices])
        
        return X_c, Y_c
    
    def get_all_data(self):
        """获取所有数据"""
        if not self.features:
            return None, None, None
        
        X = torch.stack(self.features)
        Y = torch.stack(self.labels)
        T = torch.stack(self.treatments)
        
        return X, Y, T
    
    def __len__(self):
        return len(self.features)
