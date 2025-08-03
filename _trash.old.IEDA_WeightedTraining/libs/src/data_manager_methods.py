import numpy as np
import logging

class DataManagerMethods:
    """数据管理方法类，为仿真提供数据处理方法"""
    
    def __init__(self, data_manager):
        """
        初始化数据管理方法
        
        Args:
            data_manager: 数据管理器实例
        """
        self.data_manager = data_manager
        self.logger = logging.getLogger('DataManagerMethods')
    
    def sample_users(self, batch_size, mode='train'):
        """
        随机抽样用户
        
        Args:
            batch_size: 批次大小
            mode: 'train' 或 'val'
            
        Returns:
            用户ID列表
        """
        user_list = self.data_manager.get_train_users() if mode == 'train' else self.data_manager.get_val_users()
        if len(user_list) <= batch_size:
            return user_list
        
        return np.random.choice(user_list, batch_size, replace=False).tolist()
    
    def generate_candidates(self, user_ids, n_candidate):
        """
        为一批用户生成候选视频
        
        Args:
            user_ids: 用户ID列表
            n_candidate: 每个用户的候选视频数量
            
        Returns:
            候选视频字典，格式为 {user_id: [video_id, ...]}
        """
        candidates = {}
        for user_id in user_ids:
            user_candidates = self.data_manager.get_candidate_videos(user_id, n_candidate)
            if user_candidates:
                candidates[user_id] = user_candidates
        
        return candidates
    
    def prepare_features_for_candidates(self, user_candidates):
        """
        为候选视频准备特征
        
        Args:
            user_candidates: 用户候选视频字典
            
        Returns:
            特征字典
        """
        all_user_ids = []
        all_video_ids = []
        user_video_indices = {}  # 记录每个用户的候选视频在全局列表中的索引
        
        idx = 0
        for user_id, candidates in user_candidates.items():
            user_video_indices[user_id] = []
            
            for candidate in candidates:
                all_user_ids.append(user_id)
                all_video_ids.append(candidate['video_id'])
                user_video_indices[user_id].append(idx)
                idx += 1
        
        if not all_user_ids:
            return None, user_video_indices
        
        # 准备特征
        features = self.data_manager.prepare_batch_features(all_user_ids, all_video_ids)
        
        return features, user_video_indices
    
    def get_true_labels(self, user_ids, video_ids, label_name=None):
        """
        获取真实标签
        
        Args:
            user_ids: 用户ID列表
            video_ids: 视频ID列表
            label_name: 标签名称，如果为None则尝试获取第一个可用标签
            
        Returns:
            标签数组
        """
        labels = []
        for i in range(len(user_ids)):
            label = self.get_video_label(user_ids[i], video_ids[i], label_name)
            labels.append(label if label is not None else 0.0)
        
        return np.array(labels, dtype=np.float32)
    
    def get_true_labels_dict(self, user_ids, video_ids):
        """
        获取所有真实标签的字典
        
        Args:
            user_ids: 用户ID列表
            video_ids: 视频ID列表
            
        Returns:
            标签字典，格式为 {label_name: [labels]}
        """
        # 初始化结果字典
        result = {}
        
        for i in range(len(user_ids)):
            # 获取该用户-视频对的所有标签
            video_labels = self.get_video_labels(user_ids[i], video_ids[i])
            
            # 添加到结果字典中
            for label_name, label_value in video_labels.items():
                if label_name not in result:
                    result[label_name] = [0.0] * i  # 为之前的条目填充默认值
                
                # 确保所有标签列表长度一致
                while len(result[label_name]) < i:
                    result[label_name].append(0.0)
                
                result[label_name].append(label_value)
        
        # 确保所有标签列表长度一致
        max_len = len(user_ids)
        for label_name in result:
            while len(result[label_name]) < max_len:
                result[label_name].append(0.0)
            
            # 转换为numpy数组
            result[label_name] = np.array(result[label_name], dtype=np.float32)
        
        return result
        
    def get_video_label(self, user_id, video_id, label_name=None):
        """
        获取视频的真实标签
        
        Args:
            user_id: 用户ID
            video_id: 视频ID
            label_name: 标签名称，如果为None则返回第一个标签
            
        Returns:
            标签值
        """
        videos = self.data_manager.get_user_videos(user_id)
        for video in videos:
            if video['video_id'] == video_id:
                if 'labels' in video:
                    if label_name and label_name in video['labels']:
                        return video['labels'][label_name]
                    elif video['labels']:
                        # 返回第一个可用标签
                        return next(iter(video['labels'].values()))
                elif 'label' in video:
                    # 兼容旧格式
                    return video['label']
        return None
    
    def get_video_labels(self, user_id, video_id):
        """
        获取视频的所有真实标签
        
        Args:
            user_id: 用户ID
            video_id: 视频ID
            
        Returns:
            标签字典
        """
        videos = self.data_manager.get_user_videos(user_id)
        for video in videos:
            if video['video_id'] == video_id:
                if 'labels' in video:
                    return video['labels']
                elif 'label' in video:
                    # 兼容旧格式，将单一标签转换为字典格式
                    label_name = self.data_manager.get_default_label_name()
                    return {label_name: video['label']}
        return {}
    
    def mark_videos_used(self, user_video_pairs):
        """
        标记视频为已使用
        
        Args:
            user_video_pairs: 用户-视频对列表
            
        Returns:
            成功标记的数量
        """
        count = 0
        for user_id, video_id in user_video_pairs:
            if self.data_manager.mark_video_used(user_id, video_id):
                count += 1
        
        return count

    def get_user_candidates(self, user_id, n_candidate=None):
        """
        为单个用户生成候选视频
        
        Args:
            user_id: 用户ID
            n_candidate: 候选视频数量，如果为None则使用默认值
            
        Returns:
            候选视频列表
        """
        if n_candidate is None:
            n_candidate = 10  # 默认候选数量
        
        return self.data_manager.get_candidate_videos(user_id, n_candidate)
