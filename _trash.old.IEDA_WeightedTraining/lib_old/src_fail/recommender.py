"""
推荐器模块，负责根据模型进行推荐决策
"""
import torch
import numpy as np
from src.models import PredictionModel
from typing import List, Dict, Any, Tuple, Optional


class Recommender:
    """推荐器，基于模型进行推荐决策"""
    
    def __init__(self, model: PredictionModel, alpha: float = 0.5):
        """
        初始化推荐器
        
        Args:
            model: 预测模型
            alpha: CTR和播放时间的权重参数，alpha越大越偏好CTR
        """
        self.model = model
        self.alpha = alpha
    
    def recommend(self, user_id: str, video_ids: List[str], data_manager, top_k: int = 1) -> List[str]:
        """
        为用户推荐视频
        
        Args:
            user_id: 用户ID
            video_ids: 候选视频ID列表
            data_manager: 数据管理器，用于生成特征
            top_k: 返回的推荐视频数量
        
        Returns:
            推荐视频ID列表
        """
        if not video_ids:
            return []
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 生成所有候选项的特征
        features = []
        for video_id in video_ids:
            feature = data_manager.create_features(user_id, video_id)
            features.append(feature)
        
        # 批量预测
        with torch.no_grad():
            features_batch = torch.stack(features)
            ctr_preds, playtime_preds = self.model(features_batch)
            
            # 归一化预测值
            ctr_norm = torch.sigmoid(ctr_preds)
            playtime_norm = F.normalize(playtime_preds.unsqueeze(0), p=1).squeeze(0)
            
            # 加权组合得分
            scores = self.alpha * ctr_norm + (1 - self.alpha) * playtime_norm
            
            # 获取Top-K推荐
            _, indices = torch.topk(scores, min(top_k, len(video_ids)))
            
        # 返回推荐的视频ID
        return [video_ids[idx] for idx in indices.cpu().numpy()]
    
    def get_scores(self, user_id: str, video_ids: List[str], data_manager) -> Dict[str, Dict[str, float]]:
        """
        获取用户对视频的预测得分详情
        
        Args:
            user_id: 用户ID
            video_ids: 候选视频ID列表
            data_manager: 数据管理器，用于生成特征
        
        Returns:
            视频得分字典，包含CTR、播放时间和总分
        """
        if not video_ids:
            return {}
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 生成所有候选项的特征
        features = []
        for video_id in video_ids:
            feature = data_manager.create_features(user_id, video_id)
            features.append(feature)
        
        # 批量预测
        scores_dict = {}
        with torch.no_grad():
            features_batch = torch.stack(features)
            ctr_preds, playtime_preds = self.model(features_batch)
            
            # 归一化预测值
            ctr_norm = torch.sigmoid(ctr_preds)
            playtime_norm = F.normalize(playtime_preds.unsqueeze(0), p=1).squeeze(0)
            
            # 加权组合得分
            combined_scores = self.alpha * ctr_norm + (1 - self.alpha) * playtime_norm
            
            # 构建得分字典
            for i, video_id in enumerate(video_ids):
                scores_dict[video_id] = {
                    'ctr': ctr_norm[i].item(),
                    'playtime': playtime_preds[i].item(),
                    'playtime_norm': playtime_norm[i].item(),
                    'combined_score': combined_scores[i].item()
                }
        
        return scores_dict
    
    def batch_recommend(self, 
                        user_ids: List[str], 
                        candidate_videos_per_user: Dict[str, List[str]],
                        data_manager,
                        top_k: int = 1) -> Dict[str, List[str]]:
        """
        批量为多个用户推荐视频
        
        Args:
            user_ids: 用户ID列表
            candidate_videos_per_user: 每个用户的候选视频字典
            data_manager: 数据管理器
            top_k: 每个用户的推荐视频数量
        
        Returns:
            用户到推荐视频的映射字典
        """
        recommendations = {}
        for user_id in user_ids:
            if user_id in candidate_videos_per_user and candidate_videos_per_user[user_id]:
                recommendations[user_id] = self.recommend(
                    user_id, 
                    candidate_videos_per_user[user_id], 
                    data_manager,
                    top_k
                )
            else:
                recommendations[user_id] = []
        
        return recommendations


import torch.nn.functional as F  # 导入F模块，用于归一化操作
