#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging

# 添加项目根目录到路径
def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("debug_data")

def generate_mock_data(output_dir, n_users=1000, n_videos=5000, n_interactions=50000):
    """
    生成模拟数据用于调试
    
    Args:
        output_dir: 输出目录
        n_users: 用户数量
        n_videos: 视频数量
        n_interactions: 交互数量
    """
    logger = setup_logging()
    logger.info(f"生成模拟数据: {n_users}用户, {n_videos}视频, {n_interactions}交互")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成用户数据
    user_ids = [f"u{i}" for i in range(n_users)]
    user_active_degrees = np.random.choice(['high', 'medium', 'low'], n_users)
    user_is_verified = np.random.randint(0, 2, n_users)
    user_follow_count = np.random.randint(0, 10000, n_users)
    user_fans_count = np.random.randint(0, 50000, n_users)
    user_friend_count = np.random.randint(0, 5000, n_users)
    user_video_count = np.random.randint(0, 500, n_users)
    user_favorite_count = np.random.randint(0, 1000, n_users)
    user_like_count = np.random.randint(0, 10000, n_users)
    
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'user_active_degree': user_active_degrees,
        'user_is_verified': user_is_verified,
        'user_follow_count': user_follow_count,
        'user_fans_count': user_fans_count,
        'user_friend_count': user_friend_count,
        'user_video_count': user_video_count,
        'user_favorite_count': user_favorite_count,
        'user_like_count': user_like_count
    })
    
    # 生成视频数据
    video_ids = [f"v{i}" for i in range(n_videos)]
    video_types = np.random.choice(['entertainment', 'education', 'news', 'sports', 'music'], n_videos)
    tags = np.random.choice(['funny', 'serious', 'emotional', 'informative', 'exciting'], n_videos)
    video_durations = np.random.randint(10, 600, n_videos)
    video_like_counts = np.random.randint(0, 100000, n_videos)
    video_comment_counts = np.random.randint(0, 50000, n_videos)
    video_forward_counts = np.random.randint(0, 10000, n_videos)
    
    videos_df = pd.DataFrame({
        'video_id': video_ids,
        'video_type': video_types,
        'tag': tags,
        'video_duration': video_durations,
        'video_like_count': video_like_counts,
        'video_comment_count': video_comment_counts,
        'video_forward_count': video_forward_counts
    })
    print(f"视频特征列: {list(data_manager.video_features.columns)}")
    # 生成交互数据
    interaction_user_ids = np.random.choice(user_ids, n_interactions)
    interaction_video_ids = np.random.choice(video_ids, n_interactions)
    play_times = np.random.randint(0, 600, n_interactions)
    is_clicks = np.random.randint(0, 2, n_interactions)
    
    # 确保交互对唯一
    interactions = set()
    final_user_ids = []
    final_video_ids = []
    final_play_times = []
    final_is_clicks = []
    
    for i in range(n_interactions):
        user_id = interaction_user_ids[i]
        video_id = interaction_video_ids[i]
        
        # 检查是否已存在相同的用户-视频对
        if (user_id, video_id) not in interactions:
            interactions.add((user_id, video_id))
            final_user_ids.append(user_id)
            final_video_ids.append(video_id)
            final_play_times.append(play_times[i])
            final_is_clicks.append(is_clicks[i])
    
    interactions_df = pd.DataFrame({
        'user_id': final_user_ids,
        'video_id': final_video_ids,
        'play_time': final_play_times,
        'is_click': final_is_clicks
    })
    
    # 保存数据
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    videos_df.to_csv(os.path.join(output_dir, 'videos.csv'), index=False)
    interactions_df.to_csv(os.path.join(output_dir, 'interactions.csv'), index=False)
    
    logger.info(f"数据已保存到: {output_dir}")
    logger.info(f"用户数: {len(users_df)}")
    logger.info(f"视频数: {len(videos_df)}")
    logger.info(f"交互数: {len(interactions_df)}")
    
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成模拟数据用于调试")
    parser.add_argument("--output_dir", type=str, default="data/KuaiRand/debug",
                        help="输出目录")
    parser.add_argument("--n_users", type=int, default=1000, help="用户数量")
    parser.add_argument("--n_videos", type=int, default=5000, help="视频数量")
    parser.add_argument("--n_interactions", type=int, default=50000, help="交互数量")
    args = parser.parse_args()
    
    generate_mock_data(args.output_dir, args.n_users, args.n_videos, args.n_interactions)

if __name__ == "__main__":
    main()
