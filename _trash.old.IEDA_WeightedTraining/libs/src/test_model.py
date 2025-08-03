import os
import sys
import argparse
import torch
import numpy as np
import logging
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 引入项目模块
from libs.src import init_all

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试模型")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_users", type=int, default=100, help="测试用户数量")
    parser.add_argument("--output", type=str, default="results/test_results.json", help="输出结果文件")
    return parser.parse_args()

def test_model(config_manager, data_manager_methods, model, recommender, device, args):
    """测试模型性能"""
    logger = logging.getLogger("ModelTest")
    logger.info(f"开始测试模型: {args.checkpoint}")
    
    # 加载模型检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取验证集用户
    val_users = data_manager_methods.data_manager.get_val_users()
    if len(val_users) > args.test_users:
        test_users = np.random.choice(val_users, args.test_users, replace=False).tolist()
    else:
        test_users = val_users
        
    logger.info(f"测试用户数: {len(test_users)}")
    
    # 获取配置
    global_config = config_manager.get_global_config()
    n_candidate = global_config['n_candidate']
    
    # 为用户生成候选视频
    user_candidates = {}
    for user_id in test_users:
        # 验证时不考虑used和masked状态，获取所有交互过的视频
        videos = data_manager_methods.data_manager.get_user_videos(user_id)
        if videos:
            # 随机选择n_candidate个视频
            if len(videos) > n_candidate:
                selected_videos = np.random.choice(videos, n_candidate, replace=False).tolist()
            else:
                selected_videos = videos
            user_candidates[user_id] = selected_videos
    
    # 获取推荐器配置
    alpha_T = recommender.alpha_T
    alpha_C = recommender.alpha_C
    
    # 测试Treatment组推荐
    recommendations_T = recommender.recommend(user_candidates.copy(), alpha_T, mode='val')
    
    # 测试Control组推荐
    recommendations_C = recommender.recommend(user_candidates.copy(), alpha_C, mode='val')
    
    # 收集结果
    user_ids_T = [rec[0] for rec in recommendations_T]
    video_ids_T = [rec[1] for rec in recommendations_T]
    scores_T = [float(rec[2]) for rec in recommendations_T]
    
    user_ids_C = [rec[0] for rec in recommendations_C]
    video_ids_C = [rec[1] for rec in recommendations_C]
    scores_C = [float(rec[2]) for rec in recommendations_C]
    
    # 获取真实标签
    true_labels_T = data_manager_methods.get_true_labels(user_ids_T, video_ids_T)
    true_labels_C = data_manager_methods.get_true_labels(user_ids_C, video_ids_C)
    
    # 计算指标
    if len(true_labels_T) > 0 and len(true_labels_C) > 0:
        avg_reward_T = float(true_labels_T.mean())
        avg_reward_C = float(true_labels_C.mean())
        total_reward_T = float(true_labels_T.sum())
        total_reward_C = float(true_labels_C.sum())
        gte = total_reward_T - total_reward_C
        
        logger.info(f"Treatment组平均奖励: {avg_reward_T:.4f}")
        logger.info(f"Control组平均奖励: {avg_reward_C:.4f}")
        logger.info(f"Treatment组总奖励: {total_reward_T:.4f}")
        logger.info(f"Control组总奖励: {total_reward_C:.4f}")
        logger.info(f"GTE: {gte:.4f}")
        
        # 准备输出结果
        results = {
            'checkpoint': args.checkpoint,
            'test_users': len(test_users),
            'metrics': {
                'avg_reward_T': avg_reward_T,
                'avg_reward_C': avg_reward_C,
                'total_reward_T': total_reward_T,
                'total_reward_C': total_reward_C,
                'GTE': gte
            },
            'treatment_group': {
                'user_ids': user_ids_T,
                'video_ids': video_ids_T,
                'scores': scores_T,
                'true_labels': true_labels_T.tolist()
            },
            'control_group': {
                'user_ids': user_ids_C,
                'video_ids': video_ids_C,
                'scores': scores_C,
                'true_labels': true_labels_C.tolist()
            }
        }
        
        # 输出结果
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"测试结果已保存到: {args.output}")
        
        return results
    else:
        logger.warning("没有足够的数据进行评估")
        return None

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 初始化所有模块
    config_manager, data_manager, data_manager_methods, model, recommender, trainer, device = init_all(args.config)
    
    # 测试模型
    test_model(config_manager, data_manager_methods, model, recommender, device, args)

if __name__ == "__main__":
    main()
