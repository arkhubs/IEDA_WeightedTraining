"""
模型测试脚本，用于对训练好的模型进行测试和评估
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import roc_auc_score, mean_squared_error

from src.models import PredictionModel
from src.data_manager import RecommendDataset
from src.utils import Logger, setup_seed
from src.config_manager import ConfigManager

import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path: str, model_config: Dict[str, Any], device: str = 'cuda') -> PredictionModel:
    """
    加载预训练模型
    
    Args:
        model_path: 模型路径
        model_config: 模型配置
        device: 运行设备
        
    Returns:
        加载的模型
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = PredictionModel(model_config).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已从 {model_path} 加载")
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    return model


def test_model(model: PredictionModel, 
               test_dataset: RecommendDataset, 
               device: str = 'cuda',
               batch_size: int = 128) -> Dict[str, Any]:
    """
    测试模型性能
    
    Args:
        model: 预测模型
        test_dataset: 测试数据集
        device: 运行设备
        batch_size: 批次大小
        
    Returns:
        评估指标字典
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    all_ctr_preds = []
    all_ctr_targets = []
    all_playtime_preds = []
    all_playtime_targets = []
    all_groups = []
    all_user_ids = []
    all_item_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取特征和标签
            features = {k: v.to(device) for k, v in batch['features'].items()}
            ctr_targets = batch['ctr'].to(device)
            playtime_targets = batch['playtime'].to(device)
            groups = batch['group'].cpu().numpy()
            user_ids = batch['features']['user_id'].cpu().numpy()
            item_ids = batch['features']['item_id'].cpu().numpy()
            
            # 前向传播
            ctr_preds, playtime_preds = model(features)
            ctr_preds = torch.sigmoid(ctr_preds).cpu().numpy()
            playtime_preds = playtime_preds.cpu().numpy()
            
            # 收集结果
            all_ctr_preds.extend(ctr_preds)
            all_ctr_targets.extend(ctr_targets.cpu().numpy())
            all_playtime_preds.extend(playtime_preds)
            all_playtime_targets.extend(playtime_targets.cpu().numpy())
            all_groups.extend(groups)
            all_user_ids.extend(user_ids)
            all_item_ids.extend(item_ids)
    
    # 转换为numpy数组
    all_ctr_preds = np.array(all_ctr_preds)
    all_ctr_targets = np.array(all_ctr_targets)
    all_playtime_preds = np.array(all_playtime_preds)
    all_playtime_targets = np.array(all_playtime_targets)
    all_groups = np.array(all_groups)
    all_user_ids = np.array(all_user_ids)
    all_item_ids = np.array(all_item_ids)
    
    # 分离处理组和对照组
    t_mask = (all_groups == 1)  # 处理组
    c_mask = (all_groups == 0)  # 对照组
    
    # 计算总体指标
    metrics = {}
    
    # 计算CTR指标
    if len(all_ctr_targets) > 0:
        metrics['overall'] = {
            'ctr_auc': roc_auc_score(all_ctr_targets, all_ctr_preds),
            'playtime_rmse': np.sqrt(mean_squared_error(all_playtime_targets, all_playtime_preds))
        }
    
    # 计算处理组指标
    if t_mask.sum() > 0:
        metrics['treatment'] = {
            'ctr_auc': roc_auc_score(all_ctr_targets[t_mask], all_ctr_preds[t_mask]),
            'playtime_rmse': np.sqrt(mean_squared_error(all_playtime_targets[t_mask], all_playtime_preds[t_mask]))
        }
    
    # 计算对照组指标
    if c_mask.sum() > 0:
        metrics['control'] = {
            'ctr_auc': roc_auc_score(all_ctr_targets[c_mask], all_ctr_preds[c_mask]),
            'playtime_rmse': np.sqrt(mean_squared_error(all_playtime_targets[c_mask], all_playtime_preds[c_mask]))
        }
    
    # 保存预测结果
    results = pd.DataFrame({
        'user_id': all_user_ids,
        'item_id': all_item_ids,
        'group': all_groups,
        'ctr_pred': all_ctr_preds,
        'ctr_target': all_ctr_targets,
        'playtime_pred': all_playtime_preds,
        'playtime_target': all_playtime_targets
    })
    
    return metrics, results


def visualize_results(results: pd.DataFrame, output_dir: str):
    """
    可视化测试结果
    
    Args:
        results: 测试结果DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图表风格
    sns.set(style="whitegrid")
    
    # 1. CTR预测分布
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=results, x='ctr_pred', hue='group', bins=30, 
                 element='step', stat='probability', common_norm=False)
    plt.title('CTR预测值分布')
    plt.xlabel('预测CTR值')
    plt.ylabel('概率密度')
    
    # 2. 播放时间预测分布
    plt.subplot(1, 2, 2)
    sns.histplot(data=results, x='playtime_pred', hue='group', bins=30, 
                 element='step', stat='probability', common_norm=False)
    plt.title('播放时间预测值分布')
    plt.xlabel('预测播放时间')
    plt.ylabel('概率密度')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    
    # 3. ROC曲线比较
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(10, 8))
    
    # 整体ROC
    fpr, tpr, _ = roc_curve(results['ctr_target'], results['ctr_pred'])
    plt.plot(fpr, tpr, label=f'全体数据 (AUC = {roc_auc_score(results["ctr_target"], results["ctr_pred"]):.4f})')
    
    # 处理组ROC
    t_data = results[results['group'] == 1]
    fpr_t, tpr_t, _ = roc_curve(t_data['ctr_target'], t_data['ctr_pred'])
    plt.plot(fpr_t, tpr_t, label=f'处理组 (AUC = {roc_auc_score(t_data["ctr_target"], t_data["ctr_pred"]):.4f})')
    
    # 对照组ROC
    c_data = results[results['group'] == 0]
    fpr_c, tpr_c, _ = roc_curve(c_data['ctr_target'], c_data['ctr_pred'])
    plt.plot(fpr_c, tpr_c, label=f'对照组 (AUC = {roc_auc_score(c_data["ctr_target"], c_data["ctr_pred"]):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    
    # 4. 预测值与实际值的散点图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=results, x='ctr_target', y='ctr_pred', hue='group', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('CTR: 预测值vs实际值')
    plt.xlabel('实际CTR')
    plt.ylabel('预测CTR')
    
    plt.subplot(1, 2, 2)
    max_playtime = max(results['playtime_target'].max(), results['playtime_pred'].max())
    sns.scatterplot(data=results, x='playtime_target', y='playtime_pred', hue='group', alpha=0.3)
    plt.plot([0, max_playtime], [0, max_playtime], 'k--')
    plt.title('播放时间: 预测值vs实际值')
    plt.xlabel('实际播放时间')
    plt.ylabel('预测播放时间')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    
    # 5. 不同用户和物品的性能分析
    # 按用户分组计算AUC
    user_metrics = []
    for user_id in results['user_id'].unique():
        user_data = results[results['user_id'] == user_id]
        if len(user_data) > 10 and len(user_data['ctr_target'].unique()) > 1:
            try:
                auc = roc_auc_score(user_data['ctr_target'], user_data['ctr_pred'])
                rmse = np.sqrt(mean_squared_error(user_data['playtime_target'], user_data['playtime_pred']))
                user_metrics.append({
                    'user_id': user_id,
                    'count': len(user_data),
                    'ctr_auc': auc,
                    'playtime_rmse': rmse
                })
            except:
                pass
    
    if user_metrics:
        user_df = pd.DataFrame(user_metrics)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=user_df, x='ctr_auc', bins=20)
        plt.title('用户CTR AUC分布')
        plt.xlabel('AUC')
        plt.ylabel('用户数')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=user_df, x='playtime_rmse', bins=20)
        plt.title('用户播放时间RMSE分布')
        plt.xlabel('RMSE')
        plt.ylabel('用户数')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_performance.png'))
    
    # 保存指标和图表
    plt.close('all')


def main(config_path: str, model_path: str, test_data_path: str, output_dir: str):
    """
    主函数
    
    Args:
        config_path: 配置文件路径
        model_path: 模型文件路径
        test_data_path: 测试数据路径
        output_dir: 输出目录
    """
    # 设置日志和输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(log_file=os.path.join(output_dir, 'test.log'))
    
    # 加载配置
    config_manager = ConfigManager(config_path)
    config = config_manager.get_all()
    
    # 设置随机种子
    seed = config.get('experiment', {}).get('seed', 42)
    setup_seed(seed)
    
    # 加载测试数据
    device = config.get('experiment', {}).get('device', 'cuda')
    batch_size = config.get('data', {}).get('batch_size', 128)
    
    logger.info(f"加载测试数据: {test_data_path}")
    test_dataset = RecommendDataset(test_data_path)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = load_model(model_path, config['model'], device)
    
    # 测试模型
    logger.info("开始测试模型...")
    metrics, results = test_model(model, test_dataset, device, batch_size)
    
    # 打印指标
    logger.info("测试结果:")
    for group, group_metrics in metrics.items():
        logger.info(f"{group}组:")
        for metric_name, metric_value in group_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # 保存结果
    results_path = os.path.join(output_dir, 'test_predictions.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"预测结果已保存至: {results_path}")
    
    # 可视化结果
    logger.info("生成结果可视化...")
    visualize_results(results, output_dir)
    logger.info(f"可视化结果已保存至: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试推荐模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output", type=str, default="./test_results", help="输出目录")
    
    args = parser.parse_args()
    
    main(args.config, args.model, args.data, args.output)
