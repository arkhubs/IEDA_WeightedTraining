#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score

def load_results(result_dir):
    """
    加载实验结果文件
    
    Args:
        result_dir: 结果文件目录
        
    Returns:
        结果数据字典
    """
    result_file = os.path.join(result_dir, 'result.json')
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"结果文件不存在: {result_file}")
        
    with open(result_file, 'r') as f:
        results = json.load(f)
        
    return results

def load_training_logs(result_dir):
    """
    尝试加载训练日志和结果文件，提取额外的指标
    
    Args:
        result_dir: 结果文件目录
        
    Returns:
        训练日志提取的指标
    """
    # 尝试从结果文件读取
    result_file = os.path.join(result_dir, 'result.json')
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
                
            # 从结果文件中提取指标
            metrics = {}
            
            # 处理验证指标
            if 'validation' in results:
                for label_name, val_metrics in results['validation'].items():
                    if label_name not in metrics:
                        metrics[label_name] = {}
                        
                    metrics[label_name]['val'] = {}
                    
                    if 'steps' in val_metrics:
                        metrics[label_name]['val']['steps'] = val_metrics['steps']
                    
                    # 根据标签类型添加不同的指标
                    if 'auc' in val_metrics:  # 二分类
                        metrics[label_name]['val']['auc'] = val_metrics['auc']
                    if 'accuracy' in val_metrics:
                        metrics[label_name]['val']['accuracy'] = val_metrics['accuracy']
                    if 'loss' in val_metrics:
                        metrics[label_name]['val']['loss'] = val_metrics['loss']
                    if 'rel_error' in val_metrics:
                        metrics[label_name]['val']['rel_error'] = val_metrics['rel_error']
                    if 'avg_rewards' in val_metrics:
                        metrics[label_name]['val']['avg_rewards'] = val_metrics['avg_rewards']
            
            # 处理训练指标
            if 'training' in results:
                for label_name, train_metrics in results['training'].items():
                    if label_name not in metrics:
                        metrics[label_name] = {}
                    
                    metrics[label_name]['train'] = {}
                    
                    if 'steps' in train_metrics:
                        metrics[label_name]['train']['steps'] = train_metrics['steps']
                    
                    # 添加训练指标
                    if 'auc' in train_metrics:
                        metrics[label_name]['train']['auc'] = train_metrics['auc']
                    if 'accuracy' in train_metrics:
                        metrics[label_name]['train']['accuracy'] = train_metrics['accuracy']
                    if 'loss' in train_metrics:
                        metrics[label_name]['train']['loss'] = train_metrics['loss']
                    if 'rel_error' in train_metrics:
                        metrics[label_name]['train']['rel_error'] = train_metrics['rel_error']
            
            # 处理Treatment和Control组的指标
            if 'labels' in results:
                for label_name, label_metrics in results['labels'].items():
                    if label_name not in metrics:
                        metrics[label_name] = {}
                    
                    # 添加treatment和control组的累积奖励
                    metrics[label_name]['treatment'] = {}
                    metrics[label_name]['control'] = {}
                    
                    if 'steps' in results:
                        metrics[label_name]['treatment']['steps'] = results['steps']
                        metrics[label_name]['control']['steps'] = results['steps']
                    
                    if 'total_label_T' in label_metrics:
                        metrics[label_name]['treatment']['reward'] = label_metrics['total_label_T']
                    if 'total_label_C' in label_metrics:
                        metrics[label_name]['control']['reward'] = label_metrics['total_label_C']
                
            return metrics
                
        except Exception as e:
            print(f"从结果文件加载指标失败: {str(e)}")
    
    # 如果结果文件不存在或读取失败，尝试从日志文件中提取
    log_file = os.path.join(result_dir, 'run.log')
    if not os.path.exists(log_file):
        print(f"警告: 日志文件不存在: {log_file}")
        return {}
        
    # 提取训练和验证的AUC、准确率和相对误差等指标
    metrics = {
        'click': {
            'train': {'auc': [], 'loss': [], 'steps': []},
            'val': {'auc': [], 'loss': [], 'steps': []},
            'treatment': {'auc': [], 'loss': [], 'steps': []},
            'control': {'auc': [], 'loss': [], 'steps': []}
        },
        'play_time': {
            'train': {'rel_error': [], 'loss': [], 'steps': []},
            'val': {'rel_error': [], 'loss': [], 'steps': []},
            'treatment': {'rel_error': [], 'loss': [], 'steps': []},
            'control': {'rel_error': [], 'loss': [], 'steps': []}
        },
        'weight_model': {
            'train': {'auc': [], 'accuracy': [], 'loss': [], 'steps': []},
            'val': {'auc': [], 'accuracy': [], 'loss': [], 'steps': []}
        }
    }
    
    # 日志解析逻辑（为简化起见，这里不做具体实现）
    
    return metrics

def calculate_non_zero_relative_error(predictions, true_values):
    """
    计算非零值的相对误差
    
    Args:
        predictions: 预测值数组
        true_values: 真实值数组
        
    Returns:
        平均相对误差（百分比）
    """
    # 筛选非零真实值
    mask = true_values > 0
    if not np.any(mask):
        return 0.0
        
    non_zero_true = true_values[mask]
    non_zero_pred = predictions[mask]
    
    # 计算相对误差 |y_pred - y_true| / y_true
    rel_errors = np.abs(non_zero_pred - non_zero_true) / non_zero_true
    
    # 返回平均相对误差（百分比）
    return np.mean(rel_errors) * 100.0

def plot_advanced_metrics(results, log_metrics, output_dir=None):
    """
    绘制高级指标图表
    
    Args:
        results: 结果数据字典
        log_metrics: 从日志提取的额外指标
        output_dir: 输出目录，默认与结果目录相同
    """
    plt.figure(figsize=(18, 15))
    plt.suptitle('实验结果分析', fontsize=18)
    
    # 图块1：CTR AUC
    plt.subplot(2, 3, 1)
    if 'click' in log_metrics:
        if log_metrics['click']['train']['steps']:
            plt.plot(log_metrics['click']['train']['steps'], 
                    log_metrics['click']['train']['auc'], 
                    label='Train')
        if log_metrics['click']['val']['steps']:
            plt.plot(log_metrics['click']['val']['steps'], 
                    log_metrics['click']['val']['auc'], 
                    label='Validation')
        if log_metrics['click']['treatment']['steps']:
            plt.plot(log_metrics['click']['treatment']['steps'], 
                    log_metrics['click']['treatment']['auc'], 
                    label='Treatment')
        if log_metrics['click']['control']['steps']:
            plt.plot(log_metrics['click']['control']['steps'], 
                    log_metrics['click']['control']['auc'], 
                    label='Control')
    plt.title('点击预测 AUC')
    plt.xlabel('训练步数')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    
    # 图块2：CTR Loss
    plt.subplot(2, 3, 2)
    if 'click' in log_metrics:
        if log_metrics['click']['train']['steps']:
            plt.plot(log_metrics['click']['train']['steps'], 
                    log_metrics['click']['train']['loss'], 
                    label='Train')
        if log_metrics['click']['val']['steps']:
            plt.plot(log_metrics['click']['val']['steps'], 
                    log_metrics['click']['val']['loss'], 
                    label='Validation')
        if log_metrics['click']['treatment']['steps']:
            plt.plot(log_metrics['click']['treatment']['steps'], 
                    log_metrics['click']['treatment']['loss'], 
                    label='Treatment')
        if log_metrics['click']['control']['steps']:
            plt.plot(log_metrics['click']['control']['steps'], 
                    log_metrics['click']['control']['loss'], 
                    label='Control')
    plt.title('点击预测 Loss')
    plt.xlabel('训练步数')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 图块3：Playtime 平均相对误差（百分比）
    plt.subplot(2, 3, 3)
    if 'play_time' in log_metrics:
        if log_metrics['play_time']['train']['steps']:
            plt.plot(log_metrics['play_time']['train']['steps'], 
                    log_metrics['play_time']['train']['rel_error'], 
                    label='Train')
        if log_metrics['play_time']['val']['steps']:
            plt.plot(log_metrics['play_time']['val']['steps'], 
                    log_metrics['play_time']['val']['rel_error'], 
                    label='Validation')
        if log_metrics['play_time']['treatment']['steps']:
            plt.plot(log_metrics['play_time']['treatment']['steps'], 
                    log_metrics['play_time']['treatment']['rel_error'], 
                    label='Treatment')
        if log_metrics['play_time']['control']['steps']:
            plt.plot(log_metrics['play_time']['control']['steps'], 
                    log_metrics['play_time']['control']['rel_error'], 
                    label='Control')
    plt.title('播放时长 平均相对误差 (%)')
    plt.xlabel('训练步数')
    plt.ylabel('相对误差 (%)')
    plt.grid(True)
    plt.legend()
    
    # 图块4：Playtime Loss
    plt.subplot(2, 3, 4)
    if 'play_time' in log_metrics:
        if log_metrics['play_time']['train']['steps']:
            plt.plot(log_metrics['play_time']['train']['steps'], 
                    log_metrics['play_time']['train']['loss'], 
                    label='Train')
        if log_metrics['play_time']['val']['steps']:
            plt.plot(log_metrics['play_time']['val']['steps'], 
                    log_metrics['play_time']['val']['loss'], 
                    label='Validation')
        if log_metrics['play_time']['treatment']['steps']:
            plt.plot(log_metrics['play_time']['treatment']['steps'], 
                    log_metrics['play_time']['treatment']['loss'], 
                    label='Treatment')
        if log_metrics['play_time']['control']['steps']:
            plt.plot(log_metrics['play_time']['control']['steps'], 
                    log_metrics['play_time']['control']['loss'], 
                    label='Control')
    plt.title('播放时长 Loss')
    plt.xlabel('训练步数')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 图块5：Weight Model AUC 和 Accuracy
    plt.subplot(2, 3, 5)
    if 'weight_model' in log_metrics:
        if log_metrics['weight_model']['train']['steps']:
            plt.plot(log_metrics['weight_model']['train']['steps'], 
                    log_metrics['weight_model']['train']['auc'], 
                    label='Train AUC')
            plt.plot(log_metrics['weight_model']['train']['steps'], 
                    log_metrics['weight_model']['train']['accuracy'], 
                    label='Train Accuracy')
        if log_metrics['weight_model']['val']['steps']:
            plt.plot(log_metrics['weight_model']['val']['steps'], 
                    log_metrics['weight_model']['val']['auc'], 
                    label='Val AUC')
            plt.plot(log_metrics['weight_model']['val']['steps'], 
                    log_metrics['weight_model']['val']['accuracy'], 
                    label='Val Accuracy')
    plt.title('权重模型 AUC & 准确率')
    plt.xlabel('训练步数')
    plt.ylabel('评分')
    plt.grid(True)
    plt.legend()
    
    # 图块6：Weight Model Loss
    plt.subplot(2, 3, 6)
    if 'weight_model' in log_metrics:
        if log_metrics['weight_model']['train']['steps']:
            plt.plot(log_metrics['weight_model']['train']['steps'], 
                    log_metrics['weight_model']['train']['loss'], 
                    label='Train')
        if log_metrics['weight_model']['val']['steps']:
            plt.plot(log_metrics['weight_model']['val']['steps'], 
                    log_metrics['weight_model']['val']['loss'], 
                    label='Validation')
    plt.title('权重模型 Loss')
    plt.xlabel('训练步数')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    # 统一图片保存路径为 base_dir 拼接
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/experiment.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_dir = config.get('base_dir', os.getcwd())
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'results')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, 'advanced_metrics.png')
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至: {output_path}")
    
    # 显示图表（可选）
    # plt.show()

def update_recommender_plotting(recommender_file):
    """
    更新推荐器的绘图方法，以支持高级指标
    """
    # TODO: 实现推荐器绘图方法的更新
    pass

def main():
    parser = argparse.ArgumentParser(description="绘制实验结果高级指标")
    parser.add_argument("--result_dir", type=str, required=True, help="结果目录路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录路径")
    args = parser.parse_args()
    
    # 加载结果
    try:
        results = load_results(args.result_dir)
        log_metrics = load_training_logs(args.result_dir)
        
        # 绘制高级指标图表
        plot_advanced_metrics(results, log_metrics, args.output_dir)
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
