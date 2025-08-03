#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_results import load_training_logs

def plot_training_metrics(result_dir):
    """
    绘制训练过程中各个指标的变化曲线
    
    Args:
        result_dir: 结果文件目录
    
    Returns:
        None，但会保存图像到result_dir
    """
    metrics = load_training_logs(result_dir)
    if not metrics:
        print("警告: 无法找到训练指标数据")
        return
        
    # 创建一个2行3列的大图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("训练与验证指标")
    
    # 绘制点击模型的AUC曲线
    if 'click' in metrics:
        click_metrics = metrics['click']
        ax = axes[0, 0]
        if 'train' in click_metrics and 'steps' in click_metrics['train'] and 'auc' in click_metrics['train']:
            ax.plot(click_metrics['train']['steps'], click_metrics['train']['auc'], 'b-', label='训练 AUC')
        if 'val' in click_metrics and 'steps' in click_metrics['val'] and 'auc' in click_metrics['val']:
            ax.plot(click_metrics['val']['steps'], click_metrics['val']['auc'], 'r-', label='验证 AUC')
        ax.set_title('点击模型 AUC')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('AUC')
        ax.legend()
        ax.grid(True)
    
    # 绘制点击模型的损失曲线
    ax = axes[0, 1]
    if 'click' in metrics:
        click_metrics = metrics['click']
        if 'train' in click_metrics and 'steps' in click_metrics['train'] and 'loss' in click_metrics['train']:
            ax.plot(click_metrics['train']['steps'], click_metrics['train']['loss'], 'b-', label='训练 损失')
        if 'val' in click_metrics and 'steps' in click_metrics['val'] and 'loss' in click_metrics['val']:
            ax.plot(click_metrics['val']['steps'], click_metrics['val']['loss'], 'r-', label='验证 损失')
        ax.set_title('点击模型 损失')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('损失')
        ax.legend()
        ax.grid(True)
    
    # 绘制播放时长模型的相对误差曲线
    ax = axes[0, 2]
    if 'play_time' in metrics:
        play_metrics = metrics['play_time']
        if 'train' in play_metrics and 'steps' in play_metrics['train'] and 'rel_error' in play_metrics['train']:
            ax.plot(play_metrics['train']['steps'], play_metrics['train']['rel_error'], 'b-', label='训练 相对误差')
        if 'val' in play_metrics and 'steps' in play_metrics['val'] and 'rel_error' in play_metrics['val']:
            ax.plot(play_metrics['val']['steps'], play_metrics['val']['rel_error'], 'r-', label='验证 相对误差')
        ax.set_title('播放时长模型 相对误差')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('相对误差')
        ax.legend()
        ax.grid(True)
    
    # 绘制播放时长模型的损失曲线
    ax = axes[1, 0]
    if 'play_time' in metrics:
        play_metrics = metrics['play_time']
        if 'train' in play_metrics and 'steps' in play_metrics['train'] and 'loss' in play_metrics['train']:
            ax.plot(play_metrics['train']['steps'], play_metrics['train']['loss'], 'b-', label='训练 损失')
        if 'val' in play_metrics and 'steps' in play_metrics['val'] and 'loss' in play_metrics['val']:
            ax.plot(play_metrics['val']['steps'], play_metrics['val']['loss'], 'r-', label='验证 损失')
        ax.set_title('播放时长模型 损失')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('损失')
        ax.legend()
        ax.grid(True)
    
    # 绘制处理组和对照组的累积回报
    ax = axes[1, 1]
    # 点击模型的处理组和对照组累积回报
    if 'click' in metrics:
        click_metrics = metrics['click']
        if 'treatment' in click_metrics and 'steps' in click_metrics['treatment'] and 'reward' in click_metrics['treatment']:
            ax.plot(click_metrics['treatment']['steps'], click_metrics['treatment']['reward'], 'b-', label='处理组 - 点击回报')
        if 'control' in click_metrics and 'steps' in click_metrics['control'] and 'reward' in click_metrics['control']:
            ax.plot(click_metrics['control']['steps'], click_metrics['control']['reward'], 'r-', label='对照组 - 点击回报')
    
    ax.set_title('处理组 vs 对照组 累积回报')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('累积回报')
    ax.legend()
    ax.grid(True)
    
    # 绘制播放时长的处理组和对照组累积回报
    ax = axes[1, 2]
    if 'play_time' in metrics:
        play_metrics = metrics['play_time']
        if 'treatment' in play_metrics and 'steps' in play_metrics['treatment'] and 'reward' in play_metrics['treatment']:
            ax.plot(play_metrics['treatment']['steps'], play_metrics['treatment']['reward'], 'g-', label='处理组 - 播放时长回报')
        if 'control' in play_metrics and 'steps' in play_metrics['control'] and 'reward' in play_metrics['control']:
            ax.plot(play_metrics['control']['steps'], play_metrics['control']['reward'], 'm-', label='对照组 - 播放时长回报')
    
    ax.set_title('处理组 vs 对照组 播放时长回报')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('累积回报')
    ax.legend()
    ax.grid(True)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, 'training_metrics.png'))
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="绘制训练指标")
    parser.add_argument("--result_dir", type=str, required=True, help="结果目录路径")
    args = parser.parse_args()
    
    plot_training_metrics(args.result_dir)

if __name__ == "__main__":
    main()
