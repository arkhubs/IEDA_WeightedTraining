"""
绘图脚本，用于可视化实验结果
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def plot_results(results_path: str):
    """
    绘制实验结果图
    
    Args:
        results_path: 结果JSON文件路径
    """
    # 加载结果
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if 'step' not in results or 'metrics' not in results or not results['step']:
        print("结果格式不正确，无法绘图")
        return
    
    steps = results['step']
    metrics = results['metrics']
    
    # 绘制处理组和对照组的损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 处理组CTR损失
    ax = axes[0, 0]
    train_ctr_loss_T = [m['train']['treatment']['ctr_loss'] if 'treatment' in m['train'] else np.nan for m in metrics]
    val_ctr_loss_T = [m['val']['treatment']['ctr_loss'] if m['val'] and 'treatment' in m['val'] else np.nan for m in metrics]
    ax.plot(steps, train_ctr_loss_T, 'b-', label='Train')
    ax.plot(steps, val_ctr_loss_T, 'r-', label='Val')
    ax.set_title('处理组CTR损失')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # 处理组播放时间损失
    ax = axes[0, 1]
    train_playtime_loss_T = [m['train']['treatment']['playtime_loss'] if 'treatment' in m['train'] else np.nan for m in metrics]
    val_playtime_loss_T = [m['val']['treatment']['playtime_loss'] if m['val'] and 'treatment' in m['val'] else np.nan for m in metrics]
    ax.plot(steps, train_playtime_loss_T, 'b-', label='Train')
    ax.plot(steps, val_playtime_loss_T, 'r-', label='Val')
    ax.set_title('处理组播放时间损失')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # 对照组CTR损失
    ax = axes[1, 0]
    train_ctr_loss_C = [m['train']['control']['ctr_loss'] if 'control' in m['train'] else np.nan for m in metrics]
    val_ctr_loss_C = [m['val']['control']['ctr_loss'] if m['val'] and 'control' in m['val'] else np.nan for m in metrics]
    ax.plot(steps, train_ctr_loss_C, 'b-', label='Train')
    ax.plot(steps, val_ctr_loss_C, 'r-', label='Val')
    ax.set_title('对照组CTR损失')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # 对照组播放时间损失
    ax = axes[1, 1]
    train_playtime_loss_C = [m['train']['control']['playtime_loss'] if 'control' in m['train'] else np.nan for m in metrics]
    val_playtime_loss_C = [m['val']['control']['playtime_loss'] if m['val'] and 'control' in m['val'] else np.nan for m in metrics]
    ax.plot(steps, train_playtime_loss_C, 'b-', label='Train')
    ax.plot(steps, val_playtime_loss_C, 'r-', label='Val')
    ax.set_title('对照组播放时间损失')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(results_path), 'exp_results_plot.png')
    plt.savefig(save_path)
    print(f"绘图已保存到: {save_path}")
    
    # 绘制CTR AUC曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_auc_T = [m['train']['treatment']['ctr_auc'] if 'treatment' in m['train'] and 'ctr_auc' in m['train']['treatment'] else np.nan for m in metrics]
    val_auc_T = [m['val']['treatment']['ctr_auc'] if m['val'] and 'treatment' in m['val'] and 'ctr_auc' in m['val']['treatment'] else np.nan for m in metrics]
    train_auc_C = [m['train']['control']['ctr_auc'] if 'control' in m['train'] and 'ctr_auc' in m['train']['control'] else np.nan for m in metrics]
    val_auc_C = [m['val']['control']['ctr_auc'] if m['val'] and 'control' in m['val'] and 'ctr_auc' in m['val']['control'] else np.nan for m in metrics]
    
    ax.plot(steps, train_auc_T, 'b-', label='处理组训练')
    ax.plot(steps, val_auc_T, 'b--', label='处理组验证')
    ax.plot(steps, train_auc_C, 'r-', label='对照组训练')
    ax.plot(steps, val_auc_C, 'r--', label='对照组验证')
    ax.set_title('CTR AUC')
    ax.set_xlabel('Step')
    ax.set_ylabel('AUC')
    ax.legend()
    
    plt.tight_layout()
    auc_save_path = os.path.join(os.path.dirname(results_path), 'auc_curve.png')
    plt.savefig(auc_save_path)
    print(f"AUC曲线已保存到: {auc_save_path}")
    
    # 绘制权重模型指标
    if all('weight_model' in m['train'] for m in metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weight_loss = [m['train']['weight_model']['loss'] if 'weight_model' in m['train'] else np.nan for m in metrics]
        weight_acc = [m['train']['weight_model']['accuracy'] if 'weight_model' in m['train'] else np.nan for m in metrics]
        
        ax.plot(steps, weight_loss, 'g-', label='损失')
        ax.set_xlabel('Step')
        ax.set_ylabel('损失')
        
        ax2 = ax.twinx()
        ax2.plot(steps, weight_acc, 'b-', label='准确率')
        ax2.set_ylabel('准确率')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.set_title('权重模型性能')
        
        plt.tight_layout()
        weight_save_path = os.path.join(os.path.dirname(results_path), 'weight_model_curve.png')
        plt.savefig(weight_save_path)
        print(f"权重模型曲线已保存到: {weight_save_path}")


def plot_async(results_dir: str):
    """
    异步绘图，处理结果目录中的JSON结果文件
    
    Args:
        results_dir: 结果目录
    """
    results_path = os.path.join(results_dir, 'exp_results.json')
    
    if os.path.exists(results_path):
        plot_results(results_path)
    else:
        print(f"找不到结果文件: {results_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        plot_async(results_dir)
    else:
        print("使用方法: python plot_exp_results.py <results_dir>")
        sys.exit(1)
