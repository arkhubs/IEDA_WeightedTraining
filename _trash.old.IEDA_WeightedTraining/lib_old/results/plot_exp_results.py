
import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def create_plots(result_dir, json_path=None):
    """创建实验结果可视化图表"""
    if json_path is None:
        json_path = os.path.join(result_dir, 'exp_results.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到结果文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    steps = data['step']
    # 横坐标统一用step，x_labels用epoch.step_in_epoch
    if 'epoch' in data and 'step_in_epoch' in data:
        epochs = data['epoch']
        step_in_epochs = data['step_in_epoch']
        x_labels = [f"{e}.{s}" for e, s in zip(epochs, step_in_epochs)]
    else:
        x_labels = [str(s) for s in steps]
    x_axis = steps
    
    metrics = data['metrics']
    
    # 创建4个子图: CTR-AUC, Playtime-LogMAE, Weight-Model-AUC, Weight-Model-F1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    def has_nested_metric(metrics_list, model_type, metric_name):
        """检查是否存在嵌套指标"""
        for m in metrics_list:
            if not m: continue
            train_metrics = m.get('train', {})
            if not train_metrics: continue
            if model_type in train_metrics:
                if isinstance(train_metrics[model_type], dict) and metric_name in train_metrics[model_type]:
                    return True
        return False
    
    # 检测是否有treatment/control分离模型
    has_treatment_control = any(m and 'train' in m and isinstance(m['train'], dict) and 
                             'treatment' in m['train'] and 'control' in m['train'] for m in metrics)
    
    # 检测是否有weight_model指标
    has_weight_model = any(m and 'train' in m and 'weight_model' in m['train'] for m in metrics)
    
    # 1. CTR AUC 曲线
    ax1 = axes[0,0]
    if has_treatment_control:
        # 实验组和对照组分开显示
        tr_auc_treatment = [m['train']['treatment'].get('CTR_AUC') if m and 'train' in m and 'treatment' in m['train'] else None for m in metrics]
        tr_auc_control = [m['train']['control'].get('CTR_AUC') if m and 'train' in m and 'control' in m['train'] else None for m in metrics]
        val_auc_treatment = [m['val']['treatment'].get('CTR_AUC') if m and 'val' in m and 'treatment' in m['val'] else None for m in metrics]
        val_auc_control = [m['val']['control'].get('CTR_AUC') if m and 'val' in m and 'control' in m['val'] else None for m in metrics]
        
        ax1.plot(x_axis, tr_auc_treatment, 'b-', label='Train CTR AUC (Treatment)', marker='o')
        ax1.plot(x_axis, tr_auc_control, 'g-', label='Train CTR AUC (Control)', marker='s')
        ax1.plot(x_axis, val_auc_treatment, 'b--', label='Val CTR AUC (Treatment)', marker='x')
        ax1.plot(x_axis, val_auc_control, 'g--', label='Val CTR AUC (Control)', marker='+')
    else:
        # 统一显示
        tr_auc = [m['train'].get('CTR_AUC') if m and 'train' in m else None for m in metrics]
        val_auc = [m['val'].get('CTR_AUC') if m and 'val' in m else None for m in metrics]
        ax1.plot(x_axis, tr_auc, 'b-', label='Train CTR AUC', marker='o')
        ax1.plot(x_axis, val_auc, 'r--', label='Val CTR AUC', marker='x')
    
    ax1.set_title('CTR ROC AUC')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('AUC')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    # 辅助显示epoch.step_in_epoch
    if len(x_labels) == len(x_axis) and len(x_axis) < 50:
        ax1.set_xticks(x_axis)
        ax1.set_xticklabels(x_labels, rotation=45, fontsize=8)
    
    # 2. Playtime Log-MAE 曲线
    ax2 = axes[0,1]
    if has_treatment_control:
        # 实验组和对照组分开显示
        tr_logmae_treatment = [m['train']['treatment'].get('LogMAE_play_time') if m and 'train' in m and 'treatment' in m['train'] else None for m in metrics]
        tr_logmae_control = [m['train']['control'].get('LogMAE_play_time') if m and 'train' in m and 'control' in m['train'] else None for m in metrics]
        val_logmae_treatment = [m['val']['treatment'].get('LogMAE_play_time') if m and 'val' in m and 'treatment' in m['val'] else None for m in metrics]
        val_logmae_control = [m['val']['control'].get('LogMAE_play_time') if m and 'val' in m and 'control' in m['val'] else None for m in metrics]
        
        ax2.plot(x_axis, tr_logmae_treatment, 'b-', label='Train LogMAE (Treatment)', marker='o')
        ax2.plot(x_axis, tr_logmae_control, 'g-', label='Train LogMAE (Control)', marker='s')
        ax2.plot(x_axis, val_logmae_treatment, 'b--', label='Val LogMAE (Treatment)', marker='x')
        ax2.plot(x_axis, val_logmae_control, 'g--', label='Val LogMAE (Control)', marker='+')
    else:
        # 统一显示
        tr_logmae = [m['train'].get('LogMAE_play_time') if m and 'train' in m else None for m in metrics]
        val_logmae = [m['val'].get('LogMAE_play_time') if m and 'val' in m else None for m in metrics]
        ax2.plot(x_axis, tr_logmae, 'b-', label='Train LogMAE', marker='o')
        ax2.plot(x_axis, val_logmae, 'r--', label='Val LogMAE', marker='x')
    
    ax2.set_title('Playtime Log-MAE')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Log-MAE')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    if len(x_labels) == len(x_axis) and len(x_axis) < 50:
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels(x_labels, rotation=45, fontsize=8)
    
    # 3. Weight-Model 指标曲线
    ax3 = axes[1,0]
    if has_weight_model:
        w_accuracy = [m['train']['weight_model'].get('accuracy') if m and 'train' in m and 'weight_model' in m['train'] else None for m in metrics]
        w_auc = [m['train']['weight_model'].get('auc') if m and 'train' in m and 'weight_model' in m['train'] else None for m in metrics]
        
        ax3.plot(x_axis, w_accuracy, 'b-', label='Weight Model Accuracy', marker='o')
        ax3.plot(x_axis, w_auc, 'g-', label='Weight Model AUC', marker='s')
        
        ax3.set_title('Weight Model Performance')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        if len(x_labels) == len(x_axis) and len(x_axis) < 50:
            ax3.set_xticks(x_axis)
            ax3.set_xticklabels(x_labels, rotation=45, fontsize=8)
    else:
        ax3.set_title('Weight Model Performance (Not Available)')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Score')
    
    # 4. Weight-Model F1/Precision/Recall
    ax4 = axes[1,1]
    if has_weight_model:
        w_precision = [m['train']['weight_model'].get('precision') if m and 'train' in m and 'weight_model' in m['train'] else None for m in metrics]
        w_recall = [m['train']['weight_model'].get('recall') if m and 'train' in m and 'weight_model' in m['train'] else None for m in metrics]
        w_f1 = [m['train']['weight_model'].get('f1') if m and 'train' in m and 'weight_model' in m['train'] else None for m in metrics]
        
        ax4.plot(x_axis, w_precision, 'b-', label='Weight Model Precision', marker='o')
        ax4.plot(x_axis, w_recall, 'g-', label='Weight Model Recall', marker='s')
        ax4.plot(x_axis, w_f1, 'r-', label='Weight Model F1', marker='^')
        
        ax4.set_title('Weight Model Classification Metrics')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
        if len(x_labels) == len(x_axis) and len(x_axis) < 50:
            ax4.set_xticks(x_axis)
            ax4.set_xticklabels(x_labels, rotation=45, fontsize=8)
    else:
        ax4.set_title('Weight Model Classification Metrics (Not Available)')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Score')
    
    plt.tight_layout()
    save_path = os.path.join(result_dir, 'exp_results_plot.png')
    plt.savefig(save_path, dpi=200)
    print(f"图像已保存到: {save_path}")
    return save_path

def plot_async(result_dir):
    """以非阻塞方式创建图表"""
    import subprocess
    import sys
    
    # 使用当前Python解释器启动新进程
    python_exe = sys.executable
    cmd = [python_exe, __file__, result_dir, '--async']
    subprocess.Popen(cmd)
    return None

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
        is_async = '--async' in sys.argv
    else:
        result_dir = os.path.dirname(__file__)
        is_async = False
    
    # 创建图表
    try:
        save_path = create_plots(result_dir)
        if not is_async:
            plt.show()
    except Exception as e:
        print(f"创建图表时出错: {e}")
