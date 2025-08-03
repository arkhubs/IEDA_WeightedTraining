"""
实用工具模块，提供日志、指标计算、结果保存等功能
"""
import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def setup_logging(name: str = 'experiment', 
                 log_dir: str = None, 
                 log_file: str = 'experiment.log',
                 level: int = logging.INFO) -> logging.Logger:
    """
    设置日志
    
    Args:
        name: 日志名称
        log_dir: 日志目录
        log_file: 日志文件名
        level: 日志级别
    
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    # 如果指定了日志目录，添加文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签，第一列是CTR，第二列是播放时间
        y_pred: 预测值，第一列是CTR，第二列是播放时间
    
    Returns:
        评估指标字典
    """
    metrics = {}
    
    # CTR指标（二分类）
    y_true_ctr = y_true[:, 0]
    y_pred_ctr = torch.sigmoid(torch.tensor(y_pred[:, 0])).numpy() if isinstance(y_pred, np.ndarray) else y_pred[:, 0].detach().cpu().numpy()
    
    # 设定阈值进行二分类
    y_pred_ctr_binary = (y_pred_ctr > 0.5).astype(int)
    
    try:
        metrics['ctr_accuracy'] = accuracy_score(y_true_ctr, y_pred_ctr_binary)
        metrics['ctr_precision'] = precision_score(y_true_ctr, y_pred_ctr_binary, zero_division=0)
        metrics['ctr_recall'] = recall_score(y_true_ctr, y_pred_ctr_binary, zero_division=0)
        metrics['ctr_f1'] = f1_score(y_true_ctr, y_pred_ctr_binary, zero_division=0)
        
        # 如果存在不同的类别，计算AUC
        if len(np.unique(y_true_ctr)) > 1:
            metrics['ctr_auc'] = roc_auc_score(y_true_ctr, y_pred_ctr)
        else:
            metrics['ctr_auc'] = 0.5
    except Exception as e:
        print(f"计算CTR指标时出错: {e}")
        metrics['ctr_accuracy'] = 0.0
        metrics['ctr_precision'] = 0.0
        metrics['ctr_recall'] = 0.0
        metrics['ctr_f1'] = 0.0
        metrics['ctr_auc'] = 0.5
    
    # 播放时间指标（回归）
    y_true_playtime = y_true[:, 1]
    y_pred_playtime = y_pred[:, 1]
    
    # 均方误差
    metrics['playtime_mse'] = np.mean((y_true_playtime - y_pred_playtime) ** 2)
    
    # 对数均方误差(LogMAE)
    epsilon = 1e-8
    metrics['playtime_logmae'] = np.mean(np.abs(np.log(y_pred_playtime + epsilon) - np.log(y_true_playtime + epsilon)))
    
    return metrics


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算ROC曲线和AUC值
    
    Args:
        y_true: 真实二分类标签
        y_score: 预测概率或分数
    
    Returns:
        fpr: 假阳性率
        tpr: 真阳性率
        auc: AUC值
    """
    from sklearn.metrics import roc_curve, auc
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


def compute_log_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算对数平均绝对误差(LogMAE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        LogMAE值
    """
    epsilon = 1e-8
    return np.mean(np.abs(np.log(y_pred + epsilon) - np.log(y_true + epsilon)))


class Logger:
    """
    日志记录器类，封装了日志记录功能
    """
    
    def __init__(self, log_file=None, log_level=logging.INFO):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径，如果为None则只输出到控制台
            log_level: 日志级别
        """
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # 清除现有处理器
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg):
        """记录信息级别的日志"""
        self.logger.info(msg)
    
    def warning(self, msg):
        """记录警告级别的日志"""
        self.logger.warning(msg)
    
    def error(self, msg):
        """记录错误级别的日志"""
        self.logger.error(msg)
    
    def debug(self, msg):
        """记录调试级别的日志"""
        self.logger.debug(msg)


def setup_seed(seed: int) -> None:
    """
    设置随机种子，确保实验可重复
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_results(results: Dict[str, Any], path: str):
    """
    保存实验结果
    
    Args:
        results: 结果字典
        path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 将numpy和torch值转换为Python标准类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    # 保存为JSON
    with open(path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def save_model(model: torch.nn.Module, path: str):
    """
    保存模型
    
    Args:
        model: 模型
        path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), path)


def plot_results(results: Dict[str, Any], save_path: str):
    """
    绘制实验结果
    
    Args:
        results: 结果字典，包含'step'和'metrics'键
        save_path: 保存路径
    """
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
    plt.savefig(save_path)
    plt.close()
    
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
    auc_save_path = os.path.join(os.path.dirname(save_path), 'auc_curve.png')
    plt.savefig(auc_save_path)
    plt.close()


def plot_async(results_path: str):
    """
    异步绘图，在子进程中运行
    
    Args:
        results_path: 结果文件路径，应为JSON格式
    """
    import subprocess
    import sys
    
    plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_results.py')
    
    # 如果脚本不存在，创建它
    if not os.path.exists(plot_script):
        with open(plot_script, 'w') as f:
            f.write("""#!/usr/bin/env python
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_path):
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python plot_results.py <results_json_path>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    plot_results(results_path)
""")
    
    # 构建结果文件路径
    results_json_path = os.path.join(results_path, 'exp_results.json')
    
    # 启动子进程执行绘图
    subprocess.Popen([sys.executable, plot_script, results_json_path],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
