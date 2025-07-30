
import os
import json
import logging
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def compute_roc_auc(y_true, y_score):
    """
    y_true: 1D array, 0/1
    y_score: 1D array, 概率
    返回: fpr, tpr, auc
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    unique_labels, counts = np.unique(y_true, return_counts=True)
    if unique_labels.shape[0] < 2:
        print("[AUC调试] y_true类别数不足2，实际为:", unique_labels.tolist())
        print("[AUC调试] y_true分布:", dict(zip(unique_labels.tolist(), counts.tolist())))
        print("[AUC调试] y_true前20个样本:", y_true[:20].tolist())
        raise ValueError(f"AUC计算失败：y_true类别数不足2，实际为{unique_labels.tolist()}。请检查数据采样/分割/预处理流程。")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc

def compute_log_mae(y_true, y_pred):
    """
    y_true, y_pred: 1D array, play_time_ms
    对 play_time 取 log1p 后计算 MAE
    """
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    return np.mean(np.abs(y_true_log - y_pred_log))

import logging
import os
import torch


def setup_logging(name, log_dir=None, log_file='experiment.log'):
    """设置日志
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录，默认为结果目录
        log_file: 日志文件名，默认为experiment.log
    """
    import logging
    import os
    
    if log_dir is None:
        # 日志目录统一到 /home/zhixuanhu/IEDA_WeightedTraining/results/logs/
        log_dir = '/home/zhixuanhu/IEDA_WeightedTraining/results/logs/'
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建handler
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    ch = logging.StreamHandler()
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
    return logging.getLogger()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def compute_metrics(y_true, y_pred):
    """
    计算常用指标：点击率、均方误差、长播率等
    y_true, y_pred: [N, 2] (is_click, play_time_ms)
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    click_true = y_true[:, 0]
    click_pred = torch.sigmoid(y_pred[:, 0])
    play_true = y_true[:, 1]
    play_pred = y_pred[:, 1]
    ctr = click_true.mean().item()
    ctr_pred = click_pred.mean().item()
    mse = torch.mean((play_true - play_pred) ** 2).item()
    long_view_true = (play_true > 10000).float().mean().item() # 10秒阈值
    long_view_pred = (play_pred > 10000).float().mean().item()
    return {
        'CTR_true': ctr,
        'CTR_pred': ctr_pred,
        'MSE_play_time': mse,
        'LongView_true': long_view_true,
        'LongView_pred': long_view_pred,
        'play_time_true': play_true.mean().item(),
        'play_time_pred': play_pred.mean().item()
    }
