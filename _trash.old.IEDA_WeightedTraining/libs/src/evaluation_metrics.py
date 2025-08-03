import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_non_zero_relative_error(predictions, true_values):
    """
    计算非零值的相对误差 (只考虑true_values > 0的部分)
    
    Args:
        predictions: 预测值数组
        true_values: 真实值数组
        
    Returns:
        平均相对误差（百分比）
    """
    # 将输入转换为numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
        
    predictions = np.array(predictions).flatten()
    true_values = np.array(true_values).flatten()
    
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

def calculate_binary_metrics(predictions, true_values):
    """
    计算二分类指标（AUC、准确率等）
    
    Args:
        predictions: 预测值数组（logits，未经sigmoid）
        true_values: 真实值数组（0或1）
        
    Returns:
        指标字典
    """
    # 将输入转换为numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
        
    predictions = np.array(predictions).flatten()
    true_values = np.array(true_values).flatten()
    
    # 检查数据有效性
    if len(predictions) == 0 or len(true_values) == 0:
        return {'auc': 0.5, 'accuracy': 0.0}
    
    # 将logits转换为概率
    from scipy.special import expit  # sigmoid函数
    probabilities = expit(predictions)  # sigmoid(logits)
    
    # 计算AUC（使用概率）
    try:
        # 检查是否只有一个类别
        if len(np.unique(true_values)) < 2:
            auc = 0.5  # 只有一个类别时AUC无意义
        else:
            auc = roc_auc_score(true_values, probabilities)
    except Exception as e:
        print(f"[AUC计算错误] {e}")
        auc = 0.5
    
    # 计算准确率（使用概率 > 0.5的阈值）
    binary_preds = (probabilities > 0.5).astype(int)
    accuracy = np.mean(binary_preds == true_values)
    
    # 调试信息
    print(f"[Binary Metrics] logits范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"[Binary Metrics] 概率范围: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"[Binary Metrics] 真实标签分布: {np.bincount(true_values.astype(int))}")
    print(f"[Binary Metrics] 预测标签分布: {np.bincount(binary_preds)}")
    
    return {
        'auc': auc,
        'accuracy': accuracy
    }
