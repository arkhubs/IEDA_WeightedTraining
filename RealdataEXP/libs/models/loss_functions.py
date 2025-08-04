"""
损失函数定义
包含LogMAE和其他自定义损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogMAELoss(nn.Module):
    """Log Mean Absolute Error损失函数
    用于播放时长等具有大数值范围的连续标签
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super(LogMAELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算LogMAE损失
        
        Args:
            pred: 预测值 [batch_size, 1]
            target: 真实值 [batch_size, 1]
        
        Returns:
            loss: LogMAE损失值
        """
        # 确保预测值和目标值都是正数
        pred = torch.clamp(pred, min=self.epsilon)
        target = torch.clamp(target, min=self.epsilon)
        
        # 计算log后的MAE
        log_pred = torch.log(pred + self.epsilon)
        log_target = torch.log(target + self.epsilon)
        
        loss = F.l1_loss(log_pred, log_target)
        return loss

def get_loss_function(loss_name: str, **kwargs):
    """获取损失函数"""
    if loss_name.lower() == 'logmae':
        return LogMAELoss(**kwargs)
    elif loss_name.lower() == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name.lower() == 'mse':
        return nn.MSELoss(**kwargs)
    elif loss_name.lower() == 'mae':
        return nn.L1Loss(**kwargs)
    elif loss_name.lower() == 'crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")