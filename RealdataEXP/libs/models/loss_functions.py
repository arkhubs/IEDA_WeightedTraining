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
        
        # 对于播放时长为0的情况，使用特殊处理
        # 只有当目标值大于epsilon时才使用log变换
        mask = target > self.epsilon
        
        if mask.sum() == 0:
            # 如果所有目标值都是0（或接近0），使用普通MAE
            loss = F.l1_loss(pred, target)
        else:
            # 对大于0的值使用log变换，对接近0的值使用普通MAE
            if mask.all():
                # 所有值都大于0，使用log变换
                log_pred = torch.log(pred + self.epsilon)
                log_target = torch.log(target + self.epsilon)
                loss = F.l1_loss(log_pred, log_target)
            else:
                # 混合情况
                mask = mask.squeeze()
                if pred.dim() > 1:
                    pred = pred.squeeze()
                if target.dim() > 1:
                    target = target.squeeze()
                
                # 对大于0的样本使用log变换
                if mask.sum() > 0:
                    log_pred = torch.log(pred[mask] + self.epsilon)
                    log_target = torch.log(target[mask] + self.epsilon)
                    loss_positive = F.l1_loss(log_pred, log_target)
                else:
                    loss_positive = torch.tensor(0.0, device=pred.device)
                
                # 对接近0的样本使用普通MAE
                if (~mask).sum() > 0:
                    loss_zero = F.l1_loss(pred[~mask], target[~mask])
                else:
                    loss_zero = torch.tensor(0.0, device=pred.device)
                
                # 加权平均
                loss = (mask.sum() * loss_positive + (~mask).sum() * loss_zero) / len(mask)
        
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
    elif loss_name.lower() == 'huber':
        return nn.HuberLoss(**kwargs)
    elif loss_name.lower() == 'crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")