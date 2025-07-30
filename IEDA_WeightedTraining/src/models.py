import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionModel(nn.Module):
    def __init__(self, input_dim, shared_dims, ctr_head_dims, playtime_head_dims):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        # 共享底座
        shared_layers = []
        dims = [input_dim] + shared_dims
        for i in range(len(dims)-1):
            shared_layers.append(nn.Linear(dims[i], dims[i+1]))
            shared_layers.append(nn.ReLU())
        self.shared = nn.Sequential(*shared_layers)
        # CTR分支
        ctr_layers = []
        dims = [shared_dims[-1]] + ctr_head_dims
        for i in range(len(dims)-1):
            ctr_layers.append(nn.Linear(dims[i], dims[i+1]))
            ctr_layers.append(nn.ReLU())
        ctr_layers.append(nn.Linear(ctr_head_dims[-1], 1))
        self.ctr_head = nn.Sequential(*ctr_layers)
        # Playtime分支
        playtime_layers = []
        dims = [shared_dims[-1]] + playtime_head_dims
        for i in range(len(dims)-1):
            playtime_layers.append(nn.Linear(dims[i], dims[i+1]))
            playtime_layers.append(nn.ReLU())
        playtime_layers.append(nn.Linear(playtime_head_dims[-1], 1))
        self.playtime_head = nn.Sequential(*playtime_layers)

    def forward(self, x):
        x = self.bn(x)
        shared = self.shared(x)
        ctr = self.ctr_head(shared).squeeze(-1)
        playtime = self.playtime_head(shared).squeeze(-1)
        # 输出 clip，防止极端
        ctr = torch.clamp(ctr, -10, 10)
        return ctr, playtime

    def loss_function(self, pred_click_logit, pred_play_time, Y, weights=None, loss_weights=None):
        # 获取损失权重
        if loss_weights is None:
            ctr_weight, playtime_weight = 1.0, 1.0
        else:
            ctr_weight = loss_weights.get('ctr_weight', 1.0)
            playtime_weight = loss_weights.get('playtime_weight', 1.0)
            
        # 直接用原始 play_time
        safe_weights = weights.detach() if weights is not None and weights.requires_grad else weights
        loss_click = F.binary_cross_entropy_with_logits(pred_click_logit, Y[:, 0], weight=safe_weights)
        loss_time = F.mse_loss(pred_play_time, Y[:, 1], reduction='none')
        if weights is not None:
            loss_time = (loss_time * safe_weights).mean()
        else:
            loss_time = loss_time.mean()
        
        # 应用损失权重
        total_loss = ctr_weight * loss_click + playtime_weight * loss_time
        return total_loss, loss_click, loss_time


class CTRModel(nn.Module):
    """独立的CTR预测模型"""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # 最后一层不加激活函数
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn(x)
        logit = self.model(x).squeeze(-1)
        return torch.clamp(logit, -10, 10)
    
    def loss_function(self, pred_logit, Y_click, weights=None):
        safe_weights = weights.detach() if weights is not None and weights.requires_grad else weights
        return F.binary_cross_entropy_with_logits(pred_logit, Y_click, weight=safe_weights)


class PlaytimeModel(nn.Module):
    """独立的播放时间预测模型"""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # 最后一层不加激活函数
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn(x)
        playtime = self.model(x).squeeze(-1)
        return F.relu(playtime)  # 播放时间不能为负
    
    def loss_function(self, pred_playtime, Y_playtime, weights=None):
        loss = F.mse_loss(pred_playtime, Y_playtime, reduction='none')
        if weights is not None:
            safe_weights = weights.detach() if weights is not None and weights.requires_grad else weights
            loss = (loss * safe_weights).mean()
        else:
            loss = loss.mean()
        return loss


class WeightingModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 特征归一化
        x_norm = self.bn(x)
        # logits 分布
        logits = self.net(x_norm).squeeze(-1)
        out = torch.sigmoid(logits)
        return out
