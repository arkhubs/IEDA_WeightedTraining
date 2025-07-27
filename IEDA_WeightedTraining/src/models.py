import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 特征归一化
        x_norm = self.bn(x)
        # 调试：输入特征分布
        if isinstance(x_norm, torch.Tensor):
            x_np = x_norm.detach().cpu().numpy()
            print(f"[PredictionModel] 归一化后输入: mean={x_np.mean():.4f}, std={x_np.std():.4f}, min={x_np.min():.4f}, max={x_np.max():.4f}")
        out = self.net(x_norm)
        pred_click_logit = out[:, 0]
        pred_play_time = out[:, 1]
        # 输出 clip，防止极端
        pred_click_logit = torch.clamp(pred_click_logit, -10, 10)
        # 调试：输出分布
        if isinstance(pred_click_logit, torch.Tensor):
            click_np = pred_click_logit.detach().cpu().numpy()
            print(f"[PredictionModel] click_logit(clipped): mean={click_np.mean():.4f}, std={click_np.std():.4f}, min={click_np.min():.4f}, max={click_np.max():.4f}")
        if isinstance(pred_play_time, torch.Tensor):
            time_np = pred_play_time.detach().cpu().numpy()
            print(f"[PredictionModel] play_time: mean={time_np.mean():.4f}, std={time_np.std():.4f}, min={time_np.min():.4f}, max={time_np.max():.4f}")
        return pred_click_logit, pred_play_time

    def loss_function(self, pred_click_logit, pred_play_time, Y, weights=None):
        # 直接用原始 play_time
        safe_weights = weights.detach() if weights is not None and weights.requires_grad else weights
        loss_click = F.binary_cross_entropy_with_logits(pred_click_logit, Y[:, 0], weight=safe_weights)
        loss_time = F.mse_loss(pred_play_time, Y[:, 1], reduction='none')
        if weights is not None:
            loss_time = (loss_time * safe_weights).mean()
        else:
            loss_time = loss_time.mean()
        return loss_click + loss_time

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
        # 调试：输入特征分布
        if isinstance(x_norm, torch.Tensor):
            x_np = x_norm.detach().cpu().numpy()
            print(f"[WeightingModel] 归一化后输入: mean={x_np.mean():.4f}, std={x_np.std():.4f}, min={x_np.min():.4f}, max={x_np.max():.4f}")
        # logits 分布
        logits = self.net(x_norm).squeeze(-1)
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
            print(f"[WeightingModel] logits: mean={logits_np.mean():.4f}, std={logits_np.std():.4f}, min={logits_np.min():.4f}, max={logits_np.max():.4f}")
        out = torch.sigmoid(logits)
        print(f"[WeightingModel] sigmoid输出范围: min={out.min().item()}, max={out.max().item()}")
        return out
