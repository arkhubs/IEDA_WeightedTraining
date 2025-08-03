"""
模型定义模块，包含预测模型和权重模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MLP(nn.Module):
    """通用的多层感知机实现"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation=nn.ReLU, dropout: float = 0.0, output_activation=None):
        """
        初始化多层感知机
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数，默认为ReLU
            dropout: Dropout概率
            output_activation: 输出层激活函数，默认为None
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)


class PredictionModel(nn.Module):
    """预测模型，用于预测CTR和播放时间"""
    
    def __init__(self, config):
        """
        初始化预测模型
        
        Args:
            config: 模型配置字典，包含input_dim、hidden_dims等参数
        """
        super(PredictionModel, self).__init__()
        
        # 从配置中获取参数
        input_dim = config.get('input_dim', 64)  # 默认64维输入
        hidden_dims = config.get('hidden_dims', [128, 64, 32])  # 默认隐藏层维度
        shared_dims = hidden_dims
        
        # 头部网络的维度，可以根据需要调整
        ctr_head_dims = config.get('ctr_head_dims', [16])
        playtime_head_dims = config.get('playtime_head_dims', [16])
        
        # 共享表示层
        self.shared_network = MLP(
            input_dim=input_dim,
            hidden_dims=shared_dims,
            output_dim=shared_dims[-1] if shared_dims else input_dim,
            dropout=0.1
        )
        
        shared_output_dim = shared_dims[-1] if shared_dims else input_dim
        
        # CTR预测头
        self.ctr_head = MLP(
            input_dim=shared_output_dim,
            hidden_dims=ctr_head_dims,
            output_dim=1,
            dropout=0.1
        )
        
        # 播放时间预测头
        self.playtime_head = MLP(
            input_dim=shared_output_dim,
            hidden_dims=playtime_head_dims,
            output_dim=1,
            dropout=0.1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征
        
        Returns:
            ctr_pred: CTR预测
            playtime_pred: 播放时间预测
        """
        shared_features = self.shared_network(x)
        ctr_pred = self.ctr_head(shared_features).squeeze(-1)
        playtime_pred = self.playtime_head(shared_features).squeeze(-1)
        
        return ctr_pred, playtime_pred


class WeightModel(nn.Module):
    """权重模型，用于预测样本权重"""
    
    def __init__(self, config):
        """
        初始化权重模型
        
        Args:
            config: 模型配置字典，包含input_dim和hidden_dims等参数
        """
        super(WeightModel, self).__init__()
        
        input_dim = config.get('input_dim', 64)  # 默认64维输入
        hidden_dims = config.get('hidden_dims', [32, 16])  # 默认隐藏层维度
        
        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=0.1,
            output_activation=nn.Sigmoid
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，预测处理组的概率
        
        Args:
            x: 输入特征
        
        Returns:
            预测的处理组概率
        """
        return self.model(x).squeeze(-1)
