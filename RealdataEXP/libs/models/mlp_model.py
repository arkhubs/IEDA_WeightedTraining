"""
MLP模型实现
用于单个标签的预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLPModel(nn.Module):
    """多层感知机模型"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], 
                 output_dim: int = 1, dropout: float = 0.1):
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'total_params': total_params,
            'trainable_params': trainable_params
        }