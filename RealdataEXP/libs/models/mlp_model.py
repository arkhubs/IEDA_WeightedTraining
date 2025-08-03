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
                 output_dim: int = 1, dropout: float = 0.1, 
                 batch_norm: bool = False, residual: bool = False):
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        prev_dim = input_dim
        
        # 隐藏层
        for i, hidden_dim in enumerate(hidden_layers):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # 对于残差连接，需要维度匹配
            if residual and i > 0 and prev_dim != hidden_dim:
                self.residual_proj = nn.Linear(prev_dim, hidden_dim)
            else:
                self.residual_proj = None
                
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
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
        current_input = x
        
        for i, layer in enumerate(self.layers):
            # 线性变换
            out = layer(current_input)
            
            # 批归一化
            if self.batch_norm and self.batch_norms is not None:
                out = self.batch_norms[i](out)
            
            # 激活函数
            out = F.relu(out)
            
            # Dropout
            out = F.dropout(out, p=self.dropout, training=self.training)
            
            # 残差连接
            if self.residual and i > 0:
                if current_input.shape[-1] == out.shape[-1]:
                    out = out + current_input
                elif self.residual_proj is not None:
                    out = out + self.residual_proj(current_input)
            
            current_input = out
        
        # 输出层
        output = self.output_layer(current_input)
        
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'residual': self.residual,
            'total_params': total_params,
            'trainable_params': trainable_params
        }