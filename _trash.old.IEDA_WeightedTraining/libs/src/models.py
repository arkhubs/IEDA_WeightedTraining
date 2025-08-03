import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.nn.init import xavier_normal_

class MLPModel(nn.Module):
    """基于MLP的预测模型"""
    
    def __init__(self, config, feature_info):
        """
        初始化MLP模型
        
        Args:
            config: 模型配置
            feature_info: 特征信息
        """
        super(MLPModel, self).__init__()
        self.config = config
        self.feature_info = feature_info
        
        # 特征信息
        self.numerical_features = feature_info['numerical']
        self.categorical_features = feature_info['categorical']
        self.num_numerical = len(self.numerical_features)
        self.num_categorical = len(self.categorical_features)
        self.categorical_encoders = feature_info['categorical_encoders']
        
        # 配置参数
        self.hidden_layers = config['hidden_layers']
        self.dropout_rate = config['dropout']
        self.embedding_dim = config['embedding_dim']
        
        # 计算分类特征的唯一值数量
        self.categorical_dims = []
        for feat in self.categorical_features:
            if feat in self.categorical_encoders:
                encoder = self.categorical_encoders[feat]
                if hasattr(encoder, "classes_"):
                    dim = len(encoder.classes_)
                    self.categorical_dims.append(dim)
                else:
                    # 兜底：用唯一值数量或1
                    self.categorical_dims.append(1)
            else:
                self.categorical_dims.append(1)  # 默认维度
        
        # 构建嵌入层
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(dim, self.embedding_dim)
            for dim in self.categorical_dims
        ])
        
        # 计算输入维度
        self.input_dim = self.num_numerical + self.num_categorical * self.embedding_dim
        
        # 构建MLP层
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_uniform_(m.weight.data, gain=0.1)  # 减小初始权重
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                # 嵌入层使用正态分布初始化，标准差更小
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                
    def forward(self, numerical_features, categorical_features):
        """
        前向传播
        
        Args:
            numerical_features: 数值特征 [batch_size, num_numerical]
            categorical_features: 分类特征 [batch_size, num_categorical]
            
        Returns:
            预测结果
        """
        batch_size = numerical_features.shape[0]
        
        # 处理数值特征
        numerical_output = numerical_features
        
        # 处理分类特征
        categorical_outputs = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            if i < categorical_features.shape[1]:  # 防止索引越界
                categorical_feature = categorical_features[:, i]
                embedding_output = embedding_layer(categorical_feature)
                categorical_outputs.append(embedding_output)
        
        # 如果有分类特征，则拼接所有嵌入输出
        if categorical_outputs:
            categorical_output = torch.cat(categorical_outputs, dim=1)
            categorical_output = categorical_output.view(batch_size, -1)
            # 拼接数值特征和分类特征
            combined_features = torch.cat([numerical_output, categorical_output], dim=1)
        else:
            combined_features = numerical_output
        
        # 通过MLP层
        output = self.mlp(combined_features)
        
        return output.squeeze()


class DeepFMModel(nn.Module):
    """DeepFM模型，结合了FM和深度神经网络"""
    
    def __init__(self, config, feature_info):
        """
        初始化DeepFM模型
        
        Args:
            config: 模型配置
            feature_info: 特征信息
        """
        super(DeepFMModel, self).__init__()
        self.config = config
        self.feature_info = feature_info
        
        # 特征信息
        self.numerical_features = feature_info['numerical']
        self.categorical_features = feature_info['categorical']
        self.num_numerical = len(self.numerical_features)
        self.num_categorical = len(self.categorical_features)
        self.categorical_encoders = feature_info['categorical_encoders']
        
        # 配置参数
        self.hidden_layers = config['hidden_layers']
        self.dropout_rate = config['dropout']
        self.embedding_dim = config['embedding_dim']
        
        # 计算分类特征的唯一值数量
        self.categorical_dims = []
        for feat in self.categorical_features:
            if feat in self.categorical_encoders:
                self.categorical_dims.append(len(self.categorical_encoders[feat].classes_))
            else:
                self.categorical_dims.append(1)  # 默认维度
        
        # 计算特征总数
        self.num_features = self.num_numerical + self.num_categorical
        
        # FM部分
        # 一阶特征线性部分
        self.first_order_numerical = nn.Linear(self.num_numerical, 1)
        self.first_order_embeddings = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in self.categorical_dims
        ])
        
        # 二阶交叉部分的嵌入
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(dim, self.embedding_dim)
            for dim in self.categorical_dims
        ])
        
        # 数值特征的嵌入变换
        self.numerical_embeddings = nn.Linear(self.num_numerical, self.embedding_dim)
        
        # DNN部分
        dnn_input_dim = self.num_numerical + (self.num_categorical * self.embedding_dim)
        layers = []
        prev_dim = dnn_input_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        self.dnn = nn.Sequential(*layers)
        self.dnn_output = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_uniform_(m.weight.data, gain=0.1)  # 减小初始权重
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                # 嵌入层使用正态分布初始化，标准差更小
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                
    def forward(self, numerical_features, categorical_features):
        """
        前向传播
        
        Args:
            numerical_features: 数值特征 [batch_size, num_numerical]
            categorical_features: 分类特征 [batch_size, num_categorical]
            
        Returns:
            预测结果
        """
        batch_size = numerical_features.shape[0]
        
        # FM部分 - 一阶线性
        first_order = self.first_order_numerical(numerical_features)
        for i, embedding_layer in enumerate(self.first_order_embeddings):
            if i < categorical_features.shape[1]:  # 防止索引越界
                categorical_feature = categorical_features[:, i]
                first_order += embedding_layer(categorical_feature).sum(dim=1, keepdim=True)
        
        # FM部分 - 二阶交叉
        # 处理数值特征的嵌入
        numerical_embedding = self.numerical_embeddings(numerical_features)  # [batch_size, embedding_dim]
        
        # 处理分类特征的嵌入
        categorical_embeddings = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            if i < categorical_features.shape[1]:  # 防止索引越界
                categorical_feature = categorical_features[:, i]
                embedding = embedding_layer(categorical_feature)  # [batch_size, embedding_dim]
                categorical_embeddings.append(embedding)
        
        # 组合所有嵌入
        all_embeddings = [numerical_embedding]
        if categorical_embeddings:
            all_embeddings.extend(categorical_embeddings)
        
        # 计算二阶交互项
        sum_embeddings = torch.sum(torch.stack(all_embeddings), dim=0)  # [batch_size, embedding_dim]
        sum_squared = torch.sum(torch.stack([emb ** 2 for emb in all_embeddings]), dim=0)  # [batch_size, embedding_dim]
        second_order = 0.5 * torch.sum(sum_embeddings ** 2 - sum_squared, dim=1, keepdim=True)
        
        # DNN部分
        dnn_input = torch.cat([numerical_features] + [emb.view(batch_size, -1) for emb in categorical_embeddings], dim=1)
        dnn_output = self.dnn(dnn_input)
        dnn_output = self.dnn_output(dnn_output)
        
        # 组合输出
        output = first_order + second_order + dnn_output
        
        return output.squeeze()


def create_model(config, feature_info):
    """
    创建模型
    
    Args:
        config: 模型配置
        feature_info: 特征信息
        
    Returns:
        模型实例
    """
    model_type = config.get('type', 'MLP').upper()
    
    if model_type == 'MLP':
        return MLPModel(config, feature_info)
    elif model_type == 'DEEPFM':
        return DeepFMModel(config, feature_info)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


class LogMAELoss(nn.Module):
    """
    Log Mean Absolute Error损失函数
    对于play_time等量级较大的目标更适合
    """
    
    def __init__(self, eps=1e-8):
        """
        初始化LogMAE损失
        
        Args:
            eps: 平滑参数，防止log(0)，使用更小的值以减少对数值的影响
        """
        super(LogMAELoss, self).__init__()
        self.eps = eps
        
    def forward(self, pred, target):
        """
        计算损失
        
        Args:
            pred: 预测值
            target: 真实值
            
        Returns:
            损失值
        """
        # 防止负值和异常值
        pred = torch.clamp(pred, min=0.0, max=1e6)  # 限制最大值防止溢出
        target = torch.clamp(target, min=0.0, max=1e6)
        
        # 检查是否有nan或inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"[LogMAE] 预测值包含nan或inf: nan={torch.isnan(pred).sum()}, inf={torch.isinf(pred).sum()}")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=0.0)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"[LogMAE] 目标值包含nan或inf: nan={torch.isnan(target).sum()}, inf={torch.isinf(target).sum()}")
            target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=0.0)
        
        # 计算对数MAE
        log_pred = torch.log(pred + self.eps)
        log_target = torch.log(target + self.eps)
        
        # 检查log结果
        if torch.isnan(log_pred).any() or torch.isinf(log_pred).any():
            print(f"[LogMAE] log_pred包含nan或inf")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        if torch.isnan(log_target).any() or torch.isinf(log_target).any():
            print(f"[LogMAE] log_target包含nan或inf")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        loss = F.l1_loss(log_pred, log_target)
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[LogMAE] 最终损失为nan或inf，返回0")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        return loss
