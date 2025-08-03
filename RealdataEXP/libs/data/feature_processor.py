"""
特征处理器
负责特征的预处理、编码和标准化
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """特征处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.numerical_features = config['feature']['numerical']
        self.categorical_features = config['feature']['categorical']
        
        # 编码器和缩放器
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_mappings = {}
        
        # 处理后的特征维度信息
        self.total_numerical_dim = 0
        self.categorical_dims = {}
        self.total_categorical_dim = 0
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        logger.info("[特征处理] 处理缺失值...")
        
        processed_data = data.copy()
        
        # 数值特征：用0填充
        for feature in self.numerical_features:
            if feature in processed_data.columns:
                missing_count = processed_data[feature].isna().sum()
                if missing_count > 0:
                    logger.info(f"[特征处理] {feature}: 用0填充 {missing_count} 个缺失值")
                    processed_data[feature] = processed_data[feature].fillna(0)
        
        # 分类特征：将NA作为新类别
        for feature in self.categorical_features:
            if feature in processed_data.columns:
                missing_count = processed_data[feature].isna().sum()
                if missing_count > 0:
                    logger.info(f"[特征处理] {feature}: 将 {missing_count} 个缺失值标记为'MISSING'")
                    processed_data[feature] = processed_data[feature].fillna('MISSING')
        
        logger.info("[特征处理] 缺失值处理完成")
        return processed_data
    
    def _process_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """处理分类特征，转换为one-hot编码"""
        logger.info("[特征处理] 处理分类特征...")
        
        processed_data = data.copy()
        
        for feature in self.categorical_features:
            if feature not in processed_data.columns:
                logger.warning(f"[特征处理] 特征 {feature} 不存在于数据中")
                continue
                
            if fit:
                # 训练阶段：拟合编码器
                unique_values = processed_data[feature].unique()
                logger.info(f"[特征处理] {feature}: {len(unique_values)} 个唯一值")
                
                # 创建one-hot编码
                one_hot = pd.get_dummies(processed_data[feature], prefix=feature)
                self.categorical_mappings[feature] = one_hot.columns.tolist()
                self.categorical_dims[feature] = len(one_hot.columns)
                
                # 合并到主数据框
                processed_data = pd.concat([processed_data, one_hot], axis=1)
                processed_data = processed_data.drop(feature, axis=1)
                
                logger.info(f"[特征处理] {feature} -> {len(one_hot.columns)} 个one-hot特征")
            else:
                # 预测阶段：使用已有的编码器
                if feature in self.categorical_mappings:
                    one_hot = pd.get_dummies(processed_data[feature], prefix=feature)
                    
                    # 确保所有训练时的列都存在
                    for col in self.categorical_mappings[feature]:
                        if col not in one_hot.columns:
                            one_hot[col] = 0
                    
                    # 只保留训练时的列
                    one_hot = one_hot[self.categorical_mappings[feature]]
                    
                    # 合并到主数据框
                    processed_data = pd.concat([processed_data, one_hot], axis=1)
                    processed_data = processed_data.drop(feature, axis=1)
        
        self.total_categorical_dim = sum(self.categorical_dims.values())
        logger.info(f"[特征处理] 分类特征处理完成，总维度: {self.total_categorical_dim}")
        
        return processed_data
    
    def _process_numerical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """处理数值特征，进行标准化"""
        logger.info("[特征处理] 处理数值特征...")
        
        processed_data = data.copy()
        
        # 提取数值特征
        available_numerical = [f for f in self.numerical_features if f in processed_data.columns]
        missing_numerical = [f for f in self.numerical_features if f not in processed_data.columns]
        
        if missing_numerical:
            logger.warning(f"[特征处理] 缺失的数值特征: {missing_numerical}")
            # 只使用可用的数值特征
            self.numerical_features = available_numerical
        
        if available_numerical:
            if fit:
                # 训练阶段：拟合标准化器
                self.scaler.fit(processed_data[available_numerical])
                logger.info(f"[特征处理] 数值特征标准化器已拟合，特征数: {len(available_numerical)}")
            
            # 应用标准化
            processed_data[available_numerical] = self.scaler.transform(processed_data[available_numerical])
            self.total_numerical_dim = len(available_numerical)
            
            logger.info(f"[特征处理] 数值特征标准化完成，维度: {self.total_numerical_dim}")
        else:
            logger.warning("[特征处理] 没有可用的数值特征")
            self.total_numerical_dim = 0
        
        return processed_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换特征（训练阶段）"""
        logger.info("[特征处理] 开始特征拟合和转换...")
        
        # 处理缺失值
        processed_data = self._handle_missing_values(data)
        
        # 处理分类特征
        processed_data = self._process_categorical_features(processed_data, fit=True)
        
        # 处理数值特征
        processed_data = self._process_numerical_features(processed_data, fit=True)
        
        total_dim = self.total_numerical_dim + self.total_categorical_dim
        logger.info(f"[特征处理] 特征处理完成，总维度: {total_dim} (数值: {self.total_numerical_dim}, 分类: {self.total_categorical_dim})")
        
        # 确保所有特征列都是数值类型
        feature_columns = self.get_feature_columns()
        for col in feature_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
        
        logger.info("[特征处理] 数据类型转换完成")
        return processed_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换特征（预测阶段）"""
        logger.info("[特征处理] 应用已拟合的特征转换...")
        
        # 处理缺失值
        processed_data = self._handle_missing_values(data)
        
        # 处理分类特征
        processed_data = self._process_categorical_features(processed_data, fit=False)
        
        # 处理数值特征
        processed_data = self._process_numerical_features(processed_data, fit=False)
        
        # 确保所有特征列都是数值类型
        feature_columns = self.get_feature_columns()
        for col in feature_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
        
        logger.info("[特征处理] 特征转换完成")
        return processed_data
    
    def get_feature_columns(self) -> List[str]:
        """获取处理后的特征列名"""
        feature_columns = []
        
        # 数值特征列
        available_numerical = [f for f in self.numerical_features]
        feature_columns.extend(available_numerical)
        
        # 分类特征的one-hot列
        for feature in self.categorical_features:
            if feature in self.categorical_mappings:
                feature_columns.extend(self.categorical_mappings[feature])
        
        return feature_columns
    
    def save_processors(self, save_dir: str) -> None:
        """保存特征处理器"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存标准化器
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存分类特征映射
        with open(os.path.join(save_dir, 'categorical_mappings.pkl'), 'wb') as f:
            pickle.dump(self.categorical_mappings, f)
        
        # 保存维度信息
        dim_info = {
            'total_numerical_dim': self.total_numerical_dim,
            'categorical_dims': self.categorical_dims,
            'total_categorical_dim': self.total_categorical_dim
        }
        with open(os.path.join(save_dir, 'dim_info.pkl'), 'wb') as f:
            pickle.dump(dim_info, f)
            
        logger.info(f"[特征处理] 处理器已保存到: {save_dir}")
    
    def load_processors(self, save_dir: str) -> None:
        """加载特征处理器"""
        # 加载标准化器
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 加载分类特征映射
        with open(os.path.join(save_dir, 'categorical_mappings.pkl'), 'rb') as f:
            self.categorical_mappings = pickle.load(f)
        
        # 加载维度信息
        with open(os.path.join(save_dir, 'dim_info.pkl'), 'rb') as f:
            dim_info = pickle.load(f)
            self.total_numerical_dim = dim_info['total_numerical_dim']
            self.categorical_dims = dim_info['categorical_dims']
            self.total_categorical_dim = dim_info['total_categorical_dim']
            
        logger.info(f"[特征处理] 处理器已从 {save_dir} 加载")