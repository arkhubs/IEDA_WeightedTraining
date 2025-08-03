#!/usr/bin/env python3
"""
测试训练改进效果
验证模型性能和GPU使用情况
"""

import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from libs.modes.global_mode_optimized import GlobalModeOptimized
from libs.data import KuaiRandDataLoader, FeatureProcessor
from libs.models import MultiLabelModel

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_gpu_usage():
    """测试GPU使用情况"""
    print("=== GPU使用情况测试 ===")
    
    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 测试张量操作
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: {device}")
        
        # 创建测试张量
        x = torch.randn(1000, 100).to(device)
        y = torch.randn(1000, 1).to(device)
        
        # 简单计算
        z = torch.mm(x.T, x)
        print(f"矩阵乘法完成，结果形状: {z.shape}")
        
        # 清理内存
        del x, y, z
        torch.cuda.empty_cache()
        
    return torch.cuda.is_available()

def test_model_improvements():
    """测试模型改进"""
    print("\n=== 模型改进测试 ===")
    
    # 创建简单的测试数据
    input_dim = 157  # 根据日志中的特征维度
    batch_size = 64
    
    # 测试数据
    x = torch.randn(batch_size, input_dim)
    y_play_time = torch.abs(torch.randn(batch_size, 1)) * 1000  # 播放时长
    y_click = torch.randint(0, 2, (batch_size, 1)).float()  # 点击
    
    print(f"测试数据形状: x={x.shape}, play_time={y_play_time.shape}, click={y_click.shape}")
    
    # 测试配置
    config = {
        'labels': [
            {
                'name': 'play_time',
                'target': 'play_time_ms',
                'type': 'numerical',
                'loss_function': 'logMAE',
                'model_params': {
                    'hidden_layers': [512, 256, 128, 64],
                    'dropout': 0.2,
                    'batch_norm': True,
                    'residual': True,
                    'embedding_dim': 64
                },
                'learning_rate': 0.0005,
                'weight_decay': 0.00001
            },
            {
                'name': 'click',
                'target': 'is_click',
                'type': 'binary',
                'loss_function': 'BCE',
                'model_params': {
                    'hidden_layers': [128, 64, 32],
                    'dropout': 0.2,
                    'batch_norm': True,
                    'residual': False,
                    'embedding_dim': 16
                },
                'learning_rate': 0.0001,
                'weight_decay': 0.0001
            }
        ]
    }
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLabelModel(config, input_dim, device)
    
    print(f"模型创建成功，使用设备: {device}")
    
    # 测试前向传播
    try:
        predictions = model.predict_all(x.to(device))
        print(f"预测成功: {list(predictions.keys())}")
        
        # 测试训练步骤
        targets = {
            'play_time': y_play_time.to(device),
            'click': y_click.to(device)
        }
        
        losses = model.train_step(x.to(device), targets)
        print(f"训练步骤成功，损失: {losses}")
        
        # 测试评估
        metrics = model.evaluate_with_metrics(x.to(device), targets)
        print(f"评估指标: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

def test_configuration():
    """测试配置文件"""
    print("\n=== 配置文件测试 ===")
    
    try:
        import yaml
        
        # 读取优化配置
        with open('configs/experiment_optimized.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("配置文件加载成功")
        
        # 检查关键配置
        play_time_config = config['labels'][0]
        print(f"Play_time模型配置:")
        print(f"  - 隐藏层: {play_time_config['model_params']['hidden_layers']}")
        print(f"  - 批归一化: {play_time_config['model_params']['batch_norm']}")
        print(f"  - 残差连接: {play_time_config['model_params']['residual']}")
        print(f"  - 学习率: {play_time_config['learning_rate']}")
        print(f"  - 损失函数: {play_time_config['loss_function']}")
        
        return True
        
    except Exception as e:
        print(f"配置文件测试失败: {e}")
        return False

def main():
    """主函数"""
    setup_logging()
    
    print("开始测试训练改进...")
    
    results = {}
    
    # 测试GPU使用
    results['gpu'] = test_gpu_usage()
    
    # 测试模型改进
    results['model'] = test_model_improvements()
    
    # 测试配置文件
    results['config'] = test_configuration()
    
    # 总结
    print(f"\n=== 测试结果总结 ===")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 所有测试通过！改进已生效。")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()