#!/usr/bin/env python3
"""
GPU设置测试脚本
快速验证GPU配置是否正确
"""

import os
import sys
import yaml
import torch
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

def test_basic_setup():
    """测试基本设置"""
    print("=" * 60)
    print("基本环境测试")
    print("=" * 60)
    
    # 检查PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
        
        # GPU内存测试
        device = torch.device('cuda')
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)
        print(f"✓ GPU计算测试通过")
        
        # 清理GPU内存
        del x, y, z
        torch.cuda.empty_cache()
    else:
        print("⚠️ CUDA不可用，将使用CPU")

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 60)
    print("配置文件测试")
    print("=" * 60)
    
    config_path = "configs/experiment.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ 配置文件加载成功: {config_path}")
        print(f"实验模式: {config['mode']}")
        print(f"设备配置: {config.get('device', 'auto')}")
        print(f"数据集: {config['dataset']['name']}")
        
        return config
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return None

def test_device_selection(config):
    """测试设备选择逻辑"""
    print("\n" + "=" * 60)
    print("设备选择测试")
    print("=" * 60)
    
    if config is None:
        print("✗ 跳过设备选择测试（配置加载失败）")
        return
    
    device_config = config.get('device', 'auto')
    print(f"配置中的设备设置: {device_config}")
    
    # 模拟GlobalMode的设备选择逻辑
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("⚠️ CUDA不可用，退回到CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(device_config)
    
    print(f"选择的设备: {device}")
    
    # 测试模型创建
    try:
        import torch.nn as nn
        model = nn.Linear(10, 1).to(device)
        print(f"✓ 模型创建成功，设备: {next(model.parameters()).device}")
        
        # 测试数据传输
        x = torch.randn(5, 10).to(device)
        y = model(x)
        print(f"✓ 前向传播测试通过，输出形状: {y.shape}")
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")

def test_data_paths():
    """测试数据路径"""
    print("\n" + "=" * 60)
    print("数据路径测试")
    print("=" * 60)
    
    data_dir = "data/KuaiRand/Pure"
    cache_dir = "data/KuaiRand/cache"
    
    print(f"数据目录: {data_dir}")
    print(f"缓存目录: {cache_dir}")
    
    if os.path.exists(data_dir):
        print("✓ 数据目录存在")
        files = os.listdir(data_dir)
        print(f"数据文件数: {len(files)}")
    else:
        print("⚠️ 数据目录不存在")
    
    if os.path.exists(cache_dir):
        print("✓ 缓存目录存在")
    else:
        print("⚠️ 缓存目录不存在")

def main():
    """主测试函数"""
    print("RealdataEXP GPU设置测试")
    print("时间:", torch.datetime.now() if hasattr(torch, 'datetime') else "未知")
    
    # 运行所有测试
    test_basic_setup()
    config = test_config_loading()
    test_device_selection(config)
    test_data_paths()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("🎉 GPU环境配置正确，可以使用以下命令运行实验:")
        print("   sbatch run_gpu.sh")
        print("   或")
        print("   bash run_interactive_gpu.sh")
    else:
        print("ℹ️ 当前在CPU环境，要使用GPU请在GPU节点上运行此测试")

if __name__ == "__main__":
    main()