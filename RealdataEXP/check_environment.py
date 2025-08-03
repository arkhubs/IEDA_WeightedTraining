#!/usr/bin/env python3
"""
环境检查脚本
检查PyTorch和CUDA是否正确配置
"""

import sys
import torch
import numpy as np

def check_pytorch_cuda():
    """检查PyTorch和CUDA配置"""
    print("=" * 60)
    print("PyTorch和CUDA环境检查")
    print("=" * 60)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    # CUDA信息
    print(f"\nCUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存容量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # 测试GPU计算
        print(f"\n当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
        
        # 简单的GPU测试
        print("\n进行GPU计算测试...")
        try:
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print("✓ GPU计算测试通过")
            
            # 检查内存使用
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            
        except Exception as e:
            print(f"✗ GPU计算测试失败: {e}")
    else:
        print("⚠️  CUDA不可用，将使用CPU运行")
    
    # 检查其他依赖
    print(f"\n其他依赖库检查:")
    try:
        import pandas as pd
        print(f"✓ pandas: {pd.__version__}")
    except ImportError:
        print("✗ pandas 未安装")
    
    try:
        import sklearn
        print(f"✓ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn 未安装")
    
    try:
        import yaml
        print(f"✓ PyYAML 已安装")
    except ImportError:
        print("✗ PyYAML 未安装")
    
    print("=" * 60)

if __name__ == "__main__":
    check_pytorch_cuda()