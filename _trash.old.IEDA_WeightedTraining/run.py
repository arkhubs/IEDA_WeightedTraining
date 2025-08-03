#!/usr/bin/env python
import os
import sys
import argparse
import logging
import torch
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 改变工作目录到项目根目录
os.chdir(project_root)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="IEDA 加权训练实验")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, default=None, help="实验模式，会覆盖配置文件中的mode")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--no_pretrain", action="store_true", help="跳过预训练")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 直接导入main模块
    from libs.src.main import main as main_func
    
    # 执行主程序
    main_func()
    
if __name__ == "__main__":
    main()
