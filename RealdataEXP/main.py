#!/usr/bin/env python3
"""
RealdataEXP 实验框架主入口程序
支持多种实验模式：global, weighting, splitting等
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from libs.utils import setup_logger, create_experiment_dir
from libs.modes import GlobalMode
from libs.modes.global_mode_optimized import GlobalModeOptimized

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"配置文件加载失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RealdataEXP 实验框架')
    parser.add_argument('--config', '-c', type=str, 
                      default='configs/experiment_optimized.yaml',
                      help='配置文件路径')
    parser.add_argument('--mode', '-m', type=str,
                      help='实验模式 (覆盖配置文件中的mode设置)')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)
    
    # 命令行参数覆盖配置
    if args.mode:
        config['mode'] = args.mode
    
    # 从配置文件中获取设备选择
    device_choice = config.get('device', 'auto')

    # 创建实验目录
    base_dir = config.get('base_dir', os.path.dirname(__file__))
    exp_dir = create_experiment_dir(base_dir)
    
    # 设置日志
    log_file = os.path.join(exp_dir, 'run.log')
    logger = setup_logger(log_file, config.get('logging', {}).get('level', 'INFO'))
    
    logger.info("=" * 60)
    logger.info("RealdataEXP 实验框架启动")
    logger.info("=" * 60)
    logger.info(f"实验模式: {config['mode']}")
    logger.info(f"设备选择 (来自配置): {device_choice}")
    logger.info(f"实验目录: {exp_dir}")
    logger.info(f"配置文件: {config_path}")
    
    try:
        # 根据模式运行相应的实验
        mode = config['mode'].lower()
        
        if mode == 'global':
            logger.info("[模式选择] 运行Global模式实验")
            experiment = GlobalMode(config, exp_dir, device_choice=device_choice)
            experiment.run()
            
        elif mode == 'global_optimized':
            logger.info("[模式选择] 运行Global模式优化实验")
            experiment = GlobalModeOptimized(config, exp_dir, device_choice=device_choice)
            experiment.run()
            
        elif mode == 'weighting':
            logger.error("[模式选择] Weighting模式尚未实现")
            raise NotImplementedError("Weighting模式尚未实现")
            
        elif mode == 'splitting':
            logger.error("[模式选择] Splitting模式尚未实现") 
            raise NotImplementedError("Splitting模式尚未实现")
            
        else:
            raise ValueError(f"不支持的实验模式: {mode}")
            
        logger.info("=" * 60)
        logger.info("实验成功完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"实验执行失败: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)

if __name__ == '__main__':
    main()