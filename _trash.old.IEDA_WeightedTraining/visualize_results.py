#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from libs.src.plot_results import plot_advanced_metrics, load_results, load_training_logs
from libs.src.plot_training_metrics import plot_training_metrics

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="结果可视化工具")
    parser.add_argument("--result_dir", type=str, required=True, help="结果目录路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录路径")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("visualize")
    
    try:
        # 检查结果目录
        if not os.path.exists(args.result_dir):
            logger.error(f"结果目录不存在: {args.result_dir}")
            return
            
        result_file = os.path.join(args.result_dir, 'result.json')
        if not os.path.exists(result_file):
            logger.error(f"结果文件不存在: {result_file}")
            return
            
        # 加载结果
        try:
            results = load_results(args.result_dir)
            logger.info(f"已加载结果文件: {result_file}")
        except Exception as e:
            logger.error(f"加载结果文件失败: {str(e)}")
            return
            
        # 加载训练日志指标
        log_metrics = load_training_logs(args.result_dir)
        
        # 设置输出目录
        output_dir = args.output_dir or args.result_dir
        
        # 绘制高级指标图表
        try:
            plot_advanced_metrics(results, log_metrics, output_dir)
            logger.info(f"已生成高级指标图表: {os.path.join(output_dir, 'advanced_metrics.png')}")
        except Exception as e:
            logger.error(f"绘制高级指标图表失败: {str(e)}")
            
        # 绘制训练指标图表
        try:
            plot_training_metrics(args.result_dir)
            logger.info(f"已生成训练指标图表: {os.path.join(output_dir, 'training_metrics.png')}")
        except Exception as e:
            logger.error(f"绘制训练指标图表失败: {str(e)}")
            
        logger.info("结果可视化完成")
        
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
