#!/usr/bin/env python3
"""
性能分析工具
用于分析GPU利用率日志和实验性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
from datetime import datetime

def parse_gpu_log(log_file):
    """解析GPU利用率日志"""
    try:
        df = pd.read_csv(log_file)
        # 清理列名
        df.columns = df.columns.str.strip()
        
        # 转换数据类型
        df['utilization.gpu [%]'] = pd.to_numeric(df['utilization.gpu [%]'], errors='coerce')
        df['utilization.memory [%]'] = pd.to_numeric(df['utilization.memory [%]'], errors='coerce')
        df['memory.used [MiB]'] = pd.to_numeric(df['memory.used [MiB]'], errors='coerce')
        df['memory.total [MiB]'] = pd.to_numeric(df['memory.total [MiB]'], errors='coerce')
        df['power.draw [W]'] = pd.to_numeric(df['power.draw [W]'], errors='coerce')
        df['temperature.gpu [C]'] = pd.to_numeric(df['temperature.gpu [C]'], errors='coerce')
        
        return df
    except Exception as e:
        print(f"解析GPU日志失败: {e}")
        return None

def parse_training_log(log_file):
    """解析训练日志，提取关键时间信息"""
    training_info = {
        'start_time': None,
        'end_time': None,
        'epochs': [],
        'steps': [],
        'errors': []
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取时间戳
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    
                    # 记录开始时间
                    if training_info['start_time'] is None and 'RealdataEXP 实验框架启动' in line:
                        training_info['start_time'] = timestamp
                    
                    # 记录epoch信息
                    if 'Epoch' in line and '平均损失' in line:
                        epoch_match = re.search(r'Epoch (\d+)', line)
                        if epoch_match:
                            training_info['epochs'].append({
                                'epoch': int(epoch_match.group(1)),
                                'timestamp': timestamp,
                                'line': line.strip()
                            })
                    
                    # 记录step信息
                    if 'Step' in line and '处理用户' in line:
                        step_match = re.search(r'Step (\d+)', line)
                        if step_match:
                            training_info['steps'].append({
                                'step': int(step_match.group(1)),
                                'timestamp': timestamp,
                                'line': line.strip()
                            })
                    
                    # 记录错误
                    if 'ERROR' in line or 'Error' in line:
                        training_info['errors'].append({
                            'timestamp': timestamp,
                            'line': line.strip()
                        })
                    
                    # 记录结束时间
                    if '实验完成' in line or '实验成功完成' in line:
                        training_info['end_time'] = timestamp
    
    except Exception as e:
        print(f"解析训练日志失败: {e}")
    
    return training_info

def analyze_performance(gpu_log_file, training_log_file):
    """综合性能分析"""
    print("=" * 60)
    print("性能分析报告")
    print("=" * 60)
    
    # 解析GPU日志
    gpu_df = parse_gpu_log(gpu_log_file)
    if gpu_df is not None:
        print("\n=== GPU利用率分析 ===")
        print(f"记录数量: {len(gpu_df)}")
        
        gpu_util = gpu_df['utilization.gpu [%]'].dropna()
        mem_util = gpu_df['utilization.memory [%]'].dropna()
        
        if len(gpu_util) > 0:
            print(f"GPU利用率统计:")
            print(f"  - 平均值: {gpu_util.mean():.1f}%")
            print(f"  - 最大值: {gpu_util.max():.1f}%")
            print(f"  - 最小值: {gpu_util.min():.1f}%")
            print(f"  - 标准差: {gpu_util.std():.1f}%")
            
            # 利用率分布
            high_util = (gpu_util > 80).sum()
            medium_util = ((gpu_util > 40) & (gpu_util <= 80)).sum()
            low_util = (gpu_util <= 40).sum()
            
            print(f"GPU利用率分布:")
            print(f"  - 高利用率(>80%): {high_util} 次 ({high_util/len(gpu_util)*100:.1f}%)")
            print(f"  - 中等利用率(40-80%): {medium_util} 次 ({medium_util/len(gpu_util)*100:.1f}%)")
            print(f"  - 低利用率(<=40%): {low_util} 次 ({low_util/len(gpu_util)*100:.1f}%)")
        
        if len(mem_util) > 0:
            print(f"\nGPU内存利用率:")
            print(f"  - 平均值: {mem_util.mean():.1f}%")
            print(f"  - 最大值: {mem_util.max():.1f}%")
        
        # 功耗和温度
        power = gpu_df['power.draw [W]'].dropna()
        temp = gpu_df['temperature.gpu [C]'].dropna()
        
        if len(power) > 0:
            print(f"\n功耗统计:")
            print(f"  - 平均功耗: {power.mean():.1f}W")
            print(f"  - 最大功耗: {power.max():.1f}W")
        
        if len(temp) > 0:
            print(f"\n温度统计:")
            print(f"  - 平均温度: {temp.mean():.1f}°C")
            print(f"  - 最高温度: {temp.max():.1f}°C")
    
    # 解析训练日志
    training_info = parse_training_log(training_log_file)
    
    print("\n=== 训练进度分析 ===")
    if training_info['start_time'] and training_info['end_time']:
        start = datetime.strptime(training_info['start_time'], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(training_info['end_time'], '%Y-%m-%d %H:%M:%S')
        duration = end - start
        print(f"总训练时间: {duration}")
    
    if training_info['epochs']:
        print(f"完成的Epoch数: {len(training_info['epochs'])}")
        if len(training_info['epochs']) >= 2:
            # 计算平均epoch时间
            first_epoch = datetime.strptime(training_info['epochs'][0]['timestamp'], '%Y-%m-%d %H:%M:%S')
            last_epoch = datetime.strptime(training_info['epochs'][-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
            epoch_duration = (last_epoch - first_epoch) / len(training_info['epochs'])
            print(f"平均Epoch时间: {epoch_duration}")
    
    if training_info['steps']:
        print(f"完成的Step数: {len(training_info['steps'])}")
    
    if training_info['errors']:
        print(f"\n=== 错误分析 ===")
        print(f"错误数量: {len(training_info['errors'])}")
        for error in training_info['errors'][:5]:  # 显示前5个错误
            print(f"  - {error['timestamp']}: {error['line'][:100]}...")

def create_gpu_plot(gpu_log_file, output_dir):
    """创建GPU利用率图表"""
    gpu_df = parse_gpu_log(gpu_log_file)
    if gpu_df is None:
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        # GPU利用率
        plt.subplot(2, 2, 1)
        gpu_util = gpu_df['utilization.gpu [%]'].dropna()
        if len(gpu_util) > 0:
            plt.plot(gpu_util)
            plt.title('GPU利用率')
            plt.ylabel('利用率 (%)')
            plt.grid(True)
        
        # 内存利用率
        plt.subplot(2, 2, 2)
        mem_util = gpu_df['utilization.memory [%]'].dropna()
        if len(mem_util) > 0:
            plt.plot(mem_util, color='orange')
            plt.title('GPU内存利用率')
            plt.ylabel('内存利用率 (%)')
            plt.grid(True)
        
        # 功耗
        plt.subplot(2, 2, 3)
        power = gpu_df['power.draw [W]'].dropna()
        if len(power) > 0:
            plt.plot(power, color='red')
            plt.title('GPU功耗')
            plt.ylabel('功耗 (W)')
            plt.grid(True)
        
        # 温度
        plt.subplot(2, 2, 4)
        temp = gpu_df['temperature.gpu [C]'].dropna()
        if len(temp) > 0:
            plt.plot(temp, color='green')
            plt.title('GPU温度')
            plt.ylabel('温度 (°C)')
            plt.grid(True)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, 'gpu_performance.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nGPU性能图表已保存: {plot_file}")
        
    except Exception as e:
        print(f"创建图表失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='性能分析工具')
    parser.add_argument('--job-id', type=str, required=True, help='SLURM作业ID')
    parser.add_argument('--results-dir', type=str, default='results', help='结果目录')
    
    args = parser.parse_args()
    
    # 构建文件路径
    gpu_log_file = os.path.join(args.results_dir, f'gpu_utilization_{args.job_id}.log')
    training_log_file = os.path.join(args.results_dir, f'gpu_run_{args.job_id}_detailed.log')
    
    # 检查文件是否存在
    if not os.path.exists(gpu_log_file):
        print(f"GPU日志文件不存在: {gpu_log_file}")
        return
    
    if not os.path.exists(training_log_file):
        print(f"训练日志文件不存在: {training_log_file}")
        return
    
    # 执行分析
    analyze_performance(gpu_log_file, training_log_file)
    
    # 创建图表
    create_gpu_plot(gpu_log_file, args.results_dir)

if __name__ == '__main__':
    main()
