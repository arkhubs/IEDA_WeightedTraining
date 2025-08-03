#!/usr/bin/env python3
"""
性能分析脚本
对比优化前后的训练性能
"""

import re
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_file(log_file: str) -> Dict:
    """解析日志文件，提取关键性能指标"""
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return {}
    
    metrics = {
        'epochs': [],
        'epoch_times': [],
        'losses': {'play_time': [], 'click': []},
        'steps': [],
        'step_times': [],
        'gpu_utilization': [],
        'memory_usage': [],
        'batch_sizes': [],
        'start_time': None,
        'end_time': None
    }
    
    print(f"正在解析日志文件: {log_file}")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        epoch_start_time = None
        step_start_time = None
        
        for line in lines:
            line = line.strip()
            
            # 解析时间戳
            time_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if time_match:
                current_time = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
                
                if metrics['start_time'] is None:
                    metrics['start_time'] = current_time
                metrics['end_time'] = current_time
            
            # 解析Epoch信息
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match and 'Epoch.*平均损失' not in line:
                epoch_num = int(epoch_match.group(1))
                if time_match:
                    epoch_start_time = current_time
            
            # 解析Epoch损失
            loss_match = re.search(r'Epoch \d+ 平均损失.*play_time: ([\d.]+).*click: ([\d.]+)', line)
            if loss_match:
                if epoch_start_time and time_match:
                    epoch_duration = (current_time - epoch_start_time).total_seconds()
                    metrics['epoch_times'].append(epoch_duration)
                    metrics['epochs'].append(len(metrics['epochs']) + 1)
                
                metrics['losses']['play_time'].append(float(loss_match.group(1)))
                metrics['losses']['click'].append(float(loss_match.group(2)))
            
            # 解析Step信息
            step_match = re.search(r'Step (\d+)/(\d+)', line)
            if step_match and 'Treatment' in line:
                step_num = int(step_match.group(1))
                if time_match:
                    step_start_time = current_time
            
            # 解析批次大小
            batch_match = re.search(r'批次大小[：:]\s*(\d+)', line)
            if batch_match:
                metrics['batch_sizes'].append(int(batch_match.group(1)))
            
            # 解析GPU信息（如果有nvidia-smi输出）
            gpu_match = re.search(r'(\d+)%.*(\d+)MiB / (\d+)MiB', line)
            if gpu_match:
                utilization = int(gpu_match.group(1))
                used_memory = int(gpu_match.group(2))
                total_memory = int(gpu_match.group(3))
                metrics['gpu_utilization'].append(utilization)
                metrics['memory_usage'].append(used_memory / total_memory * 100)
    
    except Exception as e:
        print(f"解析日志文件时出错: {e}")
        return {}
    
    return metrics

def analyze_performance(original_log: str, optimized_log: str):
    """分析并对比性能"""
    print("开始性能分析...")
    
    # 解析日志
    original_metrics = parse_log_file(original_log)
    optimized_metrics = parse_log_file(optimized_log)
    
    if not original_metrics or not optimized_metrics:
        print("无法解析日志文件，分析终止")
        return
    
    print("\n" + "="*60)
    print("性能对比分析报告")
    print("="*60)
    
    # 总体时间对比
    print("\n📊 总体训练时间对比:")
    if original_metrics['start_time'] and original_metrics['end_time']:
        original_duration = original_metrics['end_time'] - original_metrics['start_time']
        print(f"原版总时间: {original_duration}")
    
    if optimized_metrics['start_time'] and optimized_metrics['end_time']:
        optimized_duration = optimized_metrics['end_time'] - optimized_metrics['start_time']
        print(f"优化版总时间: {optimized_duration}")
        
        if original_metrics['start_time'] and original_metrics['end_time']:
            speedup = original_duration.total_seconds() / optimized_duration.total_seconds()
            print(f"⚡ 加速倍数: {speedup:.2f}x")
    
    # Epoch时间对比
    print("\n⏱️ Epoch训练时间对比:")
    if original_metrics['epoch_times']:
        avg_original_epoch = sum(original_metrics['epoch_times']) / len(original_metrics['epoch_times'])
        print(f"原版平均Epoch时间: {avg_original_epoch:.1f}秒")
    
    if optimized_metrics['epoch_times']:
        avg_optimized_epoch = sum(optimized_metrics['epoch_times']) / len(optimized_metrics['epoch_times'])
        print(f"优化版平均Epoch时间: {avg_optimized_epoch:.1f}秒")
        
        if original_metrics['epoch_times']:
            epoch_speedup = avg_original_epoch / avg_optimized_epoch
            print(f"⚡ Epoch加速倍数: {epoch_speedup:.2f}x")
    
    # 批次大小对比
    print("\n📦 批次大小对比:")
    if original_metrics['batch_sizes']:
        avg_original_batch = sum(original_metrics['batch_sizes']) / len(original_metrics['batch_sizes'])
        print(f"原版平均批次大小: {avg_original_batch:.0f}")
    
    if optimized_metrics['batch_sizes']:
        avg_optimized_batch = sum(optimized_metrics['batch_sizes']) / len(optimized_metrics['batch_sizes'])
        print(f"优化版平均批次大小: {avg_optimized_batch:.0f}")
        
        if original_metrics['batch_sizes']:
            batch_increase = avg_optimized_batch / avg_original_batch
            print(f"📈 批次大小提升: {batch_increase:.2f}x")
    
    # 损失收敛对比
    print("\n📉 损失收敛对比:")
    for label in ['play_time', 'click']:
        if (original_metrics['losses'][label] and 
            optimized_metrics['losses'][label]):
            
            original_final = original_metrics['losses'][label][-1]
            optimized_final = optimized_metrics['losses'][label][-1]
            
            print(f"{label} - 原版最终损失: {original_final:.6f}")
            print(f"{label} - 优化版最终损失: {optimized_final:.6f}")
            
            improvement = (original_final - optimized_final) / original_final * 100
            print(f"{label} - 损失改善: {improvement:+.2f}%")
    
    # GPU利用率对比
    if optimized_metrics['gpu_utilization']:
        print(f"\n🖥️ GPU平均利用率: {sum(optimized_metrics['gpu_utilization'])/len(optimized_metrics['gpu_utilization']):.1f}%")
    
    if optimized_metrics['memory_usage']:
        print(f"💾 GPU平均内存使用: {sum(optimized_metrics['memory_usage'])/len(optimized_metrics['memory_usage']):.1f}%")
    
    # 优化建议
    print("\n💡 优化建议:")
    if optimized_metrics['gpu_utilization']:
        avg_gpu_util = sum(optimized_metrics['gpu_utilization']) / len(optimized_metrics['gpu_utilization'])
        if avg_gpu_util < 80:
            print("- GPU利用率偏低，可考虑增大batch_size")
        elif avg_gpu_util > 95:
            print("- GPU利用率很高，优化效果良好")
    
    if optimized_metrics['batch_sizes']:
        avg_batch = sum(optimized_metrics['batch_sizes']) / len(optimized_metrics['batch_sizes'])
        if avg_batch < 128:
            print("- 批次大小较小，可尝试进一步增大")
        elif avg_batch > 512:
            print("- 批次大小很大，GPU利用充分")
    
    print("\n" + "="*60)
    print("分析完成！")

def main():
    """主函数"""
    print("GPU训练性能分析工具")
    print("="*40)
    
    # 命令行参数处理
    if len(sys.argv) < 3:
        print("用法: python performance_analysis.py <原版日志> <优化版日志>")
        print("示例: python performance_analysis.py results/gpu_run_52005_detailed.log results/gpu_run_52010_detailed.log")
        return
    
    original_log = sys.argv[1]
    optimized_log = sys.argv[2]
    
    analyze_performance(original_log, optimized_log)

if __name__ == '__main__':
    main()