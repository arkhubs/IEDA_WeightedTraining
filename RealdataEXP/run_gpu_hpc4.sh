#!/bin/bash

# GPU 作业提交脚本 - Global Mode优化实验
# 使用方法: sbatch run_gpu_optimized.sh

#SBATCH --account=sigroup     # 你的账户名
#SBATCH --time=23:30:00       # 运行时间限制 (23.5小时)
#SBATCH --partition=gpu-a30   # GPU分区 (A30 GPU)
#SBATCH --gpus-per-node=1     # 每个节点使用1个GPU
#SBATCH --cpus-per-task=32    # 每个任务使用32个CPU核心
#SBATCH --mem=256G             # 内存需求
#SBATCH --job-name=global_optimized  # 作业名称
#SBATCH --output=results/gpu_run_%j.out  # 标准输出文件
#SBATCH --error=results/gpu_run_%j.err   # 错误输出文件

echo "============================================================"
echo "作业开始时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "节点名称: $SLURM_NODELIST"
echo "GPU数量: $SLURM_GPUS_PER_NODE"
echo "CPU核心数: $SLURM_CPUS_PER_TASK"
echo "============================================================"

# 切换到项目目录
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

# 加载CUDA模块
echo "加载CUDA模块..."
module load cuda

# --- GPU利用率监控 ---
echo "启动GPU利用率监控..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10 > results/gpu_utilization_${SLURM_JOB_ID}.log &
NVIDIASMI_PID=$!

echo "GPU监控进程PID: $NVIDIASMI_PID"

# 环境检测
echo ""
echo "============================================================"
echo "=== Python环境检查 ==="
echo "============================================================"

# 检查Python和PyTorch环境
python -c "
import sys
print('Python版本:', sys.version)
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    print('GPU名称:', torch.cuda.get_device_name(0))
    print('GPU内存: {:.1f}GB'.format(torch.cuda.get_device_properties(0).total_memory/1024**3))
"

echo ""
echo "=== 开始运行优化实验 ==="
echo "配置文件: configs/experiment_optimized.yaml"
echo "开始时间: $(date)"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

# 运行优化实验
python main.py --config configs/experiment_optimized.yaml --mode global_optimized 2>&1 | tee results/gpu_run_${SLURM_JOB_ID}_detailed.log

EXPERIMENT_STATUS=$?

echo ""
echo "============================================================"
# --- 停止GPU监控 ---
echo "停止GPU利用率监控 (PID: $NVIDIASMI_PID)..."
kill $NVIDIASMI_PID 2>/dev/null

if [ $EXPERIMENT_STATUS -eq 0 ]; then
    echo "✅ 实验成功完成！"
else
    echo "❌ 实验执行出错，退出码: $EXPERIMENT_STATUS"
fi

echo "作业结束时间: $(date)"
echo "详细日志保存在: results/gpu_run_${SLURM_JOB_ID}_detailed.log"
echo "GPU利用率日志: results/gpu_utilization_${SLURM_JOB_ID}.log"

# 输出GPU利用率统计
echo ""
echo "=== GPU利用率统计 ==="
if [ -f "results/gpu_utilization_${SLURM_JOB_ID}.log" ]; then
    echo "GPU利用率文件行数: $(wc -l < results/gpu_utilization_${SLURM_JOB_ID}.log)"
    echo "最后几条GPU状态:"
    tail -5 results/gpu_utilization_${SLURM_JOB_ID}.log
fi

echo "============================================================"
