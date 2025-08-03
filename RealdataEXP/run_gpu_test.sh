#!/bin/bash

# 快速GPU测试作业脚本
# 用于测试环境，运行时间较短

#SBATCH --account=sigroup  # 你的账户名
#SBATCH --qos=a30_qos       # QoS设置
#SBATCH --time=00:30:00      # 运行时间限制 (30分钟)
#SBATCH --partition=gpu-a30  # GPU分区 (A30 GPU)
#SBATCH --gpus-per-node=1    # 每个节点使用1个GPU
#SBATCH --cpus-per-task=4    # 每个任务使用4个CPU核心
#SBATCH --mem=16G            # 内存需求
#SBATCH --job-name=gpu_test  # 作业名称
#SBATCH --output=results/gpu_test_%j.out  # 标准输出文件
#SBATCH --error=results/gpu_test_%j.err   # 错误输出文件

echo "============================================================"
echo "GPU测试作业启动"
echo "============================================================"
echo "作业开始时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "节点名称: $SLURM_NODELIST"
echo "GPU数量: $SLURM_GPUS_PER_NODE"
echo "============================================================"

# 切换到项目目录
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

# 加载CUDA模块
echo "加载CUDA模块..."
module load cuda

# 环境检测
echo ""
echo "============================================================"
echo "CUDA环境检测"
echo "============================================================"

# 检查GPU可用性
echo "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_STATUS=$?
    if [ $GPU_STATUS -eq 0 ]; then
        echo "✅ GPU检测成功"
    else
        echo "❌ GPU检测失败"
        exit 1
    fi
else
    echo "❌ nvidia-smi命令不可用"
    exit 1
fi

# 检查CUDA版本
echo ""
echo "检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✅ CUDA工具包可用"
else
    echo "⚠️ CUDA编译器不可用，但GPU可能仍然可用"
fi

# 运行Python环境检查
echo ""
echo "检查Python和PyTorch环境..."
python check_environment.py

# 检查检测结果
PYTHON_CHECK=$?
if [ $PYTHON_CHECK -ne 0 ]; then
    echo "❌ Python环境检测失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "环境检测完成 - 所有检查通过！"
echo "============================================================"

# 运行GPU设置测试
echo "运行详细的GPU设置测试..."
python test_gpu_setup.py

echo ""
echo "============================================================"
echo "如果以上所有测试都通过，你可以运行完整实验："
echo "sbatch run_gpu.sh"
echo "============================================================"

echo "============================================================"
echo "测试作业结束时间: $(date)"
echo "============================================================"