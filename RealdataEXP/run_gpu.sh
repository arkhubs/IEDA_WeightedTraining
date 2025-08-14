#!/bin/bash

# GPU 作业提交脚本 - Global Mode实验
# 使用方法: sbatch run_gpu.sh

#SBATCH --account=sigroup     # 你的账户名
#SBATCH --time=23:30:00       # 运行时间限制 (23.5小时) - 减少以提高调度优先级
#SBATCH --partition=gpu-a30   # GPU分区 (A30 GPU)
#SBATCH --gpus-per-node=1     # 每个节点使用1个GPU
#SBATCH --cpus-per-task=32     # 每个任务使用32个CPU核心
#SBATCH --mem=64G             # 内存需求
#SBATCH --job-name=global_mode_gpu  # 更明确的作业名称
#SBATCH --output=results/gpu_run_%j.out  # 标准输出文件
#SBATCH --error=results/gpu_run_%j.err   # 错误输出文件

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

# 环境检测和确认
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
    echo "❌ Python环境检测失败，继续运行但可能遇到问题"
    echo "注意：在SLURM环境中，GPU只有在作业分配后才可用"
fi

echo ""
echo "============================================================"
echo "环境检测完成 - 所有检查通过！"
echo "============================================================"

# 对于批处理作业，自动继续（不等待用户输入）
echo "批处理模式：自动开始实验..."
sleep 2

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

echo ""
echo "============================================================"
echo "开始运行实验"
echo "============================================================"
echo "使用配置文件: configs/experiment.yaml"
echo "设备配置: auto (将自动选择GPU)"

# 最终GPU检查（在SLURM分配后）
echo "SLURM作业分配后的GPU状态："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi不可用，但PyTorch应该仍能检测到GPU"
fi

echo ""
echo "开始运行Global Mode GPU实验..."
echo "预期运行时间：约60-90分钟"
echo ""

# 运行实验，增加详细输出
python main.py --config configs/experiment.yaml --mode global 2>&1 | tee results/gpu_run_${SLURM_JOB_ID}_detailed.log

EXPERIMENT_STATUS=$?

echo ""
echo "============================================================"
if [ $EXPERIMENT_STATUS -eq 0 ]; then
    echo "✅ 实验成功完成！"
else
    echo "❌ 实验执行出错，退出码: $EXPERIMENT_STATUS"
fi
echo "作业结束时间: $(date)"
echo "详细日志保存在: results/gpu_run_${SLURM_JOB_ID}_detailed.log"
echo "============================================================"