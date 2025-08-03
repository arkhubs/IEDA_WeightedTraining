#!/bin/bash
#SBATCH --account=sigroup        # 项目组账户
#SBATCH --time=03:30:00          # 运行时间限制 (1.5小时，优化后时间大幅减少)
#SBATCH --partition=gpu-a30      # GPU分区 (A30 GPU)
#SBATCH --gpus-per-node=1        # 每个节点使用1个GPU
#SBATCH --cpus-per-task=8        # 每个任务使用8个CPU核心
#SBATCH --mem=64G                # 增加内存需求，支持更大batch_size
#SBATCH --job-name=global_optimized_gpu  # 作业名称
#SBATCH --output=results/gpu_run_%j.out  # 输出文件
#SBATCH --error=results/gpu_run_%j.err   # 错误文件

# ========================================
# GPU优化版本运行脚本
# 主要优化：
# 1. 使用PyTorch DataLoader + 多线程数据加载
# 2. 混合精度训练(AMP)
# 3. 批量训练替代单样本训练
# 4. 增大batch_size充分利用GPU
# 5. GPU内存管理优化
# ========================================

echo "=========================================="
echo "GPU优化实验开始"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点名: ${SLURMD_NODENAME}"
echo "时间: $(date)"
echo "=========================================="

# 检查GPU环境
echo "=== GPU环境检查 ==="
nvidia-smi
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 进入项目目录
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/

# 创建结果目录
mkdir -p results

# 启动详细日志
LOG_DETAILED="results/gpu_run_${SLURM_JOB_ID}_detailed.log"
touch "${LOG_DETAILED}"

echo "详细日志文件: ${LOG_DETAILED}"

# 检查Python环境
echo "=== Python环境检查 ===" >> "${LOG_DETAILED}"
python -c "
import torch
import sys
print(f'Python版本: {sys.version}', flush=True)
print(f'PyTorch版本: {torch.__version__}', flush=True)
print(f'CUDA可用: {torch.cuda.is_available()}', flush=True)
print(f'GPU数量: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'GPU不可用', flush=True)
if torch.cuda.is_available():
    print(f'GPU名称: {torch.cuda.get_device_name()}', flush=True)
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB', flush=True)
" >> "${LOG_DETAILED}" 2>&1

echo "" >> "${LOG_DETAILED}"

# 运行优化实验
echo "=== 开始运行优化实验 ===" >> "${LOG_DETAILED}"
echo "配置文件: configs/experiment_optimized.yaml" >> "${LOG_DETAILED}"
echo "开始时间: $(date)" >> "${LOG_DETAILED}"
echo "" >> "${LOG_DETAILED}"

# 使用优化配置运行实验
python main.py \
    --config configs/experiment_optimized.yaml \
    --mode global_optimized \
    >> "${LOG_DETAILED}" 2>&1

EXIT_CODE=$?

echo "" >> "${LOG_DETAILED}"
echo "结束时间: $(date)" >> "${LOG_DETAILED}"
echo "退出码: ${EXIT_CODE}" >> "${LOG_DETAILED}"

# 显示GPU最终状态
echo "=== 最终GPU状态 ===" >> "${LOG_DETAILED}"
nvidia-smi >> "${LOG_DETAILED}" 2>&1

echo "=========================================="
echo "GPU优化实验完成"
echo "退出码: ${EXIT_CODE}"
echo "详细日志: ${LOG_DETAILED}"
echo "时间: $(date)"
echo "=========================================="

exit ${EXIT_CODE}