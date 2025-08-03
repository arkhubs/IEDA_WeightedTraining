#!/bin/bash

# 交互式GPU会话脚本
# 用于调试和测试
# 使用方法: bash run_interactive_gpu.sh

echo "============================================================"
echo "请求交互式GPU节点..."
echo "============================================================"

# 请求交互式GPU会话
srun --account=sigroup --time=02:00:00 --partition=gpu-a30 --gpus-per-node=1 --cpus-per-task=8 --mem=32G --pty bash << 'EOF'

echo "成功获得GPU节点访问权限!"
echo "节点信息: $(hostname)"

# 加载CUDA模块
module load cuda

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

# 切换到项目目录
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

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

# 用户确认
echo "按任意键开始实验，或按Ctrl+C退出..."
read -n 1 -s -r
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

echo ""
echo "============================================================"
echo "开始运行实验"
echo "============================================================"
echo "使用配置文件: configs/experiment.yaml"
echo "设备配置: auto (将自动选择GPU)"

# 直接运行实验
python main.py --config configs/experiment.yaml --mode global

echo ""
echo "============================================================"
echo "实验完成！如需继续使用GPU节点，请使用以下命令："
echo "python main.py --config configs/experiment.yaml --mode global"
echo "或者运行其他Python脚本"
echo "============================================================"

# 启动交互式shell供后续使用
echo "启动交互式shell..."
bash

EOF