#!/bin/bash

# 直接交互式方法 - 使用salloc分配资源然后连接

echo "============================================================"
echo "请求交互式GPU资源 (使用salloc方法)..."
echo "============================================================"

echo "正在分配GPU资源，请稍等..."
echo "如果等待时间过长，请按Ctrl+C取消"

# 使用salloc分配资源
salloc --account=sigroup \
       --partition=gpu-a30 \
       --gpus-per-node=1 \
       --cpus-per-task=4 \
       --mem=16G \
       --time=01:00:00 \
       bash -c '

echo "============================================================"
echo "成功分配GPU资源!"
echo "节点信息: $SLURMD_NODENAME"
echo "作业ID: $SLURM_JOB_ID"  
echo "============================================================"

# 加载CUDA模块
echo "加载CUDA模块..."
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
echo "实验完成！GPU资源仍然分配给你，可以继续使用："
echo "python main.py --config configs/experiment.yaml --mode global"
echo "python test_gpu_setup.py"
echo "python check_environment.py"
echo "============================================================"

# 启动交互式shell
echo "启动交互式shell (退出时将释放GPU资源)..."
bash

echo "GPU资源已释放"
'