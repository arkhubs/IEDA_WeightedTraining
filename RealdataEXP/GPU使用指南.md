# GPU 使用指南

## 概述

本指南将帮助你在HKUST HPC4集群上使用GPU运行RealdataEXP实验框架。

## 前置准备

### 1. 确认项目结构

确保你的项目位于 `/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/` 目录下。

### 2. 检查环境

首先在GPU节点上检查PyTorch和CUDA环境：

```bash
# 请求交互式GPU会话进行测试
bash run_interactive_gpu.sh
```

然后运行环境检查脚本：

```bash
python check_environment.py
```

## 运行方式

### 方式一：批处理作业（推荐用于长时间训练）

使用Slurm批处理系统提交GPU作业：

```bash
# 提交GPU作业
sbatch run_gpu.sh

# 查看作业状态
squeue -u zhixuanhu

# 查看作业输出
tail -f results/gpu_run_[job_id].out
```

### 方式二：交互式会话（推荐用于调试）

获取交互式GPU会话：

```bash
# 启动交互式GPU会话
bash run_interactive_gpu.sh

# 在GPU节点上直接运行实验
python main.py --config configs/experiment.yaml --mode global
```

## 配置说明

### GPU配置

在 `configs/experiment.yaml` 中已添加GPU配置选项：

```yaml
# GPU配置
device: 'auto'  # 设备选择: 'auto', 'cuda', 'cpu'
```

- `'auto'`: 自动选择最佳设备（有GPU时使用GPU，否则使用CPU）
- `'cuda'`: 强制使用GPU（如果GPU不可用会退回到CPU）
- `'cpu'`: 强制使用CPU

### 资源配置

在 `run_gpu.sh` 中可以调整资源需求：

```bash
#SBATCH --time=04:00:00      # 运行时间限制
#SBATCH --partition=gpu-a30  # GPU分区（A30 GPU）
#SBATCH --gpus-per-node=1    # GPU数量
#SBATCH --cpus-per-task=4    # CPU核心数
#SBATCH --mem=16G            # 内存需求
```

## 常见问题

### 1. CUDA不可用

如果遇到 "CUDA not available" 错误：

1. 确认你在GPU节点上运行（不是登录节点）
2. 检查CUDA模块是否加载：`module load cuda`
3. 检查GPU状态：`nvidia-smi`

### 2. 内存不足

如果遇到GPU内存不足：

1. 减少批处理大小：在配置文件中调整 `batch_size`
2. 减少模型复杂度：调整 `hidden_layers` 参数
3. 请求更大内存的GPU节点

### 3. 作业排队时间长

如果作业长时间排队：

1. 尝试不同的GPU分区：`gpu-a30`, `gpu-v100`等
2. 减少资源需求：降低内存、CPU核心或GPU数量
3. 选择非高峰时间提交作业

### 4. 模型加载失败

如果遇到模型加载错误：

1. 检查特征维度是否正确
2. 确认数据预处理完成
3. 查看详细错误日志

## 性能优化建议

### 1. 数据加载优化

- 使用数据缓存机制减少I/O操作
- 适当调整 `batch_size` 平衡内存使用和训练效率

### 2. 模型训练优化

- 使用混合精度训练（如果支持）
- 适当调整学习率和权重衰减
- 使用学习率调度器

### 3. 监控和调试

- 定期检查GPU使用率：`nvidia-smi`
- 监控内存使用情况
- 查看训练日志了解训练进度

## 作业监控

### 查看作业状态

```bash
# 查看你的所有作业
squeue -u zhixuanhu

# 查看作业详细信息
scontrol show job [job_id]

# 取消作业
scancel [job_id]
```

### 查看输出日志

```bash
# 实时查看输出
tail -f results/gpu_run_[job_id].out

# 查看错误日志
tail -f results/gpu_run_[job_id].err

# 查看实验日志
tail -f results/[timestamp]/run.log
```

## 示例命令

### 完整的GPU运行流程

```bash
# 1. 首先检查环境（交互式）
bash run_interactive_gpu.sh
python check_environment.py

# 2. 提交批处理作业
sbatch run_gpu.sh

# 3. 监控作业进度
squeue -u zhixuanhu
tail -f results/gpu_run_*.out
```

### 自定义配置运行

```bash
# 使用自定义配置文件
python main.py --config configs/my_config.yaml --mode global

# 强制使用CPU（用于对比测试）
# 在配置文件中设置 device: 'cpu'
```

## 联系支持

如果遇到问题：

1. 查看日志文件中的错误信息
2. 检查HPC4集群状态和公告
3. 联系集群管理员获取技术支持

---

*更新时间：2025年1月*