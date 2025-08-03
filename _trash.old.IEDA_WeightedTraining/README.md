# IEDA_WeightedTraining

基于仿真的推荐系统加权训练实验框架。

## 项目概述

本项目实现了一个模块化的仿真实验框架，用于研究推荐系统中的加权训练策略。框架主要包含以下功能：

1. 数据加载、处理与管理模块
2. 模型训练与预测模块
3. 推荐策略与仿真模块
4. 全局GTE（Global Treatment Effect）计算模块

## 主要特点

- **数据管理机制**: 实现了高效的数据加载、缓存与处理流程
- **模块化设计**: 各组件解耦，易于扩展新功能和实验模式
- **仿真实验**: 通过并行运行处理组和对照组来计算真实GTE
- **可配置性**: 丰富的配置选项，支持通过YAML文件进行实验设置
- **完整记录**: 详细的日志记录和结果可视化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 目录结构

```
IEDA_WeightedTraining/
├── configs/
│   └── experiment.yaml        # 实验配置文件
├── data/
│   └── KuaiRand/
│       ├── Pure/              # KuaiRand-Pure 原始数据集
│       ├── 1K/
│       └── 27K/
│           └── .csv           # 特征与标签数据
├── libs/
│   └── src/
│       ├── __init__.py        # 初始化模块
│       ├── config_manager.py  # 配置管理器
│       ├── data_manager.py    # 数据管理器
│       ├── models.py          # 模型定义
│       ├── recommender.py     # 推荐器
│       └── ...
└── results/                   # 实验结果目录
    └── 20250801_2210/         # 实验结果（按时间戳命名）
        ├── run.log            # 实验日志
        ├── result.json        # 实验结果指标
        └── ...
```

## 使用方法

### 运行实验

```bash
# 使用默认配置运行实验
python run.py

# 使用指定的配置文件
python run.py --config configs/custom_experiment.yaml

# 从检查点恢复训练
python run.py --resume results/20250801_2210/checkpoints/step_500.pt

# 跳过预训练
python run.py --no_pretrain
```

### 测试模型

```bash
# 测试模型
python -m libs.src.test_model --checkpoint results/20250801_2210/checkpoints/best.pt --test_users 100
```

## 配置文件说明

`experiment.yaml` 是主要的配置文件，通过修改其中的参数可以控制实验的各个方面：

```yaml
mode: 'global'  # 实验模式

# 数据集配置
dataset:
  name: "KuaiRand-27K"
  path: "data/KuaiRand/27K"

# 特征配置
feature:
  numerical: [...]
  categorical: [...]

# 标签配置
label:
  target: "play_time"
  type: "numerical"

# 全局仿真配置
global:
  user_p_val: 0.2
  batch_size: 64
  n_candidate: 10
  n_steps: 1000

# 推荐器配置
recommender:
  alpha_T: [1.0]  # 处理组权重
  alpha_C: [0.5]  # 对照组权重

# 更多配置...
```

## 结果可视化

项目现在支持丰富的结果可视化功能：

1. **训练过程指标**：包括模型AUC、损失、相对误差等指标的变化曲线
2. **对比分析**：处理组和对照组的累积回报对比
3. **多标签支持**：针对每个标签（如点击和播放时长）的单独可视化

可以通过以下命令运行可视化工具：

```bash
python visualize_results.py --result_dir [结果目录路径]
```

或者在实验完成后，结果会自动被可视化并保存在实验目录中。

## 实验模式

项目支持多种实验模式，模式实现位于 `libs/exp_modes` 目录。

### 全局模式 (global)

全局模式独立运行两次完整实验：
1. 一次全部使用处理组推荐参数 (alpha_T)
2. 一次全部使用对照组推荐参数 (alpha_C)

两次实验之间没有交互，模型训练完全独立。最后通过比较两组实验的累积收益差异计算GTE。

具体流程：
1. 初始化推荐模型
2. 运行处理组实验
   - 使用 alpha_T 参数进行推荐
   - 收集反馈并训练模型
   - 保存处理组模型和结果
3. 重置模型
4. 运行对照组实验
   - 使用 alpha_C 参数进行推荐
   - 收集反馈并训练模型
   - 保存对照组模型和结果
5. 计算最终GTE

### 计划支持的其他模式

- **weighting**: 权重训练模式
- **splitting**: 分割训练模式

## 参考

- KuaiRand 数据集：https://kuairand.com/
