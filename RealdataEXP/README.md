# RealdataEXP 实验框架

## 项目概述

RealdataEXP 是一个基于真实数据的推荐系统实验框架，用于计算全局处理效应（Global Treatment Effect, GTE）。该框架支持多种实验模式，包括global、weighting、splitting等，旨在为推荐系统的因果推断研究提供完整的实验平台。

## 项目特性

- **多模式实验支持**：支持global、weighting、splitting等多种实验模式
- **多标签预测**：同时支持点击率和播放时长等多个标签的预测
- **灵活的特征处理**：自动处理数值特征和分类特征，支持one-hot编码
- **缓存机制**：优化数据加载性能，避免重复计算
- **完整的实验记录**：详细的日志记录和结果保存
- **可扩展架构**：模块化设计，易于扩展新的实验模式

## 系统架构

```
RealdataEXP/
├── configs/                    # 配置文件
│   └── experiment.yaml        # 实验配置
├── data/                      # 数据目录
│   └── KuaiRand/             # KuaiRand数据集
│       ├── Pure/             # Pure版本数据
│       ├── 1K/               # 1K版本数据
│       ├── 27K/              # 27K版本数据
│       └── cache/            # 缓存目录
├── libs/                     # 核心代码库
│   ├── data/                 # 数据处理模块
│   │   ├── data_loader.py    # 数据加载器
│   │   ├── feature_processor.py # 特征处理器
│   │   └── cache_manager.py  # 缓存管理器
│   ├── models/               # 模型模块
│   │   ├── mlp_model.py      # MLP模型
│   │   ├── multi_label_model.py # 多标签模型
│   │   └── loss_functions.py # 损失函数
│   ├── modes/                # 实验模式
│   │   ├── global_mode.py    # Global模式
│   │   ├── weighting.py      # Weighting模式（待实现）
│   │   └── splitting.py      # Splitting模式（待实现）
│   └── utils/                # 工具模块
│       ├── logger.py         # 日志工具
│       ├── metrics.py        # 指标跟踪
│       └── experiment_utils.py # 实验工具
├── results/                  # 实验结果
│   └── [timestamp]/          # 按时间戳组织的实验结果
│       ├── run.log          # 运行日志
│       ├── result.json      # 实验结果
│       └── checkpoints/     # 模型检查点
└── main.py                  # 主入口程序
```

![RealdataEXP/实验框架v2.pdf](RealdataEXP/实验框架v2.pdf)

## 数据集

项目使用KuaiRand数据集，包含：

- **用户行为日志**：记录用户与视频的交互行为
  - 总样本数：2,622,668
  - 点击率：33.14%
  - 平均播放时长：15,676.54ms

- **用户特征**：用户的静态画像数据
  - 用户数：27,285
  - 训练用户：21,828
  - 验证用户：5,457

- **视频特征**：视频的基础信息和统计特征
  - 视频数：7,583
  - 特征维度：157（数值特征：34，分类特征：123）

## 核心模块

### 1. 数据加载与处理模块

- **数据流**：从数据集提取用户ID，查找用户交互的视频列表
- **缓存机制**：避免重复计算，提高处理效率
- **特征处理**：
  - 数值特征：标准化处理
  - 分类特征：one-hot编码（`user_active_degree`、`video_type`、`tag`）
  - 缺失值处理：数值特征用0填充，分类特征作为新类别

### 2. 多标签预测模型

- **独立模型架构**：每个标签使用独立的MLP模型
- **支持的标签**：
  - `play_time`：播放时长预测（使用logMAE损失函数）
  - `click`：点击预测（使用二元交叉熵损失函数）
- **模型参数**：
  - play_time模型：30,593参数
  - click模型：12,737参数

### 3. Global模式实验

Global模式是框架的核心，实现真实GTE的计算：

- **对称仿真**：Treatment组和Control组独立运行
- **实验流程**：
  1. 用户批次抽样
  2. 候选视频生成
  3. 模型预测与加权排序
  4. 选出胜出视频
  5. 获取真实反馈与模型训练
  6. 更新状态标记
- **标记机制**：
  - `mask`：验证集用户的视频标记
  - `used`：已推荐视频的标记（两组独立维护）

## 配置文件

`configs/experiment.yaml` 包含完整的实验配置：

```yaml
# 实验模式
mode: 'global'

# 数据集配置
dataset:
  name: "KuaiRand-Pure"
  path: "data/KuaiRand/Pure"
  cache_path: "data/KuaiRand/cache"

# 特征配置
feature:
  numerical: [数值特征列表]
  categorical: [分类特征列表]

# 多标签配置
labels:
  - name: "play_time"
    target: "play_time_ms"
    type: "numerical"
    loss_function: "logMAE"
    # ... 模型参数
  - name: "click"
    target: "is_click"
    type: "binary"
    loss_function: "BCE"
    # ... 模型参数

# Global模式配置
global:
  user_p_val: 0.2      # 验证集比例
  batch_size: 64       # 批次大小
  n_candidate: 10      # 候选视频数
  n_steps: 200         # 仿真步数
  validate_every: 25   # 验证频率
```

## 安装和使用

### 环境要求

- Python 3.7+
- PyTorch 2.0+
- pandas 2.0+
- numpy 2.0+
- scikit-learn 1.6+
- PyYAML

### 运行实验

```bash
# 运行Global模式实验
python main.py --mode global

# 使用自定义配置文件
python main.py --config configs/my_experiment.yaml

# 指定实验模式（覆盖配置文件设置）
python main.py --mode global --config configs/experiment.yaml
```

### 实验结果

实验结果保存在 `results/[timestamp]/` 目录下：

- `run.log`：完整的运行日志
- `result.json`：实验结果和指标
- `checkpoints/`：模型检查点和特征处理器

## 开发进展

### 已完成功能

- ✅ 数据加载和预处理模块
- ✅ 多标签预测模型架构
- ✅ Global模式核心逻辑
- ✅ 特征处理和缓存机制
- ✅ 实验日志和结果保存
- ✅ 配置管理系统

### 待实现功能

- ⏳ Weighting模式实验
- ⏳ Splitting模式实验
- ⏳ 实验结果可视化
- ⏳ 模型性能评估指标
- ⏳ 分布式训练支持

## 技术细节

### 数据类型处理

框架实现了强健的数据类型转换机制，确保所有特征数据都能正确转换为PyTorch张量：

```python
def ensure_float_data(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """确保数据为float类型并转换为numpy数组"""
    # 逐列强制转换为数值类型
    # 处理NaN值和非数值类型
    # 返回float32数组
```

### 缓存优化

用户-视频交互列表使用pickle缓存，避免重复计算：

```python
# 首次计算时保存缓存
cache_manager.save(user_video_lists, "user_video_lists")

# 后续运行直接加载
cached_data = cache_manager.load("user_video_lists")
```

### 损失函数

支持多种损失函数：

- **LogMAE**：用于播放时长等大数值范围的连续标签
- **BCE**：用于点击等二元分类标签
- **MSE**、**MAE**、**CrossEntropy**：其他常用损失函数

## 扩展指南

### 添加新的实验模式

1. 在 `libs/modes/` 下创建新的模式文件
2. 继承基础实验类，实现核心逻辑
3. 在 `main.py` 中添加模式分派逻辑
4. 更新配置文件模板

### 添加新的模型

1. 在 `libs/models/` 下创建模型文件
2. 实现PyTorch模型接口
3. 在多标签模型管理器中注册新模型
4. 更新配置文件中的模型参数

### 添加新的特征

1. 在配置文件中声明新特征
2. 确保数据集包含对应字段
3. 根据特征类型选择数值或分类处理
4. 测试特征处理和模型训练流程

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

*最后更新：2025年8月3日*