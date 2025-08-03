# 多标签预测代码修改说明

我们已经对项目进行了全面修改，使其能够支持多标签预测和多模型训练。以下是主要的修改点：

## 1. 数据管理相关修改
- `DataManager` 类支持多个标签的加载与处理
- `_create_user_video_map` 方法更新为支持多标签存储
- 添加 `get_default_label_name()` 和 `get_label_names()` 方法
- 原有的 `get_video_label()` 方法保留以兼容旧版代码

## 2. 数据管理方法类修改
- 添加 `get_video_label()` 和 `get_video_labels()` 方法处理多标签
- 新增 `get_true_labels_dict()` 方法以支持获取多个标签的值
- 更新现有的 `get_true_labels()` 方法以兼容多标签

## 3. 推荐器类修改
- 更新 `__init__()` 方法支持多个模型字典而非单个模型
- 修改 `recommend()` 方法接收 alpha 权重字典
- 更新 `train_step()` 方法支持多个模型、损失函数和优化器
- 修改 `validate()` 方法评估每个标签的单独性能
- 更新 `run_simulation()` 方法支持多标签训练和评估
- 改进 `_save_model()`、`_save_metrics()` 和 `_plot_metrics()` 方法

## 4. 训练器类修改
- 更新 `__init__()` 方法支持多模型
- 添加 `_get_label_specific_config()` 方法获取标签特定配置
- 修改 `create_criterion()` 和 `create_optimizer()` 方法以支持标签特定配置
- 更新 `pretrain()` 和 `resume_training()` 方法支持多个模型

## 5. 配置文件修改
- 更新 `experiment.yaml` 格式以支持多标签
- 为每个标签添加单独的模型参数和alpha权重
- 简化推荐器配置以向后兼容

## 6. 初始化和主函数修改
- 更新 `init_all()` 支持创建多个模型
- 修改 `main.py` 以支持多标签处理
- 简化 `run.py` 以直接调用主函数

## 使用方法
1. 在 `configs/experiment.yaml` 文件中配置多个标签及其对应的模型
2. 使用命令行运行实验：
   ```bash
   python run.py --config configs/experiment.yaml
   ```

## 功能特点
- 支持为每个标签训练单独的模型
- 每个标签可以有自己的损失函数和优化器参数
- 每个标签可以设置不同的alpha权重
- 训练和验证过程会报告每个标签的单独指标
- 结果保存时会单独保存每个标签的模型和指标
- 指标图表会为每个标签生成单独的图表
