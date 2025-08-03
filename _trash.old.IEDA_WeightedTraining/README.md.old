# KuaiRand Interference Experiment

本项目用于复现并扩展数据训练环干扰相关实验，基于 KuaiRand 真实数据集，采用 PyTorch 实现。

## 目录结构
- configs/ 实验参数配置
- src/     核心代码模块
- results/ 实验结果、日志、模型（自动生成 exp_results_plot.png 可视化）
- notebooks/ 数据探索与分析

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置数据路径和实验参数（见 configs/experiment_config.yaml）
3. 运行主程序：
   ```bash
   python src/main.py
   ```
4. 结果与可视化：
   - 训练过程指标保存在 results/exp_results.json
   - 运行
     ```bash
     python results/plot_exp_results.py
     ```
     自动生成并弹出训练动态曲线图，图片保存在 results/exp_results_plot.png

## 主要功能
- 支持多种训练策略（加权、池化、分割、快照）
- 训练与评估指标自动记录
- 训练过程可视化

## 注意事项
- 不要将 .conda/、venv/、results/checkpoints/ 及大数据文件提交到仓库
- 详细设计见 configs/experiment_config.yaml 和 src/ 代码注释

## 参考
- KuaiRand 数据集：https://kuairand.com/
- 论文与方法详见项目文档
