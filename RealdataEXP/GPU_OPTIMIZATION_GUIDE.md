# GPU训练优化指南

## 🚀 优化概述

针对GPU利用率低和训练速度慢的问题，我们创建了优化版本，实现了以下关键改进：

### 🔧 主要优化措施

1. **PyTorch DataLoader + 多线程数据加载**
   - 使用4个工作线程并行加载数据
   - 启用`pin_memory`和`persistent_workers`
   - 预取因子设置为2，减少GPU等待时间

2. **批量训练替代单样本训练**
   - 将单个winner样本训练改为批量训练
   - 大幅减少GPU kernel启动开销
   - 提高GPU并行计算效率

3. **混合精度训练(AMP)**
   - 自动启用CUDA AMP加速训练
   - 保持数值稳定性的同时提升速度
   - 减少GPU内存占用

4. **增大Batch Size**
   - 预训练batch_size: 64 → 256 (4x提升)
   - 仿真batch_size: 64 → 128 (2x提升)
   - 充分利用GPU并行计算能力

5. **数据预处理优化**
   - 预先转换数据为GPU张量
   - 避免重复的数据类型转换
   - 减少CPU-GPU数据传输

6. **GPU内存管理**
   - 定期清理GPU缓存
   - 优化张量操作减少内存碎片

## 📁 文件结构

```
RealdataEXP/
├── libs/modes/
│   ├── global_mode.py              # 原版训练脚本
│   └── global_mode_optimized.py    # 🆕 优化版训练脚本
├── configs/
│   ├── experiment.yaml            # 原版配置
│   └── experiment_optimized.yaml  # 🆕 优化配置
├── run_gpu.sh                     # 原版GPU脚本
├── run_gpu_optimized.sh          # 🆕 优化GPU脚本
├── monitor_gpu_optimized.sh       # 🆕 监控脚本
└── performance_analysis.py        # 🆕 性能分析工具
```

## 🚀 快速开始

### 1. 提交优化版GPU作业

```bash
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/

# 提交优化版作业
sbatch run_gpu_optimized.sh
```

### 2. 监控训练进程

```bash
# 查看作业状态
./monitor_gpu_optimized.sh status

# 监控特定作业（例如作业ID: 52010）
./monitor_gpu_optimized.sh 52010

# 查看GPU状态
./monitor_gpu_optimized.sh gpu

# 查看最新日志
./monitor_gpu_optimized.sh log
```

### 3. 实时监控

```bash
# 连接到GPU节点（假设作业ID为52010，节点为gpu01）
srun --jobid=52010 -w gpu01 --overlap --pty bash -i

# 在GPU节点上监控
nvidia-smi -l 2  # 实时GPU状态
htop             # CPU和内存状态
tail -f results/gpu_run_52010_detailed.log  # 训练日志
```

## 📊 性能预期

根据优化措施，预期性能提升：

| 指标 | 原版 | 优化版 | 提升倍数 |
|------|------|--------|----------|
| **训练速度** | ~20min/epoch | ~5-8min/epoch | **2.5-4x** |
| **GPU利用率** | ~10-30% | ~80-95% | **3-4x** |
| **Batch Size** | 64 | 256 | **4x** |
| **内存效率** | 低 | 高 | **大幅提升** |
| **CPU利用率** | 单核满载 | 多核并行 | **8x** |

## 🔍 性能分析

### 使用性能分析工具

```bash
# 对比原版和优化版性能
python performance_analysis.py \
    results/gpu_run_52005_detailed.log \
    results/gpu_run_52010_detailed.log
```

### 关键指标监控

1. **GPU利用率**: `nvidia-smi` 应显示 >80%
2. **GPU内存**: 应充分利用可用内存
3. **CPU利用率**: 多核并行，不再单核满载
4. **训练速度**: 每个epoch时间大幅减少

## ⚡ 配置对比

### 原版配置 vs 优化配置

| 配置项 | 原版 | 优化版 |
|--------|------|--------|
| 预训练batch_size | 64 | **256** |
| 预训练epochs | 10 | **5** |
| 仿真batch_size | 64 | **128** |
| 仿真steps | 200 | **100** |
| 数据加载 | 单线程 | **4线程** |
| 混合精度 | 无 | **AMP** |
| 内存需求 | 32G | **64G** |

## 🐛 故障排除

### 常见问题及解决方案

1. **GPU内存不足**
   ```bash
   # 减小batch_size
   # 在experiment_optimized.yaml中调整:
   pretrain:
     batch_size: 128  # 从256减少到128
   ```

2. **训练速度仍然慢**
   ```bash
   # 检查是否在GPU节点上运行
   nvidia-smi
   # 检查数据加载并行度
   htop  # 应看到多个Python进程
   ```

3. **作业排队时间长**
   ```bash
   # 减少资源请求
   # 在run_gpu_optimized.sh中调整:
   #SBATCH --time=01:00:00  # 减少时间
   #SBATCH --mem=32G        # 减少内存
   ```

4. **数据加载错误**
   ```bash
   # 检查数据路径和权限
   ls -la data/KuaiRand/Pure/
   # 检查缓存目录
   ls -la data/KuaiRand/cache/
   ```

## 📈 性能优化建议

### 进一步优化选项

1. **调整batch_size**
   - 根据GPU内存适当增大
   - 监控GPU利用率保持在80-95%

2. **数据预处理优化**
   - 启用更多数据加载线程
   - 使用SSD存储提升I/O性能

3. **模型并行**
   - 多GPU训练（如果可用）
   - 模型分片减少内存压力

## 🎯 使用检查清单

- [ ] 确认使用`run_gpu_optimized.sh`提交作业
- [ ] 使用`experiment_optimized.yaml`配置文件
- [ ] 监控GPU利用率 >80%
- [ ] 确认多线程数据加载工作
- [ ] 验证训练速度提升2-4倍
- [ ] 使用性能分析工具对比结果

## 📞 获取帮助

如果遇到问题：

1. **查看监控信息**
   ```bash
   ./monitor_gpu_optimized.sh <作业ID>
   ```

2. **检查详细日志**
   ```bash
   tail -f results/gpu_run_*_detailed.log
   ```

3. **验证环境**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

---

## 🏆 预期效果

使用优化版本后，你应该看到：

- ✅ GPU利用率从10-30%提升到80-95%
- ✅ 训练速度提升2.5-4倍
- ✅ CPU不再单核满载，多核并行工作
- ✅ 内存使用更加高效
- ✅ 每个epoch时间大幅减少

**第一个epoch从20分钟减少到5-8分钟！** 🚀