# RealdataEXP 项目 Claude Code 配置指南

## 🎯 项目专用配置

这个配置专门为 `/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP` 项目定制，让您可以在该项目中高效使用 Claude Code + GLM-4.5。

## 📁 项目结构

```
RealdataEXP/
├── configs/                 # 配置文件目录
├── data/                    # 数据目录  
├── libs/                    # 库文件目录
├── results/                 # 结果输出目录
├── main.py                  # 主程序入口
├── check_environment.py     # 环境检查脚本
├── performance_analysis.py  # 性能分析脚本
├── start_claude_glm.sh     # Claude Code 启动脚本
└── README.md               # 项目说明
```

## 🚀 快速启动

### 1. 获取 GLM-4.5 API Key
访问：https://bigmodel.cn/usercenter/proj-mgmt/apikeys
- 注册账户并实名认证
- 创建新的API Key

### 2. 在项目中启动 Claude Code
```bash
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
./start_claude_glm.sh YOUR_API_KEY
```

## 💡 项目专用功能

### 代码分析与优化
```
请分析 main.py 的功能，并提出优化建议
```

### 项目结构分析
```
请分析整个项目的代码结构，并绘制模块依赖图
```

### 性能优化
```
请分析 performance_analysis.py 并提出GPU优化建议
```

### 配置文件管理
```
请检查 configs/ 目录的配置文件，并建议标准化配置格式
```

### 数据处理优化
```
请分析 data/ 目录的数据处理流程，提出优化方案
```

## 🔧 Claude Code 高级用法

### 1. 项目范围限制
启动脚本已配置 `--add-dir` 参数，Claude Code 只能访问 RealdataEXP 项目目录，确保安全性。

### 2. 多任务并行处理
如果需要同时处理多个功能模块：

```bash
# 终端1：处理数据分析
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
./start_claude_glm.sh YOUR_API_KEY

# 终端2：处理性能优化
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
./start_claude_glm.sh YOUR_API_KEY
```

### 3. 会话管理
```bash
# 继续上次的对话
claude -c

# 恢复特定会话
claude -r [session_id]
```

## 📝 常用提示词模板

### 代码审查
```
请对以下文件进行代码审查，关注：
1. 代码质量和最佳实践
2. 性能优化点
3. 潜在的bug
4. 可维护性改进建议

文件：[指定文件路径]
```

### 功能实现
```
请为 RealdataEXP 项目实现以下功能：
[详细描述功能需求]

要求：
1. 遵循项目现有的代码风格
2. 与现有模块良好集成
3. 添加适当的错误处理
4. 包含必要的测试代码
```

### 性能分析
```
请分析项目的性能瓶颈：
1. 检查 main.py 的执行流程
2. 分析 libs/ 中各模块的性能
3. 识别可能的GPU优化点
4. 提出具体的优化方案
```

### 文档生成
```
请为项目生成详细的技术文档：
1. 模块功能说明
2. API接口文档  
3. 配置参数说明
4. 使用示例
```

## 🎯 项目特定的Claude Code技巧

### 1. 智能代码导航
```
# 让Claude Code帮您理解项目结构
请绘制项目的模块依赖关系图，标明各文件的作用
```

### 2. 批量文件处理
```
# 批量优化libs目录下的所有Python文件
请检查libs目录下的所有.py文件，统一代码风格并优化性能
```

### 3. 配置文件管理
```
# 统一配置格式
请标准化configs目录下的所有配置文件，使用YAML格式
```

### 4. 测试用例生成
```
# 自动生成测试用例
请为main.py中的所有函数生成对应的单元测试
```

## ⚡ 性能优化建议

1. **数据处理优化**：让Claude Code分析数据加载和处理流程
2. **GPU利用优化**：分析GPU使用效率，提出CUDA优化建议
3. **内存管理**：检查内存泄漏和优化内存使用
4. **并行计算**：识别可并行化的计算任务

## 🛠 故障排除

### Claude Code无法访问项目文件
```bash
# 确保在正确的目录启动
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
./start_claude_glm.sh YOUR_API_KEY
```

### API连接问题
1. 检查API key是否正确
2. 确认网络连接正常
3. 验证API余额充足

### 会话管理问题
```bash
# 查看所有会话
claude config list-sessions

# 清理旧会话
claude config clear-sessions
```

## 📚 学习资源

- [Claude Code官方文档](https://github.com/anthropics/claude-code)
- [GLM-4.5 API文档](https://bigmodel.cn/dev/api)
- [项目原始知乎文章](./Claude%20Code%20用法全面拆解！26%20项核心功能%20＋%20实战技巧（建议收藏！）%20-%20知乎%20(2025_8_3%2015：15：22).html)

---

**配置完成！现在您可以在 RealdataEXP 项目中享受强大的 Claude Code + GLM-4.5 编程体验！** 🚀
