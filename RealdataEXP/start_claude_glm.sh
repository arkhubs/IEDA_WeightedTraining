#!/bin/bash

# Claude Code + GLM-4.5 项目启动脚本 (专为 RealdataEXP 项目)
# 使用方法：./start_claude_glm.sh YOUR_API_KEY

# 检查是否提供了API key
if [ -z "$1" ]; then
    echo "========================================"
    echo "    RealdataEXP 项目 Claude Code 启动"
    echo "========================================"
    echo ""
    echo "使用方法: $0 <GLM_API_KEY>"
    echo "请先获取GLM-4.5 API key: https://bigmodel.cn/usercenter/proj-mgmt/apikeys"
    echo ""
    echo "示例："
    echo "  $0 glm-xxxxxxxxxxxxxxxxxxxxx"
    echo ""
    exit 1
fi

# 进入项目目录
cd /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP

# 设置环境变量
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_AUTH_TOKEN=7609083a12204da8b9c93dc3beec864d.hfeMWNl2HQB9VCKQ

echo "========================================"
echo "    RealdataEXP 项目 Claude Code 启动"
echo "========================================"
echo "项目目录: $(pwd)"
echo "Base URL: $ANTHROPIC_BASE_URL"
echo "API Token: ${ANTHROPIC_AUTH_TOKEN:0:8}..."
echo "========================================"
echo ""
echo "🚀 正在启动 Claude Code..."
echo "💡 提示：您现在在 RealdataEXP 项目中使用 Claude Code"
echo "📁 Claude Code 可以访问当前项目的所有文件"
echo "⚡ 成本：输入 0.8元/百万tokens，输出 2元/百万tokens"
echo ""
echo "常用命令提示："
echo "  - 查看项目结构：请列出项目的文件结构"
echo "  - 分析代码：请分析 main.py 的功能"
echo "  - 优化代码：请优化 libs/ 目录下的代码结构"
echo "  - 添加功能：请为项目添加 XXX 功能"
echo ""
echo "========================================"

# 启动Claude Code，限制访问目录为当前项目
claude --add-dir /home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP
