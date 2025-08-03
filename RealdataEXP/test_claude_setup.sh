#!/bin/bash

# RealdataEXP 项目环境检查和Claude Code设置脚本

echo "========================================"
echo "  RealdataEXP 项目 Claude Code 环境检查"
echo "========================================"

# 检查当前目录
current_dir=$(pwd)
expected_dir="/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP"

if [ "$current_dir" != "$expected_dir" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    echo "   当前目录: $current_dir"
    echo "   期望目录: $expected_dir"
    echo ""
    echo "解决方法："
    echo "   cd $expected_dir"
    echo "   ./test_claude_setup.sh"
    exit 1
fi

echo "✅ 目录检查：当前在正确的项目目录"

# 检查Node.js和npm
echo ""
echo "检查 Node.js 环境..."
if command -v node &> /dev/null; then
    node_version=$(node --version)
    echo "✅ Node.js: $node_version"
else
    echo "❌ Node.js 未安装"
    echo "   请运行: conda install -c conda-forge nodejs"
    exit 1
fi

if command -v npm &> /dev/null; then
    npm_version=$(npm --version)
    echo "✅ NPM: v$npm_version"
else
    echo "❌ NPM 未安装"
    exit 1
fi

# 检查Claude Code
echo ""
echo "检查 Claude Code..."
if command -v claude &> /dev/null; then
    claude_version=$(claude --version 2>&1 | head -1)
    echo "✅ Claude Code: $claude_version"
else
    echo "❌ Claude Code 未安装"
    echo "   请运行: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# 检查项目文件结构
echo ""
echo "检查项目文件结构..."
required_files=(
    "main.py"
    "configs/"
    "libs/"
    "data/"
    "start_claude_glm.sh"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "⚠️  警告：部分项目文件缺失，请检查项目完整性"
fi

# 检查脚本权限
echo ""
echo "检查脚本权限..."
if [ -x "start_claude_glm.sh" ]; then
    echo "✅ start_claude_glm.sh 可执行"
else
    echo "❌ start_claude_glm.sh 不可执行"
    echo "   修复: chmod +x start_claude_glm.sh"
    chmod +x start_claude_glm.sh
    echo "✅ 已修复执行权限"
fi

# 检查环境变量
echo ""
echo "检查环境变量..."
if [ -n "$ANTHROPIC_BASE_URL" ]; then
    echo "✅ ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
else
    echo "ℹ️  ANTHROPIC_BASE_URL: 未设置 (启动时会自动设置)"
fi

if [ -n "$ANTHROPIC_AUTH_TOKEN" ]; then
    echo "✅ ANTHROPIC_AUTH_TOKEN: ${ANTHROPIC_AUTH_TOKEN:0:8}..."
else
    echo "ℹ️  ANTHROPIC_AUTH_TOKEN: 未设置 (需要在启动时提供)"
fi

# 总结
echo ""
echo "========================================"
echo "            环境检查完成"
echo "========================================"

if [ "$all_files_exist" = true ]; then
    echo "🎉 所有检查通过！"
    echo ""
    echo "使用方法："
    echo "1. 获取GLM-4.5 API Key: https://bigmodel.cn/usercenter/proj-mgmt/apikeys"
    echo "2. 启动Claude Code:"
    echo "   ./start_claude_glm.sh YOUR_API_KEY"
    echo ""
    echo "示例："
    echo "   ./start_claude_glm.sh glm-xxxxxxxxxxxxxxxxxxxxx"
else
    echo "⚠️  环境检查发现一些问题，请先解决后再使用Claude Code"
fi

echo ""
echo "项目配置文档: Claude_GLM_项目配置.md"
echo "========================================"
