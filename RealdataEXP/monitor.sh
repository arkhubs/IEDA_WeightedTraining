#!/bin/bash

# ==============================================================================
#  实时系统资源监控脚本 (CPU, GPU, Memory)
# ==============================================================================

# --- 可配置项 ---
# 数据刷新间隔时间（秒）
SLEEP_INTERVAL=2

# --- 颜色定义 ---
# 使用 tput 来确保最大的终端兼容性
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
CYAN=$(tput setaf 6)
NC=$(tput sgr0) # No Color (恢复默认)

# --- 主循环 ---
while true; do
    # 清理屏幕，准备刷新
    clear

    # 打印标题和当前时间
    echo "${CYAN}===================== 系统资源实时监控 =====================${NC}"
    echo "            刷新时间: $(date '+%Y-%m-%d %H:%M:%S') - 刷新间隔: ${SLEEP_INTERVAL}s"
    echo ""

    # --- GPU 监控 (仅当 nvidia-smi 命令存在时) ---
    if command -v nvidia-smi &> /dev/null; then
        echo "${YELLOW}---------------------- GPU 资源监控 ----------------------${NC}"
        # 使用 nvidia-smi 查询关键指标，并格式化输出
        # 查询: GPU索引, 名称, GPU利用率, 显存使用, 总显存, 功耗, 温度
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r index name util mem_used mem_total power temp; do
            # 去除 name 中的前后空格
            name=$(echo "$name" | awk '{$1=$1};1')
            printf "${GREEN}[GPU %s]${NC}: %-20s | ${BLUE}使用率${NC}: %3s%% | ${BLUE}显存${NC}: %5s / %-5s MiB | ${BLUE}功耗${NC}: %sW | ${BLUE}温度${NC}: %s°C\n" \
                "$index" "$name" "$util" "$mem_used" "$mem_total" "$power" "$temp"
        done
        echo ""
    fi

    # --- CPU 监控 ---
    echo "${YELLOW}---------------------- CPU 资源监控 ----------------------${NC}"
    # 获取 CPU 使用率
    # 从 top 命令中提取 %Cpu(s) 行，并格式化
    CPU_STATS=$(top -b -n 1 | grep '%Cpu(s)')
    CPU_USER=$(echo "$CPU_STATS" | awk '{print $2}')
    CPU_SYS=$(echo "$CPU_STATS" | awk '{print $4}')
    CPU_IDLE=$(echo "$CPU_STATS" | awk '{print $8}')
    CPU_USED=$(printf "%.2f" $(echo "100 - $CPU_IDLE" | bc))

    # 获取系统平均负载
    LOAD_AVG=$(uptime | awk -F'load average: ' '{print $2}')

    printf "${BLUE}CPU 核心总使用率${NC}: %s%%  |  ${BLUE}用户态${NC}: %s%%  |  ${BLUE}系统态${NC}: %s%%  |  ${BLUE}空闲${NC}: %s%%\n" \
        "$CPU_USED" "$CPU_USER" "$CPU_SYS" "$CPU_IDLE"
    printf "${BLUE}系统平均负载 (1m, 5m, 15m)${NC}: %s\n" "$LOAD_AVG"
    echo ""

    # --- 内存 (RAM) 监控 ---
    echo "${YELLOW}---------------------- 内存 (RAM) 监控 ---------------------${NC}"
    # 使用 free -h 获取人类可读的内存和交换空间使用情况
    free -h | sed 's/shared/& /' # 增加一个空格让列对齐更好看
    echo ""

    # 等待下一次刷新
    sleep $SLEEP_INTERVAL
done