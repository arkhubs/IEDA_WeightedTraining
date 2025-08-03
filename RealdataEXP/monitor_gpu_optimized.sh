#!/bin/bash

# ========================================
# GPU优化实验监控脚本
# 提供实时GPU监控和性能分析
# ========================================

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <作业ID>"
    echo "示例: $0 52005"
    echo ""
    echo "或者："
    echo "  $0 status  - 查看作业状态"
    echo "  $0 gpu     - 查看GPU状态"
    echo "  $0 log     - 查看最新日志"
    exit 1
fi

JOB_ID=$1
RESULTS_DIR="/home/zhixuanhu/IEDA_WeightedTraining/RealdataEXP/results"

case $JOB_ID in
    "status")
        echo "=== 作业队列状态 ==="
        squeue -u $USER
        echo ""
        echo "=== 最近的作业 ==="
        squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %R" --sort=-i | head -10
        ;;
        
    "gpu")
        echo "=== GPU节点状态 ==="
        sinfo -p gpu-a30 -o "%n %C %G %t"
        echo ""
        echo "=== GPU使用情况 ==="
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi
        else
            echo "当前不在GPU节点上，无法查看GPU状态"
            echo "使用以下命令连接到GPU节点："
            echo "srun --partition=gpu-a30 --gpus-per-node=1 --account=sigroup --pty bash"
        fi
        ;;
        
    "log")
        echo "=== 查找最新日志文件 ==="
        LATEST_LOG=$(ls -t ${RESULTS_DIR}/gpu_run_*_detailed.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "最新日志: $LATEST_LOG"
            echo ""
            echo "=== 最新日志内容 (最后50行) ==="
            tail -50 "$LATEST_LOG"
        else
            echo "没有找到日志文件"
        fi
        ;;
        
    *)
        # 作业ID监控
        echo "=== 作业 $JOB_ID 监控面板 ==="
        echo "时间: $(date)"
        echo ""
        
        # 作业状态
        echo "=== 作业状态 ==="
        squeue -j $JOB_ID --format="%.10i %.15j %.8T %.10M %.6D %R %B" 2>/dev/null || echo "作业 $JOB_ID 不存在或已完成"
        echo ""
        
        # 作业详细信息
        echo "=== 作业详细信息 ==="
        scontrol show job $JOB_ID 2>/dev/null | grep -E "(JobId|JobName|JobState|RunTime|TimeLimit|NumNodes|NumCPUs|Partition|WorkDir)" || echo "无法获取作业详细信息"
        echo ""
        
        # 日志文件
        LOG_FILE="${RESULTS_DIR}/gpu_run_${JOB_ID}_detailed.log"
        OUT_FILE="${RESULTS_DIR}/gpu_run_${JOB_ID}.out"
        ERR_FILE="${RESULTS_DIR}/gpu_run_${JOB_ID}.err"
        
        if [ -f "$LOG_FILE" ]; then
            echo "=== 训练进度 (详细日志最后10行) ==="
            tail -10 "$LOG_FILE"
            echo ""
            
            echo "=== 性能统计 ==="
            # 提取epoch信息
            if grep -q "Epoch" "$LOG_FILE"; then
                echo "预训练进度:"
                grep "Epoch.*平均损失" "$LOG_FILE" | tail -5
                echo ""
            fi
            
            # 提取仿真步骤信息
            if grep -q "Step.*Treatment" "$LOG_FILE"; then
                echo "仿真进度:"
                grep "Step.*Treatment\|Step.*Control" "$LOG_FILE" | tail -10
                echo ""
            fi
            
            # GPU利用率统计（如果有nvidia-smi输出）
            if grep -q "GPU" "$LOG_FILE"; then
                echo "GPU状态检查:"
                grep -A 5 "GPU环境检查\|GPU状态" "$LOG_FILE" | tail -10
            fi
        else
            echo "详细日志文件不存在: $LOG_FILE"
        fi
        
        if [ -f "$OUT_FILE" ]; then
            echo "=== SLURM输出 (最后5行) ==="
            tail -5 "$OUT_FILE"
            echo ""
        fi
        
        if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
            echo "=== 错误日志 ==="
            tail -10 "$ERR_FILE"
            echo ""
        fi
        
        echo "=== 监控命令提示 ==="
        echo "实时监控训练日志: tail -f $LOG_FILE"
        echo "连接到作业节点: srun --jobid=$JOB_ID --overlap --pty bash -i"
        echo "取消作业: scancel $JOB_ID"
        echo ""
        ;;
esac

echo "监控脚本完成"