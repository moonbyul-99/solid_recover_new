#!/usr/bin/env bash
# 批量训练脚本：逐个执行 6 个训练任务（case8,9,11 × weight_true/false）
# 最多同时运行 2 个训练任务
# Usage: bash scripts/batch_train_all.sh

set -euo pipefail

# ==================== 配置 ====================
PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"
MAX_PARALLEL=2  # 最大并发任务数

# 定义 6 个训练任务
declare -a TASKS=(
    "case11_pbmc_wc_scratch.yaml:case11_pbmc"
    "case11_pbmc_wo_weight.yaml:case11_pbmc_wo_weight"
    "case8_bmmc_wc_scratch.yaml:case8_bmmc"
    "case8_bmmc_wo_weight.yaml:case8_bmmc_wo_weight"
    "case9_skin_wc_scratch.yaml:case9_skin"
    "case9_skin_wo_weight.yaml:case9_skin_wo_weight"
)

# ==================== 函数定义 ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_train_and_eval() {
    local config_file="$1"
    local task_name="$2"
    local log_file="${PROJECT_ROOT}/logs/batch_${task_name}.out"
    
    # 创建日志目录
    mkdir -p "${PROJECT_ROOT}/logs"
    
    log "=========================================="
    log "开始任务: $task_name"
    log "配置文件: $config_file"
    log "日志文件: $log_file"
    log "=========================================="
    
    cd "$PROJECT_ROOT"
    
    # 运行训练并记录日志
    python -m solid_recover.cli.main train --config "${PROJECT_ROOT}/configs/${config_file}" 2>&1 | tee "$log_file"
    
    # 从配置文件中提取 project_dir
    local project_dir
    project_dir=$(grep "project_dir:" "${PROJECT_ROOT}/configs/${config_file}" | awk '{print $2}' | tr -d '"')
    local output_dir="${PROJECT_ROOT}/${project_dir}"
    
    # 查找实际的输出目录（可能被添加了时间戳）
    local latest_dir=""
    
    # 先检查精确匹配的目录
    if [ -d "$output_dir" ] && [ -f "${output_dir}/config.yaml" ]; then
        latest_dir="$output_dir"
    else
        # 查找带时间戳的子目录
        local found_dirs
        found_dirs=$(find "${PROJECT_ROOT}/outputs" -maxdepth 1 -type d -name "$(basename "$output_dir")*" 2>/dev/null | sort)
        
        if [ -n "$found_dirs" ]; then
            # 取最后一个（最新的）
            latest_dir=$(echo "$found_dirs" | tail -n 1)
        else
            log "❌ 错误: 找不到输出目录 $output_dir"
            return 1
        fi
    fi
    
    # 验证 config.yaml 是否存在
    if [ ! -f "${latest_dir}/config.yaml" ]; then
        log "❌ 错误: 输出目录中缺少 config.yaml: ${latest_dir}"
        return 1
    fi
    
    # 运行评估
    log "开始评估: $task_name"
    python -m solid_recover.cli.main eval --output-dir "$latest_dir" --device cuda 2>&1 | tee -a "$log_file"
    
    log "✅ 任务完成: $task_name"
    log "结果保存在: $latest_dir"
}

# ==================== 主流程 ====================

log "🚀 批量训练开始"
log "任务总数: ${#TASKS[@]}"
log "最大并发数: $MAX_PARALLEL"
log "=========================================="

# 运行中的进程 PID 数组
declare -a PIDS=()
declare -a TASK_NAMES=()

# 逐个启动任务，控制并发数
for task_entry in "${TASKS[@]}"; do
    config_file="${task_entry%%:*}"
    task_name="${task_entry##*:}"
    
    # 等待如果有正在运行的任务达到上限
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        # 检查哪些进程已完成
        declare -a NEW_PIDS=()
        declare -a NEW_NAMES=()
        
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"
            name="${TASK_NAMES[$i]}"
            
            if kill -0 "$pid" 2>/dev/null; then
                # 进程仍在运行
                NEW_PIDS+=("$pid")
                NEW_NAMES+=("$name")
            else
                # 进程已完成，等待并获取退出状态
                wait "$pid" || log "⚠️  任务 $name 退出码非零"
                log "✅ 任务 $name 已完成"
            fi
        done
        
        PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
        TASK_NAMES=("${NEW_NAMES[@]+"${NEW_NAMES[@]}"}")
        
        # 如果还有空间，退出等待循环
        if [ ${#PIDS[@]} -lt $MAX_PARALLEL ]; then
            break
        fi
        
        # 否则等待一会儿再检查
        sleep 5
    done
    
    # 启动新任务
    run_train_and_eval "$config_file" "$task_name" &
    PIDS+=($!)
    TASK_NAMES+=("$task_name")
    
    log "🚀 已启动任务: $task_name (PID: $!)"
    log "当前运行任务数: ${#PIDS[@]}/$MAX_PARALLEL"
done

# 等待所有剩余任务完成
log "=========================================="
log "等待所有任务完成..."
log "=========================================="

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${TASK_NAMES[$i]}"
    
    wait "$pid" || log "⚠️  任务 $name 退出码非零"
    log "✅ 任务 $name 已完成"
done

log "=========================================="
log "🎉 所有任务完成！"
log "=========================================="
log ""
log "任务汇总："
log "  1. case8_bmmc (use_weight=true)"
log "  2. case8_bmmc (use_weight=false)"
log "  3. case9_skin (use_weight=true)"
log "  4. case9_skin (use_weight=false)"
log "  5. case11_pbmc (use_weight=true)"
log "  6. case11_pbmc (use_weight=false)"
log ""
log "日志文件位置: ${PROJECT_ROOT}/logs/"
log "=========================================="
