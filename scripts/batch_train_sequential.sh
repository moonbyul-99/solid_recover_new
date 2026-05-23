#!/usr/bin/env bash
# 批量训练脚本（顺序执行版本）：逐个执行 6 个训练任务
# 适合需要严格控制资源使用的场景
# Usage: bash scripts/batch_train_sequential.sh

set -euo pipefail

# ==================== 配置 ====================
PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"

# 定义 6 个训练任务
declare -a TASKS=(
    "case8_bmmc_wc_scratch.yaml:case8_bmmc_with_weight"
    "case8_bmmc_wo_weight.yaml:case8_bmmc_without_weight"
    "case9_skin_wc_scratch.yaml:case9_skin_with_weight"
    "case9_skin_wo_weight.yaml:case9_skin_without_weight"
    "case11_pbmc_wc_scratch.yaml:case11_pbmc_with_weight"
    "case11_pbmc_wo_weight.yaml:case11_pbmc_without_weight"
)

# ==================== 函数定义 ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_train_and_eval() {
    local config_file="$1"
    local task_name="$2"
    local log_file="${PROJECT_ROOT}/logs/seq_${task_name}.out"
    
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
    log ""
}

# ==================== 主流程 ====================

log "🚀 批量训练开始（顺序执行）"
log "任务总数: ${#TASKS[@]}"
log "=========================================="

# 顺序执行所有任务
for task_entry in "${TASKS[@]}"; do
    config_file="${task_entry%%:*}"
    task_name="${task_entry##*:}"
    
    run_train_and_eval "$config_file" "$task_name"
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
