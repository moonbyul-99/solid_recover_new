#!/usr/bin/env bash
# Train a Solid Recover model from a YAML config.
# Usage: bash scripts/train.sh configs/pair_scratch_case8.yaml [device]
#   device: 训练设备，默认为 cuda（可选）
#
# Examples:
#   bash scripts/train.sh configs/case8_bmmc_wc_scratch.yaml
#   bash scripts/train.sh configs/case8_bmmc_wc_scratch.yaml cuda
#   bash scripts/train.sh /absolute/path/to/config.yaml cpu

set -euo pipefail

# ==================== 参数检查 ====================

if [ $# -lt 1 ]; then
    echo "错误: 请提供配置文件路径"
    echo "用法: bash scripts/train.sh <config_path> [device]"
    echo "示例: bash scripts/train.sh configs/case8_bmmc_wc_scratch.yaml"
    exit 1
fi

CONFIG_PATH="$1"
DEVICE="${2:-cuda}"

# 获取项目根目录（脚本所在目录的上级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 如果配置文件路径不是绝对路径，则相对于项目根目录
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi

# 从配置文件名生成日志文件名（去掉扩展名）
CONFIG_BASENAME=$(basename "$CONFIG_PATH" .yaml)
LOG_FILE="${PROJECT_ROOT}/logs/train_${CONFIG_BASENAME}.out"

# ==================== 函数定义 ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

print_header() {
    log "=========================================="
    log "$1"
    log "=========================================="
}

# ==================== 主流程 ====================

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs"

print_header "🚀 开始训练"
log "配置文件: $CONFIG_PATH"
log "训练设备: $DEVICE"
log "日志文件: $LOG_FILE"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    log "❌ 错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

cd "$PROJECT_ROOT"

# 运行训练
python -m solid_recover.cli.main train --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log "✅ 训练完成！"
    
    # 从配置文件中提取 project_dir
    PROJECT_DIR=$(grep "project_dir:" "$CONFIG_PATH" | awk '{print $2}' | tr -d '"')
    
    if [ -n "$PROJECT_DIR" ]; then
        # 查找实际的输出目录（可能被添加了时间戳）
        BASE_DIR="${PROJECT_ROOT}/${PROJECT_DIR}"
        
        if [ -d "$BASE_DIR" ]; then
            # 检查是否有带时间戳的子目录
            LATEST_DIR=$(find "${PROJECT_ROOT}/outputs" -maxdepth 1 -type d -name "$(basename "$BASE_DIR")*" 2>/dev/null | sort | tail -n 1)
            
            if [ -z "$LATEST_DIR" ]; then
                LATEST_DIR="$BASE_DIR"
            fi
            
            log "📁 输出目录: $LATEST_DIR"
        fi
    fi
else
    log "❌ 训练失败！退出码: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

print_header "✅ 训练流程完成！"
log "📄 日志文件: $LOG_FILE"
