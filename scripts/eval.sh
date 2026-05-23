#!/usr/bin/env bash
# Evaluate all checkpoints under a training output directory.
# Usage: bash scripts/eval.sh outputs/pair_scratch_case8_YYYYMMDD [device]
#   device: 评估设备，默认为 cuda（可选）
#
# Examples:
#   bash scripts/eval.sh outputs/pair_scratch_bmmc_wc
#   bash scripts/eval.sh outputs/pair_scratch_bmmc_wc cuda
#   bash scripts/eval.sh outputs/pair_scratch_bmmc_wc cpu

set -euo pipefail

# ==================== 参数检查 ====================

if [ $# -lt 1 ]; then
    echo "错误: 请提供输出目录路径"
    echo "用法: bash scripts/eval.sh <output_dir> [device]"
    echo "示例: bash scripts/eval.sh outputs/pair_scratch_bmmc_wc"
    exit 1
fi

OUTPUT_DIR="$1"
DEVICE="${2:-cuda}"

# 获取项目根目录（脚本所在目录的上级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 如果输出目录路径不是绝对路径，则相对于项目根目录
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

# 从目录名生成日志文件名
DIR_BASENAME=$(basename "$OUTPUT_DIR")
LOG_FILE="${PROJECT_ROOT}/logs/eval_${DIR_BASENAME}.out"

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

print_header "🚀 开始评估"
log "输出目录: $OUTPUT_DIR"
log "评估设备: $DEVICE"
log "日志文件: $LOG_FILE"

# 检查输出目录是否存在
if [ ! -d "$OUTPUT_DIR" ]; then
    log "❌ 错误: 输出目录不存在: $OUTPUT_DIR"
    exit 1
fi

# 检查 config.yaml 是否存在
if [ ! -f "${OUTPUT_DIR}/config.yaml" ]; then
    log "❌ 错误: 输出目录中缺少 config.yaml: ${OUTPUT_DIR}/config.yaml"
    exit 1
fi

cd "$PROJECT_ROOT"

# 运行评估
python -m solid_recover.cli.main eval --output-dir "$OUTPUT_DIR" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    log "✅ 评估完成！"
    log "📊 结果保存在: $OUTPUT_DIR/eval_result/"
    
    # 检查是否生成了汇总图表
    if [ -f "${OUTPUT_DIR}/eval_result/metrics_summary.png" ]; then
        log "📈 汇总图表: $OUTPUT_DIR/eval_result/metrics_summary.png"
    fi
    if [ -f "${OUTPUT_DIR}/eval_result/metrics_summary.json" ]; then
        log "📄 汇总数据: $OUTPUT_DIR/eval_result/metrics_summary.json"
    fi
else
    log "❌ 评估失败！退出码: $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

print_header "✅ 评估流程完成！"
log "📄 日志文件: $LOG_FILE"
