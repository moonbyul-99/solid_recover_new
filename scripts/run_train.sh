#!/usr/bin/env bash
# 通用训练启动脚本：一键启动训练并在完成后自动评估
# Usage: bash scripts/run_train.sh <config_path> [device]
#   config_path: YAML 配置文件路径（必填）
#   device: 评估设备，默认为 cuda（可选）
#
# Examples:
#   bash scripts/run_train.sh configs/case8_bmmc_wc_scratch.yaml
#   bash scripts/run_train.sh configs/case8_bmmc_wc_scratch.yaml cuda
#   bash scripts/run_train.sh /absolute/path/to/config.yaml cpu

set -euo pipefail

# ==================== 参数检查 ====================

if [ $# -lt 1 ]; then
    echo "错误: 请提供配置文件路径"
    echo "用法: bash scripts/run_train.sh <config_path> [device]"
    echo "示例: bash scripts/run_train.sh configs/case8_bmmc_wc_scratch.yaml"
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
LOG_FILE="${PROJECT_ROOT}/${CONFIG_BASENAME}.out"

# ==================== 函数定义 ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

print_header() {
    log "=========================================="
    log "$1"
    log "=========================================="
}

run_train() {
    print_header "开始训练"
    log "配置文件: $CONFIG_PATH"
    log "项目根目录: $PROJECT_ROOT"
    
    cd "$PROJECT_ROOT"
    
    # 直接调用 CLI 模块进行训练
    python -m solid_recover.cli.main train --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
    
    log "✅ 训练完成！"
}

run_eval() {
    # 从配置文件中提取 project_dir
    PROJECT_DIR=$(grep "project_dir:" "$CONFIG_PATH" | awk '{print $2}' | tr -d '"')
    
    if [ -z "$PROJECT_DIR" ]; then
        log "⚠️  警告: 配置文件中未找到 project_dir，跳过评估"
        return 0
    fi
    
    # 查找实际的输出目录（可能被添加了时间戳）
    BASE_DIR="${PROJECT_ROOT}/${PROJECT_DIR}"
    
    if [ -d "$BASE_DIR" ]; then
        # 检查是否有带时间戳的子目录
        LATEST_DIR=$(find "${BASE_DIR}"* -maxdepth 0 -type d 2>/dev/null | sort | tail -n 1)
        
        if [ -z "$LATEST_DIR" ]; then
            LATEST_DIR="$BASE_DIR"
        fi
        
        print_header "开始评估"
        log "输出目录: $LATEST_DIR"
        log "评估设备: $DEVICE"
        
        cd "$PROJECT_ROOT"
        
        # 直接调用 CLI 模块进行评估
        python -m solid_recover.cli.main eval --output-dir "$LATEST_DIR" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
        
        log "✅ 评估完成！"
        log "📁 结果保存在: $LATEST_DIR/eval_result/"
    else
        log "⚠️  警告: 未找到输出目录 $BASE_DIR，跳过评估"
    fi
}

# ==================== 主流程 ====================

# 清空或创建日志文件
> "$LOG_FILE"

print_header "🚀 启动训练流程"
log "配置文件: $CONFIG_PATH"
log "日志文件: $LOG_FILE"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    log "❌ 错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 运行训练
run_train

# 运行评估
run_eval

print_header "✅ 全部流程完成！"
log "📄 日志文件: $LOG_FILE"
