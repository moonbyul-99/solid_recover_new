#!/usr/bin/env bash
# 训练 zws echo 模型
# Usage: bash scripts/train_zws.sh [device]
#   device: 训练设备，默认为 cuda

set -euo pipefail

DEVICE="${1:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/case_zws.yaml"
LOG_FILE="${PROJECT_ROOT}/logs/train_zws.out"

mkdir -p "${PROJECT_ROOT}/logs"

echo "==========================================" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练 zws echo" | tee -a "$LOG_FILE"
echo "  配置文件: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "  训练设备: $DEVICE" | tee -a "$LOG_FILE"
echo "  日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"

python -m solid_recover.cli.main train --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练完成！" | tee -a "$LOG_FILE"
    echo "  输出目录: outputs/zws_echo/" | tee -a "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练失败！退出码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi
