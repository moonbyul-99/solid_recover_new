#!/usr/bin/env bash
# 临时评估脚本：评估 pair_scratch_bmmc_wc 的所有 checkpoint
# Usage: bash scripts/eval_bmmc_wc.sh [device]

set -euo pipefail

# 参数
DEVICE="${1:-cuda}"
PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/pair_scratch_bmmc_wc"

echo "=========================================="
echo "开始评估 pair_scratch_bmmc_wc"
echo "输出目录: $OUTPUT_DIR"
echo "评估设备: $DEVICE"
echo "=========================================="

cd "$PROJECT_ROOT"

# 直接调用 CLI 模块进行评估
python -m solid_recover.cli.main eval --output-dir "$OUTPUT_DIR" --device "$DEVICE"

echo "=========================================="
echo "✅ 评估完成！"
echo "结果保存在: $OUTPUT_DIR/eval_result/"
echo "=========================================="
