#!/usr/bin/env bash
# 一键启动 PBMC (case_11) 三策略消融实验（后台运行，日志写入 logs/case11_strategy_ablation.out）
# 用法: bash scripts/run_case11_strategy_ablation.sh [strategies] [seeds]
#   示例:
#     bash scripts/run_case11_strategy_ablation.sh                      # 全量 3 策略 x 3 seeds
#     bash scripts/run_case11_strategy_ablation.sh scratch 0            # 冒烟: 单策略单 seed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

CONDA_SH="/home/rsun@ZHANGroup.local/anaconda3/etc/profile.d/conda.sh"
LOG_FILE="logs/case11_strategy_ablation.out"

STRATEGIES="${1:-aug_pt,scratch,train_pt}"
SEEDS="${2:-0,7,42}"

echo "启动 PBMC 三策略消融实验: strategies=$STRATEGIES seeds=$SEEDS"
echo "日志: $LOG_FILE"

nohup bash -c "source '$CONDA_SH' && conda activate snapatac && python -u scripts/run_case11_strategy_ablation.py --strategies '$STRATEGIES' --seeds '$SEEDS'" > "$LOG_FILE" 2>&1 &

echo "已后台启动 (PID: $!)"
echo "查看进度: tail -f logs/case11_strategy_ablation.out"
