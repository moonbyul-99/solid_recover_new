#!/usr/bin/env bash
# 手动完成 case11 两个任务的评估
# Usage: bash scripts/eval_case11.sh

set -euo pipefail

PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 评估 case11 with weight
log "=========================================="
log "评估 case11_pbmc (use_weight=true)"
log "=========================================="
cd "$PROJECT_ROOT"
python -m solid_recover.cli.main eval \
    --output-dir "${PROJECT_ROOT}/outputs/pair_scratch_mus_skin_wc" \
    --device cuda

log ""
log "=========================================="
log "评估 case11_pbmc (use_weight=false)"
log "=========================================="
python -m solid_recover.cli.main eval \
    --output-dir "${PROJECT_ROOT}/outputs/pair_scratch_skin_wo_weight" \
    --device cuda

log ""
log "=========================================="
log "✅ case11 两个任务评估完成！"
log "=========================================="
