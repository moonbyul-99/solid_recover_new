#!/usr/bin/env bash
# Phase 3: clip_weight sweep — 串行训练 + 评测
# Base: t=0.15, bottom_k_ratio=0.15, to_gpu: true
# Sweep: clip_weight ∈ {1, 5, 10, 40, 60, 100}
#
# Usage:
#   bash scripts/run_cw_sweep.sh [device]
#   bash scripts/run_cw_sweep.sh cuda

set -euo pipefail

DEVICE="${1:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CLIP_WEIGHTS=(1 5 10 40 60 100)

# --------------- helpers ---------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}
print_header() {
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# --------------- main ---------------
print_header "Phase 3: clip_weight sweep (t=0.15, bottom_k=0.15)"
log "Device: $DEVICE"
log "Sweep: clip_weight ∈ {${CLIP_WEIGHTS[*]}}"

FAILED=()

for cw in "${CLIP_WEIGHTS[@]}"; do
    CONFIG="configs/hca_renal_cortex_cw_${cw}.yaml"
    OUTPUT="outputs/hca_renal_cortex_cw_${cw}"

    echo ""
    print_header "clip_weight = ${cw}"

    # --- train ---
    log "Training cw_${cw} ..."
    if ! bash scripts/train.sh "$CONFIG" "$DEVICE"; then
        log "❌ Training failed for clip_weight=${cw}"
        FAILED+=("train:${cw}")
        continue
    fi
    log "✅ Training done: cw_${cw}"

    # --- eval ---
    log "Evaluating cw_${cw} ..."
    if ! bash scripts/eval.sh "$OUTPUT" "$DEVICE"; then
        log "❌ Eval failed for clip_weight=${cw}"
        FAILED+=("eval:${cw}")
        continue
    fi
    log "✅ Eval done: cw_${cw}"
done

# --------------- summary ---------------
echo ""
print_header "Sweep Complete"
if [ ${#FAILED[@]} -eq 0 ]; then
    log "✅ All 6 jobs succeeded!"
else
    log "❌ ${#FAILED[@]} failures: ${FAILED[*]}"
fi

echo ""
log "Results at:"
for cw in "${CLIP_WEIGHTS[@]}"; do
    echo "  outputs/hca_renal_cortex_cw_${cw}/eval_result/metrics_summary.json"
done
