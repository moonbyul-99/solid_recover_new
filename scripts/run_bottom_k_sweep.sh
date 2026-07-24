#!/usr/bin/env bash
# Phase 2: bottom_k_ratio sweep — 串行训练 + 评测
# Base: t=0.15, to_gpu: true
# Sweep: bottom_k_ratio ∈ {0.01, 0.05, 0.10, 0.15, 0.20}
#
# Usage:
#   bash scripts/run_bottom_k_sweep.sh [device]
#   bash scripts/run_bottom_k_sweep.sh cuda
#   bash scripts/run_bottom_k_sweep.sh cpu

set -euo pipefail

DEVICE="${1:-cuda}"

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Sweep values (ordered: fine → coarse)
RATIOS=("0.01" "0.05" "0.10" "0.15" "0.20")

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
print_header "Phase 2: bottom_k_ratio sweep (t=0.15)"
log "Device: $DEVICE"
log "Sweep: bottom_k_ratio ∈ {${RATIOS[*]}}"

FAILED=()

for ratio in "${RATIOS[@]}"; do
    rid=$(echo "$ratio" | tr -d '.')
    CONFIG="configs/hca_renal_cortex_bottom_k_${rid}.yaml"
    OUTPUT="outputs/hca_renal_cortex_bottom_k_${rid}"

    echo ""
    print_header "bottom_k_ratio = ${ratio}  (${rid})"

    # --- train ---
    log "Training ${rid} ..."
    if ! bash scripts/train.sh "$CONFIG" "$DEVICE"; then
        log "❌ Training failed for bottom_k=${ratio}"
        FAILED+=("train:${ratio}")
        continue
    fi
    log "✅ Training done: ${rid}"

    # --- eval ---
    log "Evaluating ${rid} ..."
    if ! bash scripts/eval.sh "$OUTPUT" "$DEVICE"; then
        log "❌ Eval failed for bottom_k=${ratio}"
        FAILED+=("eval:${ratio}")
        continue
    fi
    log "✅ Eval done: ${rid}"
done

# --------------- summary ---------------
echo ""
print_header "Sweep Complete"
if [ ${#FAILED[@]} -eq 0 ]; then
    log "✅ All 5 jobs succeeded!"
else
    log "❌ ${#FAILED[@]} failures: ${FAILED[*]}"
fi

echo ""
log "Results at:"
for ratio in "${RATIOS[@]}"; do
    rid=$(echo "$ratio" | tr -d '.')
    echo "  outputs/hca_renal_cortex_bottom_k_${rid}/eval_result/metrics_summary.json"
done
