#!/usr/bin/env bash
# CLIP Adaptive-Weight 基准测试：30 个随机种子并行测试
# Usage: bash scripts/run_clip_benchmark.sh
#
# 默认配置：
#   - 30 seeds, 5 并行 workers
#   - 200 epochs, snapshot_interval=20
#   - Tree 分化数据, 15 clusters

set -euo pipefail

# ==================== 路径配置 ====================
PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"
LOG_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results"

# ==================== 可调参数 ====================
SEEDS=30
WORKERS=5
EPOCHS=200
SNAPSHOT_INTERVAL=20
N_CLUSTERS=15

# ==================== 函数定义 ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ==================== 主流程 ====================

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/clip_benchmark_${TIMESTAMP}.out"

log "==============================================" | tee "$LOG_FILE"
log "CLIP Adaptive-Weight Benchmark" | tee -a "$LOG_FILE"
log "==============================================" | tee -a "$LOG_FILE"
log "Seeds:              $SEEDS" | tee -a "$LOG_FILE"
log "Workers (parallel): $WORKERS" | tee -a "$LOG_FILE"
log "Epochs:             $EPOCHS" | tee -a "$LOG_FILE"
log "Snapshot interval:  $SNAPSHOT_INTERVAL" | tee -a "$LOG_FILE"
log "Data clusters:      $N_CLUSTERS" | tee -a "$LOG_FILE"
log "Log file:           $LOG_FILE" | tee -a "$LOG_FILE"
log "==============================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"

log "Starting benchmark..." | tee -a "$LOG_FILE"

python -m examples.clip_simulation.run_benchmark \
    --seeds "$SEEDS" \
    --workers "$WORKERS" \
    --epochs "$EPOCHS" \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --n-clusters "$N_CLUSTERS" \
    --output-dir "${RESULTS_DIR}/benchmark_${TIMESTAMP}" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    log "✅ Benchmark finished successfully!" | tee -a "$LOG_FILE"
    log " Results: ${RESULTS_DIR}/benchmark_${TIMESTAMP}" | tee -a "$LOG_FILE"
    log " Analysis notebook: examples/clip_simulation/benchmark_analysis.ipynb" | tee -a "$LOG_FILE"
else
    log "❌ Benchmark failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi
