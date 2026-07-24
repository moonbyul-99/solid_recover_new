#!/usr/bin/env bash
# Phase 1: Temperature sweep (0.40, 0.50, 0.70, 1.00) - 串行训练 + 评估
set -euo pipefail

PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"
cd "$PROJECT_ROOT"

declare -a TEMPS=("0.40" "0.50" "0.70" "1.00")
declare -a CONFIGS=(
    "configs/hca_renal_cortex_pretrain_t040.yaml"
    "configs/hca_renal_cortex_pretrain_t050.yaml"
    "configs/hca_renal_cortex_pretrain_t070.yaml"
    "configs/hca_renal_cortex_pretrain_t100.yaml"
)
declare -a PROJECT_DIRS=(
    "outputs/hca_renal_cortex_pretrain_t040"
    "outputs/hca_renal_cortex_pretrain_t050"
    "outputs/hca_renal_cortex_pretrain_t070"
    "outputs/hca_renal_cortex_pretrain_t100"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

for i in "${!TEMPS[@]}"; do
    TEMP="${TEMPS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    PROJECT_DIR="${PROJECT_DIRS[$i]}"
    LOGFILE="${PROJECT_ROOT}/logs/train_hca_renal_cortex_pretrain_t$(echo $TEMP | tr -d '.').out"
    
    log "=========================================="
    log "Phase 1: 开始训练 t=${TEMP}"
    log "Config: ${CONFIG}"
    log "Output: ${PROJECT_DIR}"
    log "=========================================="
    
    # Training
    bash scripts/train.sh "${CONFIG}" cuda 2>&1 | tee "${LOGFILE}"
    
    log "训练完成 t=${TEMP}，开始评估..."
    
    # Find actual output directory (may have timestamp suffix)
    ACTUAL_DIR=$(find "${PROJECT_ROOT}/outputs" -maxdepth 1 -type d -name "$(basename ${PROJECT_DIR})*" 2>/dev/null | sort | tail -n 1)
    if [ -z "${ACTUAL_DIR}" ]; then
        ACTUAL_DIR="${PROJECT_ROOT}/${PROJECT_DIR}"
    fi
    
    # Evaluation
    bash scripts/eval.sh "${ACTUAL_DIR}" cuda 2>&1 | tee -a "${LOGFILE}"
    
    log "=========================================="
    log "Phase 1: t=${TEMP} 训练+评估完成"
    log "=========================================="
    log ""
done

log "=========================================="
log "Phase 1 全部完成！请开始汇总报告。"
log "=========================================="
