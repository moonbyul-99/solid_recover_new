#!/usr/bin/env bash
# v4 一键编排：预训练（10K步, v2 LR快速退火） + 各 ckpt 对齐
# Usage: bash scripts/run_pretrain_pair_v4_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

NOW() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=========================================="
echo "[$(NOW)] 🚀 v4 全量实验启动"
echo "=========================================="

# --------------------------------------------------
# 阶段 1: 预训练（10K 步）+ 10K ckpt 配对训练
# --------------------------------------------------
echo ""
echo "[$(NOW)] >>> 阶段 1: 单组学预训练 + 10K ckpt 配对训练"
python scripts/run_pretrain_pair.py configs/hca_renal_cortex_pretrain_pair_v4.yaml 

echo "[$(NOW)] 10K ckpt 训练完成，开始评测"
bash scripts/eval.sh outputs/hca_renal_cortex_pretrain_pair_v4/pair_pretrain 

# --------------------------------------------------
# 阶段 2: 2K / 4K / 6K / 8K ckpt 配对训练
# --------------------------------------------------
for k in 2000 4000 6000 8000; do
  CONFIG="configs/hca_renal_cortex_pt_v4_ckpt${k}.yaml"
  OUTDIR="outputs/hca_renal_cortex_pt_v4_ckpt${k}"
  
  echo ""
  echo "[$(NOW)] >>> ckpt ${k}: 配对训练"
  bash scripts/train.sh "$CONFIG" cuda
  
  echo "[$(NOW)] ckpt ${k}: 评测"
  bash scripts/eval.sh "$OUTDIR" cuda
done

echo ""
echo "=========================================="
echo "[$(NOW)] ✅ v4 全量实验完成"
echo "=========================================="
