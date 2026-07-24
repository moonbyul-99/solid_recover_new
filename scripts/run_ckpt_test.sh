#!/usr/bin/env bash
# 串行
for k in 2000 4000 6000 8000; do
  bash scripts/train.sh configs/hca_renal_cortex_pt_v3_ckpt${k}.yaml cuda
  bash scripts/eval.sh outputs/hca_renal_cortex_pt_v3_ckpt${k} cuda
done