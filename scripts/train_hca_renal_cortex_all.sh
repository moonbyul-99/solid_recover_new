#!/usr/bin/env bash
# HCA Renal Cortex 完整训练流程脚本（串行执行）
# 包含: SR (with/without adaptive weight), pretrain+pair, 以及对比方法 (scpair, multivi, cobolt, scb)
# 所有训练使用 nohup 挂起，输出到对应的 .out 文件
# 采用串行执行方式，逐个任务完成后再启动下一个

set -euo pipefail

# ==================== 配置 ====================
PROJECT_ROOT="/home/rsun@ZHANGroup.local/solid_recover_main"
COMPARE_ROOT="/home/rsun@ZHANGroup.local/solid-recover/compare_method"

LOG() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

HEADER() {
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "============================================================"
    echo ""
}

# ==================== 任务 1-3 已完成，以下为注释 ====================
: << 'SKIP_1_3'
# ==================== 任务 1: SR with adaptive weight scratch ====================
HEADER "任务 1: SR with adaptive weight scratch 训练"

LOG "工作目录: ${PROJECT_ROOT}"
LOG "配置文件: ${PROJECT_ROOT}/configs/case_renal_cortex.yaml"
LOG "日志文件: ${PROJECT_ROOT}/logs/sr_aw_scratch_train.out"

cd "${PROJECT_ROOT}"
conda activate snapatac 2>/dev/null || source activate snapatac 2>/dev/null || true

nohup bash scripts/run_train.sh configs/case_renal_cortex.yaml cuda \
    > logs/sr_aw_scratch_train.out 2>&1 &

SR_AW_PID=$!
LOG "✅ SR-AW-Scratch 训练已启动 (PID: ${SR_AW_PID})"
LOG "⏳ 等待任务完成..."
wait ${SR_AW_PID}
LOG "✅ SR-AW-Scratch 训练完成！"
LOG "📄 日志: ${PROJECT_ROOT}/logs/sr_aw_scratch_train.out"

# ==================== 任务 2: SR without adaptive weight scratch ====================
HEADER "任务 2: SR without adaptive weight scratch 训练"

# 创建临时配置文件（修改 use_weight 和 loss 相关参数）
TMP_CONFIG="${PROJECT_ROOT}/configs/case_renal_cortex_wo_weight.yaml"
LOG "生成无权重配置文件: ${TMP_CONFIG}"

cat > "${TMP_CONFIG}" << 'EOF'
# pair_config.yaml - 适配重构后项目的 mus_kidney_sex 配置（无 adaptive weight）

task: "pair_scratch"

data:
  train_data_path: "/home/rsun@ZHANGroup.local/solid-recover/data/multi_omics/hca_renal_cortex/train.h5mu"
  test_data_path: "/home/rsun@ZHANGroup.local/solid-recover/data/multi_omics/hca_renal_cortex/test.h5mu"
  key_1: "rna_count"
  key_2: "peak_count"
  batch_size: 512

model:
  feature_num_1: 14577
  feature_num_2: 101123
  hidden_params_1: [1024, 256]
  hidden_params_2: [1024, 256]
  embed_dim: 64
  use_rmsnorm: true
  use_residual: true
  dropout_p: 0.0

training:
  project_dir: "outputs/hca_renal_cortex_wo_weight"
  train_steps: 6000
  eval_points: 200
  save_points: 500
  device: "cuda"

optimizer:
  lr: 1e-3
  warmup_steps: 500
  steady_1_steps: 500
  cosine_anneal_steps: 5000
  min_lr: 1e-4

loss:
  vae_beta_1: 1.0
  vae_beta_2: 1.0
  clip_weight: 20.0
  cross_recon_1: 0.75
  cross_recon_2: 0.75
  temperature: 0.12
  use_weight: false
EOF

LOG "工作目录: ${PROJECT_ROOT}"
LOG "配置文件: ${TMP_CONFIG}"
LOG "日志文件: ${PROJECT_ROOT}/logs/sr_wo_aw_scratch_train.out"

cd "${PROJECT_ROOT}"
nohup bash scripts/run_train.sh "${TMP_CONFIG}" cuda \
    > logs/sr_wo_aw_scratch_train.out 2>&1 &

SR_WO_AW_PID=$!
LOG "✅ SR-WO-AW-Scratch 训练已启动 (PID: ${SR_WO_AW_PID})"
LOG "⏳ 等待任务完成..."
wait ${SR_WO_AW_PID}
LOG "✅ SR-WO-AW-Scratch 训练完成！"
LOG "📄 日志: ${PROJECT_ROOT}/logs/sr_wo_aw_scratch_train.out"

# ==================== 任务 3: SR with adaptive weight pretrain+pair ====================
HEADER "任务 3: SR with adaptive weight pretrain+pair 训练"

LOG "工作目录: ${PROJECT_ROOT}"
LOG "配置文件: ${PROJECT_ROOT}/configs/case_renal_cortex_pretrain_pair.yaml"
LOG "日志文件: ${PROJECT_ROOT}/logs/sr_aw_pretrain.out"

cd "${PROJECT_ROOT}"
conda activate snapatac 2>/dev/null || source activate snapatac 2>/dev/null || true

nohup python scripts/run_pretrain_pair.py configs/case_renal_cortex_pretrain_pair.yaml --device cuda \
    > logs/sr_aw_pretrain.out 2>&1 &

SR_PRETRAIN_PID=$!
LOG "✅ SR-AW-Pretrain 训练已启动 (PID: ${SR_PRETRAIN_PID})"
LOG "⏳ 等待任务完成..."
wait ${SR_PRETRAIN_PID}
LOG "✅ SR-AW-Pretrain 训练完成！"
LOG "📄 日志: ${PROJECT_ROOT}/logs/sr_aw_pretrain.out"
SKIP_1_3
# ==================== 任务 1-3 已完成 ====================

# ==================== 任务 4: SCPAIR 训练 ====================
HEADER "任务 4: SCPAIR 训练"

SCPAIR_DIR="${COMPARE_ROOT}/scpair"
LOG "工作目录: ${SCPAIR_DIR}"
LOG "日志文件: ${SCPAIR_DIR}/hca.out"

cd "${SCPAIR_DIR}"
conda activate scpair 2>/dev/null || source activate scpair 2>/dev/null || true

# 依次运行训练、绘图、评估脚本
nohup bash -c "
    set -e
    echo '[$(date)] 开始 scpair train...'
    python train_scpair.py
    echo '[$(date)] 开始 scpair plot_loss...'
    python plot_loss.py
    echo '[$(date)] 开始 scpair eval...'
    python eval_scpair.py
    echo '[$(date)] 开始 scpair eval_predict...'
    python eval_predict.py
    echo '[$(date)] SCPAIR 全部流程完成 ✅'
" > hca.out 2>&1 &

SCPAIR_PID=$!
LOG "✅ SCPAIR 训练已启动 (PID: ${SCPAIR_PID})"
LOG "⏳ 等待任务完成..."
wait ${SCPAIR_PID}
LOG "✅ SCPAIR 训练完成！"
LOG "📄 日志: ${SCPAIR_DIR}/hca.out"

# ==================== 任务 5: MULTIVI 训练 ====================
HEADER "任务 5: MULTIVI 训练"

MVI_DIR="${COMPARE_ROOT}/multivi"
LOG "工作目录: ${MVI_DIR}"
LOG "日志文件: ${MVI_DIR}/hca.out"

cd "${MVI_DIR}"
conda activate scpair 2>/dev/null || source activate scpair 2>/dev/null || true
nohup bash -c "
    set -e
    echo '[\$(date)] 开始 multivi train...'
    python train_mvi.py
    echo '[\$(date)] 开始 multivi eval...'
    python mvi_eval.py
    echo '[\$(date)] 开始 multivi eval_predict...'
    python eval_predict.py
    echo '[\$(date)] MULTIVI 全部流程完成 ✅'
" > hca.out 2>&1 &

MVI_PID=$!
LOG "✅ MULTIVI 训练已启动 (PID: ${MVI_PID})"
LOG "⏳ 等待任务完成..."
wait ${MVI_PID}
LOG "✅ MULTIVI 训练完成！"
LOG "📄 日志: ${MVI_DIR}/hca.out"

# ==================== 任务 6: COBOLT 训练 ====================
HEADER "任务 6: COBOLT 训练"

COBOLT_DIR="${COMPARE_ROOT}/cobolt"
LOG "工作目录: ${COBOLT_DIR}"
LOG "日志文件: ${COBOLT_DIR}/hca.out"

cd "${COBOLT_DIR}"
conda activate cobolt 2>/dev/null || source activate cobolt 2>/dev/null || true

nohup bash -c "
    set -e
    echo '[\$(date)] 开始 cobolt train...'
    python train_cobolt.py
    echo '[\$(date)] 开始 cobolt eval...'
    python eval_cobolt.py
    echo '[\$(date)] 开始 cobolt get_embed...'
    python get_embed.py
    echo '[\$(date)] COBOLT 全部流程完成 ✅'
" > hca.out 2>&1 &

COBOLT_PID=$!
LOG "✅ COBOLT 训练已启动 (PID: ${COBOLT_PID})"
LOG "⏳ 等待任务完成..."
wait ${COBOLT_PID}
LOG "✅ COBOLT 训练完成！"
LOG "📄 日志: ${COBOLT_DIR}/hca.out"

# ==================== 任务 7: SCBUTTERFLY 训练 ====================
HEADER "任务 7: SCBUTTERFLY 训练"

SCB_DIR="${COMPARE_ROOT}/scb"
LOG "工作目录: ${SCB_DIR}"
LOG "日志文件: ${SCB_DIR}/hca.out"

cd "${SCB_DIR}"
conda activate scbutterfly 2>/dev/null || source activate scbutterfly 2>/dev/null || true

nohup bash -c "
    set -e
    echo '[\$(date)] 开始 scb train...'
    python train_scb.py
    echo '[\$(date)] 开始 scb eval...'
    python eval_scb.py
    echo '[\$(date)] 开始 scb eval_predict...'
    python eval_predict.py
    echo '[\$(date)] SCBUTTERFLY 全部流程完成 ✅'
" > hca.out 2>&1 &

SCB_PID=$!
LOG "✅ SCBUTTERFLY 训练已启动 (PID: ${SCB_PID})"
LOG "⏳ 等待任务完成..."
wait ${SCB_PID}
LOG "✅ SCBUTTERFLY 训练完成！"
LOG "📄 日志: ${SCB_DIR}/hca.out"

# ==================== 总结 ====================
HEADER "🎉 所有训练任务已全部完成！"

echo "任务完成列表:"
echo "  ✅ 1. SR-AW-Scratch      日志: ${PROJECT_ROOT}/logs/sr_aw_scratch_train.out"
echo "  ✅ 2. SR-WO-AW-Scratch   日志: ${PROJECT_ROOT}/logs/sr_wo_aw_scratch_train.out"
echo "  ✅ 3. SR-AW-Pretrain     日志: ${PROJECT_ROOT}/logs/sr_aw_pretrain.out"
echo "  ✅ 4. SCPAIR             日志: ${SCPAIR_DIR}/hca.out"
echo "  ✅ 5. MULTIVI            日志: ${MVI_DIR}/hca.out"
echo "  ✅ 6. COBOLT             日志: ${COBOLT_DIR}/hca.out"
echo "  ✅ 7. SCBUTTERFLY        日志: ${SCB_DIR}/hca.out"
echo ""
echo "所有任务已串行完成，可以查看各日志文件了解详细结果。"
