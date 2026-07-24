# HCA Renal Cortex 消融实验自动执行方案

## 环境与约定

### 环境激活
```bash
conda activate snapatac
cd /home/rsun@ZHANGroup.local/solid_recover_main
```

### 训练与评估命令
```bash
# 训练
bash scripts/train.sh configs/<config_name>.yaml cuda

# 评估（train 完成后执行）
bash scripts/eval.sh outputs/<project_dir> cuda
```

### GitHub 提交规范
- 分支：`exp-reports`（已存在，远程 `origin/exp-reports`）
- 每次实验完成后，将 `summary.md` + `metrics_summary.json` 提交到此分支
- 目录结构：`exp-reports/<YYYY-MM-DD>_<phase_name>/<experiment_name>/`
- 提交信息格式：`[exp] <phase>：<关键结论>`

```bash
# 提交模板
git checkout exp-reports
git add exp-reports/<phase_dir>/
git commit -m "[exp] <phase>: <result_summary>"
git push origin exp-reports
```

### 主指标：top_1_hit

top_1_hit 是唯一主指标，直接反映跨模态检索的精确匹配能力。所有实验的优度判定、阶段间最优参数选择，均以 top_1_hit 为准。
foscttm 作为辅助参考，仅在 top_1_hit 持平（差距 < 0.0005）时用于 tie-breaking（取 foscttm 较低者）。

### 最优实验判定规则
1. 从 `outputs/<project_dir>/eval_result/metrics_summary.json` 读取
2. 在所有 step 中取 top_1_hit 最大值，记录对应的 step、top_1_hit、foscttm
3. 若多个实验 top_1_hit 差距 < 0.0005，则以 foscttm 较低者胜出

### 完整结果归档规则（必须执行）
消融实验的目标是全面记录每个参数的全部影响，不可只保存最优值：
1. 将 `outputs/<project_dir>/eval_result/metrics_summary.json` **完整复制**到 `exp-reports/<phase_dir>/<exp_name>/`
2. 如有 TB 日志（`outputs/<project_dir>/logs/`），也一并复制
3. summary.md 中必须包含该实验**所有 step 的 foscttm 和 top_1 逐步轨迹表**，并标注最优 step 加粗

### 基线配置模板
所有实验基于 `configs/hca_renal_cortex_pretrain_t030.yaml` 修改。核心固定参数：
- `task: pair_pretrain`
- `data.train_data_path / test_data_path`：不变
- `data.to_gpu: true`
- `model.feature_num_1: 14577, feature_num_2: 101123`
- `model.use_rmsnorm: true, use_residual: true, dropout_p: 0.0`
- `training.train_steps: 6000, eval_points: 200, save_points: 500`
- `optimizer.lr: 2e-4, warmup_steps: 500, steady_1_steps: 500, cosine_anneal_steps: 5000, min_lr: 5e-5`
- `loss.vae_beta_1: 1.0, vae_beta_2: 1.0, clip_weight: 20.0, cross_recon_1: 0.75, cross_recon_2: 0.75`
- `loss.use_weight: true, top_k_ratio: 0.1`
- `ckpt.omic_1: outputs/hca_renal_cortex_pretrain_pair_v2/rna_pretrain/models/ckpt_4000.pth`
- `ckpt.omic_2: outputs/hca_renal_cortex_pretrain_pair_v2/atac_pretrain/models/ckpt_4000.pth`

---

## Phase 1: Temperature 扩展扫描

**目标**：确定全局最优 CLIP temperature

**变参**：`loss.temperature` ∈ {0.4, 0.5, 0.7, 1.0}

**实验列表**：

| 实验ID | temperature | project_dir | 配置名 |
|--------|:-----------:|------|------|
| t040 | 0.40 | `outputs/hca_renal_cortex_pretrain_t040` | `hca_renal_cortex_pretrain_t040.yaml` |
| t050 | 0.50 | `outputs/hca_renal_cortex_pretrain_t050` | `hca_renal_cortex_pretrain_t050.yaml` |
| t070 | 0.70 | `outputs/hca_renal_cortex_pretrain_t070` | `hca_renal_cortex_pretrain_t070.yaml` |
| t100 | 1.00 | `outputs/hca_renal_cortex_pretrain_t100` | `hca_renal_cortex_pretrain_t100.yaml` |

**执行步骤**：

1. 复制 `configs/hca_renal_cortex_pretrain_t030.yaml` 为每个新配置
2. 修改 `loss.temperature` 和 `training.project_dir`
3. 逐个训练 + 评估（可并行启动 4 个训练，但 eval 需等训练完成）
4. 从 `metrics_summary.json` 提取各实验的最优 foscttm 和 top_1
5. 结合阶段一已有的 t=0.07/0.10/0.12/0.15/0.20/0.30 结果，汇总排名
6. 确定最优温度 `T_best`
7. 提交报告到 `exp-reports/2026-07-23_temperature_sweep/`

**报告生成**：在 `exp-reports/2026-07-23_temperature_sweep/summary.md` 中写入：

```markdown
# Temperature 扫描汇总

| temperature | best_step | top_1 | foscttm | top_5 |
|------------:|----------:|------:|--------:|------:|
| 0.07 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

## 逐步轨迹（每个实验一个子表）

### t=0.40
| step | top_1 | foscttm |
|-----:|------:|--------:|
| 500 | ... | ... |
| ... | ... | ... |
| **2500** | **0.xxxx** | 0.xxxx |  ← top_1 最优

## 结论
最优温度: **T_best = X.XX**, top_1=..., foscttm=...
```

---

## Phase 2: bottom_k_ratio 扫描

**目标**：在最优温度 `T_best` 下确定最优 bottom_k_ratio

**基线**：Phase 1 最优温度的配置

**变参**：`loss.bottom_k_ratio` ∈ {0.01, 0.02, 0.05, 0.10, 0.15, 0.20}

**实验列表**：

| 实验ID | bottom_k_ratio | project_dir |
|--------|:---:|------|
| bk001 | 0.01 | `outputs/hca_renal_cortex_bk001` |
| bk002 | 0.02 | `outputs/hca_renal_cortex_bk002` |
| bk005 | 0.05 | `outputs/hca_renal_cortex_bk005` |
| bk010 | 0.10 | `outputs/hca_renal_cortex_bk010` |
| bk015 | 0.15 | `outputs/hca_renal_cortex_bk015` |
| bk020 | 0.20 | `outputs/hca_renal_cortex_bk020` |

**执行**：同上流程（生成yaml → 训练 → eval → 提取最优值 → 排名 → 提交）

---

## Phase 3: weight_top 扫描

**基线**：Phase 1 最优温度 + Phase 2 最优 `bottom_k_ratio`

**变参**：`loss.weight_top` ∈ {0.01, 0.05, 0.1}

| 实验ID | weight_top | project_dir |
|--------|:---:|------|
| wt001 | 0.01 | `outputs/hca_renal_cortex_wt001` |
| wt005 | 0.05 | `outputs/hca_renal_cortex_wt005` |
| wt010 | 0.10 | `outputs/hca_renal_cortex_wt010` |

---

## Phase 4: weight_bottom 扫描

**基线**：Phase 1~3 最优配置

**变参**：`loss.weight_bottom` ∈ {2.0, 5.0, 10.0}

| 实验ID | weight_bottom | project_dir |
|--------|:---:|------|
| wb2 | 2.0 | `outputs/hca_renal_cortex_wb2` |
| wb5 | 5.0 | `outputs/hca_renal_cortex_wb5` |
| wb10 | 10.0 | `outputs/hca_renal_cortex_wb10` |

---

## Phase 5: 模型容量（embed_dim）扫描

**基线**：Phase 1~4 最优配置

**变参**：`model.embed_dim` ∈ {16, 32, 64, 128, 256}，同步调整 `model.hidden_params_1/2`

| 实验ID | embed_dim | hidden_params | project_dir |
|--------|:---:|------|------|
| e016 | 16 | [1024, 64] | `outputs/hca_renal_cortex_e016` |
| e032 | 32 | [1024, 128] | `outputs/hca_renal_cortex_e032` |
| e064 | 64 | [1024, 256] | `outputs/hca_renal_cortex_e064` |
| e128 | 128 | [1024, 512] | `outputs/hca_renal_cortex_e128` |
| e256 | 256 | [1024, 512] | `outputs/hca_renal_cortex_e256` |

> 注意：embed_dim 变化时，`hidden_params` 最后一层至少为 embed_dim 的 2 倍。embed_dim=256 与 128 共用 [1024,512] 因为 512/256 = 2。

---

## Phase 6: 预训练步数扫描

**目标**：确认最优的单组学预训练步数

**方式**：
1. 先跑一次 RNA 单组学预训练（10000 步，save_points=2000），产出 ckpt_2000/4000/6000/8000/10000
2. 同样跑一次 ATAC 单组学预训练（10000 步，save_points=2000）
3. 基于 Phase 1~5 最优配置，分别用每对 ckpt 跑 pair_pretrain

**Step 6a：预训练配置生成**

创建 `configs/hca_renal_cortex_single_pretrain_rna.yaml`：
```yaml
task: single_pretrain
data:
  train_data_path: "/home/rsun@ZHANGroup.local/solid_recover_main/paper_figure/hca_renal_cortex/train.h5mu"
  test_data_path: "/home/rsun@ZHANGroup.local/solid_recover_main/paper_figure/hca_renal_cortex/test.h5mu"
  key_1: rna_count
  batch_size: 256
  to_gpu: true
model:
  feature_num: 14577
  hidden_params: [1024, 256]
  embed_dim: 64
  use_rmsnorm: true
  use_residual: true
  dropout_p: 0.0
training:
  project_dir: "outputs/hca_renal_cortex_pretrain_10k/rna_pretrain"
  train_steps: 10000
  eval_points: 500
  save_points: 2000
  device: "cuda"
optimizer:
  lr: 1e-3
  warmup_steps: 400
  steady_1_steps: 0
  cosine_anneal_steps: 9600
  min_lr: 1e-4
loss:
  vae_beta: 1.0
```

> ATAC 同理，修改 key_1→key_2, feature_num→101123, project_dir→atac_pretrain

**Step 6b：pair_pretrain 实验**

| 实验ID | ckpt_step | project_dir |
|--------|:---:|------|
| pt2k | 2000 | `outputs/hca_renal_cortex_pt2k` |
| pt4k | 4000 | `outputs/hca_renal_cortex_pt4k` |
| pt6k | 6000 | `outputs/hca_renal_cortex_pt6k` |
| pt8k | 8000 | `outputs/hca_renal_cortex_pt8k` |
| pt10k | 10000 | `outputs/hca_renal_cortex_pt10k` |

每个配置中 `ckpt.omic_1` 和 `ckpt.omic_2` 指向对应步数的 checkpoint。

> 注意：CLI 的 `single_pretrain` 任务使用的数据集格式可能与 pair_pretrain 不同。如果 `solid-recover train single --config ...` 不支持直接传入 h5mu 路径，需要检查 `solid_recover/cli/main.py` 中的 single_pretrain 数据加载逻辑，或直接复用 `case_renal_cortex_pretrain_pair.yaml` 中 `pretrain_rna`/`pretrain_atac` 段落格式（如果 CLI 支持此 legacy 模式）。

---

## Phase 7: Loss 权重调整

**基线**：Phase 1~6 最优配置

**变参**：

| 实验ID | clip_weight | cross_recon | project_dir |
|--------|:---:|:---:|------|
| lw_clip10 | 10.0 | 0.75 | `outputs/hca_renal_cortex_lw_clip10` |
| lw_clip40 | 40.0 | 0.75 | `outputs/hca_renal_cortex_lw_clip40` |
| lw_cr05 | 20.0 | 0.50 | `outputs/hca_renal_cortex_lw_cr05` |
| lw_cr10 | 20.0 | 1.00 | `outputs/hca_renal_cortex_lw_cr10` |

> 先测 clip_weight（保持 cross_recon=0.75），确定最优后测 cross_recon。

---

## Phase 8: Batch Size 扫描

**基线**：Phase 1~7 最优配置

**关键规则**：
- `train_steps` 与 `batch_size` 成反比，保持总样本量 `bs * steps = 512 * 6000 = 3,072,000` 恒定
- LR schedule 参数（warmup_steps/steady_1_steps/cosine_anneal_steps）不变

| 实验ID | batch_size | train_steps | 预计耗时 | project_dir |
|--------|:---:|:---:|:---:|------|
| bs128 | 128 | 24000 | ~8h | `outputs/hca_renal_cortex_bs128` |
| bs256 | 256 | 12000 | ~4h | `outputs/hca_renal_cortex_bs256` |
| bs512 | 512 | 6000 | ~2h | `outputs/hca_renal_cortex_bs512` |
| bs1024 | 1024 | 3000 | ~1h | `outputs/hca_renal_cortex_bs1024` |

> `save_points` 按比例调整：bs128→2000, bs256→1000, bs512→500, bs1024→250

---

## 单次实验执行流程

```
1. 生成 YAML 配置（从模板修改目标参数）
2. bash scripts/train.sh configs/<name>.yaml cuda
3. 等待训练完成（检查 exit code = 0）
4. bash scripts/eval.sh outputs/<project_dir> cuda
5. 从 `outputs/<project_dir>/eval_result/metrics_summary.json` 完整复制到 `exp-reports/<phase_dir>/<exp_name>/metrics_summary.json`
6. 提取所有 step 的 foscttm/top_1 逐步轨迹，标注最优 step
7. 生成 summary.md（包含排名表 + 每个实验的逐步轨迹表），写入 `exp-reports/<phase_dir>/`
8. git checkout exp-reports → git add exp-reports/<phase_dir>/ → git commit → git push
9. git checkout main（切回 main 继续下一个实验）
```

## 各阶段最优判定与汇总

每阶段完成后，汇总报告包含：
- 该阶段所有实验的 top_1_hit 排名表（按 top_1_hit 降序，附带 foscttm 参考列）
- 每个实验的完整逐步 top_1_hit / foscttm 轨迹表（最优 step 加粗）
- 每个实验的 metrics_summary.json 完整归档
- 与上一阶段基线的 top_1_hit 对比
- 确定为该阶段最优的参数值（以 top_1_hit 为准）
- 标注进入下一阶段的基线参数

最终全部完成后，在 `exp-reports/final_summary.md` 中汇总：
1. 全链路最优配置（每阶段选出的最优参数串联）
2. 每阶段的 top_1_hit 增益表（该阶段最优 top_1_hit 相对于进入该阶段基线的变化，同时记录 foscttm 变化）
3. 最终 top_1_hit 与 multivi (0.01418) 的差距

## 注意事项

1. **to_gpu 偏差**：所有实验均启用 `to_gpu: true`，与 Phase 1 已完成的 6 组实验口径一致
2. 消融并行会爆显存，所有任务串行。可以并行一个测试和一个训练，但不能并行两个以上的训练。
3. **GPU 显存**：embed_dim=256 / batch_size=1024 时注意 OOM，必要时降 bs
4. **异常处理**：训练或 eval 失败时，先检查日志 `logs/train_*.out` 和 `logs/eval_*.out`，修复后重试；重试 3 次仍失败则跳过该实验并在报告中标注
5. **Phase 6 特殊处理**：如果 `single_pretrain` CLI 任务不能直接处理 h5mu 输入，改为手动实现单组学 VAE 训练逻辑（从 `solid_recover/models/single.py::SinglePretrain` 调用）
