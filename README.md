# Solid Recover

单细胞多组学整合与跨模态预测框架。支持：

- 单组学大规模预训练（VAE）
- 配对多组学联合训练（CLIP + VAE）
- 交叉模态预测（任意双组学，字段命名为 `omic_1_*` / `omic_2_*`）
- 嵌入提取与多种匹配指标（top-k hit、FOSCTTM、matchscore）
- **调控程序解耦分析**（ElasticNet 矩阵分解 + GRN 构建）

本目录为原 `src/` + 顶层入口脚本的工程化重构版，提供 **YAML+CLI** 与 **Python API** 双轨使用方式。

## 安装

```bash
# 推荐：可编辑安装
pip install -e .

# 包含测试依赖（pytest）
pip install -e .[test]

# 包含分析模块依赖（snapatac2, GRN构建）
pip install -e .[analysis]

# 或仅安装运行依赖
pip install -e . --no-deps
pip install torch numpy pandas scipy scikit-learn tqdm pyyaml tensorboard \
            anndata muon scanpy datasets joblib matplotlib seaborn
```

Python >= 3.9，PyTorch >= 2.0。

**注意**：GRN 构建功能需要 `snapatac2`，仅在使用分析模块时需要安装。

## 快速开始

### Python API（Notebook 友好）

```python
import muon as mu
from solid_recover import PairScratch
from solid_recover.evaluation import get_paired_embedding, predict_cross_modality

mdata = mu.read_h5mu("data/case_4/train.h5mu")

model = PairScratch(
    feature_num_1=mdata["rna_count"].shape[1],
    feature_num_2=mdata["peak_count"].shape[1],
    hidden_params_1=[1024, 384],
    hidden_params_2=[1024, 384],
    embed_dim=384,
    use_rmsnorm=True,
    use_residual=True,
    dropout_p=0.0,
)

# 1) 构造 Dataset 并挂到 DataLoader
train_ds, test_ds = model.create_dataset(mdata, key_1="rna_count", key_2="peak_count", test_size=0.1)
model.setup_data(train_ds, test_ds, batch_size=512)

# 2) 损失 / 优化器 / 输出目录
model.set_loss(vae_beta_1=1.0, vae_beta_2=1.0, clip_weight=100.0,
               cross_recon_1=0.8, cross_recon_2=0.8, temperature=0.12)
model.configure_optimizer(lr=8e-4, warmup_steps=2000,
                          steady_1_steps=8000, cosine_anneal_steps=12000, min_lr=1e-4)
model.set_project("outputs/demo")

# 3) 训练
model.train(train_steps=16000, eval_points=80, save_points=300, device="cuda")

# 4) 提取嵌入 / 做交叉模态预测
embeds = get_paired_embedding(model, test_ds, device="cuda")           # {z_mu_1, z_mu_2}
pred   = predict_cross_modality(model, test_ds, device="cuda")         # {omic_1_pred, omic_2_pred}
```

### YAML + CLI（规模训练）

```bash
# 单组学预训练
solid-recover train single --config configs/single_pretrain_example.yaml

# 配对从头训练
solid-recover train pair-scratch --config configs/pair_scratch_example.yaml

# 配对微调（基于单组学预训练 ckpt）
solid-recover train pair-pretrain --config configs/pair_pretrain_example.yaml

# 评估某次训练输出目录（按 ckpt 遍历或指定 step）
solid-recover eval pair --output-dir outputs/demo

# 交叉模态预测
solid-recover predict --output-dir outputs/demo --ckpt 4000 \
                      --data data/case_4/test.h5mu --out outputs/demo/pred.h5mu
```

也可使用 `scripts/*.sh` 薄包装直接启动。

### 调控程序解耦分析（新增）

从训练好的模型中提取可解释的调控程序，构建基因调控网络（GRN）。

#### Python API

```python
from solid_recover import PairScratch
from solid_recover.analysis import decompose_latent_to_features, GRNBuilder
from solid_recover.evaluation import get_paired_embedding
import muon as mu

# 1. 加载模型和数据
mdata = mu.read_h5mu("data/case_4/test.h5mu")
model = PairScratch.from_pretrained("outputs/demo", ckpt_steps=4000)

# 2. 提取潜空间嵌入
embeds = get_paired_embedding(model, test_dataset, device="cuda")
Z_rna = embeds['z_mu_1']  # (n_cells, n_latent)

# 3. 矩阵分解：将基因表达分解到潜空间维度
X_rna = mdata['rna_count'].X
W_rna = decompose_latent_to_features(
    X_rna, Z_rna,
    feature_names=mdata['rna_count'].var_names.tolist(),
    alpha=0.1, l1_ratio=0.9
)

# 4. 获取特定 latent 维度的 top 基因
top_genes = W_rna['latent_63'].sort_values(ascending=False).head(1000)

# 5. 构建 GRN（需要 snapatac2）
top_peaks = W_atac['latent_63'].sort_values(ascending=False).head(3000).index.tolist()
grn = GRNBuilder(genome="hg38")
df_links = grn.build_peak_gene_network(top_peaks, top_genes)
df_tf_re = grn.add_tf_binding(tf_filter=top_genes)

# 6. 保存结果
W_rna.to_csv("W_rna.csv")
df_links.to_csv("re_tg.csv")
df_tf_re.to_csv("tf_re.csv")
```

#### CLI 使用

```bash
# 完整流程：从模型到 GRN
python examples/decomposition_example.py \
    --model-dir outputs/demo \
    --data data/case_4/test.h5mu \
    --ckpt 4000 \
    --out-dir results/decomposition \
    --latent-idx 63 \
    --n-top-genes 1000 \
    --n-top-peaks 3000 \
    --genome hg38 \
    --device cuda
```

输出文件：
- `W_rna.csv`: RNA 特征权重矩阵 (genes × latent_dims)
- `W_atac.csv`: ATAC 特征权重矩阵 (peaks × latent_dims)
- `re_tg.csv`: Peak-gene 调控关系
- `tf_re.csv`: TF-motif-peak 调控关系

## 目录结构

```
sr_new/
├── solid_recover/         # 主包
│   ├── nn/                # 网络模块（FCBlock / Encoder / VAE / PairVAE）
│   ├── losses/            # 损失（Recon / VAE / CLIP / WeightedCLIP / VAEClip）
│   ├── data/              # AnnData 适配 + Dataset + prepare
│   ├── training/          # Scheduler + Trainer
│   ├── models/            # 门面（SinglePretrain / PairScratch / PairPretrain）
│   ├── evaluation/        # metrics / embeddings / prediction / reporting
│   ├── analysis/          # 调控程序解耦（矩阵分解 + GRN 构建）[新增]
│   ├── config/            # YAML schema + loader
│   └── cli/               # 统一命令行入口
├── configs/               # 示例 YAML
├── scripts/               # shell 启动器
├── notebooks/             # 快速上手 / 评估可视化
├── examples/              # 使用示例（含 decomposition_example.py）
├── tests/                 # 单元测试
├── context/               # 重构过程文档
├── pyproject.toml
└── README.md
```

## 从旧版迁移

| 旧版 | 新版 |
| --- | --- |
| `src/sr_net.py::fc_net` | `solid_recover.nn.blocks.FCBlock` |
| `src/sr_grn.py::sr_grn` | `solid_recover.analysis.GRNBuilder`（v3 新增） |
| `src/metrics.py::matching_metrics` | `solid_recover.evaluation.metrics.matching_metrics` |
| `pred_evaluation/get_sr_pred.py::sr_pipe` | `solid_recover.evaluation.predict_cross_modality` |
| `result_analysis/utils.py::get_sr_embed` | `solid_recover.evaluation.get_paired_embedding` |
| `result_analysis/utils.py::ckpt_merge` | `solid_recover.evaluation.evaluate_output_dir` |
| `src/sr_net.py::sr_vae` | `solid_recover.nn.vae.SRVAE` |
| `src/sr_net.py::sr_pair_vae` | `solid_recover.nn.pair_vae.SRPairVAE` |
| `src/sr_loss.py::VAE_clip_loss` | `solid_recover.losses.composite.VAEClipLoss` |
| `src/sr_model.py::pair_sr_scratch` | `solid_recover.models.pair.PairScratch` |
| `src/sr_model.py::pair_sr_pretrain` | `solid_recover.models.pair.PairPretrain` |
| `src/sr_model.py::single_sr` | `solid_recover.models.single.SinglePretrain` |
| `src/sr_dataset.py::pair_data` | `solid_recover.data.datasets.PairDataset` |
| `src/lr_scheduler.py::sr_scheduler` | `solid_recover.training.scheduler.SRScheduler` |
| `pair_scratch.py` | `solid-recover train pair-scratch` |
| `pair_pretrain.py` | `solid-recover train pair-pretrain` |
| `pair_eval.py` | `solid-recover eval pair` |
| `single_omic_pretrain.py` | `solid-recover train single` |

**checkpoint 兼容**：旧 `outputs/xxx/models/ckpt_*.pth` 可直接被 `PairScratch.from_pretrained(project_dir, ckpt_steps)` 加载。

## v3 主要变更（2026-05 更新）

### 新增分析模块

- **新增 `solid_recover.analysis` 子包**：提供调控程序解耦和 GRN 构建功能
  - `decompose_latent_to_features()`: 使用 ElasticNet 将原始特征分解到潜空间
  - `GRNBuilder`: 基于 snapatac2 构建基因调控网络
  - 支持 RNA 和 ATAC 双模态分析
- **新增 `examples/decomposition_example.py`**: 完整的 CLI 示例脚本
- **新增可选依赖 `analysis`**: `pip install -e .[analysis]` 安装 snapatac2 等生物信息学工具

### 设计特点

- **核心与可视化分离**：分析模块专注数据生成，可视化代码保留在 `SR_fig4/plot.ipynb`
- **延迟导入**：snapatac2 仅在调用 GRN 功能时导入，避免强制安装大型生物信息学库
- **兼容旧版输出**：生成的 CSV 格式与原有 notebook 工作流完全兼容

## v2 主要变更（2026-05）

详见 [`context/项目修改文档.md`](./context/项目修改文档.md) 的 “v2 修订” 章节。破坏性改动摘要：

- 删除 `analysis/` 子包、`nn.vae.SRAE`、`losses.vae.AELoss`
- `ModelConfig.vae_model` / `LossConfig.trainable_clip_temperature` / `PairPretrain.model_type` 等冗余字段删除；旧 YAML 中的该等字段会被加载时 warning 并丢弃
- `SRPairVAE.forward` 返回结构：cross-recon 从顶层 `x1_c_recon`/`x2_c_recon` 移入 `x1["cross_recon"]` / `x2["cross_recon"]`
- `predict_cross_modality` 返回键：`rna2atac`/`atac2rna` → `omic_*_cross_recon`；CLI `predict` 输出字段：`rna_pred`/`atac_pred` → `omic_1_pred`/`omic_2_pred`
- `CLIPLoss.logit_scale` / `WeightedCLIPLoss.logit_scale` 从 `nn.Parameter` 改为 `register_buffer`；旧 ckpt 通过 `strict=False` 加载
- `BaseModel.setup_data(train_dataset, test_dataset=None, ...)`：test set 不再强制；`train(..., training_config=...)` 可用于将 in-memory 改动后的 config 落盘
- 新增 `tests/test_smoke.py` CPU mock 用例；`pip install -e .[test]` 后 `pytest tests/` 即可

## 重构背景与决策

请参阅 [`context/`](./context/) 下的三份文档：

- [`项目理解文档.md`](./context/项目理解文档.md)：原项目结构、数据流、已知问题清单
- [`项目修改文档.md`](./context/项目修改文档.md)：逐 Task 的改动对照
- [`当前工作进展.md`](./context/当前工作进展.md)：实时进度
