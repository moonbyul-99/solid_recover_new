# Solid Recover — Project Skills & Architecture Guide

> Quick-reference for agents and developers encountering this codebase for the first time.

## 1. Project Overview

Solid Recover is a **single-cell multi-omics integration and cross-modality prediction** framework built on a **CLIP + VAE** hybrid objective. Core capabilities:

- **Single-omic pretraining** — large-scale VAE on one modality (RNA, ATAC, etc.)
- **Paired multi-omic training** — two modalities jointly trained with CLIP contrastive alignment, cross-modality reconstruction, and per-modality VAE losses
- **Cross-modality prediction** — generate one modality from another via shared latent space

## 2. Directory Layout

```
solid_recover/
├── nn/                # Low-level neural network building blocks
│   ├── blocks.py      #   ResidualBlock, RMSNorm helpers
│   ├── encoder.py     #   FeatureEncoder / FeatureDecoder (MLP stacks)
│   ├── vae.py         #   SRVAE — single-omic VAE backbone
│   └── pair_vae.py    #   SRPairVAE — two SRVAE modules + cross-recon forward
├── losses/            # Loss function modules
│   ├── recon.py       #   ReconLoss (MSE)
│   ├── vae.py         #   VAELoss (recon + KL)
│   ├── clip.py        #   CLIPLoss / WeightedCLIPLoss (InfoNCE)
│   └── composite.py   #   VAEClipLoss — full composite loss (VAE+CLIP+cross-recon)
├── models/            # High-level facade classes (scvi-style API)
│   ├── base.py        #   BaseModel — setup_data, train, save, load
│   ├── single.py      #   SinglePretrain — single-omic VAE facade
│   └── pair.py        #   PairScratch / PairPretrain — paired-omic facades
├── data/              # Data pipeline
│   ├── adata_utils.py #   adata_to_tensor (sparse/dense -> float32 tensor)
│   ├── datasets.py    #   SingleDataset / PairDataset (torch Dataset)
│   └── prepare.py     #   prepare_pair_data / split_and_save_mudata (MuData I/O)
├── training/          # Training loop
│   ├── scheduler.py   #   SRScheduler (warmup → steady → cosine anneal)
│   └── trainer.py     #   Trainer — generic loop, checkpointing, tensorboard
├── evaluation/        # Post-training evaluation
│   ├── embeddings.py  #   extract_embeddings (batch-wise z_mu extraction)
│   ├── metrics.py     #   top-k hit rate, FOSCTTM, matchscore
│   ├── prediction.py  #   cross-modality prediction utilities
│   ├── reporting.py   #   evaluate_checkpoint, metric aggregation
│   └── runner.py      #   evaluate_output_dir — top-level eval entry point
├── analysis/          # Downstream analysis (post-integration)
│   ├── decomposition.py #   matrix decomposition (NMF/PCA on latent)
│   └── grn_builder.py   #   gene regulatory network inference from latent
├── config/            # Configuration layer
│   ├── schema.py      #   TrainConfig / DataConfig / ModelConfig / LossConfig dataclasses
│   └── loader.py      #   load_train_config / dump_train_config (YAML <-> dataclass)
├── cli/               # Command-line interface
│   └── main.py        #   solid-recover train / eval entry point
└── _logging.py        # Shared logger setup
```

## 3. Three Training Modes

| Mode | CLI task key | Entry class | Data source | Loss |
|------|-------------|-------------|-------------|------|
| Single-omic pretrain | `single_pretrain` | `SinglePretrain` | HuggingFace `datasets` (on-disk) | `VAELoss` (recon + KL) |
| Paired scratch | `pair_scratch` | `PairScratch` | `.h5mu` MuData (train/test) | `VAEClipLoss` (VAE + CLIP + cross-recon ) |
| Paired pretrain | `pair_pretrain` | `PairPretrain` | `.h5mu` + two single-omic checkpoints | Same as pair_scratch |

## 4. Core Data Flow (Paired Training)

```
.h5mu file
  └─ muon.read_h5mu -> MuData
       └─ MuData[key_1], MuData[key_2]    # two AnnData (e.g. RNA + ATAC)
            └─ adata_to_tensor            # sparse/dense -> torch.float32 tensor
                 └─ PairDataset           # stores omic_1, omic_2, optional cell_type
                      └─ DataLoader (batch dict: omic_1, omic_2,)
                           └─ SRPairVAE.forward(x1, x2)
                                ├─ model_1.get_embedding(x1) -> z, z_mu, z_logvar, z_embed
                                ├─ model_2.get_embedding(x2) -> z, z_mu, z_logvar, z_embed
                                ├─ self-recon: decoder_1(z_embed_1), decoder_2(z_embed_2)
                                └─ cross-recon: decoder_1(z_embed_2), decoder_2(z_embed_1)
                           └─ VAEClipLoss(x1, x2, outputs, cell_type_labels)
                                ├─ VAELoss_1 + VAELoss_2
                                ├─ CLIPLoss(z_embed_1, z_embed_2)  [cross-modality alignment]
                                ├─ cross_recon_loss_1 + cross_recon_loss_2
```

## 5. Composite Loss Formula

```
L = cross_recon_1 * recon_cross_1 + (1 - cross_recon_1) * recon_self_1
  + cross_recon_2 * recon_cross_2 + (1 - cross_recon_2) * recon_self_2
  + vae_beta_1 * KL_1
  + vae_beta_2 * KL_2
  + clip_weight * CLIP(z_embed_1, z_embed_2)
```

## 6. Configuration Schema

A single YAML file drives one training run. Key sections:

```yaml
task: "pair_scratch"  # or "single_pretrain" / "pair_pretrain"

data:
  train_data_path: "..."    # .h5mu for paired tasks
  test_data_path: "..."
  key_1: "rna_count"        # MuData mod key for modality 1
  key_2: "atac_count"       # MuData mod key for modality 2
  batch_size: 512

model:
  embed_dim: 64
  feature_num_1: 17701
  feature_num_2: 141400
  hidden_params_1: [1024, 256]
  hidden_params_2: [1024, 256]
  use_rmsnorm: true
  use_residual: true
  dropout_p: 0.0

loss:
  clip_weight: 40.0
  cross_recon_1: 0.75
  cross_recon_2: 0.75
  temperature: 0.12
  use_weight: true          # use WeightedCLIPLoss


optimizer:
  lr: 8e-4
  warmup_steps: 1000
  steady_1_steps: 1000
  cosine_anneal_steps: 6000
  min_lr: 8e-5

training:
  project_dir: "outputs/my_run"
  train_steps: 10000
  eval_points: 500
  save_points: 2000
  device: "cuda"
```

## 7. CLI Usage

```bash
# Train from YAML config
solid-recover train --config configs/my_config.yaml

# Evaluate all checkpoints in an output directory
solid-recover eval --output-dir outputs/my_run --device cuda
```

## 8. Python API Usage

```python
from solid_recover.config.loader import load_train_config
from solid_recover.data.prepare import prepare_pair_data
from solid_recover.models.pair import PairScratch

cfg = load_train_config("configs/my_config.yaml")
train_ds, test_ds = prepare_pair_data(
    train_data_path=cfg.data.train_data_path,
    test_data_path=cfg.data.test_data_path,
    key_1=cfg.data.key_1,
    key_2=cfg.data.key_2,
)

model = PairScratch(
    feature_num_1=cfg.model.feature_num_1,
    feature_num_2=cfg.model.feature_num_2,
    hidden_params_1=cfg.model.hidden_params_1,
    hidden_params_2=cfg.model.hidden_params_2,
    embed_dim=cfg.model.embed_dim,
)
model.setup_data(train_ds, test_ds, batch_size=cfg.data.batch_size)
model.set_loss(
    clip_weight=cfg.loss.clip_weight,
    #...other params
)
model.configure_optimizer(lr=cfg.optimizer.lr, ...)
model.set_project(cfg.training.project_dir)
model.train(train_steps=cfg.training.train_steps, ...)
```

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `forward` is pure (no loss) | Loss composition lives in `VAEClipLoss`, keeping inference and training decoupled |
| `logit_scale` is a fixed buffer (not learnable) | Trainable-temperature ablation was retired; avoids stateful checkpoint edge cases |
| Checkpoint format: `{'model_state_dict': ...}` | Preserves backward compatibility with legacy checkpoints |
| `BaseModel` is not `nn.Module` | Delegates `nn.Module` to `self.net`; `BaseModel` is a facade managing data/optimizer/training |

## 10. Checkpoint Compatibility

On-disk format: `{'model_state_dict': net.state_dict()}`

- `SRVAE` state dict keys: `encoder.*`, `mu_proj.*`, `logvar_proj.*`, `decoder.*`
- `SRPairVAE` state dict keys: `model_1.*`, `model_2.*` (same as legacy)

## 11. Running Without Installation

The project supports running without `pip install` by prepending the project root to `PYTHONPATH`:

```bash
cd /path/to/solid_recover_main
PYTHONPATH=. python -m solid_recover.cli.main train --config configs/my_config.yaml
```
