# Solid Recover 迭代报告 — Round 2: GRL_adversarial_weight_0.01

**生成时间**: 2026-05-23 15:52:24
**配置文件**: `outputs/hca_brain_dev_grl_20260523_1433/config.yaml`
**策略**: GRL_adversarial_weight_0.01

## 一、训练概况

| 项目 | 值 |
|------|----|
| train_steps | 6000 |
| batch_size | 512 |
| embed_dim | 64 |
| num_batches | 38 |
| batch_embed_dim | 0 |

## 二、超参数

| 参数 | 值 |
|------|----|
| lr | 0.001 |
| warmup_steps | 500 |
| clip_weight | 40.0 |
| cross_recon_1 | 0.75 |
| cross_recon_2 | 0.75 |
| temperature | 0.12 |
| adversarial_batch_weight | 0.01 |
| batch_alignment_weight | 0.0 |

## 三、定量评估结果 (ARI / NMI)

| Checkpoint | Class ARI | Class NMI | Subclass ARI | Subclass NMI | cell_type ARI | cell_type NMI |
|------|------|------|------|------|------|------|
| 1500 | 0.1292 | 0.4124 | 0.2675 | 0.5560 | 0.4036 | 0.6081 |
| 3000 | 0.1360 | 0.4036 | 0.2609 | 0.5613 | 0.3860 | 0.5894 |
| 4500 | 0.1411 | 0.4036 | 0.2583 | 0.5545 | 0.3955 | 0.5962 |
| 5928 | 0.1526 | 0.3913 | 0.3038 | 0.5662 | 0.3952 | 0.5991 |

> 最佳 Checkpoint: **5928** 步

## 四、UMAP 可视化

### Checkpoint 1500

| Class | Subclass | cell_type |
|------|------|------|
| ![Class](umap_ckpt1500_Class.png) | ![Subclass](umap_ckpt1500_Subclass.png) | ![cell_type](umap_ckpt1500_cell_type.png) |

| Group | Sample_ID | Region |
|------|------|------|
| ![Group](umap_ckpt1500_Group.png) | ![Sample_ID](umap_ckpt1500_Sample_ID.png) | ![Region](umap_ckpt1500_Region.png) |

### Checkpoint 3000

| Class | Subclass | cell_type |
|------|------|------|
| ![Class](umap_ckpt3000_Class.png) | ![Subclass](umap_ckpt3000_Subclass.png) | ![cell_type](umap_ckpt3000_cell_type.png) |

| Group | Sample_ID | Region |
|------|------|------|
| ![Group](umap_ckpt3000_Group.png) | ![Sample_ID](umap_ckpt3000_Sample_ID.png) | ![Region](umap_ckpt3000_Region.png) |

### Checkpoint 4500

| Class | Subclass | cell_type |
|------|------|------|
| ![Class](umap_ckpt4500_Class.png) | ![Subclass](umap_ckpt4500_Subclass.png) | ![cell_type](umap_ckpt4500_cell_type.png) |

| Group | Sample_ID | Region |
|------|------|------|
| ![Group](umap_ckpt4500_Group.png) | ![Sample_ID](umap_ckpt4500_Sample_ID.png) | ![Region](umap_ckpt4500_Region.png) |

### Checkpoint 5928

| Class | Subclass | cell_type |
|------|------|------|
| ![Class](umap_ckpt5928_Class.png) | ![Subclass](umap_ckpt5928_Subclass.png) | ![cell_type](umap_ckpt5928_cell_type.png) |

| Group | Sample_ID | Region |
|------|------|------|
| ![Group](umap_ckpt5928_Group.png) | ![Sample_ID](umap_ckpt5928_Sample_ID.png) | ![Region](umap_ckpt5928_Region.png) |

---
*报告由 `scripts/run_iteration.py` 自动生成于 2026-05-23 15:52:24*