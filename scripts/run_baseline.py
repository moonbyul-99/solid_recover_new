#!/usr/bin/env python3
"""Round 0 baseline evaluation: use existing SR embeddings from obsm (no training)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import muon as mu
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solid_recover.evaluation.batch_metrics import (
    compute_ari_nmi_leiden,
    generate_umap_plots,
)
from solid_recover.config.loader import load_train_config

# ---------- constants ----------
OUTPUT_DIR = str(PROJECT_ROOT / "reports" / "round_0_baseline")
LABEL_KEYS = ["Class", "Subclass", "cell_type"]
COLOR_KEYS = ["Class", "Subclass", "cell_type", "Group", "Sample_ID", "Region"]
CKPT_STEP = 10000  # Use final checkpoint embedding


def main() -> None:
    # 1. Load sub-config and sub indices
    sub_cfg_path = str(PROJECT_ROOT / "configs" / "case_brain_dev_sub.yaml")
    cfg = load_train_config(sub_cfg_path)
    # Get original data path from the original config
    orig_cfg = load_train_config(str(PROJECT_ROOT / "configs" / "case_brain_dev.yaml"))
    train_path = orig_cfg.data.train_data_path
    key_1 = cfg.data.key_1

    train_idx = np.load(str(PROJECT_ROOT / "train_sub_indices.npy"))
    print(f"Loading original train data: {train_path}")
    print(f"Using {len(train_idx)} cell indices for baseline evaluation")

    # 2. Load original h5mu and extract embedding + obs
    mdata = mu.read_h5mu(train_path)
    obsm_key = "X_embed_sr_ckpt_10000"
    if obsm_key not in mdata[key_1].obsm:
        print(f"ERROR: {obsm_key} not found in obsm. Available keys:")
        print(list(mdata[key_1].obsm.keys()))
        sys.exit(1)

    embedding = mdata[key_1].obsm[obsm_key][train_idx, :]
    obs_sub = mdata[key_1].obs.iloc[train_idx].copy()

    print(f"  Embedding shape: {embedding.shape}")
    n_dims = embedding.shape[1]
    print(f"  Embed dim: {n_dims}")

    # 3. Compute ARI/NMI per label
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics: dict = {}

    for lkey in LABEL_KEYS:
        if lkey not in obs_sub.columns:
            print(f"  [WARN] Label '{lkey}' not found in obs — skipping")
            continue
        labels = obs_sub[lkey].values
        result = compute_ari_nmi_leiden(embedding, labels)
        metrics[lkey] = result
        print(f"  {lkey}: ARI={result['best_ari']:.4f}  NMI={result['best_nmi']:.4f}  "
              f"(res={result['best_resolution']})")

    # 4. Generate UMAP plots
    print(f"Generating UMAP plots (min_dist=0.2) ...")
    umap_paths = generate_umap_plots(
        embedding, obs_sub, OUTPUT_DIR, CKPT_STEP,
        color_keys=COLOR_KEYS, min_dist=0.2,
    )
    for k, p in umap_paths.items():
        print(f"  {k}: {p}")

    # 5. Wrap results for sending
    all_metrics = {CKPT_STEP: metrics}
    all_umaps = {CKPT_STEP: umap_paths}

    round_info = {
        "round": 0,
        "strategy": "Baseline — 原始 SR 模型 (ckpt 10000, 不使用批次信息)",
        "config": sub_cfg_path,
        "notes": "直接使用 obsm 中已有 embedding，未重新训练",
    }

    # 6. Send report
    from scripts.send_report import send_report

    send_report(
        round_info=round_info,
        metrics=all_metrics,
        umap_paths=all_umaps,
        output_dir=OUTPUT_DIR,
        hypers={
            "embed_dim": n_dims,
            "n_cells": embedding.shape[0],
            "atac_features": cfg.model.feature_num_2,
            "checkpoint": "ckpt_10000 (原始模型)",
        },
    )
    print(f"\nBaseline report saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
