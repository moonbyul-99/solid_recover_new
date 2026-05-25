#!/usr/bin/env python3
"""
对 hca_brain_dev_cvae_large 训练的各个 checkpoint 进行推理，提取
train/test 的 embedding（z_mu_1 / z_mu_2），并生成 RNA embedding 的
UMAP 可视化。

输出结构:
  reports/round_3_CVAE_large/
  ├── embeddings/
  │   ├── train/   (ckpt_{step}_z_mu_1.npy, ckpt_{step}_z_mu_2.npy)
  │   └── test/
  ├── umap/
  │   ├── train/ckpt_{step}/
  │   │   ├── umap_coords.npy
  │   │   ├── umap_ckpt{step}_Class.png
  │   │   ├── umap_ckpt{step}_Subclass.png
  │   │   ├── umap_ckpt{step}_cell_type.png
  │   │   ├── umap_ckpt{step}_Group.png
  │   │   ├── umap_ckpt{step}_Sample_ID.png
  │   │   └── umap_ckpt{step}_Region.png
  │   └── test/ckpt_{step}/
  └── config.yaml

Usage:
    cd /home/rsun@ZHANGroup.local/solid_recover_dev
    conda activate snapatac
    python scripts/inference_umap_cvae_large.py
"""

from __future__ import annotations

import gc
import glob
import os
import re
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import yaml
from tqdm import tqdm

from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import PairDataset
from solid_recover.evaluation.embeddings import get_paired_embedding
from solid_recover.evaluation.batch_metrics import generate_umap_plots
from solid_recover.models.pair import PairScratch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/rsun@ZHANGroup.local/solid_recover_dev"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "hca_brain_dev_cvae_large")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.yaml")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "round_3_CVAE_large")

# UMAP parameters (matching existing reports style)
UMAP_MIN_DIST = 0.2
UMAP_RANDOM_STATE = 42
COLOR_KEYS = ["Class", "Subclass", "cell_type", "Group", "Sample_ID", "Region"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 3)


def find_checkpoints(models_dir: str) -> List[Tuple[int, str]]:
    """Return sorted list of (step, path) for all ckpt_*.pth files."""
    ckpt_files = glob.glob(os.path.join(models_dir, "ckpt_*.pth"))
    result = []
    for path in ckpt_files:
        match = re.search(r"ckpt_(\d+)\.pth", os.path.basename(path))
        if match:
            result.append((int(match.group(1)), path))
    result.sort(key=lambda x: x[0])
    return result


def build_model(cfg: dict) -> PairScratch:
    m = cfg["model"]
    d = cfg["data"]
    return PairScratch(
        feature_num_1=m["feature_num_1"],
        feature_num_2=m["feature_num_2"],
        hidden_params_1=m["hidden_params_1"],
        hidden_params_2=m["hidden_params_2"],
        embed_dim=m["embed_dim"],
        use_rmsnorm=m.get("use_rmsnorm", True),
        use_residual=m.get("use_residual", False),
        dropout_p=m.get("dropout_p", 0.0),
        num_batches=d.get("num_batches", 0),
        batch_embed_dim=m.get("batch_embed_dim", 0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- 1. Load config ---------------------------------------------------
    print(f"[{_now()}] Loading config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    train_path = cfg["data"]["train_data_path"]
    test_path = cfg["data"]["test_data_path"]
    key_1 = cfg["data"]["key_1"]   # "rna_count"
    key_2 = cfg["data"]["key_2"]   # "atac_count"

    print(f"  train data: {train_path}")
    print(f"  test  data: {test_path}")
    print(f"  key_1: {key_1},  key_2: {key_2}")
    print(f"  device: {DEVICE}")

    # --- 2. Build model & discover ckpts ---------------------------------
    model = build_model(cfg)
    ckpts = find_checkpoints(MODELS_DIR)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt_*.pth files in {MODELS_DIR}")
    print(f"\nFound {len(ckpts)} checkpoints:")
    for step, _ in ckpts:
        print(f"  ckpt_{step}")

    # --- 3. Load & densify data (once) -----------------------------------
    from muon import read_h5mu

    print(f"\n[{_now()}] Loading train data ...")
    train_mdata = read_h5mu(train_path)
    print(f"  train: {train_mdata}")

    print(f"[{_now()}] Loading test data ...")
    test_mdata = read_h5mu(test_path)
    print(f"  test:  {test_mdata}")

    # Extract obs DataFrames for UMAP coloring
    train_obs = train_mdata[key_1].obs.copy()
    test_obs = test_mdata[key_1].obs.copy()

    print(f"\n[{_now()}] Densifying data (one-time) ...")
    t0 = time.time()

    print(f"[{_now()}]   [train] densifying ...")
    train_tensor_1 = adata_to_tensor(train_mdata[key_1])
    train_tensor_2 = adata_to_tensor(train_mdata[key_2])
    print(f"[{_now()}]   [train] done (RSS={_rss_gb():.1f} GB)")

    print(f"[{_now()}]   [test]  densifying ...")
    test_tensor_1 = adata_to_tensor(test_mdata[key_1])
    test_tensor_2 = adata_to_tensor(test_mdata[key_2])
    print(f"[{_now()}]   [test]  done (RSS={_rss_gb():.1f} GB)")

    print(f"[{_now()}]   >> Densify all: {time.time() - t0:.1f}s")

    # Free the heavy MuData (keep tensors + obs)
    del train_mdata, test_mdata
    gc.collect()
    print(f"[{_now()}]   MuData freed (RSS={_rss_gb():.1f} GB)")

    # --- 4. Create output directory structure -----------------------------
    for sub in [
        "embeddings/train", "embeddings/test",
        "umap/train", "umap/test",
    ]:
        os.makedirs(os.path.join(REPORT_DIR, sub), exist_ok=True)

    # Copy config for reference
    import shutil
    shutil.copy2(CONFIG_PATH, os.path.join(REPORT_DIR, "config.yaml"))
    print(f"\nOutput dir: {REPORT_DIR}")

    # --- 5. Process each checkpoint --------------------------------------
    for step, ckpt_path in ckpts:
        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{_now()}] Checkpoint: ckpt_{step}")

        # -- Load weights --
        t0 = time.time()
        model.load_state_dict(ckpt_path)
        model.to(DEVICE)
        model.eval()
        print(f"[{_now()}]   load weights: {time.time() - t0:.1f}s")

        # -- Train embeddings --
        t0 = time.time()
        print(f"[{_now()}]   [train] extracting embeddings ...")
        train_ds = PairDataset(train_tensor_1, train_tensor_2)
        train_emb = get_paired_embedding(
            model, train_ds, device=DEVICE, batch_size=BATCH_SIZE,
            desc=f"[train ckpt_{step}]",
        )
        t_emb = time.time() - t0
        print(f"[{_now()}]   [train] inference: {t_emb:.1f}s  "
              f"RNA={train_emb['z_mu_1'].shape}  ATAC={train_emb['z_mu_2'].shape}")

        # -- Save train embeddings --
        emb_train_dir = os.path.join(REPORT_DIR, "embeddings", "train")
        np.save(os.path.join(emb_train_dir, f"ckpt_{step}_z_mu_1.npy"), train_emb["z_mu_1"])
        np.save(os.path.join(emb_train_dir, f"ckpt_{step}_z_mu_2.npy"), train_emb["z_mu_2"])

        # -- Train UMAP --
        t0 = time.time()
        print(f"[{_now()}]   [train] computing UMAP ...")
        train_umap_dir = os.path.join(REPORT_DIR, "umap", "train", f"ckpt_{step}")
        os.makedirs(train_umap_dir, exist_ok=True)

        adata_train = sc.AnnData(train_emb["z_mu_1"])
        sc.pp.neighbors(adata_train, use_rep="X", n_neighbors=15, random_state=UMAP_RANDOM_STATE)
        sc.tl.umap(adata_train, min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE)
        np.save(os.path.join(train_umap_dir, "umap_coords.npy"), adata_train.obsm["X_umap"])

        # Generate UMAP PNGs
        adata_train.obs = train_obs.iloc[:train_emb["z_mu_1"].shape[0]].copy()
        for key in COLOR_KEYS:
            if key not in adata_train.obs.columns:
                print(f"  [WARN] train obs column '{key}' not found — skipping")
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            sc.pl.umap(adata_train, color=key, ax=ax, show=False, frameon=False, legend_loc="right margin")
            fname = f"umap_ckpt{step}_{key}.png"
            fig.savefig(os.path.join(train_umap_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
        t_umap = time.time() - t0
        print(f"[{_now()}]   [train] UMAP done: {t_umap:.1f}s")

        # Free train intermediates
        del train_ds, train_emb, adata_train
        gc.collect()

        # -- Test embeddings --
        t0 = time.time()
        print(f"[{_now()}]   [test]  extracting embeddings ...")
        test_ds = PairDataset(test_tensor_1, test_tensor_2)
        test_emb = get_paired_embedding(
            model, test_ds, device=DEVICE, batch_size=BATCH_SIZE,
            desc=f"[test  ckpt_{step}]",
        )
        t_emb = time.time() - t0
        print(f"[{_now()}]   [test]  inference: {t_emb:.1f}s  "
              f"RNA={test_emb['z_mu_1'].shape}  ATAC={test_emb['z_mu_2'].shape}")

        # -- Save test embeddings --
        emb_test_dir = os.path.join(REPORT_DIR, "embeddings", "test")
        np.save(os.path.join(emb_test_dir, f"ckpt_{step}_z_mu_1.npy"), test_emb["z_mu_1"])
        np.save(os.path.join(emb_test_dir, f"ckpt_{step}_z_mu_2.npy"), test_emb["z_mu_2"])

        # -- Test UMAP --
        t0 = time.time()
        print(f"[{_now()}]   [test]  computing UMAP ...")
        test_umap_dir = os.path.join(REPORT_DIR, "umap", "test", f"ckpt_{step}")
        os.makedirs(test_umap_dir, exist_ok=True)

        adata_test = sc.AnnData(test_emb["z_mu_1"])
        sc.pp.neighbors(adata_test, use_rep="X", n_neighbors=15, random_state=UMAP_RANDOM_STATE)
        sc.tl.umap(adata_test, min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE)
        np.save(os.path.join(test_umap_dir, "umap_coords.npy"), adata_test.obsm["X_umap"])

        adata_test.obs = test_obs.iloc[:test_emb["z_mu_1"].shape[0]].copy()
        for key in COLOR_KEYS:
            if key not in adata_test.obs.columns:
                print(f"  [WARN] test obs column '{key}' not found — skipping")
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            sc.pl.umap(adata_test, color=key, ax=ax, show=False, frameon=False, legend_loc="right margin")
            fname = f"umap_ckpt{step}_{key}.png"
            fig.savefig(os.path.join(test_umap_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
        t_umap = time.time() - t0
        print(f"[{_now()}]   [test]  UMAP done: {t_umap:.1f}s")

        # Free test intermediates
        del test_ds, test_emb, adata_test
        gc.collect()

        # Summary
        t_total = time.time() - t_start
        print(f"[{_now()}]   >> ckpt_{step} total: {t_total:.1f}s  (RSS={_rss_gb():.1f} GB)")

    # --- 6. Summary ------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[{_now()}] All done! Results saved to: {REPORT_DIR}")

    # Print overview
    for split in ["train", "test"]:
        emb_dir = os.path.join(REPORT_DIR, "embeddings", split)
        umap_base = os.path.join(REPORT_DIR, "umap", split)
        npy_files = sorted(glob.glob(os.path.join(emb_dir, "*.npy")))
        umap_dirs = sorted(glob.glob(os.path.join(umap_base, "ckpt_*")))

        total_emb_mb = sum(os.path.getsize(f) for f in npy_files) / (1024 ** 2)
        total_umap_mb = 0
        for d in umap_dirs:
            for f in glob.glob(os.path.join(d, "*")):
                total_umap_mb += os.path.getsize(f) / (1024 ** 2)

        print(f"\n  [{split}]")
        print(f"    Embeddings: {len(npy_files)} files, {total_emb_mb:.1f} MB")
        print(f"    UMAP dirs:  {len(umap_dirs)} checkpoints, {total_umap_mb:.1f} MB")


if __name__ == "__main__":
    main()
