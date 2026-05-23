#!/usr/bin/env python3
"""Prepare brain development subset: ATAC peak downsampling + cell subsampling."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import muon as mu
import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solid_recover.config.loader import load_train_config, dump_train_config

RANDOM_SEED = 42
TOP_K_ATAC = 100_000
N_TRAIN_CELLS = 40_000
N_TEST_CELLS = 20_000
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "case_brain_dev.yaml"
TARGET_CONFIG = PROJECT_ROOT / "configs" / "case_brain_dev_sub.yaml"


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Read config
    cfg = load_train_config(str(SOURCE_CONFIG))
    train_path = cfg.data.train_data_path
    test_path = cfg.data.test_data_path
    key_1 = cfg.data.key_1
    key_2 = cfg.data.key_2

    # 2. Load train data, compute peak selection
    print(f"Loading train: {train_path}")
    mdata_train = mu.read_h5mu(train_path)
    adata_train_atac = mdata_train[key_2]  # AnnData view

    print("Computing ATAC peak non-zero ratios ...")
    X = adata_train_atac.X
    n_cells = X.shape[0]
    if sp.issparse(X):
        nz_per_col = np.asarray(X.getnnz(axis=0)).ravel()
    else:
        nz_per_col = (X != 0).sum(axis=0)
    nz_ratio = nz_per_col / n_cells
    order = np.argsort(-nz_ratio)
    top_idx = order[:min(TOP_K_ATAC, len(order))]
    print(f"  Kept {len(top_idx)} / {X.shape[1]} peaks")

    # 3a. Train: sample cells, extract adata, subset peaks, save
    n_train_orig = mdata_train.n_obs
    train_cell_idx = np.sort(rng.choice(n_train_orig, size=N_TRAIN_CELLS, replace=False))
    adata1_train = mdata_train[key_1][train_cell_idx, :].copy()
    adata2_train = mdata_train[key_2][train_cell_idx, :][:, top_idx].copy()
    train_sub = mu.MuData({key_1: adata1_train, key_2: adata2_train})
    # Copy obs from RNA modality (cell-level metadata)
    train_sub.obs = mdata_train[key_1][train_cell_idx, :].obs.copy()
    train_out = str(PROJECT_ROOT / "train_sub.h5mu")
    mu.write_h5mu(train_out, train_sub)
    print(f"Saved {train_out}: {train_sub.n_obs} cells")
    del mdata_train, adata_train_atac, train_sub
    gc.collect()

    # 3b. Test: sample cells, extract adata, subset peaks, save
    print(f"Loading test: {test_path}")
    mdata_test = mu.read_h5mu(test_path)
    n_test_orig = mdata_test.n_obs
    test_cell_idx = np.sort(rng.choice(n_test_orig, size=N_TEST_CELLS, replace=False))
    adata1_test = mdata_test[key_1][test_cell_idx, :].copy()
    adata2_test = mdata_test[key_2][test_cell_idx, :][:, top_idx].copy()
    test_sub = mu.MuData({key_1: adata1_test, key_2: adata2_test})
    test_sub.obs = mdata_test[key_1][test_cell_idx, :].obs.copy()
    test_out = str(PROJECT_ROOT / "test_sub.h5mu")
    mu.write_h5mu(test_out, test_sub)
    print(f"Saved {test_out}: {test_sub.n_obs} cells")

    # 4. Save indices
    np.save(str(PROJECT_ROOT / "train_sub_indices.npy"), train_cell_idx)
    np.save(str(PROJECT_ROOT / "test_sub_indices.npy"), test_cell_idx)
    print("Saved train_sub_indices.npy, test_sub_indices.npy")

    # 5. Create sub-config
    cfg.data.train_data_path = train_out
    cfg.data.test_data_path = test_out
    cfg.model.feature_num_2 = len(top_idx)
    cfg.training.train_steps = 6000
    cfg.training.save_points = 1500
    cfg.training.eval_points = 500
    cfg.optimizer.warmup_steps = 500
    cfg.optimizer.steady_1_steps = 500
    cfg.optimizer.cosine_anneal_steps = 5000

    dump_train_config(cfg, str(TARGET_CONFIG))
    print(f"Config saved: {TARGET_CONFIG}")
    print("Done.")


if __name__ == "__main__":
    main()
