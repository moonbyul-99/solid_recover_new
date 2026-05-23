#!/usr/bin/env python3
"""
Extract SR model multi-omics embeddings and compute UMAP for each checkpoint.

For each checkpoint of the PairScratch model under mus_kidney_sex:
  1. Extract RNA (z_mu_1) and ATAC (z_mu_2) embeddings for both train and test data.
  2. Compute UMAP on the RNA embeddings using scanpy (min_dist=0.5).
  3. Write all results back into the corresponding .h5mu files as obsm entries.

Usage (in snapatac conda env):
    conda activate snapatac
    cd /home/rsun@ZHANGroup.local/solid_recover_main
    python scripts/extract_embeddings_umap.py
"""

from __future__ import annotations

import glob
import gc
import os
import re
import resource
import time
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
import torch
import yaml
from muon import read_h5mu

from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import PairDataset
from solid_recover.evaluation.embeddings import get_paired_embedding
from solid_recover.models.pair import PairScratch


# ---------------------------------------------------------------------------
# Configuration – adjust these paths if needed
# ---------------------------------------------------------------------------
OUTPUT_DIR = "/home/rsun@ZHANGroup.local/solid_recover_main/outputs/mus_kidney_sex_20260522_0946"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.yaml")

# UMAP parameters
UMAP_MIN_DIST = 0.5
UMAP_RANDOM_STATE = 42


def load_config(config_path: str) -> dict:
    """Load the YAML training config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg: dict) -> PairScratch:
    """Build a PairScratch model matching the config architecture."""
    m = cfg["model"]
    return PairScratch(
        feature_num_1=m["feature_num_1"],
        feature_num_2=m["feature_num_2"],
        hidden_params_1=m["hidden_params_1"],
        hidden_params_2=m["hidden_params_2"],
        embed_dim=m["embed_dim"],
        use_rmsnorm=m.get("use_rmsnorm", True),
        use_residual=m.get("use_residual", False),
        dropout_p=m.get("dropout_p", 0.0),
    )


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


def compute_umap(
    embedding: np.ndarray,
    min_dist: float = UMAP_MIN_DIST,
    random_state: int = UMAP_RANDOM_STATE,
) -> np.ndarray:
    """Compute UMAP coordinates from an embedding matrix using scanpy."""
    adata = sc.AnnData(embedding)
    sc.pp.neighbors(adata, use_rep="X", random_state=random_state)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    return adata.obsm["X_umap"].copy()


def _now() -> str:
    """Return current wall-clock time as 'HH:MM:SS'."""
    return time.strftime("%H:%M:%S", time.localtime())


# def _rss_gb() -> float:
#     """Current process RSS in GB."""
#     return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
import psutil

def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def process_checkpoint(
    model: PairScratch,
    ckpt_path: str,
    step: int,
    train_tensor_1: torch.Tensor,
    train_tensor_2: torch.Tensor,
    test_tensor_1: torch.Tensor,
    test_tensor_2: torch.Tensor,
    train_mdata,
    test_mdata,
    key_1: str,
    key_2: str,
    device: str,
) -> None:
    """
    Extract embeddings + UMAP for one checkpoint and write into h5mu objects.

    Stored keys in obsm:
      - mod[key_1].obsm["X_sr_embed_ckpt_{step}"]  → RNA embedding
      - mod[key_1].obsm["X_sr_umap_ckpt_{step}"]   → UMAP of RNA embedding
      - mod[key_2].obsm["X_sr_embed_ckpt_{step}"]  → ATAC embedding
    """
    suffix = f"sr_ckpt_{step}"
    t_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"[{_now()}] Checkpoint: ckpt_{step}  ({ckpt_path})")

    # --- Load weights ---
    t0 = time.time()
    model.load_state_dict(ckpt_path)
    model.to(device)
    model.eval()
    t_load = time.time() - t0
    print(f"[{_now()}]   load weights:  {t_load:.1f}s")

    # --- Process train data ---
    t0 = time.time()
    print(f"[{_now()}]   [train] Extracting embeddings (DataLoader + GPU) ...")
    train_ds = PairDataset(train_tensor_1, train_tensor_2)
    train_emb = get_paired_embedding(model, train_ds, device=device, desc=f"[train ckpt_{step}]")
    t_train_emb = time.time() - t0
    print(f"[{_now()}]   [train] inference done:  {t_train_emb:.1f}s")
    print(f"    RNA embed  shape: {train_emb['z_mu_1'].shape}")
    print(f"    ATAC embed shape: {train_emb['z_mu_2'].shape}")

    train_mdata.mod[key_1].obsm[f"X_embed_{suffix}"] = train_emb["z_mu_1"]
    train_mdata.mod[key_2].obsm[f"X_embed_{suffix}"] = train_emb["z_mu_2"]

    t0 = time.time()
    print(f"[{_now()}]   [train] Computing UMAP (min_dist={UMAP_MIN_DIST}) ...")
    train_umap = compute_umap(train_emb["z_mu_1"])
    t_train_umap = time.time() - t0
    print(f"[{_now()}]   [train] UMAP done:  {t_train_umap:.1f}s")
    train_mdata.mod[key_1].obsm[f"X_umap_{suffix}"] = train_umap
    print(f"    UMAP shape: {train_umap.shape}")

    # --- Free train data to avoid memory pressure on test phase ---
    print(f"[{_now()}]   [train] freeing tensors (RSS before={_rss_gb():.1f} GB) ...")
    del train_ds, train_emb, train_umap
    gc.collect()
    print(f"[{_now()}]   [train] memory freed  (RSS after={_rss_gb():.1f} GB)")

    # --- Process test data ---
    t0 = time.time()
    print(f"[{_now()}]   [test]  Extracting embeddings (DataLoader + GPU) ...")
    test_ds = PairDataset(test_tensor_1, test_tensor_2)
    test_emb = get_paired_embedding(model, test_ds, device=device, desc=f"[test  ckpt_{step}]")
    
    t_test_emb = time.time() - t0
    print(f"[{_now()}]   [test]  inference done:  {t_test_emb:.1f}s")
    print(f"    RNA embed  shape: {test_emb['z_mu_1'].shape}")
    print(f"    ATAC embed shape: {test_emb['z_mu_2'].shape}")

    test_mdata.mod[key_1].obsm[f"X_embed_{suffix}"] = test_emb["z_mu_1"]
    test_mdata.mod[key_2].obsm[f"X_embed_{suffix}"] = test_emb["z_mu_2"]

    t0 = time.time()
    print(f"[{_now()}]   [test]  Computing UMAP (min_dist={UMAP_MIN_DIST}) ...")
    test_umap = compute_umap(test_emb["z_mu_1"])
    t_test_umap = time.time() - t0
    print(f"[{_now()}]   [test]  UMAP done:  {t_test_umap:.1f}s")
    test_mdata.mod[key_1].obsm[f"X_umap_{suffix}"] = test_umap
    print(f"    UMAP shape: {test_umap.shape}")

    # --- Summary ---
    t_total = time.time() - t_start
    print(f"[{_now()}]   >> ckpt_{step} total: {t_total:.1f}s  "
          f"(load={t_load:.1f}s, train_emb={t_train_emb:.1f}s, "
          f"train_umap={t_train_umap:.1f}s, "
          f"test_emb={t_test_emb:.1f}s, test_umap={t_test_umap:.1f}s)")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    print(f"Loading config: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)
    train_path = cfg["data"]["train_data_path"]
    test_path = cfg["data"]["test_data_path"]
    key_1 = cfg["data"]["key_1"]  # e.g. "rna_count"
    key_2 = cfg["data"]["key_2"]  # e.g. "atac_count"
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")
    print(f"  key_1: {key_1}")
    print(f"  key_2: {key_2}")

    # ------------------------------------------------------------------
    # 2. Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ------------------------------------------------------------------
    # 3. Build model and find checkpoints
    # ------------------------------------------------------------------
    model = build_model_from_config(cfg)
    ckpts = find_checkpoints(MODELS_DIR)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt_*.pth files found in {MODELS_DIR}")
    print(f"\nFound {len(ckpts)} checkpoints:")
    for step, path in ckpts:
        print(f"  ckpt_{step}")

    # ------------------------------------------------------------------
    # 4. Load h5mu data (once)
    # ------------------------------------------------------------------
    print(f"\nLoading train data ...")
    train_mdata = read_h5mu(train_path)
    print(f"  train: {train_mdata}")

    print(f"Loading test data ...")
    test_mdata = read_h5mu(test_path)
    print(f"  test:  {test_mdata}")

    # ------------------------------------------------------------------
    # 5. Densify ALL data upfront (one-time cost)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[{_now()}] Densifying ALL data (one-time cost) ...")
    t0 = time.time()
    
    print(f"[{_now()}]   [train] Densifying train data ...")
    train_tensor_1 = adata_to_tensor(train_mdata[key_1])
    train_tensor_2 = adata_to_tensor(train_mdata[key_2])
    print(f"[{_now()}]   [train] done  (RSS={_rss_gb():.1f} GB)")
    
    print(f"[{_now()}]   [test]  Densifying test data ...")
    test_tensor_1 = adata_to_tensor(test_mdata[key_1])
    test_tensor_2 = adata_to_tensor(test_mdata[key_2])
    print(f"[{_now()}]   [test]  done  (RSS={_rss_gb():.1f} GB)")
    
    t_densify_all = time.time() - t0
    print(f"[{_now()}]   >> All densify done: {t_densify_all:.1f}s")

    # ------------------------------------------------------------------
    # 6. Process each checkpoint (no more repeated densify)
    # ------------------------------------------------------------------
    for step, ckpt_path in ckpts:
        process_checkpoint(
            model=model,
            ckpt_path=ckpt_path,
            step=step,
            train_tensor_1=train_tensor_1,
            train_tensor_2=train_tensor_2,
            test_tensor_1=test_tensor_1,
            test_tensor_2=test_tensor_2,
            train_mdata=train_mdata,
            test_mdata=test_mdata,
            key_1=key_1,
            key_2=key_2,
            device=device,
        )

    # # ------------------------------------------------------------------
    # # 7. Write back modified h5mu files
    # # ------------------------------------------------------------------
    # print(f"\n{'=' * 60}")
    # train_path = os.path.join(OUTPUT_DIR, "train.h5mu")
    # print(f"Writing train data → {train_path}")
    # train_mdata.write(train_path)

    # test_path = os.path.join(OUTPUT_DIR, "test.h5mu")
    # print(f"Writing test data  → {test_path}")
    # test_mdata.write(test_path)

    # print(f"\n{'=' * 60}")
    # print("Done!  Summary of keys written to each h5mu file:")
    # for name, mdata in [("train", train_mdata), ("test", test_mdata)]:
    #     print(f"\n  [{name}] mod['{key_1}'].obsm keys:")
    #     for k in sorted(mdata.mod[key_1].obsm.keys()):
    #         print(f"    {k}: {mdata.mod[key_1].obsm[k].shape}")
    #     print(f"  [{name}] mod['{key_2}'].obsm keys:")
    #     for k in sorted(mdata.mod[key_2].obsm.keys()):
    #         print(f"    {k}: {mdata.mod[key_2].obsm[k].shape}")
    # ------------------------------------------------------------------
    # 7. Write lightweight h5mu (embeddings + UMAP only, no count data)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    from anndata import AnnData
    import muon as mu

    train_path = os.path.join(OUTPUT_DIR, "train_embeddings.h5mu")
    test_path = os.path.join(OUTPUT_DIR, "test_embeddings.h5mu")

    for name, mdata, out_path in [
        ("train", train_mdata, train_path),
        ("test", test_mdata, test_path),
    ]:
        print(f"Writing {name} embeddings → {out_path}")
        mod_dict = {}
        for mod_key in mdata.mod.keys():
            obsm_keys = [k for k in mdata.mod[mod_key].obsm.keys() if f"sr_ckpt_" in k]
            adata = AnnData(
                obs=mdata.mod[mod_key].obs.copy(),
            )
            for k in obsm_keys:
                adata.obsm[k] = mdata.mod[mod_key].obsm[k].copy()
            mod_dict[mod_key] = adata
        out_mdata = mu.MuData(mod_dict)
        mu.write_h5mu(out_path, out_mdata)

    # Free the heavy mdata objects
    del train_mdata, test_mdata
    gc.collect()

    print(f"\n{'=' * 60}")
    print("Done!  Summary of keys written to each h5mu file:")
    for name, path in [("train", train_path), ("test", test_path)]:
        m = mu.read_h5mu(path)
        for mod_key in m.mod.keys():
            print(f"\n  [{name}] mod['{mod_key}'].obsm keys:")
            for k in sorted(m.mod[mod_key].obsm.keys()):
                print(f"    {k}: {m.mod[mod_key].obsm[k].shape}")


if __name__ == "__main__":
    main()