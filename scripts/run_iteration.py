#!/usr/bin/env python3
"""Per-round evaluation and report generation for Solid Recover iteration.

Loads all checkpoints from a training output directory, extracts RNA
embeddings from **train data only**, computes ARI/NMI via Leiden clustering,
generates UMAP plots, and produces a Markdown report (optionally emailed).

The kNN graph (``sc.pp.neighbors``) and UMAP projection are computed **once**
per checkpoint and shared between metrics and visualisation, avoiding
redundant O(n^2) computation.

Usage::

    python scripts/run_iteration.py \\
        --output-dir outputs/hca_brain_dev_cvae \\
        --round 1 \\
        --strategy "CVAE (batch_embed_dim=8)" \\
        --data-dir .          # where train_sub.h5mu / test_sub.h5mu live
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import muon as mu
import numpy as np
import scanpy as sc
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solid_recover.config.loader import load_train_config
from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import PairDataset
from solid_recover.evaluation.batch_metrics import (
    compute_ari_nmi_leiden,
    generate_umap_plots,
)
from solid_recover.evaluation.embeddings import get_paired_embedding
from solid_recover.models.pair import PairScratch
from scripts.send_report import send_report

# ---------- defaults ----------
LABEL_KEYS = ["Class", "Subclass", "cell_type"]
COLOR_KEYS = ["Class", "Subclass", "cell_type", "Group", "Sample_ID", "Region"]


# ---------- helpers ----------

def _find_checkpoints(output_dir: str) -> List[str]:
    """Locate checkpoint files (``ckpt_*.pth``) sorted by step number."""
    pattern = os.path.join(output_dir, "models", "ckpt_*.pth")
    paths = sorted(glob.glob(pattern), key=_parse_ckpt_step)
    return paths


def _parse_ckpt_step(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"ckpt_(\d+)", base)
    return int(m.group(1)) if m else 0


def _build_model_from_config(cfg, device: str = "cuda") -> PairScratch:
    """Recreate a PairScratch from a TrainConfig."""
    return PairScratch(
        feature_num_1=cfg.model.feature_num_1 or 0,
        feature_num_2=cfg.model.feature_num_2 or 0,
        hidden_params_1=cfg.model.hidden_params_1,
        hidden_params_2=cfg.model.hidden_params_2,
        embed_dim=cfg.model.embed_dim,
        use_rmsnorm=cfg.model.use_rmsnorm,
        use_residual=cfg.model.use_residual,
        dropout_p=cfg.model.dropout_p,
        num_batches=cfg.data.num_batches,
        batch_embed_dim=cfg.model.batch_embed_dim,
    )


# ---------- main ----------

def evaluate_round(
    output_dir: str,
    round_num: int,
    strategy: str,
    data_dir: str,
    device: str = "cuda",
    batch_size: int = 128,
    resolutions: Optional[Sequence[float]] = None,
    recipient: Optional[str] = None,
    smtp_config: Optional[Dict] = None,
    notes: str = "",
) -> str:
    """Evaluate all checkpoints in *output_dir* and produce a report.

    Only train data is used for embedding extraction, ARI/NMI, and UMAP.
    Test data is **not** loaded or processed.

    Returns
    -------
    str — path to the generated report directory.
    """
    # ---- 1. Load config ----
    config_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] config.yaml not found in {output_dir}")
        sys.exit(1)
    cfg = load_train_config(config_path)

    key_1 = cfg.data.key_1
    key_2 = cfg.data.key_2

    train_path = os.path.join(data_dir, os.path.basename(cfg.data.train_data_path))

    # ---- 2. Build model ----
    print(f"Building model: num_batches={cfg.data.num_batches}, batch_embed_dim={cfg.model.batch_embed_dim}")
    model = _build_model_from_config(cfg, device=device)
    model.net.to(device)

    # ---- 3. Load train data (test data is NOT loaded) ----
    print(f"Loading train data: {train_path}")
    mdata_train = mu.read_h5mu(train_path)

    ds_train = PairDataset(
        adata_to_tensor(mdata_train[key_1]),
        adata_to_tensor(mdata_train[key_2]),
    )
    obs_train = mdata_train[key_1].obs

    # ---- 4. Evaluate each checkpoint ----
    ckpts = _find_checkpoints(output_dir)
    if not ckpts:
        print(f"[ERROR] No checkpoints found in {output_dir}/models/")
        sys.exit(1)
    print(f"Found {len(ckpts)} checkpoints: {[_parse_ckpt_step(c) for c in ckpts]}")

    report_dir = os.path.join("reports", f"round_{round_num}_{strategy.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}")
    os.makedirs(report_dir, exist_ok=True)

    all_metrics: Dict[int, Dict] = {}
    all_umaps: Dict[int, Dict] = {}

    for ckpt_path in ckpts:
        step = _parse_ckpt_step(ckpt_path)
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {step}")
        print(f"{'='*60}")

        # Load weights
        ckpt_data = torch.load(ckpt_path, map_location=device)
        model.net.load_state_dict(ckpt_data["model_state_dict"])
        model.net.eval()

        # ---- Extract embeddings (train only) ----
        print("Extracting train embeddings ...")
        embeds_train = get_paired_embedding(
            model, ds_train, device=device, batch_size=batch_size, desc="train embed",
        )
        z_train = embeds_train["z_mu_1"]

        # ---- Build shared AnnData: neighbors + UMAP computed ONCE ----
        print("Computing kNN graph ...")
        adata = sc.AnnData(z_train, obs=obs_train.copy())
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, random_state=42)

        print("Computing UMAP ...")
        sc.tl.umap(adata, min_dist=0.2, random_state=42)

        # ---- ARI/NMI via Leiden (reuses the shared kNN graph) ----
        step_metrics: Dict = {}
        for lkey in LABEL_KEYS:
            if lkey not in obs_train.columns:
                print(f"  [WARN] Label '{lkey}' not in obs — skipping")
                continue
            labels = obs_train[lkey].values
            result = compute_ari_nmi_leiden(z_train, labels, resolutions=resolutions, adata_sc=adata)
            step_metrics[lkey] = result
            print(f"  {lkey}: ARI={result['best_ari']:.4f}  NMI={result['best_nmi']:.4f}  (res={result['best_resolution']})")

        all_metrics[step] = step_metrics

        # ---- UMAP plots (reuses the shared AnnData) ----
        umap_dir = os.path.join(report_dir, "umap")
        umap_paths = generate_umap_plots(
            z_train, obs_train, umap_dir, step,
            color_keys=COLOR_KEYS, min_dist=0.2, adata_sc=adata,
        )
        all_umaps[step] = umap_paths
        for k, p in umap_paths.items():
            print(f"  UMAP {k}: {p}")

    # ---- 5. Generate report ----
    round_info = {
        "round": round_num,
        "strategy": strategy,
        "config": config_path,
        "notes": notes,
    }

    training_info = {
        "train_steps": cfg.training.train_steps,
        "batch_size": cfg.data.batch_size,
        "embed_dim": cfg.model.embed_dim,
        "num_batches": cfg.data.num_batches,
        "batch_embed_dim": cfg.model.batch_embed_dim,
    }

    hypers = {
        "lr": cfg.optimizer.lr,
        "warmup_steps": cfg.optimizer.warmup_steps,
        "clip_weight": cfg.loss.clip_weight,
        "cross_recon_1": cfg.loss.cross_recon_1,
        "cross_recon_2": cfg.loss.cross_recon_2,
        "temperature": cfg.loss.temperature,
        "adversarial_batch_weight": cfg.loss.adversarial_batch_weight,
        "batch_alignment_weight": cfg.loss.batch_alignment_weight,
    }

    report_path = send_report(
        round_info=round_info,
        metrics=all_metrics,
        umap_paths=all_umaps,
        output_dir=report_dir,
        recipient=recipient or "sunrui171@mails.ucas.edu.cn",
        smtp_config=smtp_config,
        hypers=hypers,
        training_info=training_info,
    )
    print(f"\nReport saved to: {report_path}")
    return report_dir


# ---------- CLI ----------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Solid Recover per-round evaluation and report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir", required=True, help="Training output directory")
    p.add_argument("--round", type=int, required=True, help="Round number")
    p.add_argument("--strategy", required=True, help='Strategy name, e.g. "CVAE (dim=8)"')
    p.add_argument("--data-dir", default=".", help="Directory containing train_sub.h5mu / test_sub.h5mu")
    p.add_argument("--device", default="cuda", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=128, help="Embedding extraction batch size")
    p.add_argument("--recipient", help="Email recipient (default: sunrui171@mails.ucas.edu.cn)")
    p.add_argument("--notes", default="", help="Additional notes for report")
    p.add_argument("--smtp-host", help="SMTP host")
    p.add_argument("--smtp-port", type=int, default=587)
    p.add_argument("--smtp-user", help="SMTP username")
    p.add_argument("--smtp-pass", help="SMTP password")
    args = p.parse_args(argv)

    smtp_config = None
    if args.smtp_host:
        smtp_config = {
            "host": args.smtp_host,
            "port": args.smtp_port,
            "username": args.smtp_user,
            "password": args.smtp_pass,
            "from_addr": args.smtp_user,
        }

    evaluate_round(
        output_dir=args.output_dir,
        round_num=args.round,
        strategy=args.strategy,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        recipient=args.recipient,
        smtp_config=smtp_config,
        notes=args.notes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
