#!/usr/bin/env python3
"""Standalone pseudo-pair self-training for :class:`PairScratch`.

Reads a standard pair-training YAML config (e.g. ``case_renal_cortex.yaml``).
Treats ``train.h5mu`` as real paired data and ``test.h5mu`` as **unmatched**
single-omics data (the original row-to-row pairing is deliberately ignored).

Training schedule
-----------------
1. **Warmup** (steps 0 … ``warmup_steps``): standard paired training.
2. **Self-training** (steps ``warmup_steps`` … ``train_steps``): every
   ``rematch_interval`` steps the model encodes all cells, finds nearest
   neighbours in embedding space, and constructs pseudo cross-modality pairs.
   These are concatenated with the real pairs into a combined DataLoader.

Usage
-----
::

    python scripts/run_pseudo_pair.py --config configs/case_renal_cortex.yaml
    python scripts/run_pseudo_pair.py --config configs/case_renal_cortex.yaml \\
        --warmup-steps 2000 --rematch-interval 500 --k-neighbors 3
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import muon as mu
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -- project imports ---------------------------------------------------------
from solid_recover._logging import get_logger
from solid_recover.config.loader import load_train_config
from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import PairDataset
from solid_recover.models.pair import PairScratch
from solid_recover.training.pseudo_pair import (
    PseudoPairDataset,
    build_pseudo_pairs,
    build_pseudo_pairs_mnn,
    compute_topk_hit,
)
from solid_recover.training.scheduler import SRScheduler

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_project_dir(project_dir: str) -> str:
    """Create or timestamp-suffix the project directory."""
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
        return project_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    suffixed = f"{project_dir}_{ts}"
    os.makedirs(suffixed, exist_ok=True)
    return suffixed


def _log_scalars(writer: SummaryWriter, loss_dic: dict, step: int, phase: str) -> None:
    """Write loss components to TensorBoard."""
    for key, value in loss_dic.items():
        if "loss" in key or "logit_scale" in key:
            scalar = value.item() if isinstance(value, torch.Tensor) else value
            writer.add_scalar(f"{key}/{phase}", scalar, step)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pseudo-pair self-training for PairScratch"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/case_renal_cortex.yaml)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2000,
        help="Steps of pure paired training before first rematch (T).",
    )
    parser.add_argument(
        "--rematch-interval", type=int, default=500,
        help="Steps between pseudo-pair re-matching (K).",
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=3,
        help="Number of nearest neighbours to aggregate per pseudo-pair.",
    )
    parser.add_argument(
        "--mnn", action="store_true", default=False,
        help="Use mutual nearest neighbours only (ignores --k-neighbors).",
    )
    args = parser.parse_args(argv)

    # ---- 1. Load config ---------------------------------------------------
    cfg = load_train_config(args.config)
    _logger.info(
        "Config loaded: task=%s train_steps=%d embed_dim=%d",
        cfg.task, cfg.training.train_steps, cfg.model.embed_dim,
    )

    # ---- 2. Load data -----------------------------------------------------
    _logger.info("Loading train data: %s", cfg.data.train_data_path)
    train_mdata = mu.read_h5mu(cfg.data.train_data_path)
    X1_real = adata_to_tensor(train_mdata[cfg.data.key_1])
    X2_real = adata_to_tensor(train_mdata[cfg.data.key_2])
    _logger.info("Real paired data: %s, %s", tuple(X1_real.shape), tuple(X2_real.shape))

    _logger.info(
        "Loading test data (unmatched for training, real-pairing for eval): %s",
        cfg.data.test_data_path,
    )
    test_mdata = mu.read_h5mu(cfg.data.test_data_path)
    X1_unmatched = adata_to_tensor(test_mdata[cfg.data.key_1])
    X2_unmatched = adata_to_tensor(test_mdata[cfg.data.key_2])
    _logger.info(
        "Test/eval data: %s (mod-1), %s (mod-2)",
        tuple(X1_unmatched.shape), tuple(X2_unmatched.shape),
    )

    # ---- 2b. Optionally move tensors to GPU (per cfg.data.to_gpu) ---------
    if cfg.data.to_gpu:
        gpu_device = cfg.training.device
        _logger.info("Moving all tensors to %s (cfg.data.to_gpu=True) …", gpu_device)
        X1_real = X1_real.to(gpu_device)
        X2_real = X2_real.to(gpu_device)
        X1_unmatched = X1_unmatched.to(gpu_device)
        X2_unmatched = X2_unmatched.to(gpu_device)
        _logger.info("All tensors on %s.", gpu_device)

    # ---- 3. Build model ---------------------------------------------------
    feat_1 = cfg.model.feature_num_1 or X1_real.shape[1]
    feat_2 = cfg.model.feature_num_2 or X2_real.shape[1]

    model = PairScratch(
        feature_num_1=feat_1,
        feature_num_2=feat_2,
        hidden_params_1=cfg.model.hidden_params_1,  # type: ignore[arg-type]
        hidden_params_2=cfg.model.hidden_params_2,  # type: ignore[arg-type]
        embed_dim=cfg.model.embed_dim,
        use_rmsnorm=cfg.model.use_rmsnorm,
        use_residual=cfg.model.use_residual,
        dropout_p=cfg.model.dropout_p,
    )
    model.set_loss(
        vae_beta_1=cfg.loss.vae_beta_1,
        vae_beta_2=cfg.loss.vae_beta_2,
        clip_weight=cfg.loss.clip_weight,
        cross_recon_1=cfg.loss.cross_recon_1,
        cross_recon_2=cfg.loss.cross_recon_2,
        temperature=cfg.loss.temperature,
        use_weight=cfg.loss.use_weight,
        top_k_ratio=cfg.loss.top_k_ratio,
        bottom_k_ratio=cfg.loss.bottom_k_ratio,
        weight_top=cfg.loss.weight_top,
        weight_bottom=cfg.loss.weight_bottom,
    )

    device = cfg.training.device
    model.net.to(device)
    model.loss_fn.to(device)

    # ---- 4. Optimizer & scheduler -----------------------------------------
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=cfg.optimizer.lr)
    scheduler = SRScheduler(
        optimizer,
        warmup_steps=cfg.optimizer.warmup_steps,
        steady_1_steps=cfg.optimizer.steady_1_steps,
        cosine_anneal_steps=cfg.optimizer.cosine_anneal_steps,
        min_lr=cfg.optimizer.min_lr,
    )

    # ---- 5. Project dir & logging -----------------------------------------
    project_dir = _ensure_project_dir(cfg.training.project_dir)
    model_dir = os.path.join(project_dir, "models")
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    _logger.info("Project dir: %s", project_dir)

    # ---- 6. Hyper-params --------------------------------------------------
    batch_size = cfg.data.batch_size
    train_steps = cfg.training.train_steps
    eval_points = cfg.training.eval_points
    save_points = cfg.training.save_points
    warmup_steps = args.warmup_steps
    rematch_interval = args.rematch_interval
    k_neighbors = args.k_neighbors
    use_mnn = args.mnn

    _logger.info(
        "Pseudo-pair schedule: warmup=%d  rematch_interval=%d  k=%d  mnn=%s  to_gpu=%s",
        warmup_steps, rematch_interval, k_neighbors, use_mnn, cfg.data.to_gpu,
    )

    # ---- 7. Eval helper ---------------------------------------------------
    def _run_eval(step: int) -> None:
        """Cross-modality top-k hit on test set's real (hidden) pairing."""
        model.net.eval()
        t0 = datetime.now()
        metrics = compute_topk_hit(
            model, X1_unmatched, X2_unmatched,
            k_values=(1, 5, 10), batch_size=batch_size,
        )
        model.net.train()
        elapsed = (datetime.now() - t0).total_seconds()
        for name, value in metrics.items():
            writer.add_scalar(f"{name}/eval", value, step)
        _logger.info(
            "[step %d] Eval (%.1fs) — top1=%.4f  top5=%.4f  top10=%.4f",
            step, elapsed,
            metrics.get("top1_hit", 0),
            metrics.get("top5_hit", 0),
            metrics.get("top10_hit", 0),
        )

    # ---- 8. Warmup phase (real pairs only) --------------------------------
    real_dataset = PairDataset(X1_real, X2_real)
    train_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(train_loader)

    _logger.info(">>> Warmup phase: %d steps (real pairs only)", warmup_steps)
    model.net.train()
    pbar = tqdm(total=warmup_steps, desc="warmup")

    for step in range(warmup_steps):
        # fetch next batch (rebuild iterator if exhausted)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        _, loss_dic = model._process_batch(batch, device)
        loss = loss_dic["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        _log_scalars(writer, loss_dic, step, "train")
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], step)

        if (step + 1) % eval_points == 0:
            _logger.info(
                "[step %d] loss=%.4f  clip=%.4f",
                step + 1, loss.item(), loss_dic.get("clip_loss", 0),
            )
            _run_eval(step + 1)
        if (step + 1) % save_points == 0:
            ckpt_path = os.path.join(model_dir, f"ckpt_{step + 1}.pth")
            torch.save({"model_state_dict": model.net.state_dict()}, ckpt_path)
            _logger.info("Saved checkpoint: %s", ckpt_path)

        pbar.update(1)
    pbar.close()

    # ---- 9. Self-training phase (real + pseudo pairs) ---------------------
    _logger.info(
        ">>> Self-training phase: steps %d → %d (rematch every %d steps)",
        warmup_steps, train_steps, rematch_interval,
    )

    # Initial pseudo-pair build before entering the loop.
    _logger.info("Building initial pseudo-pairs (mnn=%s) …", use_mnn)
    t0 = datetime.now()
    if use_mnn:
        X1_all, X2_all, real_mask = build_pseudo_pairs_mnn(
            model, X1_real, X2_real, X1_unmatched, X2_unmatched,
            batch_size=batch_size,
        )
    else:
        X1_all, X2_all, real_mask = build_pseudo_pairs(
            model, X1_real, X2_real, X1_unmatched, X2_unmatched,
            k_neighbors=k_neighbors, batch_size=batch_size,
        )
    _logger.info(
        "Pseudo-pairs built in %.1fs: combined dataset %s (real=%d pseudo=%d)",
        (datetime.now() - t0).total_seconds(),
        tuple(X1_all.shape),
        int(real_mask.sum().item()),
        int((~real_mask).sum().item()),
    )

    combined_dataset = PseudoPairDataset(X1_all, X2_all, real_mask)
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(train_loader)

    model.net.train()
    pbar = tqdm(total=train_steps - warmup_steps, desc="self-train")
    next_rematch = warmup_steps + rematch_interval

    for step in range(warmup_steps, train_steps):
        # ---- Periodic rematch ---------------------------------------------
        if step == next_rematch:
            _logger.info("[step %d] Rebuilding pseudo-pairs …", step)
            t0 = datetime.now()
            # Release old combined tensors before building new ones
            del combined_dataset, train_loader, loader_iter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.net.eval()
            if use_mnn:
                X1_all, X2_all, real_mask = build_pseudo_pairs_mnn(
                    model, X1_real, X2_real, X1_unmatched, X2_unmatched,
                    batch_size=batch_size,
                )
            else:
                X1_all, X2_all, real_mask = build_pseudo_pairs(
                    model, X1_real, X2_real, X1_unmatched, X2_unmatched,
                    k_neighbors=k_neighbors, batch_size=batch_size,
                )
            model.net.train()
            _logger.info(
                "[step %d] Rematch done in %.1fs. Total samples: %d (real=%d pseudo=%d)",
                step,
                (datetime.now() - t0).total_seconds(),
                X1_all.shape[0],
                int(real_mask.sum().item()),
                int((~real_mask).sum().item()),
            )

            combined_dataset = PseudoPairDataset(X1_all, X2_all, real_mask)
            train_loader = DataLoader(
                combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            )
            loader_iter = iter(train_loader)
            next_rematch += rematch_interval

        # ---- Training step ------------------------------------------------
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        _, loss_dic = model._process_batch(batch, device)
        loss = loss_dic["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        _log_scalars(writer, loss_dic, step, "train")
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], step)

        if (step + 1) % eval_points == 0:
            _logger.info(
                "[step %d] loss=%.4f  recon1=%.4f  recon2=%.4f  clip=%.4f",
                step + 1,
                loss.item(),
                loss_dic.get("recon_loss_1", 0),
                loss_dic.get("recon_loss_2", 0),
                loss_dic.get("clip_loss", 0),
            )
            _run_eval(step + 1)
        if (step + 1) % save_points == 0:
            ckpt_path = os.path.join(model_dir, f"ckpt_{step + 1}.pth")
            torch.save({"model_state_dict": model.net.state_dict()}, ckpt_path)
            _logger.info("Saved checkpoint: %s", ckpt_path)

        pbar.update(1)
    pbar.close()

    # ---- 10. Final eval & checkpoint --------------------------------------
    _logger.info("Final eval …")
    _run_eval(train_steps)

    final_ckpt = os.path.join(model_dir, f"ckpt_{train_steps}.pth")
    torch.save({"model_state_dict": model.net.state_dict()}, final_ckpt)
    _logger.info("Final checkpoint saved: %s", final_ckpt)
    _logger.info("Training finished (%d steps). Output dir: %s", train_steps, project_dir)

    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
