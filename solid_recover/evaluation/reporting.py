"""Per-checkpoint evaluation reports (matching metrics + hit rates)."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable

import numpy as np

from solid_recover._logging import get_logger
from solid_recover.evaluation.embeddings import get_paired_embedding
from solid_recover.evaluation.metrics import (
    calculate_hit_rate,
    calculate_hit_rates_batch,
    matching_metrics,
)
from solid_recover.models.pair import PairScratch

_logger = get_logger(__name__)

DEFAULT_HIT_KS = (1, 5, 10, 15, 20, 30, 50, 100)


def evaluate_checkpoint(
    model: PairScratch,
    dataset,
    eval_res_dir: str,
    ckpt_steps: int,
    device: str = "cpu",
    hit_ks: Iterable[int] = DEFAULT_HIT_KS,
    metric: str = "cosine",
) -> Dict[str, float]:
    """Run matching-metric evaluation for one checkpoint, persist JSON result.

    Assumes ``model`` has already had the target checkpoint loaded via
    :meth:`PairScratch.load_state_dict`. Returns the metric dict for in-memory
    consumption; also writes ``<eval_res_dir>/match_metric.json``.
    
    Optimized to reuse intermediate results and avoid redundant computations:
    - Uses calculate_hit_rates_batch to fit NearestNeighbors once with max(k)
    - Reuses the neighbor indices for all smaller k values
    """
    os.makedirs(eval_res_dir, exist_ok=True)

    # Get embeddings
    embeds = get_paired_embedding(model, dataset, device=device)
    z_mu, y_mu = embeds["z_mu_1"], embeds["z_mu_2"]

    res: Dict[str, float] = {}
    
    # Optimized: Calculate all hit rates in one pass (fits NearestNeighbors once)
    hit_rate_results = calculate_hit_rates_batch(z_mu, y_mu, hit_ks, metric=metric)
    for k, hit_rate in hit_rate_results.items():
        res[f"top_{k}_hit"] = float(hit_rate)
    
    # Compute foscttm
    fs = matching_metrics(x=z_mu, y=y_mu, metric=metric)
    res["foscttm"] = fs
    res["ckpt_steps"] = ckpt_steps

    # Save results
    with open(os.path.join(eval_res_dir, "match_metric.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    _logger.info("eval ckpt=%d foscttm=%.4f", ckpt_steps, fs)
    return res


__all__ = ["evaluate_checkpoint", "DEFAULT_HIT_KS"]
