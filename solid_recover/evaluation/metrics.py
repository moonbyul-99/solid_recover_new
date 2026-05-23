"""Matching and prediction metrics used for cross-modality evaluation.

Direct port of :mod:`src.metrics` with dead / commented-out code removed and
dependencies trimmed (no implicit matplotlib / seaborn imports at module load
time).
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors


def matching_metrics(
    similarity: np.ndarray = None,
    x: np.ndarray = None,
    y: np.ndarray = None,
    metric: str = "euclidean",
) -> float:
    """Compute foscttm on a similarity matrix.

    Either pass a pre-computed ``similarity`` (``(N, N)``) or two equal-shape
    embedding matrices ``x`` / ``y`` plus a ``metric`` in ``{"cosine", "euclidean"}``.
    
    Returns:
        foscttm: Fraction of samples closer than true match (lower is better)
    """
    if similarity is None:
        if x is None or y is None:
            raise ValueError("Provide either similarity or (x, y).")
        if x.shape != y.shape:
            raise ValueError("Shapes of x and y do not match!")

        if metric == "cosine":
            x_norm = np.linalg.norm(x, axis=1, keepdims=True)
            y_norm = np.linalg.norm(y, axis=1, keepdims=True)
            similarity = np.dot(x, y.T) / (x_norm * y_norm.T + 1e-8)
        elif metric == "euclidean":
            similarity = -cdist(x, y, metric="euclidean")
        else:
            raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    if not isinstance(similarity, torch.Tensor):
        similarity = torch.from_numpy(similarity)

    with torch.no_grad():
        foscttm_x = (similarity > torch.diag(similarity)).float().mean(axis=1).mean().item()
        foscttm_y = (similarity > torch.diag(similarity)).float().mean(axis=0).mean().item()
        foscttm = (foscttm_x + foscttm_y) / 2
    
    return float(foscttm)


def calculate_hit_rate(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    k: int,
    metric: str = "euclidean",
    nbrs_a=None,
    nbrs_b=None,
) -> float:
    """Average top-K cross-modal nearest-neighbour hit rate.

    Returns the mean of A->B and B->A hit rates; values are in ``[0, 1]``.
    
    Args:
        embeddings_a: Embeddings from modality A (N, D)
        embeddings_b: Embeddings from modality B (N, D)
        k: Number of nearest neighbors
        metric: Distance metric
        nbrs_a: Pre-fitted NearestNeighbors for embeddings_a (optional, for reuse)
        nbrs_b: Pre-fitted NearestNeighbors for embeddings_b (optional, for reuse)
    
    Returns:
        hit_rate: Average bidirectional hit rate
    """
    n = embeddings_a.shape[0]
    if embeddings_b.shape[0] != n:
        raise ValueError("The two modalities must have the same number of samples")

    # Fit or reuse NearestNeighbors for B
    if nbrs_b is None:
        nbrs_b = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto").fit(embeddings_b)
    indices_a2b = nbrs_b.kneighbors(embeddings_a, return_distance=False)
    hits_a2b = sum(1 for i in range(n) if i in indices_a2b[i])

    # Fit or reuse NearestNeighbors for A
    if nbrs_a is None:
        nbrs_a = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto").fit(embeddings_a)
    indices_b2a = nbrs_a.kneighbors(embeddings_b, return_distance=False)
    hits_b2a = sum(1 for i in range(n) if i in indices_b2a[i])

    return (hits_a2b / n + hits_b2a / n) / 2


def calculate_hit_rates_batch(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    hit_ks: Iterable[int] = (1, 5, 10, 15, 20, 30, 50, 100),
    metric: str = "euclidean",
) -> Dict[int, float]:
    """Calculate hit rates for multiple k values efficiently.
    
    Optimized version that fits NearestNeighbors once with max(k) and reuses
    the results for all smaller k values.
    
    Args:
        embeddings_a: Embeddings from modality A (N, D)
        embeddings_b: Embeddings from modality B (N, D)
        hit_ks: Iterable of k values to compute hit rates for
        metric: Distance metric
    
    Returns:
        Dictionary mapping k -> hit_rate
    """
    hit_ks_list = sorted(hit_ks)
    max_k = max(hit_ks_list)
    n = embeddings_a.shape[0]
    
    if embeddings_b.shape[0] != n:
        raise ValueError("The two modalities must have the same number of samples")
    
    # Fit NearestNeighbors once with max_k
    nbrs_b = NearestNeighbors(n_neighbors=max_k, metric=metric, algorithm="auto").fit(embeddings_b)
    indices_a2b_all = nbrs_b.kneighbors(embeddings_a, return_distance=False)
    
    nbrs_a = NearestNeighbors(n_neighbors=max_k, metric=metric, algorithm="auto").fit(embeddings_a)
    indices_b2a_all = nbrs_a.kneighbors(embeddings_b, return_distance=False)
    
    # Calculate hit rates for each k by slicing the results
    results = {}
    for k in hit_ks_list:
        # For each sample, check if true match is in top-k neighbors
        hits_a2b = sum(1 for i in range(n) if i in indices_a2b_all[i, :k])
        hits_b2a = sum(1 for i in range(n) if i in indices_b2a_all[i, :k])
        results[k] = (hits_a2b / n + hits_b2a / n) / 2
    
    return results


def pearson_corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation; returns an ``(n_features,)`` array."""
    p = a.shape[1]
    corrs = np.empty(p)
    for i in range(p):
        corrs[i], _ = pearsonr(a[:, i], b[:, i])
    return corrs


__all__ = ["matching_metrics", "calculate_hit_rate", "pearson_corr_columns"]
