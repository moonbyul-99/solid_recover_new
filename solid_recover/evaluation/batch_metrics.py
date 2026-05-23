"""Batch-integration evaluation metrics (ARI/NMI via Leiden clustering + UMAP plots).

Extends the existing ``evaluation`` module with clustering-quality metrics
tailored to multi-sample integration assessment.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

matplotlib.use("Agg")  # headless-safe


# ---------------------------------------------------------------------------
# ARI / NMI via Leiden clustering
# ---------------------------------------------------------------------------

def compute_ari_nmi_leiden(
    embedding: np.ndarray,
    labels: np.ndarray,
    resolutions: Optional[Sequence[float]] = None,
    n_neighbors: int = 15,
    random_state: int = 42,
    adata_sc: Optional["sc.AnnData"] = None,
) -> Dict:
    """Compute ARI and NMI by Leiden clustering over multiple resolutions.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, embed_dim)
        RNA latent representation.
    labels : np.ndarray or pd.Series, shape (n_cells,)
        Ground-truth categorical labels (Class / Subclass / cell_type).
    resolutions : sequence of float, optional
        Leiden resolutions to scan.  Default: ``np.arange(0.1, 1.55, 0.1)``.
    n_neighbors : int
        KNN graph neighbours for ``scanpy.pp.neighbors``.
    random_state : int
        Seed for reproducibility.
    adata_sc : sc.AnnData or None, optional
        Pre-computed AnnData with ``neighbors`` already run.  If provided,
        ``sc.pp.neighbors`` is skipped.  This avoids redundant kNN graph
        construction.

    Returns
    -------
    dict with keys: ``best_resolution``, ``best_ari``, ``best_nmi``,
    ``ari_nmi_sum``, ``all_results``.
    """
    if resolutions is None:
        resolutions = list(np.arange(0.1, 1.55, 0.1))

    if adata_sc is not None:
        adata = adata_sc
    else:
        adata = sc.AnnData(embedding)
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors, random_state=random_state)

    # Filter out NaN / "unknown" / "nan" entries
    if isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    labels = np.asarray(labels, dtype=str)
    valid_mask = ~np.isin(labels, ["nan", "NaN", "unknown", "Unknown", ""])
    labels_valid = labels[valid_mask]

    all_results: List[Dict] = []
    best_sum = -1.0
    best_entry: Dict = {}

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=random_state)
        cluster_labels = np.asarray(adata.obs["leiden"])
        cluster_valid = cluster_labels[valid_mask]

        ari = adjusted_rand_score(labels_valid, cluster_valid)
        nmi = normalized_mutual_info_score(labels_valid, cluster_valid)
        entry = {"resolution": round(res, 1), "ari": round(ari, 4), "nmi": round(nmi, 4)}
        all_results.append(entry)

        if ari + nmi > best_sum:
            best_sum = ari + nmi
            best_entry = entry

    return {
        "best_resolution": best_entry.get("resolution"),
        "best_ari": best_entry.get("ari"),
        "best_nmi": best_entry.get("nmi"),
        "ari_nmi_sum": round(best_sum, 4),
        "all_results": all_results,
    }


# ---------------------------------------------------------------------------
# UMAP plot generation
# ---------------------------------------------------------------------------

def generate_umap_plots(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    output_dir: str,
    ckpt_step: int,
    color_keys: Optional[Sequence[str]] = None,
    min_dist: float = 0.2,
    random_state: int = 42,
    dpi: int = 150,
    adata_sc: Optional["sc.AnnData"] = None,
) -> Dict[str, str]:
    """Compute UMAP and save per-category coloured plots.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, embed_dim)
    obs : pd.DataFrame
        Cell-level metadata (must contain columns named by *color_keys*).
    output_dir : str
        Directory to write PNG files.
    ckpt_step : int
        Checkpoint step number used in filename.
    color_keys : sequence of str, optional
        Columns in *obs* to colour UMAP by.
        Default: ``["Class", "Subclass", "cell_type", "Group", "Sample_ID", "Region"]``.
    min_dist : float
        UMAP ``min_dist`` parameter.
    random_state : int
        UMAP random seed.
    dpi : int
        Output resolution.
    adata_sc : sc.AnnData or None, optional
        Pre-computed AnnData with ``neighbors`` and ``umap`` already run.
        If provided, both steps are skipped.  This avoids redundant kNN and
        UMAP computation.

    Returns
    -------
    dict mapping color_key → absolute PNG path.
    """
    if color_keys is None:
        color_keys = ["Class", "Subclass", "cell_type", "Group", "Sample_ID", "Region"]

    os.makedirs(output_dir, exist_ok=True)

    if adata_sc is not None:
        adata = adata_sc
    else:
        adata = sc.AnnData(embedding, obs=obs.copy())
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, random_state=random_state)
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    paths: Dict[str, str] = {}
    for key in color_keys:
        if key not in adata.obs.columns:
            print(f"  [WARN] obs column '{key}' not found — skipping")
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata, color=key, ax=ax, show=False, frameon=False, legend_loc="right margin")
        fname = f"umap_ckpt{ckpt_step}_{key}.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[key] = fpath

    return paths


# ---------------------------------------------------------------------------
# Per-checkpoint batch evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint_batch(
    model,
    ckpt_path: str,
    mdata,
    key_1: str,
    key_2: str,
    device: str = "cuda",
    batch_size: int = 128,
    label_keys: Optional[Sequence[str]] = None,
    resolutions: Optional[Sequence[float]] = None,
) -> Dict:
    """Load a checkpoint, extract RNA embeddings, and compute ARI/NMI per label.

    Parameters
    ----------
    model : PairScratch
        Pre-initialised model (same architecture as the checkpoint).
    ckpt_path : str
        Path to ``ckpt_N.pth``.
    mdata : mu.MuData
        MuData containing training data.
    key_1, key_2 : str
        Modality keys.
    device : str
        Torch device string.
    batch_size : int
        DataLoader batch size.
    label_keys : sequence of str, optional
        obs columns to evaluate.  Default: ``["Class", "Subclass", "cell_type"]``.
    resolutions : sequence of float, optional
        Leiden resolutions for ARI/NMI scan.

    Returns
    -------
    dict with keys ``ckpt_step`` and per-label ARI/NMI results.
    """
    from solid_recover.data.adata_utils import adata_to_tensor
    from solid_recover.data.datasets import PairDataset
    from solid_recover.evaluation.embeddings import get_paired_embedding

    if label_keys is None:
        label_keys = ["Class", "Subclass", "cell_type"]

    # Load weights
    model.load_state_dict(ckpt_path)

    # Build dataset from MuData
    ds = PairDataset(
        adata_to_tensor(mdata[key_1]),
        adata_to_tensor(mdata[key_2]),
    )

    # Extract embeddings
    embeds = get_paired_embedding(model, ds, device=device, batch_size=batch_size)
    z_rna = embeds["z_mu_1"]

    # Obs for labels
    obs_df = mdata[key_1].obs

    # Ensure embedding and obs row counts match
    assert z_rna.shape[0] == obs_df.shape[0], (
        f"Embedding rows ({z_rna.shape[0]}) != obs rows ({obs_df.shape[0]})"
    )

    # Compute per-label metrics
    results: Dict = {"ckpt_step": _parse_ckpt_step(ckpt_path)}
    for lkey in label_keys:
        if lkey not in obs_df.columns:
            print(f"  [WARN] obs '{lkey}' not found — skipping")
            continue
        labels = obs_df[lkey].values
        metrics = compute_ari_nmi_leiden(z_rna, labels, resolutions=resolutions)
        results[lkey] = metrics

    return results


def _parse_ckpt_step(ckpt_path: str) -> int:
    """Extract step number from a checkpoint filename like ``ckpt_1500.pth``."""
    base = os.path.basename(ckpt_path)
    # ckpt_1500.pth → 1500
    stem = base.replace(".pth", "").replace("ckpt_", "")
    try:
        return int(stem)
    except ValueError:
        return 0


__all__ = [
    "compute_ari_nmi_leiden",
    "generate_umap_plots",
    "evaluate_checkpoint_batch",
]
