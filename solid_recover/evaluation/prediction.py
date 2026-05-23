"""Cross-modality prediction utilities (mini-batch, no filesystem coupling)."""

from __future__ import annotations

from typing import Dict, Optional

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from solid_recover.data.datasets import PairDataset
from solid_recover.models.pair import PairScratch


@torch.no_grad()
def predict_cross_modality(
    model: PairScratch,
    dataset: PairDataset,
    device: str = "cuda",
    batch_size: int = 128,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Run ``model.net(x1, x2)`` in mini-batches and collect cross reconstructions.

    Returns
    -------
    dict with keys:
        - ``omic_1_pred``: modality-1 reconstructed from modality-2 embedding
          (``sr_pair_out['x1']['cross_recon']``), stacked across batches.
        - ``omic_2_pred``: modality-2 reconstructed from modality-1 embedding
          (``sr_pair_out['x2']['cross_recon']``), stacked across batches.

    v2: renamed from the legacy RNA/ATAC-specific ``rna_pred`` / ``atac_pred``
    so downstream code does not have to pretend that every paired task is
    RNA+ATAC.
    """
    model.net.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    omic_1_pred, omic_2_pred = [], []
    iterator = tqdm(loader, desc="predict_cross_modality") if progress else loader
    for batch in iterator:
        x1 = batch["omic_1"].to(device)
        x2 = batch["omic_2"].to(device)
        outputs = model.net(x1, x2)
        omic_1_pred.append(outputs["x1"]["cross_recon"].detach().cpu().numpy())
        omic_2_pred.append(outputs["x2"]["cross_recon"].detach().cpu().numpy())

    return {
        "omic_1_pred": np.concatenate(omic_1_pred, axis=0),
        "omic_2_pred": np.concatenate(omic_2_pred, axis=0),
    }


def predictions_to_anndata(
    predictions: Dict[str, np.ndarray],
    reference_obs: Optional[ad.AnnData] = None,
    reference_var: Optional[ad.AnnData] = None,
    modality: str = "omic_1",
) -> ad.AnnData:
    """Wrap a prediction matrix as ``AnnData`` with optional ``obs`` / ``var`` copy.

    ``reference_obs`` / ``reference_var`` should be the original omics
    ``AnnData`` slices; only ``.obs`` and ``.var`` metadata are copied.
    ``modality`` should be one of ``"omic_1"`` / ``"omic_2"`` matching the
    keys returned by :func:`predict_cross_modality`.
    """
    key = f"{modality}_pred"
    if key not in predictions:
        raise KeyError(f"Expected '{key}' in predictions; got {list(predictions)}")
    adata = ad.AnnData(predictions[key])
    if reference_obs is not None:
        adata.obs = reference_obs.obs.copy()
    if reference_var is not None:
        adata.var = reference_var.var.copy()
    return adata


__all__ = ["predict_cross_modality", "predictions_to_anndata"]
