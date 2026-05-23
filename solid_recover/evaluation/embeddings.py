"""Extract paired embeddings (``z_mu``) from a trained :class:`PairScratch` model."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from solid_recover.data.datasets import PairDataset
from solid_recover.models.pair import PairScratch


@torch.no_grad()
def get_paired_embedding(
    model: PairScratch,
    dataset: PairDataset,
    device: str = "cuda",
    batch_size: int = 128,
    desc: Optional[str] = None,
    num_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """Return ``{'z_mu_1': ..., 'z_mu_2': ...}`` for every row in ``dataset``.

    Replaces the mini-batch loops scattered across the legacy
    ``pred_evaluation/get_sr_pred.py`` and ``result_analysis/get_all_embed.py``
    scripts with a single re-usable function that does not depend on any
    filesystem layout or hard-coded paths.
    """
    model.net.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    z1_chunks, z2_chunks = [], []
    for batch in tqdm(loader, desc=desc or "embed"):
        x1 = batch["omic_1"].to(device)
        x2 = batch["omic_2"].to(device)
        outputs = model.net(x1, x2)
        z1_chunks.append(outputs["x1"]["z_mu"].detach().cpu().numpy())
        z2_chunks.append(outputs["x2"]["z_mu"].detach().cpu().numpy())

    return {
        "z_mu_1": np.concatenate(z1_chunks, axis=0),
        "z_mu_2": np.concatenate(z2_chunks, axis=0),
    }


__all__ = ["get_paired_embedding"]
