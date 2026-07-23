"""Pseudo-pair construction via nearest-neighbor matching in embedding space.

Prototype module for the pseudo-pair self-training strategy. Given a partially
trained paired model, encodes unmatched single-omics data and finds the closest
real paired cells in embedding space to construct synthetic cross-modality pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PseudoPairDataset(Dataset):
    """Dataset yielding ``{"omic_1": ..., "omic_2": ...}``, compatible with
    :meth:`solid_recover.models.pair.PairScratch._process_batch`.

    Parameters
    ----------
    omic_1, omic_2:
        Feature tensors of shape ``(N, D1)`` and ``(N, D2)``.
    real_mask:
        Optional boolean tensor of length N indicating which rows are real
        (True) vs. pseudo (False). Kept for future loss-weighting use.
    """

    def __init__(
        self,
        omic_1: torch.Tensor,
        omic_2: torch.Tensor,
        real_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.omic_1 = omic_1
        self.omic_2 = omic_2
        self.real_mask = real_mask

    def __len__(self) -> int:
        return self.omic_1.shape[0]

    def __getitem__(self, idx: int):
        return {"omic_1": self.omic_1[idx], "omic_2": self.omic_2[idx]}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_device(model) -> torch.device:
    """Return the device the model's parameters are on."""
    return next(model.net.parameters()).device


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_dataset(
    model,
    X: torch.Tensor,
    modality: int,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Encode *X* through the model's modality encoder, returning ``z_mu``.

    Encodings stay on the model's device.  The caller can ``.cpu()`` or
    ``.to(...)`` afterwards as needed.

    Parameters
    ----------
    model:
        A :class:`~solid_recover.models.pair.PairScratch` instance.
    X:
        Input tensor of shape ``(N, D)``.
    modality:
        1 or 2 — which SRVAE sub-module to use.
    batch_size:
        Encoding batch size.
    """
    model_device = _model_device(model)
    model.net.eval()
    encoder = model.net.model_1 if modality == 1 else model.net.model_2

    embeddings: list[torch.Tensor] = []
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        batch = X[start:end].to(model_device)
        _, z_mu, _, _ = encoder.get_embedding(batch)
        embeddings.append(z_mu)  # stays on model_device
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def compute_topk_hit(
    model,
    X1: torch.Tensor,
    X2: torch.Tensor,
    k_values: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 1024,
) -> dict[str, float]:
    """Cross-modality top-k hit rates on row-aligned paired *X1*, *X2*.

    For cell *i*, the correct partner is cell *i* of the other modality.
    Returns the fraction of cells whose partner ranks in the top-k most
    similar embeddings.
    """
    model_device = _model_device(model)

    Z1 = encode_dataset(model, X1, modality=1, batch_size=batch_size)
    Z2 = encode_dataset(model, X2, modality=2, batch_size=batch_size)
    Z1 = F.normalize(Z1, dim=-1)
    Z2 = F.normalize(Z2, dim=-1)

    N = Z1.shape[0]
    target = torch.arange(N, device=model_device).unsqueeze(1)  # [N, 1]
    max_k = max(k_values)

    # Compute similarity in chunks to avoid O(N^2) memory
    sim_chunk = 2048
    topk_chunks: list[torch.Tensor] = []
    for start in range(0, N, sim_chunk):
        end = min(start + sim_chunk, N)
        sim = Z1[start:end] @ Z2.T  # [chunk, N]
        _, idx = sim.topk(min(max_k, N), dim=1)  # [chunk, max_k]
        topk_chunks.append(idx)
    topk_all = torch.cat(topk_chunks, dim=0)  # [N, max_k]

    results: dict[str, float] = {}
    for k in k_values:
        hits = (topk_all[:, :k] == target).any(dim=1).float().mean().item()
        results[f"top{k}_hit"] = hits
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _index_aggregate(
    topk_idx: torch.Tensor,
    X_real: torch.Tensor,
    X_unmatched: torch.Tensor,
) -> torch.Tensor:
    """Aggregate pseudo-features by indexing directly into *X_real* and
    *X_unmatched*, avoiding the creation of a giant concatenated pool tensor.

    Parameters
    ----------
    topk_idx:
        Integer indices of shape ``(B, k)`` referencing rows in the
        conceptual pool ``[X_real; X_unmatched]``.  May be on any device;
        will be moved to *X_real*'s device automatically.
    X_real, X_unmatched:
        Real and unmatched feature tensors of shapes ``(N_real, D)`` and
        ``(M, D)``, on the same device.

    Returns
    -------
    Aggregated tensor of shape ``(B, D)``, computed as the **mean** over
    the ``k`` neighbours.
    """
    N_real = X_real.shape[0]
    device = X_real.device
    k = topk_idx.shape[1]

    idx = topk_idx.to(device)                     # [B, k]

    real_mask = idx < N_real                      # [B, k]
    unmatched_mask = ~real_mask

    # safe indices for gather (dummy 0 where mask is False)
    real_idx = idx.clone()
    real_idx[unmatched_mask] = 0
    unmatched_idx = (idx - N_real).clamp(min=0)

    contrib_real = X_real[real_idx] * real_mask.unsqueeze(-1).float()
    contrib_unmatched = X_unmatched[unmatched_idx] * unmatched_mask.unsqueeze(-1).float()

    return (contrib_real + contrib_unmatched).sum(dim=1) / k


# ---------------------------------------------------------------------------
# Core pseudo-pair builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_pseudo_pairs_mnn(
    model,
    X1_real: torch.Tensor,
    X2_real: torch.Tensor,
    X1_unmatched: torch.Tensor,
    X2_unmatched: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pseudo-pairs using **mutual nearest neighbors** only.

    An unmatched RNA cell *i* and ATAC cell *j* are paired iff:
    1. RNA cell *i*'s nearest ATAC neighbour (in Z2_pool) is ATAC cell *j*,
       **and** ATAC cell *j* is in the unmatched pool.
    2. ATAC cell *j*'s nearest RNA neighbour (in Z1_pool) is RNA cell *i*,
       **and** RNA cell *i* is in the unmatched pool.

    Only MNN-confirmed cells participate in pseudo-pairs.  Others are
    discarded for the current rematch cycle.

    Returns
    -------
    X1_all, X2_all:
        Combined tensors of shape ``(N_real + N_mnn, D1/D2)``.
    real_mask:
        Boolean tensor; ``True`` for real-pair rows.
    """
    N_real = X1_real.shape[0]
    M1 = X1_unmatched.shape[0]
    M2 = X2_unmatched.shape[0]
    storage_device = X1_real.device

    # ---- 1. Encode --------------------------------------------------------
    Z1_real = encode_dataset(model, X1_real, modality=1, batch_size=batch_size)
    Z2_real = encode_dataset(model, X2_real, modality=2, batch_size=batch_size)
    Z1_real = F.normalize(Z1_real, dim=-1)
    Z2_real = F.normalize(Z2_real, dim=-1)

    Z1_unmatched = encode_dataset(model, X1_unmatched, modality=1, batch_size=batch_size)
    Z2_unmatched = encode_dataset(model, X2_unmatched, modality=2, batch_size=batch_size)
    Z1_unmatched = F.normalize(Z1_unmatched, dim=-1)
    Z2_unmatched = F.normalize(Z2_unmatched, dim=-1)

    # ---- 2. Cross-modality nearest-neighbour search -----------------------
    Z1_pool = torch.cat([Z1_real, Z1_unmatched], dim=0)   # [N_real+M1, E]
    Z2_pool = torch.cat([Z2_real, Z2_unmatched], dim=0)   # [N_real+M2, E]

    # RNA unmatched -> ATAC pool
    nn_A = _sim_argmax_chunked(Z1_unmatched, Z2_pool, chunk=batch_size)   # [M1]
    # ATAC unmatched -> RNA pool
    nn_B = _sim_argmax_chunked(Z2_unmatched, Z1_pool, chunk=batch_size)   # [M2]

    # ---- 3. Mutual nearest-neighbour detection ----------------------------
    # nn_A[i] is the position in Z2_pool.
    # We only consider unmatched->unmatched MNN.
    mnn_i: list[int] = []
    mnn_j: list[int] = []

    for i in range(M1):
        j_pool = int(nn_A[i].item())
        if j_pool < N_real:
            continue                           # nearest is a real cell — skip
        j = j_pool - N_real                     # unmatched ATAC index
        if 0 <= j < M2:
            i_pool = int(nn_B[j].item())
            if i_pool >= N_real and (i_pool - N_real) == i:
                mnn_i.append(i)
                mnn_j.append(j)

    N_mnn = len(mnn_i)

    if N_mnn == 0:
        # Fallback: no MNN found, return real pairs only
        from solid_recover._logging import get_logger
        _log = get_logger(__name__)
        _log.warning("build_pseudo_pairs_mnn: zero MNN pairs — returning real pairs only")
        real_mask = torch.ones(N_real, dtype=torch.bool, device=storage_device)
        return X1_real, X2_real, real_mask

    # ---- 4. Build pseudo pairs from MNN -----------------------------------
    mnn_i_t = torch.tensor(mnn_i, device=storage_device, dtype=torch.long)
    mnn_j_t = torch.tensor(mnn_j, device=storage_device, dtype=torch.long)

    X1_pseudo = X1_unmatched[mnn_i_t].to(storage_device)   # [N_mnn, D1]
    X2_pseudo = X2_unmatched[mnn_j_t].to(storage_device)   # [N_mnn, D2]

    X1_all = torch.cat([X1_real, X1_pseudo], dim=0)
    X2_all = torch.cat([X2_real, X2_pseudo], dim=0)

    real_mask = torch.cat([
        torch.ones(N_real, dtype=torch.bool, device=storage_device),
        torch.zeros(N_mnn, dtype=torch.bool, device=storage_device),
    ])

    from solid_recover._logging import get_logger
    _log = get_logger(__name__)
    _log.info(
        "MNN pseudo-pairs: %d / %d RNA cells and %d / %d ATAC cells paired",
        N_mnn, M1, N_mnn, M2,
    )

    return X1_all, X2_all, real_mask


# ---------------------------------------------------------------------------
# Helper: chunked argmax (GPU-friendly)
# ---------------------------------------------------------------------------

def _sim_argmax_chunked(
    Q: torch.Tensor, K: torch.Tensor, chunk: int = 2048
) -> torch.Tensor:
    """Compute ``argmax(Q @ K.T, dim=1)`` in chunks, return [Q.shape[0]] CPU tensor."""
    argmax_chunks: list[torch.Tensor] = []
    for start in range(0, Q.shape[0], chunk):
        end = min(start + chunk, Q.shape[0])
        sim = Q[start:end] @ K.T
        argmax_chunks.append(sim.argmax(dim=1).cpu())
    return torch.cat(argmax_chunks, dim=0)


# ---------------------------------------------------------------------------
# Original k-NN pseudo-pair builder (kept for backward compatibility)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_pseudo_pairs(
    model,
    X1_real: torch.Tensor,
    X2_real: torch.Tensor,
    X1_unmatched: torch.Tensor,
    X2_unmatched: torch.Tensor,
    k_neighbors: int = 3,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pseudo-pairs via cross-modality nearest-neighbour matching and
    return the combined (real + pseudo) dataset tensors.

    The output tensors are placed on the **same device as X1_real**.

    Algorithm
    ---------
    1. Encode all real paired cells → Z1_real, Z2_real (L2-normalised).
    2. Encode all unmatched cells → Z1_unmatched, Z2_unmatched.
    3. For each unmatched RNA cell, find top-k NN in Z2_pool
       (Z2_real + Z2_unmatched), then aggregate the neighbours' X2
       features as pseudo_X2 (cross-modality retrieval).
    4. Symmetrically, for each unmatched ATAC cell, find top-k NN in
       Z1_pool (Z1_real + Z1_unmatched), aggregate X1 features as
       pseudo_X1.
    5. Concatenate::

         X1_all = [X1_real,      X1_unmatched,  pseudo_X1]
         X2_all = [X2_real,      pseudo_X2,     X2_unmatched]

    Parameters
    ----------
    model:
        :class:`~solid_recover.models.pair.PairScratch` instance.
    X1_real, X2_real:
        Real paired tensors, shapes ``(N_real, D1)`` and ``(N_real, D2)``.
    X1_unmatched, X2_unmatched:
        Unmatched single-omics tensors, shapes ``(M1, D1)`` and ``(M2, D2)``.
    k_neighbors:
        Number of nearest neighbours to aggregate.
    batch_size:
        Query batch size during NN search.

    Returns
    -------
    X1_all, X2_all:
        Combined tensors of shape ``(N_real + M1 + M2, D1/D2)``.
    real_mask:
        Boolean tensor of length ``N_real + M1 + M2``; ``True`` for real rows.
    """
    N_real = X1_real.shape[0]
    M1 = X1_unmatched.shape[0]
    M2 = X2_unmatched.shape[0]
    storage_device = X1_real.device  # where to place output tensors

    # ---- 1. Encode real cells (on model device) ---------------------------
    Z1_real = encode_dataset(model, X1_real, modality=1, batch_size=batch_size)
    Z2_real = encode_dataset(model, X2_real, modality=2, batch_size=batch_size)
    Z1_real = F.normalize(Z1_real, dim=-1)
    Z2_real = F.normalize(Z2_real, dim=-1)

    # ---- 2. Encode unmatched cells (on model device) ----------------------
    Z1_unmatched = encode_dataset(model, X1_unmatched, modality=1, batch_size=batch_size)
    Z1_unmatched = F.normalize(Z1_unmatched, dim=-1)
    Z2_unmatched = encode_dataset(model, X2_unmatched, modality=2, batch_size=batch_size)
    Z2_unmatched = F.normalize(Z2_unmatched, dim=-1)

    # ---- 3. Modality-1 unmatched → pseudo X2 (cross-modality) -------------
    # For each unmatched RNA cell, find the most similar ATAC embeddings
    # (Z2_pool) and aggregate their X2 features.
    Z2_pool = torch.cat([Z2_real, Z2_unmatched], dim=0)  # [N_real + M2, E]

    pseudo_X2_chunks: list[torch.Tensor] = []
    for start in range(0, M1, batch_size):
        end = min(start + batch_size, M1)
        z_batch = Z1_unmatched[start:end]              # [B, E] on model device
        sim = z_batch @ Z2_pool.T                       # [B, N_real + M2]
        k = min(k_neighbors, sim.shape[1])
        _, topk_idx = sim.topk(k, dim=1)
        pseudo = _index_aggregate(topk_idx, X2_real, X2_unmatched)  # [B, D2]
        pseudo_X2_chunks.append(pseudo)
    pseudo_X2 = torch.cat(pseudo_X2_chunks, dim=0)  # [M1, D2]

    # ---- 4. Modality-2 unmatched → pseudo X1 (cross-modality) -------------
    Z1_pool = torch.cat([Z1_real, Z1_unmatched], dim=0)  # [N_real + M1, E]

    pseudo_X1_chunks: list[torch.Tensor] = []
    for start in range(0, M2, batch_size):
        end = min(start + batch_size, M2)
        z_batch = Z2_unmatched[start:end]
        sim = z_batch @ Z1_pool.T                       # [B, N_real + M1]
        k = min(k_neighbors, sim.shape[1])
        _, topk_idx = sim.topk(k, dim=1)
        pseudo = _index_aggregate(topk_idx, X1_real, X1_unmatched)  # [B, D1]
        pseudo_X1_chunks.append(pseudo)
    pseudo_X1 = torch.cat(pseudo_X1_chunks, dim=0)  # [M2, D1]

    # ---- 5. Concatenate (on storage_device) -------------------------------
    X1_all = torch.cat(
        [X1_real, X1_unmatched.to(storage_device), pseudo_X1], dim=0
    )
    X2_all = torch.cat(
        [X2_real, pseudo_X2, X2_unmatched.to(storage_device)], dim=0
    )

    real_mask = torch.cat(
        [
            torch.ones(N_real, dtype=torch.bool, device=storage_device),
            torch.zeros(M1 + M2, dtype=torch.bool, device=storage_device),
        ]
    )

    return X1_all, X2_all, real_mask
