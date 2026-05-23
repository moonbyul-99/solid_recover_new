"""Helpers to convert AnnData ``.X`` into ``torch.float32`` tensors."""

from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData
from scipy import sparse


def adata_to_tensor(adata: AnnData) -> torch.Tensor:
    """Convert ``adata.X`` to a ``torch.float32`` tensor of shape ``(n_obs, n_vars)``.

    Accepts the three concrete container types we ever hand off:

    - existing ``torch.Tensor``: returned as ``.float()``
    - ``scipy.sparse`` matrix: densified via a single ``.toarray()`` call
    - ``numpy.ndarray``: contiguous copy, then ``torch.from_numpy``

    v2 simplification
    -----------------
    The legacy ``Base_sr._adata_format`` densified sparse matrices in
    ``sparse_chunk_size``-row chunks under the (unverified) assumption that
    ``.toarray()`` on the whole thing would OOM. In practice for every
    dataset we ship the split has already been sliced to <=~100k rows and a
    single ``.toarray()`` has lower peak memory than ``chunks + concatenate``
    (which transiently holds both the chunk list and the final dense array).
    The chunking / ``tqdm`` scaffolding was dropped accordingly.
    """
    x = adata.X

    if isinstance(x, torch.Tensor):
        return x.float()

    if sparse.issparse(x):
        x = x.toarray()

    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
        return torch.from_numpy(x).float()

    raise TypeError(
        f"Unsupported type for adata.X: {type(x)}. "
        "Expected torch.Tensor, scipy.sparse matrix, or numpy.ndarray."
    )
