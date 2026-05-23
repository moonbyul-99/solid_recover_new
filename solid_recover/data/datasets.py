"""Torch Datasets used by Solid Recover."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset


def _check_tensor(data: torch.Tensor, name: str) -> None:
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(data)}")
    if data.dtype != torch.float32:
        raise TypeError(f"{name} dtype must be torch.float32, got {data.dtype}")
    if torch.isnan(data).any():
        raise ValueError(f"{name} contains NaN values")


class SingleDataset(Dataset):
    """Single-omic dataset yielding ``{'feature': tensor_row}``."""

    def __init__(self, data: torch.Tensor) -> None:
        _check_tensor(data, "SingleDataset data")
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return {"feature": self.data[idx, :]}

    def to_gpu(self, device: str = "cuda") -> "SingleDataset":
        """Move the backing tensor to ``device`` *in place*.

        Lets the caller skip per-batch host->device copies when the whole
        tensor fits in GPU memory (the common case for pretrain feature
        matrices). See :meth:`PairDataset.to_gpu` for the rationale.
        """
        self.data = self.data.to(device)
        return self


class PairDataset(Dataset):
    """Paired two-omic dataset yielding ``{'omic_1': ..., 'omic_2': ...}``."""

    def __init__(
        self,
        omic_1: torch.Tensor,
        omic_2: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> None:
        _check_tensor(omic_1, "omic_1")
        _check_tensor(omic_2, "omic_2")
        if omic_1.shape[0] != omic_2.shape[0]:
            raise ValueError("omic_1 and omic_2 must have the same number of samples")
        self.omic_1 = omic_1
        self.omic_2 = omic_2
        self.batch_indices = batch_indices

    def __len__(self) -> int:
        return self.omic_1.shape[0]

    def __getitem__(self, idx: int):
        item = {"omic_1": self.omic_1[idx, :], "omic_2": self.omic_2[idx, :]}
        if self.batch_indices is not None:
            item["batch_idx"] = self.batch_indices[idx]
        return item

    def to_gpu(self, device: str = "cuda") -> "PairDataset":
        """Move both modality tensors (and optional batch_indices) to *device* *in place*."""
        self.omic_1 = self.omic_1.to(device)
        self.omic_2 = self.omic_2.to(device)
        if self.batch_indices is not None:
            self.batch_indices = self.batch_indices.to(device)
        return self
