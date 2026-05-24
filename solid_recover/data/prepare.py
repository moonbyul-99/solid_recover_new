"""Data preparation pipeline for paired multi-omics (MuData -> Datasets).

Refactored from ``src/load_eval_data.py``:

- Split into explicit public functions (no more ``eval_mode`` double-return hack)
- Dropped the ~100 lines of commented-out legacy code and the hard-coded
  ``__main__`` block that targeted a specific lab machine
- Renamed helper functions from ``_qc`` / ``_filter_data`` / ``split_data`` /
  ``data_prepare``
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import muon as mu
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData

from solid_recover._logging import get_logger
from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import PairDataset

_logger = get_logger(__name__)


def qc_and_normalize(adata: AnnData, min_cell_fraction: float = 0.01) -> AnnData:
    """Filter genes present in fewer than ``min_cell_fraction`` of cells, then log1p normalise."""
    _logger.info("QC start: %s", adata.shape)
    sc.pp.filter_genes(adata, min_cells=int(min_cell_fraction * adata.shape[0]))
    _logger.info("QC done:  %s", adata.shape)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def filter_by_train_vars(
    train_adata: AnnData, test_adata: AnnData, min_cell_fraction: float = 0.01
) -> Tuple[AnnData, AnnData]:
    """QC on train, then subset test to train's ``var.index`` and normalise test."""
    _logger.info("Train data QC / normalisation")
    train_adata = qc_and_normalize(train_adata, min_cell_fraction=min_cell_fraction)

    _logger.info("Filtering test data to train vars")
    test_adata = test_adata[:, train_adata.var.index]
    sc.pp.normalize_total(test_adata, target_sum=1e4)
    sc.pp.log1p(test_adata)

    _logger.info("Sizes after filtering -> train: %s, test: %s", train_adata.shape, test_adata.shape)
    return train_adata, test_adata


def split_and_save_mudata(
    data_path: str,
    train_split_path: str,
    test_split_path: str,
    save_dir: str,
    min_cell_fraction: float = 0.01,
) -> None:
    """Split a ``.h5mu`` by index arrays, per-modality QC, and persist to disk."""
    os.makedirs(save_dir, exist_ok=True)

    mdata = mu.read_h5mu(data_path)
    train_idx = np.load(train_split_path)
    test_idx = np.load(test_split_path)

    mdata_train = mdata[train_idx]
    mdata_test = mdata[test_idx]

    train_dic, test_dic = {}, {}
    for key in mdata.mod_names:
        train_adata = mdata_train[key]
        test_adata = mdata_test[key]
        train_adata, test_adata = filter_by_train_vars(
            train_adata, test_adata, min_cell_fraction=min_cell_fraction
        )
        train_dic[key] = train_adata
        test_dic[key] = test_adata

    mu.write_h5mu(os.path.join(save_dir, "train.h5mu"), mu.MuData(train_dic))
    mu.write_h5mu(os.path.join(save_dir, "test.h5mu"), mu.MuData(test_dic))
    _logger.info("Split-and-save finished under %s", save_dir)


def prepare_pair_data(
    train_data_path: str,
    test_data_path: str,
    key_1: str,
    key_2: str,
    to_gpu: bool = False,
) -> Tuple[PairDataset, PairDataset]:
    """Load train/test ``.h5mu`` files and return ``PairDataset`` tuple.

    Parameters
    ----------
    train_data_path : str
        Path to training ``.h5mu``.
    test_data_path : str
        Path to test ``.h5mu``.
    key_1 : str
        Modality key for omic-1 (e.g. ``"rna_count"``).
    key_2 : str
        Modality key for omic-2 (e.g. ``"atac_count"``).
    to_gpu : bool
        If True, move tensors to GPU.
    """
    train_data = mu.read_h5mu(train_data_path)
    test_data = mu.read_h5mu(test_data_path)


    train_dataset = PairDataset(
        adata_to_tensor(train_data[key_1]),
        adata_to_tensor(train_data[key_2]),
    )
    test_dataset = PairDataset(
        adata_to_tensor(test_data[key_1]),
        adata_to_tensor(test_data[key_2]),
    )

    if to_gpu:
        train_dataset.to_gpu()
        test_dataset.to_gpu()

    return train_dataset, test_dataset


def prepare_pair_data_from_single(
    data_path: str,
    key_1: str,
    key_2: str,
    test_size: float = 0.1,
    seed: int = 42,
    to_gpu: bool = False,
) -> Tuple[PairDataset, PairDataset]:
    """Load a single ``.h5mu``, random-split into train/test, and return ``PairDataset`` tuples.

    No QC/normalisation is performed — it is assumed the data has already been
    preprocessed.  Random split uses a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    mdata = mu.read_h5mu(data_path)
    n = mdata.n_obs

    # Random split
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_test = max(1, int(n * test_size))
    train_idx = idxs[n_test:]
    test_idx = idxs[:n_test]

    _logger.info("Auto-split: %d train / %d test (test_size=%.3f, seed=%d)",
                 len(train_idx), len(test_idx), test_size, seed)

    train_data_1 = mdata[key_1][train_idx, :]
    test_data_1 = mdata[key_1][test_idx, :]
    train_data_2 = mdata[key_2][train_idx, :]
    test_data_2 = mdata[key_2][test_idx, :]

    train_dataset = PairDataset(
        adata_to_tensor(train_data_1),
        adata_to_tensor(train_data_2),
    )
    test_dataset = PairDataset(
        adata_to_tensor(test_data_1),
        adata_to_tensor(test_data_2),
    )

    if to_gpu:
        train_dataset.to_gpu()
        test_dataset.to_gpu()

    return train_dataset, test_dataset


def prepare_pair_test_only(
    test_data_path: str,
    key_1: str,
    key_2: str,
    to_gpu: bool = False,
) -> PairDataset:
    """Evaluation-only loader (replaces the legacy ``eval_mode=True`` hack)."""
    test_data = mu.read_h5mu(test_data_path)
    test_dataset = PairDataset(
        adata_to_tensor(test_data[key_1]),
        adata_to_tensor(test_data[key_2]),
    )
    if to_gpu:
        test_dataset.to_gpu()
    return test_dataset


def split_and_save_from_args(args: Optional[list] = None) -> None:  # pragma: no cover
    """Convenience CLI helper mirroring the legacy ``__main__`` block.

    Intentionally kept thin: advanced users should call
    :func:`split_and_save_mudata` directly from a notebook or their own script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Split a .h5mu file into train/test.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--min_cell_fraction", type=float, default=0.01)
    ns = parser.parse_args(args)

    split_and_save_mudata(
        data_path=ns.data_path,
        train_split_path=ns.train_split,
        test_split_path=ns.test_split,
        save_dir=ns.save_dir,
        min_cell_fraction=ns.min_cell_fraction,
    )
