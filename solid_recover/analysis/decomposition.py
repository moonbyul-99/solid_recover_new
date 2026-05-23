"""Latent space decomposition via regularized linear regression.

This module provides utilities to decompose high-dimensional feature spaces
(RNA genes, ATAC peaks) into learned latent embeddings using ElasticNet
regression, enabling interpretable analysis of which latent dimensions
contribute to each feature.

The core idea: X ≈ Z W^T, where
- X: (n_cells, n_features) original feature matrix
- Z: (n_cells, n_latent_dims) learned embedding
- W: (n_features, n_latent_dims) weight matrix to be estimated

By fitting each feature independently against Z with sparsity constraints
(L1 regularization), we obtain an interpretable weight matrix that reveals
which latent programs drive each gene/peak's expression.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from tqdm import tqdm


def decompose_latent_to_features(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: list,
    alpha: float = 0.1,
    l1_ratio: float = 0.9,
    fit_intercept: bool = False,
    tol: float = 1e-6,
    progress: bool = True,
) -> pd.DataFrame:
    """Decompose feature matrix into latent space via ElasticNet.

    For each feature (gene/peak), fit an ElasticNet model against the latent
    embeddings to obtain interpretable weights. This reveals which latent
    dimensions contribute most to each feature's variation.

    Parameters
    ----------
    X : np.ndarray
        Original feature matrix of shape (n_cells, n_features).
        Can be dense or sparse (will be converted to dense if needed).
    Z : np.ndarray
        Latent embedding matrix of shape (n_cells, n_latent_dims).
        Typically obtained from a trained model's encoder (z_mu).
    feature_names : list
        List of feature names (gene symbols or peak coordinates),
        length must equal n_features.
    alpha : float, default=0.1
        Regularization strength. Higher values increase sparsity.
    l1_ratio : float, default=0.9
        ElasticNet mixing parameter (1.0 = pure Lasso, 0.0 = pure Ridge).
    fit_intercept : bool, default=False
        Whether to fit an intercept term. Usually False for centered data.
    tol : float, default=1e-6
        Tolerance for optimization convergence.
    progress : bool, default=True
        Whether to show a progress bar during fitting.

    Returns
    -------
    pd.DataFrame
        Weight matrix of shape (n_features, n_latent_dims).
        - Index: feature names
        - Columns: ['latent_1', 'latent_2', ..., 'latent_n']
        - Values: regression coefficients (importance of each latent dimension)

    Examples
    --------
    >>> from solid_recover.analysis import decompose_latent_to_features
    >>> # X_rna: (1000 cells, 15000 genes)
    >>> # Z_embed: (1000 cells, 64 latent dims)
    >>> W_rna = decompose_latent_to_features(
    ...     X_rna, Z_embed,
    ...     feature_names=gene_names,
    ...     alpha=0.1, l1_ratio=0.9
    ... )
    >>> # Get weights for latent dimension 63
    >>> latent_63_weights = W_rna['latent_63'].sort_values(ascending=False)
    """
    n_cells, n_features = X.shape
    n_latent = Z.shape[1]

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X.shape[1] ({n_features})"
        )

    if n_cells != Z.shape[0]:
        raise ValueError(
            f"X and Z must have same number of cells: "
            f"X has {n_cells}, Z has {Z.shape[0]}"
        )

    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Ensure float64 for numerical stability
    X = X.astype(np.float64)
    Z = Z.astype(np.float64)

    # Fit ElasticNet for each feature
    weight_matrix = np.zeros((n_features, n_latent), dtype=np.float64)

    iterator = range(n_features)
    if progress:
        iterator = tqdm(iterator, desc="Fitting ElasticNet", leave=False)

    for i in iterator:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            tol=tol,
            max_iter=10000,
        )
        model.fit(Z, X[:, i])
        weight_matrix[i, :] = model.coef_

    # Create labeled DataFrame
    column_names = [f"latent_{i+1}" for i in range(n_latent)]
    W = pd.DataFrame(weight_matrix, index=feature_names, columns=column_names)

    return W


def compute_reconstruction_error(
    W: pd.DataFrame,
    Z: np.ndarray,
    X: np.ndarray,
) -> float:
    """Compute mean squared error between original and reconstructed features.

    Calculates ||X - Z W^T||^2 / (n_cells * n_features)

    Parameters
    ----------
    W : pd.DataFrame
        Weight matrix from decompose_latent_to_features().
    Z : np.ndarray
        Latent embedding matrix (n_cells, n_latent_dims).
    X : np.ndarray
        Original feature matrix (n_cells, n_features).

    Returns
    -------
    float
        Mean squared reconstruction error.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()

    W_np = W.values
    Z = Z.astype(np.float64)
    X = X.astype(np.float64)

    X_pred = Z @ W_np.T
    mse = np.mean((X_pred - X) ** 2)
    return float(mse)


def compare_with_mlp_error(
    W: pd.DataFrame,
    Z: np.ndarray,
    X: np.ndarray,
    X_pred_mlp: np.ndarray,
) -> Dict[str, float]:
    """Compare linear decomposition error with MLP-based reconstruction error.

    This helps assess whether the linear approximation captures most of the
    variance explained by the full nonlinear model.

    Parameters
    ----------
    W : pd.DataFrame
        Weight matrix from decompose_latent_to_features().
    Z : np.ndarray
        Latent embedding matrix (n_cells, n_latent_dims).
    X : np.ndarray
        Original feature matrix (n_cells, n_features).
    X_pred_mlp : np.ndarray
        Reconstructed features from the MLP decoder
        (e.g., from model's layers['rna_pred'] or layers['atac_pred']).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'linear_error': MSE from linear decomposition
        - 'mlp_error': MSE from MLP reconstruction
        - 'ratio': linear_error / mlp_error (closer to 1.0 is better)
    """
    linear_err = compute_reconstruction_error(W, Z, X)

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = X.astype(np.float64)
    X_pred_mlp = X_pred_mlp.astype(np.float64)
    mlp_err = float(np.mean((X_pred_mlp - X) ** 2))

    return {
        "linear_error": linear_err,
        "mlp_error": mlp_err,
        "ratio": linear_err / mlp_err if mlp_err > 0 else float("inf"),
    }


def get_top_features_by_latent(
    W: pd.DataFrame,
    latent_idx: int,
    n_top: int = 100,
    ascending: bool = False,
) -> pd.Series:
    """Get top features ranked by weight in a specific latent dimension.

    Parameters
    ----------
    W : pd.DataFrame
        Weight matrix from decompose_latent_to_features().
    latent_idx : int
        Latent dimension index (1-based, e.g., 63 for 'latent_63').
    n_top : int, default=100
        Number of top features to return.
    ascending : bool, default=False
        If True, return features with lowest (most negative) weights.
        If False, return features with highest (most positive) weights.

    Returns
    -------
    pd.Series
        Top features sorted by weight, index=feature names, values=weights.

    Examples
    --------
    >>> top_genes = get_top_features_by_latent(W_rna, latent_idx=63, n_top=1000)
    >>> top_gene_names = top_genes.index.tolist()
    """
    column_name = f"latent_{latent_idx}"
    if column_name not in W.columns:
        raise ValueError(
            f"Latent dimension {latent_idx} not found. "
            f"Available columns: {W.columns.tolist()}"
        )

    weights = W[column_name].sort_values(ascending=ascending)
    return weights.head(n_top)


__all__ = [
    "decompose_latent_to_features",
    "compute_reconstruction_error",
    "compare_with_mlp_error",
    "get_top_features_by_latent",
]
