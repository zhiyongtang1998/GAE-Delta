"""RNA-seq preprocessing: log2(FPKM+1) transformation and z-score normalization."""
from __future__ import annotations

import numpy as np


def preprocess_rna(
    expression: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Apply log2(x+1) and z-score normalization.

    Parameters
    ----------
    expression : (n_patients, n_genes) raw FPKM values or pre-log-transformed
    train_mask : boolean mask for training patients

    Returns
    -------
    Normalized expression matrix (n_patients, n_genes).
    """
    # log2(x + 1) — safe for already-transformed data
    x = np.log2(expression + 1.0).astype(np.float32)

    # z-score using training statistics
    train_data = x[train_mask]
    mu = train_data.mean(axis=0, keepdims=True)
    sigma = train_data.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0  # avoid division by zero

    x = (x - mu) / sigma
    return x
