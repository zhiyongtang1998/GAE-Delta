"""Copy-number variation preprocessing: z-score normalization."""
from __future__ import annotations

import numpy as np


def preprocess_cnv(
    copy_ratios: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Z-score normalize CNV log2 copy ratios.

    Parameters
    ----------
    copy_ratios : (n_patients, n_genes) gene-level log2 copy ratios
    train_mask : boolean mask for training patients

    Returns
    -------
    Normalized CNV matrix (n_patients, n_genes).
    """
    x = copy_ratios.astype(np.float32)
    train_data = x[train_mask]

    mu = train_data.mean(axis=0, keepdims=True)
    sigma = train_data.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    x = (x - mu) / sigma
    return x
