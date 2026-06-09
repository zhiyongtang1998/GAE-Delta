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
    # impute NaN with column mean (training-data only, no leakage)
    train_data = x[train_mask]
    col_mean = np.nanmean(train_data, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(x)
    x[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    train_data = x[train_mask]
    mu = train_data.mean(axis=0, keepdims=True)
    sigma = train_data.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    x = (x - mu) / sigma
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
