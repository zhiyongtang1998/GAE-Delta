"""DNA methylation preprocessing: variance filtering and z-score normalization."""
from __future__ import annotations

import numpy as np


def preprocess_methylation(
    beta_values: np.ndarray,
    train_mask: np.ndarray,
    variance_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess methylation beta values.

    Steps:
    1. Filter genes with variance < threshold (using training set)
    2. Z-score normalize using training statistics

    Parameters
    ----------
    beta_values : (n_patients, n_genes) gene-level mean beta values
    train_mask : boolean mask for training patients
    variance_threshold : minimum variance to retain a gene

    Returns
    -------
    normalized : (n_patients, n_retained_genes)
    gene_mask : boolean mask of retained genes (n_genes,)
    """
    train_data = beta_values[train_mask]

    # variance filtering on training data
    var = train_data.var(axis=0)
    gene_mask = var >= variance_threshold

    # apply filter
    filtered = beta_values[:, gene_mask].astype(np.float32)
    train_filtered = filtered[train_mask]

    # z-score
    mu = train_filtered.mean(axis=0, keepdims=True)
    sigma = train_filtered.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    normalized = (filtered - mu) / sigma
    return normalized, gene_mask
