"""KNN residual correction for embedding shifts.

Uses C++ extension for fast KNN computation, with scikit-learn fallback.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def knn_residual_correction(
    raw_shifts: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Apply KNN residual correction to raw embedding shifts.

    For each gene g:
    1. Find K nearest neighbors of Δz_g in the shift space
    2. Compute KNN-predicted shift: mean of neighbor shifts
    3. Residual: ε_g = Δz_g - predicted(Δz_g)

    Parameters
    ----------
    raw_shifts : (n_genes, d) raw shift vectors
    k : number of nearest neighbors

    Returns
    -------
    residuals : (n_genes, d) residual shift vectors
    """
    shifts = raw_shifts.astype(np.float32)

    # Try C++ extension first
    try:
        from gae_delta.core.shift._knn_ext import knn_smooth_residuals
        residuals = knn_smooth_residuals(shifts, k)
        return residuals
    except ImportError:
        logger.warning(
            "C++ KNN extension not available, using scikit-learn fallback. "
            "Run 'make cpp' to build the optimized extension."
        )

    # Fallback: scikit-learn KNN
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(shifts)
    distances, indices = nn.kneighbors(shifts)

    # indices[:, 0] is the point itself; use indices[:, 1:] for neighbors
    neighbor_indices = indices[:, 1:]  # (n_genes, k)

    # Compute mean of neighbor shifts
    predicted = np.mean(shifts[neighbor_indices], axis=1)  # (n_genes, d)
    residuals = shifts - predicted

    return residuals.astype(np.float32)
