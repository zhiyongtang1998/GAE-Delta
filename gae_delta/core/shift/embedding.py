"""Embedding normalization and phenotype-specific shift computation."""
from __future__ import annotations

import numpy as np


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row (gene embedding) to unit length.

    z ← z / ||z||_2
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return (embeddings / norms).astype(np.float32)


def compute_embedding_shift(
    z_good: np.ndarray,
    z_poor: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the raw embedding shift between good and poor outcome graphs.

    Δz_g = z_g^(poor) - z_g^(good)

    Parameters
    ----------
    z_good : (n_genes, d) embeddings from good-outcome graph
    z_poor : (n_genes, d) embeddings from poor-outcome graph
    normalize : whether to L2-normalize embeddings before computing shift

    Returns
    -------
    raw_shift : (n_genes, d) raw shift vectors
    """
    if normalize:
        z_good = l2_normalize(z_good)
        z_poor = l2_normalize(z_poor)

    raw_shift = (z_poor - z_good).astype(np.float32)
    return raw_shift
