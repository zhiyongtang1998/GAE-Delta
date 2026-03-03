"""Multi-omics shift fusion: concatenate residual shifts across modalities."""
from __future__ import annotations

from typing import Dict

import numpy as np


def fuse_multiomics_shifts(
    modality_residuals: Dict[str, np.ndarray],
    modality_order: tuple[str, ...] = ("rna", "methylation", "cnv"),
) -> np.ndarray:
    """Concatenate residual shift embeddings from multiple modalities.

    s_g = [s_g^(RNA), s_g^(Meth), s_g^(CNV)] ∈ R^{3d}

    Parameters
    ----------
    modality_residuals : dict mapping modality name → (n_genes, d) residual shifts
    modality_order : order of concatenation

    Returns
    -------
    fused : (n_genes, 3*d) multi-omics shift representation
    """
    arrays = []
    for mod in modality_order:
        if mod not in modality_residuals:
            raise KeyError(f"Missing modality: {mod}. Available: {list(modality_residuals)}")
        arrays.append(modality_residuals[mod])

    fused = np.concatenate(arrays, axis=1).astype(np.float32)
    return fused
