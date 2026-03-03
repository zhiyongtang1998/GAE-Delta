"""Stage 4: Gene selection and patient embedding construction."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from gae_delta.core.shift.fusion import fuse_multiomics_shifts
from gae_delta.core.selection.isolation_forest import select_shift_genes, compute_patient_embeddings

logger = logging.getLogger(__name__)


def run_stage4(
    modality_residuals: Dict[str, np.ndarray],
    omics_values: Dict[str, np.ndarray],
    modality_order: tuple[str, ...] = ("rna", "methylation", "cnv"),
    n_top_genes: int = 100,
    embedding_dim: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse shifts, select genes, and construct patient embeddings.

    Returns
    -------
    patient_embeddings : (n_patients, 3*d)
    selected_indices : (n_top_genes,)
    fused_shifts : (n_genes, 3*d)
    """
    logger.info("Stage 4: Gene selection and patient embedding...")

    # Fuse multi-omics shifts
    fused_shifts = fuse_multiomics_shifts(modality_residuals, modality_order)

    # Gene selection
    selected_indices, scores = select_shift_genes(
        fused_shifts, n_top=n_top_genes, random_state=seed
    )

    # Patient embeddings
    patient_embeddings = compute_patient_embeddings(
        fused_shifts, selected_indices, omics_values,
        modality_order, embedding_dim,
    )

    logger.info(
        "Stage 4 complete: %d genes selected, patient embedding dim=%d",
        len(selected_indices), patient_embeddings.shape[1],
    )
    return patient_embeddings, selected_indices, fused_shifts
