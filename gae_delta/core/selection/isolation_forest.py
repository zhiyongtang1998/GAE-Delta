"""Gene selection via Isolation Forest on multi-omics shift representations."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def select_shift_genes(
    fused_shifts: np.ndarray,
    n_top: int = 100,
    n_estimators: int = 500,
    max_samples: int = 256,
    contamination: str | float = "auto",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select genes with pronounced outcome-specific role shifts using Isolation Forest.

    Genes are ranked by anomaly score (more anomalous = more distinctive shift).

    Parameters
    ----------
    fused_shifts : (n_genes, 3*d) multi-omics shift representations
    n_top : number of top genes to select
    n_estimators, max_samples, contamination, random_state : IsolationForest params

    Returns
    -------
    selected_indices : (n_top,) indices of selected genes
    anomaly_scores : (n_genes,) anomaly scores for all genes
    """
    # Standardize across genes to equalize contributions from different modalities
    scaler = StandardScaler()
    shifts_scaled = scaler.fit_transform(fused_shifts)

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, shifts_scaled.shape[0]),
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_forest.fit(shifts_scaled)

    # Score: more negative = more anomalous
    # We negate so that higher = more anomalous (for ranking)
    anomaly_scores = -iso_forest.score_samples(shifts_scaled)

    # Select top-N
    n_top = min(n_top, len(anomaly_scores))
    selected_indices = np.argsort(anomaly_scores)[-n_top:][::-1]

    logger.info(
        "Selected %d / %d genes (score range: %.3f – %.3f)",
        n_top, len(anomaly_scores),
        anomaly_scores[selected_indices[-1]],
        anomaly_scores[selected_indices[0]],
    )
    return selected_indices.astype(np.int64), anomaly_scores.astype(np.float32)


def compute_patient_embeddings(
    fused_shifts: np.ndarray,
    selected_gene_indices: np.ndarray,
    omics_values: dict[str, np.ndarray],
    modality_order: tuple[str, ...] = ("rna", "methylation", "cnv"),
    embedding_dim: int = 16,
) -> np.ndarray:
    """Construct patient-level embeddings by weighting gene shifts by omics abundances.

    u_p = Σ_{g∈G} [v_{p,g}^(RNA)·s_g^(RNA) ⊕ v_{p,g}^(Meth)·s_g^(Meth) ⊕ v_{p,g}^(CNV)·s_g^(CNV)]

    Parameters
    ----------
    fused_shifts : (n_genes, 3*d) multi-omics shift representations
    selected_gene_indices : (n_top,) indices of selected genes
    omics_values : dict mapping modality → (n_patients, n_genes) standardized values
    modality_order : order matching the fused_shifts concatenation
    embedding_dim : d, dimension per modality segment

    Returns
    -------
    patient_embeddings : (n_patients, 3*d) patient-level embeddings
    """
    n_modalities = len(modality_order)
    total_dim = n_modalities * embedding_dim

    # Extract selected gene shifts (n_top, 3*d)
    s_selected = fused_shifts[selected_gene_indices]  # (n_top, 3*d)

    # Split into per-modality segments
    shift_segments = {}
    for m_idx, mod in enumerate(modality_order):
        start = m_idx * embedding_dim
        end = start + embedding_dim
        shift_segments[mod] = s_selected[:, start:end]  # (n_top, d)

    # Get number of patients from first modality
    first_mod = modality_order[0]
    n_patients = omics_values[first_mod].shape[0]
    patient_emb = np.zeros((n_patients, total_dim), dtype=np.float32)

    for m_idx, mod in enumerate(modality_order):
        start = m_idx * embedding_dim
        end = start + embedding_dim

        # v_{p,g}^(m): (n_patients, n_top)
        v = omics_values[mod][:, selected_gene_indices]

        # s_g^(m): (n_top, d)
        s = shift_segments[mod]

        # Weighted sum: (n_patients, n_top) @ (n_top, d) → (n_patients, d)
        # But we want element-wise: Σ_g v_{p,g} * s_g
        # This is equivalent to matrix multiplication: v @ s
        patient_emb[:, start:end] = v @ s

    return patient_emb
