"""Stage 3: Train GAE and compute embedding shifts."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from gae_delta.core.graph.builder import OutcomeGraph
from gae_delta.core.model.gae import OutcomeGAE, train_gae
from gae_delta.core.shift.embedding import compute_embedding_shift
from gae_delta.core.shift.knn_residual import knn_residual_correction

logger = logging.getLogger(__name__)


def run_stage3(
    graph_good: OutcomeGraph,
    graph_poor: OutcomeGraph,
    gae_cfg: dict,
    knn_k: int = 15,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train GAEs on group-specific graphs and compute residual shifts.

    Returns
    -------
    raw_shift : (n_genes, d) raw embedding shift
    residual : (n_genes, d) KNN-corrected residual shift
    z_good, z_poor : raw embeddings (for debugging/analysis)
    """
    logger.info("Stage 3: Training GAEs for %s...", graph_good.modality)

    # Train good-outcome GAE
    gae_good = OutcomeGAE(
        in_channels=gae_cfg.get("in_channels", 4),
        hidden_channels=gae_cfg.get("hidden_channels", 32),
        out_channels=gae_cfg.get("out_channels", 16),
        dropout=gae_cfg.get("dropout", 0.3),
    )
    _, z_good = train_gae(
        gae_good, graph_good.node_features, graph_good.edge_index,
        lr=gae_cfg.get("lr", 1e-3),
        weight_decay=gae_cfg.get("weight_decay", 1e-4),
        max_epochs=gae_cfg.get("max_epochs", 300),
        patience=gae_cfg.get("patience", 30),
        device=device,
    )

    # Train poor-outcome GAE
    gae_poor = OutcomeGAE(
        in_channels=gae_cfg.get("in_channels", 4),
        hidden_channels=gae_cfg.get("hidden_channels", 32),
        out_channels=gae_cfg.get("out_channels", 16),
        dropout=gae_cfg.get("dropout", 0.3),
    )
    _, z_poor = train_gae(
        gae_poor, graph_poor.node_features, graph_poor.edge_index,
        lr=gae_cfg.get("lr", 1e-3),
        weight_decay=gae_cfg.get("weight_decay", 1e-4),
        max_epochs=gae_cfg.get("max_epochs", 300),
        patience=gae_cfg.get("patience", 30),
        device=device,
    )

    # Compute shift
    raw_shift = compute_embedding_shift(z_good, z_poor, normalize=True)
    residual = knn_residual_correction(raw_shift, k=knn_k)

    logger.info("Stage 3 complete for %s.", graph_good.modality)
    return raw_shift, residual, z_good
