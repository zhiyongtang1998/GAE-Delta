"""Stage 2: Group-specific graph construction."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from gae_delta.core.graph.builder import OutcomeGraph, OutcomeGraphBuilder

logger = logging.getLogger(__name__)


def run_stage2(
    omics_data: np.ndarray,
    good_mask: np.ndarray,
    poor_mask: np.ndarray,
    fi_edges: np.ndarray,
    modality: str,
    pcc_threshold: float = 0.5,
) -> Tuple[OutcomeGraph, OutcomeGraph]:
    """Build group-specific graphs for a single modality.

    Returns
    -------
    graph_good : OutcomeGraph for good-outcome cohort
    graph_poor : OutcomeGraph for poor-outcome cohort
    """
    logger.info("Stage 2: Building %s graphs...", modality)

    builder = OutcomeGraphBuilder(fi_edges, pcc_threshold)
    graph_good = builder.build(omics_data, good_mask, "good", modality)
    graph_poor = builder.build(omics_data, poor_mask, "poor", modality)

    logger.info("Stage 2 complete for %s.", modality)
    return graph_good, graph_poor
