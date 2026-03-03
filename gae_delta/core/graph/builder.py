"""Group-specific graph builder: orchestrates PCC computation, FI filtering, and feature construction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class OutcomeGraph:
    """A single group-specific gene interaction graph."""
    edge_index: torch.LongTensor   # (2, n_edges)
    edge_weight: torch.FloatTensor  # (n_edges,)
    node_features: torch.FloatTensor  # (n_genes, 4)
    n_nodes: int
    outcome_label: str  # "good" or "poor"
    modality: str


class OutcomeGraphBuilder:
    """Build FI-constrained group-specific gene graphs.

    For each (modality, phenotypic_group), the builder:
    1. Computes pairwise Pearson correlations among genes (Cython-accelerated)
    2. Filters edges by FI prior and |PCC| > threshold
    3. Constructs 4D node features
    """

    def __init__(
        self,
        fi_edges: np.ndarray,
        pcc_threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        fi_edges : (n_edges, 2) int64 array of gene index pairs in the FI network
        pcc_threshold : minimum |PCC| to retain an edge (default 0.5)
        """
        self.fi_edges = fi_edges
        self.pcc_threshold = pcc_threshold

    def build(
        self,
        omics_data: np.ndarray,
        patient_mask: np.ndarray,
        outcome_label: str,
        modality: str,
    ) -> OutcomeGraph:
        """Build one group-specific graph.

        Parameters
        ----------
        omics_data : (n_patients, n_genes) standardized omics matrix
        patient_mask : boolean mask selecting patients in this outcome group
        outcome_label : "good" or "poor"
        modality : "rna", "methylation", or "cnv"

        Returns
        -------
        OutcomeGraph
        """
        n_genes = omics_data.shape[1]
        subset_data = omics_data[patient_mask].astype(np.float32)

        # --- Compute PCC ---
        corr_matrix = self._compute_correlation(subset_data)

        # --- Build FI-constrained adjacency ---
        edge_index_np, edge_weight_np = self._build_adjacency(corr_matrix)

        # --- Compute node features ---
        from gae_delta.core.graph.features import compute_node_features
        node_features = compute_node_features(
            omics_data, patient_mask, edge_index_np, n_genes
        )

        n_edges = edge_index_np.shape[1] // 2  # undirected
        logger.info(
            "Built %s-%s graph: %d nodes, %d edges (undirected)",
            outcome_label, modality, n_genes, n_edges,
        )

        return OutcomeGraph(
            edge_index=torch.from_numpy(edge_index_np),
            edge_weight=torch.from_numpy(edge_weight_np),
            node_features=torch.from_numpy(node_features),
            n_nodes=n_genes,
            outcome_label=outcome_label,
            modality=modality,
        )

    def _compute_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute pairwise Pearson correlation using Cython or NumPy fallback."""
        try:
            from gae_delta.core.graph._correlation import pairwise_pearson
            return pairwise_pearson(data)
        except ImportError:
            logger.warning(
                "Cython _correlation not available, falling back to NumPy. "
                "Run 'make cython' to build optimized extensions."
            )
            return np.corrcoef(data, rowvar=False).astype(np.float32)

    def _build_adjacency(self, corr_matrix: np.ndarray):
        """Filter FI edges by PCC threshold using Cython or Python fallback."""
        try:
            from gae_delta.core.graph._adjacency import build_fi_constrained_adjacency
            return build_fi_constrained_adjacency(
                corr_matrix, self.fi_edges, self.pcc_threshold
            )
        except ImportError:
            logger.warning(
                "Cython _adjacency not available, falling back to Python. "
                "Run 'make cython' to build optimized extensions."
            )
            return self._build_adjacency_python(corr_matrix)

    def _build_adjacency_python(self, corr_matrix: np.ndarray):
        """Pure Python fallback for adjacency construction."""
        src_list, dst_list, weight_list = [], [], []
        for i in range(self.fi_edges.shape[0]):
            g1, g2 = int(self.fi_edges[i, 0]), int(self.fi_edges[i, 1])
            pcc = abs(float(corr_matrix[g1, g2]))
            if pcc > self.pcc_threshold:
                src_list.extend([g1, g2])
                dst_list.extend([g2, g1])
                weight_list.extend([pcc, pcc])
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_weight = np.array(weight_list, dtype=np.float32)
        if len(src_list) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_weight = np.empty(0, dtype=np.float32)
        return edge_index, edge_weight
