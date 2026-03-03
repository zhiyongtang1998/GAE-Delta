"""Reactome Functional Interaction network loader and utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_fi_network(
    filepath: str | Path,
) -> Set[Tuple[str, str]]:
    """Load FI network edge list from a tab-separated file.

    Expected format (tab-separated, no header or with header):
        GeneA\\tGeneB
    or
        src\\tdest

    Returns
    -------
    Set of undirected gene pairs (alphabetically ordered tuples).
    """
    filepath = Path(filepath)
    edges: Set[Tuple[str, str]] = set()

    with open(filepath, "r") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src, dest = parts[0].strip(), parts[1].strip()
            # skip header-like lines
            if line_no == 0 and (src.lower() in ("src", "gene1", "genea", "source")):
                continue
            # store as sorted tuple for undirected edges
            edge = tuple(sorted([src, dest]))
            edges.add(edge)

    logger.info("Loaded FI network: %d undirected edges", len(edges))
    return edges


def build_gene_to_index(gene_universe: np.ndarray) -> Dict[str, int]:
    """Create a mapping from gene symbol to index in the gene universe."""
    return {
        (g if isinstance(g, str) else g.decode()): i
        for i, g in enumerate(gene_universe)
    }


def fi_edges_to_index_pairs(
    fi_edges: Set[Tuple[str, str]],
    gene_to_idx: Dict[str, int],
) -> np.ndarray:
    """Convert FI edges to index pairs based on gene universe.

    Only includes edges where both genes are in the universe.

    Returns
    -------
    edge_array : (n_valid_edges, 2) int64 array of gene index pairs
    """
    pairs = []
    for g1, g2 in fi_edges:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            pairs.append((gene_to_idx[g1], gene_to_idx[g2]))
    edge_array = np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
    logger.info(
        "FI edges mapped to gene universe: %d / %d retained",
        len(edge_array), len(fi_edges),
    )
    return edge_array
