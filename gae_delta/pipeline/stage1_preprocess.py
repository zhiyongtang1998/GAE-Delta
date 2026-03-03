"""Stage 1: Data loading and validation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Set, Tuple

import numpy as np

from gae_delta.data.tcga.loader import MultiOmicsDataset, load_hdf5_dataset
from gae_delta.data.network.reactome_fi import load_fi_network, build_gene_to_index, fi_edges_to_index_pairs

logger = logging.getLogger(__name__)


def run_stage1(
    hdf5_path: str,
    fi_network_path: str,
    modalities: tuple[str, ...] = ("rna", "methylation", "cnv"),
) -> Tuple[MultiOmicsDataset, np.ndarray]:
    """Load and validate multi-omics data and FI network.

    Returns
    -------
    dataset : MultiOmicsDataset
    fi_edges : (n_edges, 2) int64 array of gene index pairs
    """
    logger.info("Stage 1: Loading data...")

    # Load HDF5 dataset
    dataset = load_hdf5_dataset(hdf5_path, modalities)

    # Load FI network
    if dataset.fi_edge_list is not None:
        fi_edges = dataset.fi_edge_list
        logger.info("Using FI edges from HDF5 metadata: %d edges", len(fi_edges))
    else:
        fi_edge_set = load_fi_network(fi_network_path)
        gene_to_idx = build_gene_to_index(dataset.gene_universe)
        fi_edges = fi_edges_to_index_pairs(fi_edge_set, gene_to_idx)

    # Validation
    n_genes = dataset.n_genes
    n_patients = len(dataset.clinical.patient_ids)
    logger.info("Dataset: %d patients, %d genes, %d FI edges", n_patients, n_genes, len(fi_edges))

    for mod_name in modalities:
        mod = dataset.get_modality(mod_name)
        assert mod.n_genes == n_genes, (
            f"Gene count mismatch in {mod_name}: {mod.n_genes} vs {n_genes}"
        )

    logger.info("Stage 1 complete.")
    return dataset, fi_edges
