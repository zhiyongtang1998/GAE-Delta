"""4D node feature computation for outcome-specific graphs."""
from __future__ import annotations

import numpy as np

from gae_delta.data.transforms.normalize import FeatureStandardizer


def compute_node_features(
    omics_data: np.ndarray,
    patient_mask: np.ndarray,
    edge_index: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Compute the 4-dimensional node feature vector for each gene.

    Features:
    1. Local Mean: mean of standardized omics across outcome-group patients
    2. Local Std: std of standardized omics across outcome-group patients
    3. Node Degree: number of edges in the FI-constrained graph
    4. Log-transformed Degree: log(1 + degree)

    All 4 features are standardized across nodes.

    Parameters
    ----------
    omics_data : (n_patients, n_genes) standardized omics values
    patient_mask : boolean mask selecting patients in this outcome group
    edge_index : (2, n_edges) edge index array
    n_genes : total number of genes in the universe

    Returns
    -------
    features : (n_genes, 4) standardized node features
    """
    subset = omics_data[patient_mask]

    # Feature 1: local mean
    local_mean = subset.mean(axis=0)  # (n_genes,)

    # Feature 2: local std
    local_std = subset.std(axis=0)    # (n_genes,)

    # Feature 3: node degree
    degree = np.zeros(n_genes, dtype=np.float32)
    if edge_index.shape[1] > 0:
        src = edge_index[0]
        np.add.at(degree, src, 1.0)

    # Feature 4: log(1 + degree)
    log_degree = np.log1p(degree)

    # Stack into (n_genes, 4) matrix
    features = np.column_stack([local_mean, local_std, degree, log_degree]).astype(np.float32)

    # Standardize across nodes
    standardizer = FeatureStandardizer()
    features = standardizer.fit_transform(features)

    return features
