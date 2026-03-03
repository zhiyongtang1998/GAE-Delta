# cython: boundscheck=False, wraparound=False, cdivision=True
"""FI-constrained adjacency matrix construction using Cython."""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

cnp.import_array()


def build_fi_constrained_adjacency(
    cnp.ndarray[cnp.float32_t, ndim=2] corr_matrix,
    cnp.ndarray[cnp.int64_t, ndim=2] fi_edges,
    float pcc_threshold=0.5,
):
    """Build adjacency from FI edges filtered by correlation threshold.

    Parameters
    ----------
    corr_matrix : (n_genes, n_genes) Pearson correlation matrix
    fi_edges : (n_fi_edges, 2) int64 array of gene index pairs from FI network
    pcc_threshold : minimum |PCC| to retain an edge

    Returns
    -------
    edge_index : (2, n_retained_edges) int64 array (PyG format, undirected)
    edge_weight : (n_retained_edges,) float32 array of |PCC| values
    """
    cdef Py_ssize_t n_edges = fi_edges.shape[0]
    cdef Py_ssize_t i
    cdef int g1, g2
    cdef float pcc_val

    # First pass: count retained edges
    cdef Py_ssize_t count = 0
    for i in range(n_edges):
        g1 = fi_edges[i, 0]
        g2 = fi_edges[i, 1]
        pcc_val = fabs(corr_matrix[g1, g2])
        if pcc_val > pcc_threshold:
            count += 1

    # Allocate output (undirected → 2 entries per edge)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] edge_index = np.empty((2, count * 2), dtype=np.int64)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] edge_weight = np.empty(count * 2, dtype=np.float32)

    # Second pass: fill
    cdef Py_ssize_t idx = 0
    for i in range(n_edges):
        g1 = fi_edges[i, 0]
        g2 = fi_edges[i, 1]
        pcc_val = fabs(corr_matrix[g1, g2])
        if pcc_val > pcc_threshold:
            # Forward edge
            edge_index[0, idx] = g1
            edge_index[1, idx] = g2
            edge_weight[idx] = pcc_val
            idx += 1
            # Backward edge (undirected)
            edge_index[0, idx] = g2
            edge_index[1, idx] = g1
            edge_weight[idx] = pcc_val
            idx += 1

    return edge_index, edge_weight
