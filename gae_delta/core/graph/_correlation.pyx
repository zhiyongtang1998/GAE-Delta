# cython: boundscheck=False, wraparound=False, cdivision=True
"""Fast pairwise Pearson correlation computation using Cython typed memoryviews."""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


def pairwise_pearson(cnp.ndarray[cnp.float32_t, ndim=2] X):
    """Compute pairwise Pearson correlation matrix for columns of X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), float32
        Data matrix where each column is a gene.

    Returns
    -------
    corr : ndarray of shape (n_features, n_features), float32
        Pearson correlation matrix.
    """
    cdef Py_ssize_t n = X.shape[0]  # n_samples (patients)
    cdef Py_ssize_t p = X.shape[1]  # n_features (genes)
    cdef Py_ssize_t i, j, k

    # Centre columns (subtract mean)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] Xc = np.empty((n, p), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] col_mean = np.zeros(p, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] col_std = np.zeros(p, dtype=np.float32)

    # Compute means
    for j in range(p):
        for i in range(n):
            col_mean[j] += X[i, j]
        col_mean[j] /= n

    # Centre and compute std
    cdef float diff, s
    for j in range(p):
        s = 0.0
        for i in range(n):
            diff = X[i, j] - col_mean[j]
            Xc[i, j] = diff
            s += diff * diff
        s = sqrt(s)
        if s < 1e-10:
            s = 1.0
        col_std[j] = s

    # Normalize columns
    for j in range(p):
        for i in range(n):
            Xc[i, j] /= col_std[j]

    # Compute correlation via dot products
    cdef cnp.ndarray[cnp.float32_t, ndim=2] corr = np.zeros((p, p), dtype=np.float32)
    cdef float dot_val

    for i in range(p):
        corr[i, i] = 1.0
        for j in range(i + 1, p):
            dot_val = 0.0
            for k in range(n):
                dot_val += Xc[k, i] * Xc[k, j]
            dot_val /= n
            corr[i, j] = dot_val
            corr[j, i] = dot_val

    return corr
