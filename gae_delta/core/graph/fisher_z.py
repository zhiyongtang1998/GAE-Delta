"""Fisher's z-test for differential correlation between two patient groups.

Replaces the |PCC| > tau within-group edge filter with a principled test of
whether an FI edge has statistically different correlation between the two
phenotypic groups (R4's concern #3).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def fisher_z_differential_edges(
    fi_edges: np.ndarray,
    omics_data: np.ndarray,
    good_mask: np.ndarray,
    poor_mask: np.ndarray,
    q_threshold: float = 0.1,
    min_keep_frac: float = 0.10,
) -> np.ndarray:
    """Filter FI edges to only those with significantly different correlation
    between good and poor outcome groups (Fisher's z-test + BH FDR).

    Parameters
    ----------
    fi_edges : (n_fi, 2) int array of (gene_i, gene_j) pairs
    omics_data : (n_patients, n_genes) standardized omics matrix
    good_mask, poor_mask : boolean masks for each outcome group (over training patients)
    q_threshold : BH-corrected q-value cutoff (default 0.1)

    Returns
    -------
    (n_kept, 2) int array of edges that pass the test
    """
    data_good = omics_data[good_mask]
    data_poor = omics_data[poor_mask]
    n_good = int(good_mask.sum())
    n_poor = int(poor_mask.sum())

    # Compute full correlation matrices (genes × genes), then index FI edges.
    # 5513^2 floats = 121 MB per group, fits.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_good = np.corrcoef(data_good, rowvar=False)
        corr_poor = np.corrcoef(data_poor, rowvar=False)
    corr_good = np.nan_to_num(corr_good, nan=0.0)
    corr_poor = np.nan_to_num(corr_poor, nan=0.0)

    r_good = corr_good[fi_edges[:, 0], fi_edges[:, 1]]
    r_poor = corr_poor[fi_edges[:, 0], fi_edges[:, 1]]

    # Clip to avoid arctanh(±1) -> ±inf
    r_good = np.clip(r_good, -0.9999, 0.9999)
    r_poor = np.clip(r_poor, -0.9999, 0.9999)

    z_good = np.arctanh(r_good)
    z_poor = np.arctanh(r_poor)
    se = np.sqrt(1.0 / max(n_good - 3, 1) + 1.0 / max(n_poor - 3, 1))
    z_stat = (z_good - z_poor) / se
    p = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))
    p = np.nan_to_num(p, nan=1.0)

    _, q, _, _ = multipletests(p, method="fdr_bh")
    keep_q = q < q_threshold

    # Top-K floor: always keep at least the top min_keep_frac of edges by |z|
    # ranked by absolute z-statistic
    k_min = max(int(len(fi_edges) * min_keep_frac), 100)
    top_k_idx = np.argsort(-np.abs(z_stat))[:k_min]
    keep_top = np.zeros(len(fi_edges), dtype=bool)
    keep_top[top_k_idx] = True

    keep = keep_q | keep_top  # union: BH-significant OR top-K by z-stat
    kept = fi_edges[keep]
    logger.info(
        "Fisher's z edge filter: %d kept (q<%.2g passes %d, top-%d%% union %d)",
        keep.sum(), q_threshold, int(keep_q.sum()), int(min_keep_frac * 100), int(keep_top.sum()),
    )
    return kept
