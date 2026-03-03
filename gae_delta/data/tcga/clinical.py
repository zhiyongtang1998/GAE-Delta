"""Clinical data utilities: outcome stratification and patient splitting."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_outcome_labels(
    os_days: np.ndarray,
    os_status: np.ndarray,
    patient_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """Stratify patients into good (0) and poor (1) outcome groups.

    The threshold is the median OS of the provided patient subset.
    Patients with OS > median → good (0); OS <= median → poor (1).

    Parameters
    ----------
    os_days : overall survival in days
    os_status : vital status (1 = deceased, 0 = censored)
    patient_mask : boolean mask selecting a subset (e.g., training set)

    Returns
    -------
    labels : array of 0/1 labels for ALL patients
    threshold : the median OS used for splitting
    """
    if patient_mask is not None:
        subset_days = os_days[patient_mask]
    else:
        subset_days = os_days

    threshold = float(np.median(subset_days))
    labels = (os_days <= threshold).astype(np.int32)  # 1 = poor (short OS)

    n_good = int((labels == 0).sum())
    n_poor = int((labels == 1).sum())
    logger.info(
        "Outcome split at %.1f days: %d good, %d poor", threshold, n_good, n_poor
    )
    return labels, threshold


def stratified_kfold_split(
    n_samples: int,
    labels: np.ndarray,
    n_folds: int = 10,
    random_state: int = 42,
) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate stratified k-fold splits with an 80/20 train-val split within
    each fold's non-test portion.

    Returns
    -------
    List of (train_idx, val_idx, test_idx) for each fold.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(n_samples)
    splits = []

    for train_val_idx, test_idx in skf.split(indices, labels):
        train_val_labels = labels[train_val_idx]
        # further split train_val into 80% train, 20% val (stratified)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        for sub_train_idx, sub_val_idx in sss.split(train_val_idx, train_val_labels):
            train_idx = train_val_idx[sub_train_idx]
            val_idx = train_val_idx[sub_val_idx]

        splits.append((train_idx, val_idx, test_idx))
        logger.debug(
            "Fold: train=%d, val=%d, test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )

    return splits
