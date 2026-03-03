"""Fold-aware normalization transforms."""
from __future__ import annotations

import numpy as np


class FoldZScoreNormalizer:
    """Z-score normalizer that fits on training data and transforms all splits.

    Ensures no information leakage from validation/test sets.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "FoldZScoreNormalizer":
        """Fit mean and std from training data.

        Parameters
        ----------
        data : (n_train_samples, n_features)
        """
        self.mean_ = data.mean(axis=0, keepdims=True).astype(np.float32)
        self.std_ = data.std(axis=0, keepdims=True).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply learned z-score normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer must be fit before transform.")
        return ((data - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class FeatureStandardizer:
    """Standardize node features across genes within a single graph."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Standardize features (n_nodes, n_features) to zero mean, unit variance."""
        self.mean_ = features.mean(axis=0, keepdims=True)
        self.std_ = features.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-8] = 1.0
        return ((features - self.mean_) / self.std_).astype(np.float32)
