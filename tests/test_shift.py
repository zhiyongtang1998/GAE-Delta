"""Tests for shift computation."""
import numpy as np
import pytest

from gae_delta.core.shift.embedding import l2_normalize, compute_embedding_shift
from gae_delta.core.shift.knn_residual import knn_residual_correction
from gae_delta.core.shift.fusion import fuse_multiomics_shifts


def test_l2_normalize():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(x)

    # First row: norm should be 1
    assert abs(np.linalg.norm(normed[0]) - 1.0) < 1e-5
    # Zero row: should remain safe (not NaN)
    assert not np.any(np.isnan(normed))
    # Third row: already unit-ish
    assert abs(np.linalg.norm(normed[2]) - 1.0) < 1e-5


def test_compute_embedding_shift():
    rng = np.random.RandomState(42)
    z_good = rng.randn(50, 16).astype(np.float32)
    z_poor = rng.randn(50, 16).astype(np.float32)

    shift = compute_embedding_shift(z_good, z_poor, normalize=True)
    assert shift.shape == (50, 16)
    assert shift.dtype == np.float32


def test_knn_residual_correction():
    rng = np.random.RandomState(42)
    raw_shifts = rng.randn(50, 16).astype(np.float32)

    residuals = knn_residual_correction(raw_shifts, k=5)
    assert residuals.shape == (50, 16)
    # Residuals should be different from raw shifts (KNN smoothing removes global trend)
    assert not np.allclose(residuals, raw_shifts)


def test_fuse_multiomics_shifts():
    rng = np.random.RandomState(42)
    residuals = {
        "rna": rng.randn(50, 16).astype(np.float32),
        "methylation": rng.randn(50, 16).astype(np.float32),
        "cnv": rng.randn(50, 16).astype(np.float32),
    }

    fused = fuse_multiomics_shifts(residuals)
    assert fused.shape == (50, 48)

    # Check that concatenation order is correct
    np.testing.assert_array_equal(fused[:, :16], residuals["rna"])
    np.testing.assert_array_equal(fused[:, 16:32], residuals["methylation"])
    np.testing.assert_array_equal(fused[:, 32:48], residuals["cnv"])


def test_fuse_missing_modality_raises():
    residuals = {"rna": np.zeros((10, 16), dtype=np.float32)}
    with pytest.raises(KeyError):
        fuse_multiomics_shifts(residuals)
