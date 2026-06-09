"""GAE-Δ: Phenotype-specific gene role shifts in multi-omics data.

The public API is exposed lazily (PEP 562) so that ``import gae_delta`` stays
cheap and does not pull in torch / torch-geometric until a heavy entry point is
actually used.

Examples
--------
>>> import gae_delta
>>> gae_delta.__version__
>>> from gae_delta import run_cross_validation, compute_embedding_shift
"""
from gae_delta._version import __version__

# name -> "module:attr" it lives in. Resolved on first access by __getattr__.
_LAZY = {
    # Full cross-validation pipeline (Algorithm 1)
    "run_cross_validation": "gae_delta.evaluation.cross_validation:run_cross_validation",
    "CVResult": "gae_delta.evaluation.cross_validation:CVResult",
    # Core method building blocks
    "OutcomeGraphBuilder": "gae_delta.core.graph.builder:OutcomeGraphBuilder",
    "compute_embedding_shift": "gae_delta.core.shift.embedding:compute_embedding_shift",
    "knn_residual_correction": "gae_delta.core.shift.knn_residual:knn_residual_correction",
    "fuse_multiomics_shifts": "gae_delta.core.shift.fusion:fuse_multiomics_shifts",
    "select_shift_genes": "gae_delta.core.selection.isolation_forest:select_shift_genes",
    # Data
    "MultiOmicsDataset": "gae_delta.data.tcga.loader:MultiOmicsDataset",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    """Lazily import and cache a public attribute on first access."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = target.split(":")
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(__all__)
