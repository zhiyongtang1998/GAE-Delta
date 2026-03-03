"""Hydra-based pipeline runner for the full GAE-Δ framework."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def resolve_device(cfg_device: str) -> str:
    """Resolve device string."""
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the GAE-Δ cross-validation pipeline."""
    from gae_delta.utils.logging import setup_logging
    from gae_delta.utils.seed import set_global_seed
    from gae_delta.utils.io import save_results, ensure_dir
    from gae_delta.pipeline.stage1_preprocess import run_stage1
    from gae_delta.evaluation.cross_validation import run_cross_validation

    setup_logging("INFO")
    logger.info("GAE-Δ Pipeline")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed
    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    # Resolve device
    device = resolve_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Resolve paths
    hdf5_path = cfg.data.hdf5_path
    fi_network_path = cfg.data.fi_network_path
    output_dir = cfg.data.get("output_dir", "outputs")

    # Stage 1: Load data
    modalities = tuple(cfg.get("modalities", ["rna", "methylation", "cnv"]))
    dataset, fi_edges = run_stage1(hdf5_path, fi_network_path, modalities)

    # GAE config
    gae_cfg = {
        "in_channels": cfg.model.gae.encoder.get("in_channels", 4),
        "hidden_channels": cfg.model.gae.encoder.get("hidden_channels", 32),
        "out_channels": cfg.model.gae.encoder.get("out_channels", 16),
        "dropout": cfg.model.gae.encoder.get("dropout", 0.3),
        "lr": cfg.model.gae.training.get("lr", 1e-3),
        "weight_decay": cfg.model.gae.training.get("weight_decay", 1e-4),
        "max_epochs": cfg.model.gae.training.get("max_epochs", 300),
        "patience": cfg.model.gae.training.get("patience", 30),
    }

    # MLP config
    mlp_cfg = {
        "hidden_dim": cfg.model.mlp.architecture.get("hidden_dim", 64),
        "dropout": cfg.model.mlp.architecture.get("dropout", 0.3),
        "lr": cfg.model.mlp.training.get("lr", 1e-3),
        "weight_decay": cfg.model.mlp.training.get("weight_decay", 1e-4),
        "max_epochs": cfg.model.mlp.training.get("max_epochs", 200),
        "patience": cfg.model.mlp.training.get("patience", 20),
        "batch_size": cfg.model.mlp.training.get("batch_size", 32),
    }

    # Experiment config
    exp_cfg = cfg.get("experiment", {})
    n_folds = cfg.get("n_folds", 10)
    n_top_genes = cfg.get("n_top_genes", 100)
    pcc_threshold = exp_cfg.get("graph", {}).get("pcc_threshold", 0.5)
    knn_k = exp_cfg.get("shift", {}).get("knn_k", 15)

    # Run cross-validation
    cv_result = run_cross_validation(
        dataset=dataset,
        fi_edges=fi_edges,
        modalities=modalities,
        n_folds=n_folds,
        n_top_genes=n_top_genes,
        pcc_threshold=pcc_threshold,
        knn_k=knn_k,
        gae_cfg=gae_cfg,
        mlp_cfg=mlp_cfg,
        device=device,
        seed=seed,
    )

    # Save results
    ensure_dir(output_dir)
    results = {
        "mean_auc": cv_result.mean_auc,
        "std_auc": cv_result.std_auc,
        "fold_aucs": [m.auc_roc for m in cv_result.fold_metrics],
        "fold_f1s": [m.f1 for m in cv_result.fold_metrics],
        "summary": cv_result.summary(),
    }
    save_results(results, Path(output_dir) / "cv_results.json")
    logger.info("Results saved to %s", output_dir)
    logger.info("Final: %s", cv_result.summary())


if __name__ == "__main__":
    main()
