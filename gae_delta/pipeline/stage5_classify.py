"""Stage 5: MLP training and evaluation."""
from __future__ import annotations

import logging

import numpy as np
import torch

from gae_delta.core.model.mlp import OutcomeClassifier, train_classifier
from gae_delta.evaluation.metrics import evaluate_predictions, ClassificationMetrics

logger = logging.getLogger(__name__)


def run_stage5(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    mlp_cfg: dict,
    device: str = "cpu",
) -> ClassificationMetrics:
    """Train MLP and evaluate on test set.

    Returns
    -------
    ClassificationMetrics on the test set
    """
    logger.info("Stage 5: Training MLP classifier...")

    input_dim = train_emb.shape[1]
    mlp = OutcomeClassifier(
        input_dim=input_dim,
        hidden_dim=mlp_cfg.get("hidden_dim", 64),
        dropout=mlp_cfg.get("dropout", 0.3),
    )

    mlp = train_classifier(
        mlp, train_emb, train_labels, val_emb, val_labels,
        lr=mlp_cfg.get("lr", 1e-3),
        weight_decay=mlp_cfg.get("weight_decay", 1e-4),
        max_epochs=mlp_cfg.get("max_epochs", 200),
        patience=mlp_cfg.get("patience", 20),
        batch_size=mlp_cfg.get("batch_size", 32),
        device=device,
    )

    # Predict on test set
    test_X = torch.from_numpy(test_emb).float().to(device)
    test_probs = mlp.predict_proba(test_X).cpu().numpy()

    metrics = evaluate_predictions(test_labels, test_probs)
    logger.info(
        "Stage 5 complete: AUC=%.3f, F1=%.3f, Precision=%.3f, Recall=%.3f",
        metrics.auc_roc, metrics.f1, metrics.precision, metrics.recall,
    )
    return metrics
