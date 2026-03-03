"""10-fold stratified cross-validation orchestrator for the full GAE-Δ pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gae_delta.data.tcga.loader import MultiOmicsDataset
from gae_delta.data.tcga.clinical import compute_outcome_labels, stratified_kfold_split
from gae_delta.data.tcga.rna import preprocess_rna
from gae_delta.data.tcga.methylation import preprocess_methylation
from gae_delta.data.tcga.cnv import preprocess_cnv
from gae_delta.data.network.reactome_fi import build_gene_to_index, fi_edges_to_index_pairs
from gae_delta.core.graph.builder import OutcomeGraphBuilder
from gae_delta.core.model.gae import OutcomeGAE, train_gae
from gae_delta.core.model.mlp import OutcomeClassifier, train_classifier
from gae_delta.core.shift.embedding import compute_embedding_shift
from gae_delta.core.shift.knn_residual import knn_residual_correction
from gae_delta.core.shift.fusion import fuse_multiomics_shifts
from gae_delta.core.selection.isolation_forest import select_shift_genes, compute_patient_embeddings
from gae_delta.evaluation.metrics import evaluate_predictions, ClassificationMetrics

logger = logging.getLogger(__name__)

PREPROCESS_FN = {
    "rna": preprocess_rna,
    "methylation": preprocess_methylation,
    "cnv": preprocess_cnv,
}


@dataclass
class CVResult:
    """Cross-validation results across all folds."""
    fold_metrics: List[ClassificationMetrics] = field(default_factory=list)

    @property
    def mean_auc(self) -> float:
        return float(np.mean([m.auc_roc for m in self.fold_metrics]))

    @property
    def std_auc(self) -> float:
        return float(np.std([m.auc_roc for m in self.fold_metrics]))

    def summary(self) -> str:
        aucs = [m.auc_roc for m in self.fold_metrics]
        f1s = [m.f1 for m in self.fold_metrics]
        return (
            f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | "
            f"F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
        )


def run_cross_validation(
    dataset: MultiOmicsDataset,
    fi_edges: np.ndarray,
    modalities: Tuple[str, ...] = ("rna", "methylation", "cnv"),
    n_folds: int = 10,
    n_top_genes: int = 100,
    pcc_threshold: float = 0.5,
    knn_k: int = 5,
    gae_cfg: Optional[dict] = None,
    mlp_cfg: Optional[dict] = None,
    device: str = "cpu",
    seed: int = 42,
) -> CVResult:
    """Run full GAE-Δ cross-validation pipeline.

    Implements Algorithm 1 from the paper.
    """
    gae_cfg = gae_cfg or {}
    mlp_cfg = mlp_cfg or {}

    # Use clinical patient IDs as reference
    clinical = dataset.clinical
    n_patients = len(clinical.patient_ids)

    # Initial outcome labels (will be recomputed per fold)
    initial_labels, _ = compute_outcome_labels(clinical.os_days, clinical.os_status)

    # Generate fold splits
    splits = stratified_kfold_split(n_patients, initial_labels, n_folds, seed)

    result = CVResult()
    embedding_dim = gae_cfg.get("out_channels", 16)

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        logger.info("=== Fold %d / %d ===", fold_idx + 1, n_folds)

        # Step 1: Compute outcome labels using training median OS
        train_mask = np.zeros(n_patients, dtype=bool)
        train_mask[train_idx] = True
        labels, threshold = compute_outcome_labels(
            clinical.os_days, clinical.os_status, train_mask
        )

        good_train_mask = train_mask & (labels == 0)
        poor_train_mask = train_mask & (labels == 1)

        # Step 2: Preprocess each modality
        preprocessed: Dict[str, np.ndarray] = {}
        for mod_name in modalities:
            mod_data = dataset.get_modality(mod_name)
            if mod_name == "rna":
                preprocessed[mod_name] = preprocess_rna(mod_data.data, train_mask)
            elif mod_name == "methylation":
                normed, gene_mask = preprocess_methylation(mod_data.data, train_mask)
                preprocessed[mod_name] = normed
            elif mod_name == "cnv":
                preprocessed[mod_name] = preprocess_cnv(mod_data.data, train_mask)

        # Step 3-5: For each modality, build graphs, train GAEs, compute shifts
        modality_residuals: Dict[str, np.ndarray] = {}

        for mod_name in modalities:
            omics_data = preprocessed[mod_name]
            n_genes = omics_data.shape[1]

            # Build outcome-specific graphs
            builder = OutcomeGraphBuilder(fi_edges, pcc_threshold)
            graph_good = builder.build(omics_data, good_train_mask, "good", mod_name)
            graph_poor = builder.build(omics_data, poor_train_mask, "poor", mod_name)

            # Train GAEs
            gae_good = OutcomeGAE(
                in_channels=4,
                hidden_channels=gae_cfg.get("hidden_channels", 32),
                out_channels=embedding_dim,
                dropout=gae_cfg.get("dropout", 0.3),
            )
            gae_poor = OutcomeGAE(
                in_channels=4,
                hidden_channels=gae_cfg.get("hidden_channels", 32),
                out_channels=embedding_dim,
                dropout=gae_cfg.get("dropout", 0.3),
            )

            _, z_good = train_gae(
                gae_good, graph_good.node_features, graph_good.edge_index,
                lr=gae_cfg.get("lr", 1e-3),
                weight_decay=gae_cfg.get("weight_decay", 1e-4),
                max_epochs=gae_cfg.get("max_epochs", 300),
                patience=gae_cfg.get("patience", 30),
                device=device,
            )
            _, z_poor = train_gae(
                gae_poor, graph_poor.node_features, graph_poor.edge_index,
                lr=gae_cfg.get("lr", 1e-3),
                weight_decay=gae_cfg.get("weight_decay", 1e-4),
                max_epochs=gae_cfg.get("max_epochs", 300),
                patience=gae_cfg.get("patience", 30),
                device=device,
            )

            # Compute shift and KNN residual
            raw_shift = compute_embedding_shift(z_good, z_poor, normalize=True)
            residual = knn_residual_correction(raw_shift, k=knn_k)
            modality_residuals[mod_name] = residual

        # Step 6: Fuse multi-omics shifts
        fused_shifts = fuse_multiomics_shifts(modality_residuals, modalities)

        # Step 7: Gene selection via Isolation Forest
        selected_indices, _ = select_shift_genes(
            fused_shifts, n_top=n_top_genes, random_state=seed
        )

        # Step 8: Compute patient embeddings
        train_emb = compute_patient_embeddings(
            fused_shifts, selected_indices,
            {mod: preprocessed[mod][train_idx] for mod in modalities},
            modalities, embedding_dim,
        )
        val_emb = compute_patient_embeddings(
            fused_shifts, selected_indices,
            {mod: preprocessed[mod][val_idx] for mod in modalities},
            modalities, embedding_dim,
        )
        test_emb = compute_patient_embeddings(
            fused_shifts, selected_indices,
            {mod: preprocessed[mod][test_idx] for mod in modalities},
            modalities, embedding_dim,
        )

        train_labels = labels[train_idx].astype(np.float32)
        val_labels = labels[val_idx].astype(np.float32)
        test_labels = labels[test_idx].astype(np.float32)

        # Step 9: Train MLP classifier
        mlp = OutcomeClassifier(
            input_dim=fused_shifts.shape[1],
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

        # Step 10: Evaluate on test set
        test_X = torch.from_numpy(test_emb).float().to(device)
        mlp.eval()
        with torch.no_grad():
            test_probs = mlp.predict_proba(test_X).cpu().numpy()

        fold_metrics = evaluate_predictions(test_labels, test_probs)
        result.fold_metrics.append(fold_metrics)
        logger.info(
            "Fold %d: AUC=%.3f, F1=%.3f", fold_idx + 1,
            fold_metrics.auc_roc, fold_metrics.f1,
        )

    logger.info("CV Result: %s", result.summary())
    return result
