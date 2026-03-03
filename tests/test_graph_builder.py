"""Tests for graph construction."""
import numpy as np
import pytest

from gae_delta.core.graph.builder import OutcomeGraphBuilder
from gae_delta.core.graph.features import compute_node_features


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    n_patients, n_genes = 30, 50
    omics = rng.randn(n_patients, n_genes).astype(np.float32)
    fi_edges = np.array(
        [(i, j) for i in range(0, 40, 2) for j in range(1, 41, 2) if i != j][:100],
        dtype=np.int64,
    )
    mask = np.zeros(n_patients, dtype=bool)
    mask[:15] = True
    return omics, fi_edges, mask, n_genes


def test_graph_builder_produces_valid_output(sample_data):
    omics, fi_edges, mask, n_genes = sample_data
    builder = OutcomeGraphBuilder(fi_edges, pcc_threshold=0.3)
    graph = builder.build(omics, mask, "good", "rna")

    assert graph.n_nodes == n_genes
    assert graph.edge_index.shape[0] == 2
    assert graph.node_features.shape == (n_genes, 4)
    assert graph.outcome_label == "good"
    assert graph.modality == "rna"


def test_node_features_shape(sample_data):
    omics, fi_edges, mask, n_genes = sample_data
    edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    features = compute_node_features(omics, mask, edge_index, n_genes)

    assert features.shape == (n_genes, 4)
    # Features should be standardized (approx zero mean)
    assert abs(features.mean()) < 0.5


def test_high_threshold_produces_fewer_edges(sample_data):
    omics, fi_edges, mask, n_genes = sample_data
    builder_low = OutcomeGraphBuilder(fi_edges, pcc_threshold=0.1)
    builder_high = OutcomeGraphBuilder(fi_edges, pcc_threshold=0.8)

    graph_low = builder_low.build(omics, mask, "good", "rna")
    graph_high = builder_high.build(omics, mask, "good", "rna")

    assert graph_high.edge_index.shape[1] <= graph_low.edge_index.shape[1]
