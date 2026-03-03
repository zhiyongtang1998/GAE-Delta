"""Tests for GAE model."""
import numpy as np
import torch
import pytest

from gae_delta.core.model.encoder import GCNEncoder
from gae_delta.core.model.decoder import InnerProductDecoder
from gae_delta.core.model.gae import OutcomeGAE
from gae_delta.core.model.mlp import OutcomeClassifier


@pytest.fixture
def small_graph():
    n_nodes = 20
    x = torch.randn(n_nodes, 4)
    # Simple ring graph
    src = list(range(n_nodes))
    dst = [(i + 1) % n_nodes for i in range(n_nodes)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return x, edge_index, n_nodes


def test_encoder_output_shape(small_graph):
    x, edge_index, n_nodes = small_graph
    encoder = GCNEncoder(in_channels=4, hidden_channels=32, out_channels=16)
    z = encoder(x, edge_index)
    assert z.shape == (n_nodes, 16)


def test_decoder_output_shape(small_graph):
    x, edge_index, n_nodes = small_graph
    z = torch.randn(n_nodes, 16)
    decoder = InnerProductDecoder()
    probs = decoder(z, edge_index)
    assert probs.shape == (edge_index.shape[1],)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_gae_forward_and_loss(small_graph):
    x, edge_index, n_nodes = small_graph
    gae = OutcomeGAE(in_channels=4, hidden_channels=16, out_channels=8)
    z = gae.encode(x, edge_index)
    assert z.shape == (n_nodes, 8)

    loss = gae.recon_loss(z, edge_index)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_mlp_classifier():
    model = OutcomeClassifier(input_dim=48, hidden_dim=64, dropout=0.3)
    x = torch.randn(10, 48)
    logits = model(x)
    assert logits.shape == (10, 1)

    probs = model.predict_proba(x)
    assert probs.shape == (10,)
    assert (probs >= 0).all() and (probs <= 1).all()
