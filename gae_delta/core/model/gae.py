"""Graph Autoencoder: encoder + decoder + training loop."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, to_undirected

from gae_delta.core.model.encoder import GCNEncoder
from gae_delta.core.model.decoder import InnerProductDecoder

logger = logging.getLogger(__name__)


class OutcomeGAE(nn.Module):
    """Graph Autoencoder for learning group-specific gene embeddings.

    Combines a GCN encoder with an inner-product decoder, trained to
    reconstruct edges via binary cross-entropy loss.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        out_channels: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.decoder = InnerProductDecoder()

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode nodes into latent space."""
        return self.encoder(x, edge_index)

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        """Decode edge probabilities."""
        return self.decoder(z, edge_index, sigmoid=True)

    def recon_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.LongTensor,
        neg_edge_index: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Binary cross-entropy reconstruction loss with 1:1 negative sampling.

        Parameters
        ----------
        z : (n_nodes, d) latent embeddings
        pos_edge_index : (2, n_pos_edges) positive (observed) edges
        neg_edge_index : (2, n_neg_edges) negative edges; sampled if None

        Returns
        -------
        loss : scalar tensor
        """
        n_nodes = z.size(0)
        pos_scores = self.decoder(z, pos_edge_index, sigmoid=True)

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=n_nodes,
                num_neg_samples=pos_edge_index.size(1),
            )
        neg_scores = self.decoder(z, neg_edge_index, sigmoid=True)

        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        return pos_loss + neg_loss


def train_gae(
    model: OutcomeGAE,
    x: torch.Tensor,
    edge_index: torch.LongTensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 30,
    val_edge_ratio: float = 0.1,
    device: str = "cpu",
) -> Tuple[OutcomeGAE, np.ndarray]:
    """Train GAE with early stopping on validation reconstruction loss.

    Parameters
    ----------
    model : OutcomeGAE instance
    x : (n_nodes, in_channels) node features
    edge_index : (2, n_edges) full edge index (undirected)
    lr, weight_decay : optimizer parameters
    max_epochs, patience : training schedule
    val_edge_ratio : fraction of edges held out for validation
    device : "cpu" or "cuda"

    Returns
    -------
    model : trained model
    embeddings : (n_nodes, out_channels) numpy array of final embeddings
    """
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    # Split edges into train/val
    n_edges = edge_index.size(1) // 2  # undirected count
    perm = torch.randperm(n_edges)
    n_val = max(1, int(n_edges * val_edge_ratio))
    n_train = n_edges - n_val

    # Extract unique undirected edges (take first half)
    # edge_index is [src; dst], with pairs at [i, i+1] for undirected
    all_src = edge_index[0, :n_edges * 2:2]
    all_dst = edge_index[1, :n_edges * 2:2]

    train_src = all_src[perm[:n_train]]
    train_dst = all_dst[perm[:n_train]]
    val_src = all_src[perm[n_train:]]
    val_dst = all_dst[perm[n_train:]]

    train_edge_index = to_undirected(torch.stack([train_src, train_dst], dim=0))
    val_edge_index = torch.stack([val_src, val_dst], dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_edge_index)
        loss = model.recon_loss(z, train_edge_index)
        loss.backward()
        optimizer.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            z_val = model.encode(x, train_edge_index)
            val_loss = model.recon_loss(z_val, val_edge_index).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "Early stopping at epoch %d (val_loss=%.4f)", epoch, best_val_loss
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Extract final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(x.to(device), edge_index.to(device)).cpu().numpy()

    return model, embeddings
