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

        eps = 1e-7
        pos_scores = torch.nan_to_num(pos_scores, nan=0.5).clamp(eps, 1 - eps)
        neg_scores = torch.nan_to_num(neg_scores, nan=0.5).clamp(eps, 1 - eps)
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
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    return model, embeddings


def train_shared_gae(
    model: "OutcomeGAE",
    x_good: torch.Tensor, edge_good: torch.LongTensor,
    x_poor: torch.Tensor, edge_poor: torch.LongTensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 30,
    val_edge_ratio: float = 0.1,
    device: str = "cpu",
) -> Tuple["OutcomeGAE", np.ndarray, np.ndarray]:
    """Train ONE GAE jointly on both group-specific graphs.

    Returns shared model + (z_good, z_poor) on the SAME latent space.
    No KNN residual correction needed downstream — latent alignment is by construction.
    """
    model = model.to(device)
    x_good = x_good.to(device); edge_good = edge_good.to(device)
    x_poor = x_poor.to(device); edge_poor = edge_poor.to(device)

    def split_edges(edge_index):
        n_edges = edge_index.size(1) // 2
        perm = torch.randperm(n_edges)
        n_val = max(1, int(n_edges * val_edge_ratio))
        n_train = n_edges - n_val
        all_src = edge_index[0, :n_edges * 2:2]
        all_dst = edge_index[1, :n_edges * 2:2]
        tr_src = all_src[perm[:n_train]]; tr_dst = all_dst[perm[:n_train]]
        va_src = all_src[perm[n_train:]]; va_dst = all_dst[perm[n_train:]]
        tr_edge = to_undirected(torch.stack([tr_src, tr_dst], dim=0))
        va_edge = torch.stack([va_src, va_dst], dim=0)
        return tr_edge.to(device), va_edge.to(device)

    tr_good, va_good = split_edges(edge_good)
    tr_poor, va_poor = split_edges(edge_poor)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf"); best_state = None; wait = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        z_g = model.encode(x_good, tr_good)
        z_p = model.encode(x_poor, tr_poor)
        loss = model.recon_loss(z_g, tr_good) + model.recon_loss(z_p, tr_poor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            zg_v = model.encode(x_good, tr_good)
            zp_v = model.encode(x_poor, tr_poor)
            v_loss = (model.recon_loss(zg_v, va_good).item() +
                      model.recon_loss(zp_v, va_poor).item())
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Shared-encoder early stop at epoch %d (val_loss=%.4f)", epoch, best_val)
                break

    if best_state: model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        z_good = model.encode(x_good, edge_good).cpu().numpy()
        z_poor = model.encode(x_poor, edge_poor).cpu().numpy()
    z_good = np.nan_to_num(z_good, nan=0.0, posinf=0.0, neginf=0.0)
    z_poor = np.nan_to_num(z_poor, nan=0.0, posinf=0.0, neginf=0.0)
    return model, z_good, z_poor
