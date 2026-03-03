"""Two-layer GCN encoder for the graph autoencoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """Two-layer Graph Convolutional Network encoder.

    Architecture: X → GCN(F→32) → ReLU → Dropout → GCN(32→d) → Z

    The encoder uses the symmetrically normalized adjacency with self-loops
    (handled internally by PyG's GCNConv).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        out_channels: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode node features into latent embeddings.

        Parameters
        ----------
        x : (n_nodes, in_channels) node feature matrix
        edge_index : (2, n_edges) edge index

        Returns
        -------
        z : (n_nodes, out_channels) latent embeddings
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return z
