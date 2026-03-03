"""Inner product decoder for edge reconstruction."""
from __future__ import annotations

import torch
import torch.nn as nn


class InnerProductDecoder(nn.Module):
    """Decode edges via inner product: Â_ij = σ(z_i^T z_j)."""

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.LongTensor,
        sigmoid: bool = True,
    ) -> torch.Tensor:
        """Compute edge reconstruction probabilities.

        Parameters
        ----------
        z : (n_nodes, d) latent embeddings
        edge_index : (2, n_edges) edges to score
        sigmoid : whether to apply sigmoid activation

        Returns
        -------
        prob : (n_edges,) reconstruction probabilities
        """
        src, dst = edge_index[0], edge_index[1]
        value = (z[src] * z[dst]).sum(dim=1)
        if sigmoid:
            value = torch.sigmoid(value)
        return value

    def forward_all(self, z: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        """Reconstruct full adjacency matrix (for evaluation)."""
        adj = torch.matmul(z, z.t())
        if sigmoid:
            adj = torch.sigmoid(adj)
        return adj
