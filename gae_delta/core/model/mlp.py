"""MLP classifier for patient outcome prediction."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class OutcomeClassifier(nn.Module):
    """Two-layer MLP for binary outcome classification.

    Architecture: input_dim → hidden_dim (ReLU, Dropout) → 1 (Sigmoid)
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch_size, input_dim)

        Returns
        -------
        logits : (batch_size, 1) pre-sigmoid outputs
        """
        h = self.fc1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h)
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of poor outcome."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


def train_classifier(
    model: OutcomeClassifier,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
) -> OutcomeClassifier:
    """Train MLP classifier with early stopping.

    Parameters
    ----------
    model : OutcomeClassifier
    train_X, train_y : training data and labels
    val_X, val_y : validation data and labels
    lr, weight_decay, max_epochs, patience, batch_size : training params
    device : "cpu" or "cuda"

    Returns
    -------
    trained model
    """
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(train_X).float(),
        torch.from_numpy(train_y).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_X_t = torch.from_numpy(val_X).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t).squeeze(-1)
            val_loss = criterion(val_logits, val_y_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("MLP early stop at epoch %d (val_loss=%.4f)", epoch, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)
