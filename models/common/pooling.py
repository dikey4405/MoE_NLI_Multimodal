from __future__ import annotations

import torch
from torch import nn


class MeanModePooling(nn.Module):
    """Shared mode pooling: mean over four mode representations."""

    def forward(self, mode_embeddings: torch.Tensor) -> torch.Tensor:
        if mode_embeddings.ndim != 3:
            raise ValueError(f"mode_embeddings must have shape [B, M, H], got {tuple(mode_embeddings.shape)}")
        return mode_embeddings.mean(dim=1)
