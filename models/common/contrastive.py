from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveProjectionHead(nn.Module):
    """Projection head shared by all models before contrastive loss."""

    def __init__(self, hidden_dim: int, contrastive_hidden_dim: int, contrastive_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, contrastive_hidden_dim),
            nn.GELU(),
            nn.Linear(contrastive_hidden_dim, contrastive_dim),
        )

    def forward(self, mode_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(mode_embeddings)


class MultiViewContrastiveLoss(nn.Module):
    """Multi-positive NT-Xent loss over four modality views of each sample."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = temperature

    def forward(self, mode_embeddings: torch.Tensor) -> torch.Tensor:
        if mode_embeddings.ndim != 3:
            raise ValueError(f"mode_embeddings must have shape [B, M, D], got {tuple(mode_embeddings.shape)}")

        batch_size, num_modes, _ = mode_embeddings.shape
        if batch_size < 2 or num_modes < 2:
            return mode_embeddings.sum() * 0.0

        flat = F.normalize(mode_embeddings.reshape(batch_size * num_modes, -1), p=2, dim=-1)
        logits = torch.matmul(flat, flat.transpose(0, 1)) / self.temperature

        num_views = batch_size * num_modes
        eye = torch.eye(num_views, dtype=torch.bool, device=mode_embeddings.device)
        sample_ids = torch.arange(batch_size, device=mode_embeddings.device).repeat_interleave(num_modes)
        positive_mask = sample_ids.unsqueeze(0).eq(sample_ids.unsqueeze(1)) & ~eye

        logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        positive_counts = positive_mask.sum(dim=1).clamp_min(1)
        positive_log_probs = log_probs.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_counts
        return -positive_log_probs.mean()
