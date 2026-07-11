from __future__ import annotations

import torch
from torch import nn


class TopKRouter(nn.Module):
    """Softmax router with normalized top-k expert weights."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return {
            "router_logits": router_logits,
            "router_probs": router_probs,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }
