from __future__ import annotations

import torch
from torch import nn

from ..common.contrastive import ContrastiveProjectionHead
from ..common.expert import FeedForwardExpert, InputProjection, NLIClassifier
from ..common.losses import compute_load_balancing_loss
from ..common.pooling import MeanModePooling
from ..common.router import TopKRouter
from ..common.routing_utils import combine_topk_expert_outputs, reshape_router_tensor, validate_feature_tensor


class DeepSeekMoENLIModel(nn.Module):
    """DeepSeek-style MoE with routed experts plus always-on shared experts."""

    def __init__(
        self,
        *,
        input_dim: int = 4096,
        num_modes: int = 4,
        hidden_dim: int = 1024,
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        routed_top_k: int = 3,
        expert_ffn_dim: int = 1024,
        num_labels: int = 3,
        dropout: float = 0.2,
        use_contrastive_loss: bool = True,
        contrastive_hidden_dim: int = 512,
        contrastive_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_shared_experts < 1:
            raise ValueError("DeepSeekMoENLIModel requires at least one shared expert.")

        self.input_dim = input_dim
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.routed_top_k = routed_top_k
        self.use_contrastive_loss = use_contrastive_loss

        self.input_projection = InputProjection(input_dim, hidden_dim, dropout)
        self.router = TopKRouter(hidden_dim, num_routed_experts, routed_top_k)
        self.routed_experts = nn.ModuleList(
            [FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_shared_experts)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mode_pooling = MeanModePooling()
        self.classifier = NLIClassifier(hidden_dim, num_labels, dropout)
        self.contrastive_head = (
            ContrastiveProjectionHead(hidden_dim, contrastive_hidden_dim, contrastive_dim)
            if use_contrastive_loss
            else None
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_modes = validate_feature_tensor(
            features,
            input_dim=self.input_dim,
            num_modes=self.num_modes,
        )

        projected = self.input_projection(features)
        flat = projected.reshape(batch_size * num_modes, self.hidden_dim)
        routing = self.router(flat)
        routed_flat = combine_topk_expert_outputs(
            flat,
            self.routed_experts,
            routing["topk_indices"],
            routing["topk_weights"],
        )
        shared_flat = torch.zeros_like(flat)
        for shared_expert in self.shared_experts:
            shared_flat = shared_flat + shared_expert(flat)

        moe_flat = self.output_norm(routed_flat + shared_flat)
        routed_output = routed_flat.reshape(batch_size, num_modes, self.hidden_dim)
        shared_output = shared_flat.reshape(batch_size, num_modes, self.hidden_dim)
        moe_output = moe_flat.reshape(batch_size, num_modes, self.hidden_dim)
        fused = self.mode_pooling(moe_output)
        logits = self.classifier(fused)

        load_balancing_loss = compute_load_balancing_loss(
            routing["router_probs"],
            routing["topk_indices"],
            self.num_routed_experts,
        )

        output = {
            "logits": logits,
            "fused": fused,
            "moe_output": moe_output,
            "router_logits": reshape_router_tensor(routing["router_logits"], batch_size, num_modes),
            "router_probs": reshape_router_tensor(routing["router_probs"], batch_size, num_modes),
            "topk_indices": reshape_router_tensor(routing["topk_indices"], batch_size, num_modes),
            "topk_weights": reshape_router_tensor(routing["topk_weights"], batch_size, num_modes),
            "load_balancing_loss": load_balancing_loss,
            "routed_output": routed_output,
            "shared_output": shared_output,
        }
        if self.contrastive_head is not None:
            output["contrastive_embeddings"] = self.contrastive_head(moe_output)
        return output
