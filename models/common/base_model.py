from __future__ import annotations

import torch
from torch import nn

from ...pair_features import PAIR_FEATURE_COMPONENTS, build_pair_feature, split_pair_feature
from .contrastive import PairAlignmentProjection
from .expert import FeedForwardExpert, InputProjection, NLIClassifier
from .losses import compute_load_balancing_loss
from .pooling import MeanModePooling
from .router import TopKRouter
from .routing_utils import combine_topk_expert_outputs, reshape_router_tensor, validate_feature_tensor


class RoutedMoENLIModel(nn.Module):
    """Common routed-MoE implementation shared by all NLI model variants."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_modes: int,
        hidden_dim: int,
        num_routed_experts: int,
        routed_top_k: int,
        expert_ffn_dim: int,
        num_labels: int,
        dropout: float,
        use_relation_contrastive_loss: bool,
        alignment_hidden_dim: int,
        num_shared_experts: int = 0,
        normalize_combined_output: bool = False,
    ) -> None:
        super().__init__()
        if input_dim % PAIR_FEATURE_COMPONENTS != 0:
            raise ValueError(
                f"input_dim must be divisible by {PAIR_FEATURE_COMPONENTS}, got {input_dim}."
            )
        self.input_dim = input_dim
        self.embedding_dim = input_dim // PAIR_FEATURE_COMPONENTS
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.routed_top_k = routed_top_k
        self.use_relation_contrastive_loss = use_relation_contrastive_loss

        self.alignment_projection = (
            PairAlignmentProjection(self.embedding_dim, alignment_hidden_dim, dropout)
            if use_relation_contrastive_loss
            else None
        )

        self.input_projection = InputProjection(input_dim, hidden_dim, dropout)
        self.router = TopKRouter(hidden_dim, num_routed_experts, routed_top_k)
        self.routed_experts = nn.ModuleList(
            [FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_routed_experts)]
        )

        if num_shared_experts > 0:
            self.num_shared_experts = num_shared_experts
            self.shared_experts = nn.ModuleList(
                [FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_shared_experts)]
            )
            self.output_norm = nn.LayerNorm(hidden_dim) if normalize_combined_output else nn.Identity()
        elif normalize_combined_output:
            raise ValueError("normalize_combined_output requires at least one shared expert.")

        self.mode_pooling = MeanModePooling()
        self.classifier = NLIClassifier(hidden_dim, num_labels, dropout)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_modes = validate_feature_tensor(
            features,
            input_dim=self.input_dim,
            num_modes=self.num_modes,
        )

        aligned_premise: torch.Tensor | None = None
        aligned_hypothesis: torch.Tensor | None = None
        if self.alignment_projection is not None:
            premise_embeddings, hypothesis_embeddings = split_pair_feature(
                features,
                expected_embedding_dim=self.embedding_dim,
            )
            aligned_premise = self.alignment_projection(premise_embeddings)
            aligned_hypothesis = self.alignment_projection(hypothesis_embeddings)
            features = build_pair_feature(
                aligned_premise,
                aligned_hypothesis,
                expected_embedding_dim=self.embedding_dim,
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

        moe_flat = routed_flat
        shared_flat: torch.Tensor | None = None
        if hasattr(self, "shared_experts"):
            shared_flat = sum((expert(flat) for expert in self.shared_experts), torch.zeros_like(flat))
            moe_flat = self.output_norm(routed_flat + shared_flat)

        moe_output = moe_flat.reshape(batch_size, num_modes, self.hidden_dim)
        fused = self.mode_pooling(moe_output)
        output = {
            "logits": self.classifier(fused),
            "fused": fused,
            "moe_output": moe_output,
            "router_logits": reshape_router_tensor(routing["router_logits"], batch_size, num_modes),
            "router_probs": reshape_router_tensor(routing["router_probs"], batch_size, num_modes),
            "topk_indices": reshape_router_tensor(routing["topk_indices"], batch_size, num_modes),
            "topk_weights": reshape_router_tensor(routing["topk_weights"], batch_size, num_modes),
            "load_balancing_loss": compute_load_balancing_loss(
                routing["router_probs"],
                routing["topk_indices"],
                self.num_routed_experts,
            ),
        }

        if shared_flat is not None:
            output["routed_output"] = routed_flat.reshape(batch_size, num_modes, self.hidden_dim)
            output["shared_output"] = shared_flat.reshape(batch_size, num_modes, self.hidden_dim)
        if aligned_premise is not None and aligned_hypothesis is not None:
            output["aligned_premise_embeddings"] = aligned_premise
            output["aligned_hypothesis_embeddings"] = aligned_hypothesis
        return output
