from __future__ import annotations

from ..common.base_model import RoutedMoENLIModel


class DeepSeekMoENLIModel(RoutedMoENLIModel):
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
        use_relation_contrastive_loss: bool = True,
        alignment_hidden_dim: int = 512,
    ) -> None:
        if num_shared_experts < 1:
            raise ValueError("DeepSeekMoENLIModel requires at least one shared expert.")

        super().__init__(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim,
            num_routed_experts=num_routed_experts,
            routed_top_k=routed_top_k,
            expert_ffn_dim=expert_ffn_dim,
            num_labels=num_labels,
            dropout=dropout,
            use_relation_contrastive_loss=use_relation_contrastive_loss,
            alignment_hidden_dim=alignment_hidden_dim,
            num_shared_experts=num_shared_experts,
            normalize_combined_output=True,
        )
