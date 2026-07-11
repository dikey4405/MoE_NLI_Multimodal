from __future__ import annotations

from typing import Any

from torch import nn


MODEL_KWARGS = (
    "input_dim",
    "num_modes",
    "hidden_dim",
    "num_routed_experts",
    "num_shared_experts",
    "routed_top_k",
    "expert_ffn_dim",
    "num_labels",
    "dropout",
    "use_contrastive_loss",
    "contrastive_hidden_dim",
    "contrastive_dim",
)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate exactly one model architecture from config."""

    model_name = config["model_name"]
    kwargs = {key: config[key] for key in MODEL_KWARGS if key in config}
    if float(config.get("contrastive_loss_coef", 0.0)) == 0.0:
        kwargs["use_contrastive_loss"] = False

    if model_name == "conventional_moe":
        from ..models.conventional_moe.model import ConventionalMoENLIModel

        return ConventionalMoENLIModel(**kwargs)
    if model_name == "fine_grained_moe":
        from ..models.fine_grained_moe.model import FineGrainedMoENLIModel

        return FineGrainedMoENLIModel(**kwargs)
    if model_name == "deepseek_moe":
        from ..models.deepseek_moe.model import DeepSeekMoENLIModel

        return DeepSeekMoENLIModel(**kwargs)

    raise ValueError(f"Unsupported model_name: {model_name}")
