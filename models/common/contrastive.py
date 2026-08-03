from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PairAlignmentProjection(nn.Module):
    """Shared trainable adapter for premise and hypothesis SONAR embeddings."""

    def __init__(self, embedding_dim: int, alignment_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, alignment_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(alignment_hidden_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.output_norm(embeddings + self.net(embeddings))


class NLIRelationContrastiveLoss(nn.Module):
    """Pull entailment pairs together and separate contradiction pairs by a margin."""

    def __init__(
        self,
        *,
        margin: float = 0.5,
        entailment_label_id: int = 0,
        contradiction_label_id: int = 2,
    ) -> None:
        super().__init__()
        if margin <= 0 or margin > 2:
            raise ValueError(f"margin must be in (0, 2], got {margin}.")
        if entailment_label_id == contradiction_label_id:
            raise ValueError("Entailment and contradiction label ids must be different.")
        self.margin = margin
        self.entailment_label_id = entailment_label_id
        self.contradiction_label_id = contradiction_label_id

    def forward(
        self,
        premise_embeddings: torch.Tensor,
        hypothesis_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        _validate_relation_inputs(premise_embeddings, hypothesis_embeddings, labels)

        cosine_distance = 1.0 - F.cosine_similarity(
            premise_embeddings,
            hypothesis_embeddings,
            dim=-1,
        )
        entailment_mask = labels.eq(self.entailment_label_id)
        contradiction_mask = labels.eq(self.contradiction_label_id)

        entailment_loss = _masked_sample_mean(cosine_distance.square(), entailment_mask)
        contradiction_penalty = F.relu(self.margin - cosine_distance).square()
        contradiction_loss = _masked_sample_mean(contradiction_penalty, contradiction_mask)
        return entailment_loss + contradiction_loss


def _validate_relation_inputs(
    premise_embeddings: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if premise_embeddings.ndim != 3:
        raise ValueError(
            "premise_embeddings must have shape [B, M, D], "
            f"got {tuple(premise_embeddings.shape)}."
        )
    if premise_embeddings.shape != hypothesis_embeddings.shape:
        raise ValueError(
            "Premise and hypothesis embeddings must have the same shape, "
            f"got {tuple(premise_embeddings.shape)} and {tuple(hypothesis_embeddings.shape)}."
        )
    if labels.ndim != 1 or labels.shape[0] != premise_embeddings.shape[0]:
        raise ValueError(
            f"labels must have shape [{premise_embeddings.shape[0]}], got {tuple(labels.shape)}."
        )


def _masked_sample_mean(values: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = sample_mask.unsqueeze(-1).expand_as(values).to(values.dtype)
    return (values * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)
