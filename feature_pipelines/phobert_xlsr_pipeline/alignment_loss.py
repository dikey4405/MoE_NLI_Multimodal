from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SymmetricTextSpeechAlignmentLoss(nn.Module):
    """Symmetric in-batch InfoNCE for matched text-speech sentence pairs."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,
        speech_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        _validate_alignment_pair(text_embeddings, speech_embeddings)
        if text_embeddings.shape[0] == 1:
            return (text_embeddings.sum() + speech_embeddings.sum()) * 0.0

        text = F.normalize(text_embeddings, p=2, dim=-1)
        speech = F.normalize(speech_embeddings, p=2, dim=-1)
        logits = text @ speech.transpose(0, 1) / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def compute_alignment_metrics(
    text_embeddings: torch.Tensor,
    speech_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return retrieval accuracy and matched/unmatched cosine statistics."""

    _validate_alignment_pair(text_embeddings, speech_embeddings)
    text = F.normalize(text_embeddings, p=2, dim=-1)
    speech = F.normalize(speech_embeddings, p=2, dim=-1)
    similarities = text @ speech.transpose(0, 1)
    targets = torch.arange(similarities.shape[0], device=similarities.device)
    positive_similarity = similarities.diagonal().mean()
    if similarities.shape[0] > 1:
        negative_mask = ~torch.eye(
            similarities.shape[0], dtype=torch.bool, device=similarities.device
        )
        negative_similarity = similarities.masked_select(negative_mask).mean()
    else:
        negative_similarity = similarities.sum() * 0.0
    return {
        "text_to_speech_accuracy": similarities.argmax(dim=1).eq(targets).float().mean(),
        "speech_to_text_accuracy": similarities.argmax(dim=0).eq(targets).float().mean(),
        "positive_cosine_similarity": positive_similarity,
        "negative_cosine_similarity": negative_similarity,
    }


def _validate_alignment_pair(
    text_embeddings: torch.Tensor,
    speech_embeddings: torch.Tensor,
) -> None:
    if text_embeddings.ndim != 2:
        raise ValueError(
            f"text_embeddings must have shape [B, D], got {tuple(text_embeddings.shape)}."
        )
    if text_embeddings.shape != speech_embeddings.shape:
        raise ValueError(
            "Text and speech embeddings must have the same shape, "
            f"got {tuple(text_embeddings.shape)} and {tuple(speech_embeddings.shape)}."
        )
