from __future__ import annotations

import torch

from .utils import validate_embedding_shape, validate_feature_shape


def build_pair_feature(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Build concat(u, v, |u - v|, u * v) for one mode only."""

    validate_embedding_shape(u, name="premise embedding")
    validate_embedding_shape(v, name="hypothesis embedding")
    if u.shape != v.shape:
        raise ValueError(f"Premise and hypothesis embeddings must have the same shape, got {tuple(u.shape)} and {tuple(v.shape)}")

    feature = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
    validate_feature_shape(feature)
    return feature
