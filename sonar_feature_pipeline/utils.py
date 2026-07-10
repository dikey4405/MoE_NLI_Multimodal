from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.utils import (
    build_audio_index,
    get_label,
    parse_audio_filename,
    read_json_or_jsonl,
    resolve_audio_path,
    validate_raw_nli_record,
)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_pt(payload: Any, path: str | Path) -> None:
    torch.save(payload, Path(path))


def load_pt(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu")


def validate_embedding_shape(tensor: torch.Tensor, *, expected_dim: int = 1024, name: str = "embedding") -> None:
    if tensor.ndim not in {1, 2}:
        raise ValueError(
            f"{name} must have shape [{expected_dim}] or [batch_size, {expected_dim}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {tuple(tensor.shape)}")


def validate_feature_shape(tensor: torch.Tensor, *, expected_dim: int = 4096, name: str = "feature") -> None:
    if tensor.ndim not in {1, 2}:
        raise ValueError(
            f"{name} must have shape [{expected_dim}] or [batch_size, {expected_dim}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {tuple(tensor.shape)}")


__all__ = [
    "build_audio_index",
    "ensure_output_dir",
    "get_label",
    "load_pt",
    "parse_audio_filename",
    "read_json_or_jsonl",
    "resolve_audio_path",
    "save_pt",
    "validate_embedding_shape",
    "validate_feature_shape",
    "validate_raw_nli_record",
]
