from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path, *, config_path: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if candidate.parts and candidate.parts[0] == "KLTN":
        return cwd_candidate
    if cwd_candidate.exists() or config_path is None:
        return cwd_candidate
    return Path(config_path).resolve().parent / candidate


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    expert = sum(param.numel() for name, param in model.named_parameters() if "expert" in name)
    contrastive = sum(param.numel() for name, param in model.named_parameters() if "contrastive_head" in name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "expert_parameters": expert,
        "contrastive_head_parameters": contrastive,
    }


class Timer:
    """Simple context manager for epoch/training time logging."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
