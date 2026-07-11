from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..data.config import DEFAULT_LABEL_MAPPING
from ..models.common.routing_utils import MODE_NAMES


class FeatureTensorDataset(Dataset):
    """Dataset over saved SONAR pair features."""

    def __init__(
        self,
        feature_path: str | Path,
        *,
        label_mapping: dict[str, int] | None = None,
    ) -> None:
        self.feature_path = Path(feature_path)
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_path}")
        self.label_mapping = label_mapping or dict(DEFAULT_LABEL_MAPPING)
        payload = torch.load(self.feature_path, map_location="cpu")
        if not isinstance(payload, list):
            raise ValueError(f"Feature file must contain a list of samples: {self.feature_path}")
        self.samples = payload

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        features_by_mode = sample["features"]
        features = torch.stack([features_by_mode[mode].float() for mode in MODE_NAMES], dim=0)
        label = sample["label"]
        label_id = self.label_mapping[label] if isinstance(label, str) else int(label)
        return {
            "id": sample.get("id"),
            "sample_index": sample.get("sample_index", index),
            "features": features,
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_text": label,
        }


class FeatureBatchCollator:
    """Collate saved feature samples into [B, 4, 4096] tensors."""

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "ids": [sample["id"] for sample in samples],
            "sample_indices": [sample["sample_index"] for sample in samples],
            "features": torch.stack([sample["features"] for sample in samples], dim=0),
            "labels": torch.stack([sample["label"] for sample in samples], dim=0),
            "label_texts": [sample["label_text"] for sample in samples],
        }
