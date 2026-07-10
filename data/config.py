from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .schemas import InputMode


DEFAULT_LABEL_MAPPING = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


@dataclass(frozen=True)
class DatasetConfig:
    data_path: str | Path
    input_mode: InputMode | str = InputMode.TEXT_TEXT
    audio_root: str | Path | None = None
    label_mapping: dict[str, int] | None = None
    validate_audio_exists: bool = True
    allow_ambiguous_audio: bool = True


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 8
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False


@dataclass(frozen=True)
class SplitDataConfig:
    train: DatasetConfig | None = None
    val: DatasetConfig | None = None
    test: DatasetConfig | None = None
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
