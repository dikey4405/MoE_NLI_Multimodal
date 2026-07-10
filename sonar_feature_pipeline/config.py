from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..data.config import DEFAULT_LABEL_MAPPING


KLTN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = KLTN_ROOT / "Data"
DEFAULT_OUTPUT_DIR = KLTN_ROOT / "data_sonar_features"

DEFAULT_SPLIT_FILES = {
    "train": Path("Train") / "train.json",
    "dev": Path("Dev") / "dev.json",
    "test": Path("Test") / "test.json",
}


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for SONAR feature extraction only."""

    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
    split_files: dict[str, str | Path] = field(default_factory=lambda: dict(DEFAULT_SPLIT_FILES))
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"
    text_encoder_name_or_path: str = "text_sonar_basic_encoder"
    text_tokenizer_name_or_path: str = "text_sonar_basic_encoder"
    speech_encoder_name_or_path: str = "sonar_speech_encoder_vie"
    text_source_lang: str = "vie_Latn"
    encoder_batch_size: int = 8
    label_mapping: dict[str, int] | None = None
    validate_audio_exists: bool = True
    allow_ambiguous_audio: bool = True

    def get_split_path(self, split: str) -> Path:
        if split not in self.split_files:
            available = ", ".join(sorted(self.split_files))
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}")

        split_path = Path(self.split_files[split])
        if split_path.is_absolute():
            return split_path
        return Path(self.raw_data_dir) / split_path


__all__ = [
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RAW_DATA_DIR",
    "DEFAULT_SPLIT_FILES",
    "FeatureExtractionConfig",
]
