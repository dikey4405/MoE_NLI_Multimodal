from __future__ import annotations

from pathlib import Path

from KLTN.source.training.trainer import run_training_from_config


if __name__ == "__main__":
    run_training_from_config(Path("KLTN/source/configs/conventional_moe.yaml"))
