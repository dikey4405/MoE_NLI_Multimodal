from __future__ import annotations

from torch.utils.data import DataLoader

from .collator import NLICollator
from .config import DataLoaderConfig, DatasetConfig, SplitDataConfig
from .dataset import NLIMultimodalDataset


def build_dataloader(
    dataset_config: DatasetConfig,
    dataloader_config: DataLoaderConfig | None = None,
    *,
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a PyTorch DataLoader for one split."""

    loader_config = dataloader_config or DataLoaderConfig()
    dataset = NLIMultimodalDataset(
        data_path=dataset_config.data_path,
        input_mode=dataset_config.input_mode,
        audio_root=dataset_config.audio_root,
        label_mapping=dataset_config.label_mapping,
        validate_audio_exists=dataset_config.validate_audio_exists,
        allow_ambiguous_audio=dataset_config.allow_ambiguous_audio,
    )
    collator = NLICollator(label_mapping=dataset_config.label_mapping)

    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle if shuffle is None else shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        drop_last=loader_config.drop_last,
        collate_fn=collator,
    )


def build_train_val_test_dataloaders(config: SplitDataConfig) -> dict[str, DataLoader]:
    """Build available train/validation/test dataloaders from split configs."""

    loaders: dict[str, DataLoader] = {}
    if config.train is not None:
        loaders["train"] = build_dataloader(config.train, config.dataloader, shuffle=True)
    if config.val is not None:
        loaders["val"] = build_dataloader(config.val, config.dataloader, shuffle=False)
    if config.test is not None:
        loaders["test"] = build_dataloader(config.test, config.dataloader, shuffle=False)
    return loaders
