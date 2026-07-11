from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data.config import DEFAULT_LABEL_MAPPING
from ..models.common.contrastive import MultiViewContrastiveLoss
from ..models.common.routing_utils import compute_routing_statistics
from .feature_dataset import FeatureBatchCollator, FeatureTensorDataset
from .model_factory import build_model
from .utils import Timer, count_parameters, load_yaml_config, resolve_path, set_random_seed


def differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_training_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    aux_loss_coef: float,
    contrastive_loss_coef: float,
    contrastive_criterion: MultiViewContrastiveLoss,
    use_contrastive_loss: bool,
) -> dict[str, torch.Tensor]:
    """Compute classification + load-balancing + optional contrastive losses."""

    classification_loss = F.cross_entropy(outputs["logits"], labels)
    load_balancing_loss = outputs["load_balancing_loss"]
    should_use_contrastive = (
        use_contrastive_loss
        and contrastive_loss_coef > 0
        and "contrastive_embeddings" in outputs
    )
    if should_use_contrastive:
        contrastive_loss = contrastive_criterion(outputs["contrastive_embeddings"])
    else:
        contrastive_loss = differentiable_zero(outputs["logits"])

    total_loss = (
        classification_loss
        + aux_loss_coef * load_balancing_loss
        + contrastive_loss_coef * contrastive_loss
    )
    return {
        "classification_loss": classification_loss,
        "load_balancing_loss": load_balancing_loss,
        "contrastive_loss": contrastive_loss,
        "total_loss": total_loss,
    }


class Trainer:
    """Shared trainer that depends only on the common model output interface."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader | None,
        config: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._build_logger()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.contrastive_criterion = MultiViewContrastiveLoss(
            temperature=float(config["contrastive_temperature"])
        )

    def train(self) -> None:
        best_dev_loss = float("inf")
        epochs_without_improvement = 0
        num_epochs = int(self.config["num_epochs"])
        self.logger.info("parameter_counts=%s", count_parameters(self.model))

        with Timer() as total_timer:
            for epoch in range(1, num_epochs + 1):
                with Timer() as epoch_timer:
                    train_metrics = self._run_epoch(training=True)
                    dev_metrics = self._run_epoch(training=False) if self.dev_loader is not None else {}

                self.logger.info(
                    "epoch=%d train=%s dev=%s epoch_time=%.2fs",
                    epoch,
                    train_metrics,
                    dev_metrics,
                    epoch_timer.elapsed,
                )
                dev_loss = dev_metrics.get("total_loss", train_metrics["total_loss"])
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    epochs_without_improvement = 0
                    self._save_checkpoint("best_model.pt", epoch, best_dev_loss)
                else:
                    epochs_without_improvement += 1

                if torch.cuda.is_available():
                    self.logger.info("peak_gpu_memory_mb=%.2f", torch.cuda.max_memory_allocated() / (1024**2))
                if epochs_without_improvement >= int(self.config["patience"]):
                    self.logger.info("early_stopping_epoch=%d", epoch)
                    break

        self.logger.info("total_training_time=%.2fs", total_timer.elapsed)

    def _run_epoch(self, *, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.dev_loader
        if loader is None:
            return {}

        self.model.train(training)
        totals: dict[str, float] = {
            "classification_loss": 0.0,
            "load_balancing_loss": 0.0,
            "contrastive_loss": 0.0,
            "total_loss": 0.0,
            "accuracy": 0.0,
        }
        num_batches = 0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.set_grad_enabled(training):
                outputs = self.model(features)
                losses = compute_training_losses(
                    outputs,
                    labels,
                    aux_loss_coef=float(self.config["aux_loss_coef"]),
                    contrastive_loss_coef=float(self.config["contrastive_loss_coef"]),
                    contrastive_criterion=self.contrastive_criterion,
                    use_contrastive_loss=bool(self.config["use_contrastive_loss"]),
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total_loss"].backward()
                    self.optimizer.step()

            predictions = outputs["logits"].argmax(dim=-1)
            totals["accuracy"] += (predictions == labels).float().mean().item()
            for key in ("classification_loss", "load_balancing_loss", "contrastive_loss", "total_loss"):
                totals[key] += losses[key].detach().item()
            num_batches += 1

        return {key: value / max(num_batches, 1) for key, value in totals.items()}

    def collect_routing_statistics(self, loader: DataLoader) -> list[dict[str, Any]]:
        self.model.eval()
        stats: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch["features"].to(self.device))
                stat = compute_routing_statistics(
                    outputs["router_probs"],
                    outputs["topk_indices"],
                    outputs["topk_weights"],
                )
                stat["load_balancing_loss"] = outputs["load_balancing_loss"].detach().cpu()
                stats.append(stat)
        return stats

    def _save_checkpoint(self, file_name: str, epoch: int, score: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "score": score,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            self.output_dir / file_name,
        )

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"trainer.{self.output_dir.name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger


def build_feature_loader(path: str | Path, config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    dataset = FeatureTensorDataset(path, label_mapping=DEFAULT_LABEL_MAPPING)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        collate_fn=FeatureBatchCollator(),
    )


def run_training_from_config(config_path: str | Path) -> None:
    config = load_yaml_config(config_path)
    set_random_seed(int(config["seed"]))
    model = build_model(config)
    train_loader = build_feature_loader(
        resolve_path(config["train_path"], config_path=config_path),
        config,
        shuffle=True,
    )
    dev_loader = build_feature_loader(
        resolve_path(config["dev_path"], config_path=config_path),
        config,
        shuffle=False,
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=config,
        output_dir=resolve_path(config["output_dir"], config_path=config_path),
    )
    trainer.train()
