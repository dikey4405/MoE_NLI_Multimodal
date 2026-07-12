from __future__ import annotations

from typing import Any


class NLICollator:
    """Collate raw NLI samples without encoding, padding embeddings, or fusion."""

    def __init__(self, label_mapping: dict[str, int] | None = None) -> None:
        self.label_mapping = label_mapping

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Group normalized samples into a batch dictionary for the embedding stage."""

        labels = [sample["label"] for sample in samples]
        batch: dict[str, Any] = {
            "sample_indices": [sample["sample_index"] for sample in samples],
            "ids": [sample.get("id") for sample in samples],
            "premise": [sample["premise"] for sample in samples],
            "hypothesis": [sample["hypothesis"] for sample in samples],
            "premise_modality": [sample["premise_modality"] for sample in samples],
            "hypothesis_modality": [sample["hypothesis_modality"] for sample in samples],
            "input_mode": [sample["input_mode"] for sample in samples],
            "labels": labels,
        }
        if samples and "input_pairs" in samples[0]:
            batch["input_pairs"] = self._collate_input_pairs(samples)

        if self.label_mapping is not None:
            batch["label_ids"] = [self.label_mapping[label] for label in labels]

        return batch

    @staticmethod
    def _collate_input_pairs(samples: list[dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
        modes = samples[0]["input_pairs"].keys()
        return {
            mode: {
                "premise": [sample["input_pairs"][mode]["premise"] for sample in samples],
                "hypothesis": [sample["input_pairs"][mode]["hypothesis"] for sample in samples],
                "premise_modality": [
                    sample["input_pairs"][mode]["premise_modality"] for sample in samples
                ],
                "hypothesis_modality": [
                    sample["input_pairs"][mode]["hypothesis_modality"] for sample in samples
                ],
                "input_mode": [sample["input_pairs"][mode]["input_mode"] for sample in samples],
            }
            for mode in modes
        }
