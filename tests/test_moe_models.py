from __future__ import annotations

import unittest

import torch

from KLTN.source.models.conventional_moe.model import ConventionalMoENLIModel
from KLTN.source.models.deepseek_moe.model import DeepSeekMoENLIModel
from KLTN.source.models.fine_grained_moe.model import FineGrainedMoENLIModel
from KLTN.source.models.common.contrastive import MultiViewContrastiveLoss
from KLTN.source.training.model_factory import build_model
from KLTN.source.training.trainer import compute_training_losses


class MoEModelTestMixin:
    model_cls = None
    model_kwargs: dict = {}
    expected_top_k = 0
    has_shared_expert = False

    def _build_model(self, **overrides):
        kwargs = {
            "input_dim": 4096,
            "num_modes": 4,
            "hidden_dim": 64,
            "expert_ffn_dim": 128,
            "num_labels": 3,
            "dropout": 0.0,
            "use_contrastive_loss": True,
            "contrastive_hidden_dim": 32,
            "contrastive_dim": 16,
        }
        kwargs.update(self.model_kwargs)
        kwargs.update(overrides)
        return self.model_cls(**kwargs)

    def test_forward_loss_and_backward(self):
        torch.manual_seed(7)
        model = self._build_model()
        features = torch.randn(4, 4, 4096)
        labels = torch.tensor([0, 1, 2, 0])

        outputs = model(features)
        self.assertEqual(outputs["logits"].shape, (4, 3))
        self.assertEqual(outputs["fused"].shape, (4, 64))
        self.assertEqual(outputs["moe_output"].shape, (4, 4, 64))
        self.assertEqual(outputs["router_probs"].shape[-1], self.model_kwargs["num_routed_experts"])
        self.assertFalse(torch.isnan(outputs["router_probs"]).any())
        self.assertTrue(torch.allclose(outputs["router_probs"].sum(dim=-1), torch.ones(4, 4), atol=1e-5))
        self.assertEqual(outputs["topk_indices"].shape, (4, 4, self.expected_top_k))
        self.assertTrue(torch.allclose(outputs["topk_weights"].sum(dim=-1), torch.ones(4, 4), atol=1e-5))
        self.assertEqual(outputs["load_balancing_loss"].ndim, 0)
        self.assertIn("contrastive_embeddings", outputs)

        criterion = MultiViewContrastiveLoss(temperature=0.07)
        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.01,
            contrastive_loss_coef=0.1,
            contrastive_criterion=criterion,
            use_contrastive_loss=True,
        )
        for key in ("classification_loss", "load_balancing_loss", "contrastive_loss", "total_loss"):
            self.assertEqual(losses[key].ndim, 0)
            self.assertFalse(torch.isnan(losses[key]))
            self.assertFalse(torch.isinf(losses[key]))

        losses["total_loss"].backward()
        self.assertTrue(_has_gradient(model.router))
        self.assertTrue(_has_gradient(model.routed_experts))
        self.assertTrue(_has_gradient(model.contrastive_head))

        if self.has_shared_expert:
            self.assertTrue(_has_gradient(model.shared_experts))
            self.assertLess(int(outputs["topk_indices"].max()), model.num_routed_experts)
            self.assertEqual(outputs["shared_output"].shape, (4, 4, 64))
            self.assertEqual(outputs["routed_output"].shape, (4, 4, 64))
        else:
            self.assertFalse(hasattr(model, "shared_experts"))

        self.assertFalse(hasattr(model, "models"))

    def test_bad_input_shape_raises(self):
        model = self._build_model()
        with self.assertRaises(ValueError):
            model(torch.randn(4, 16384))

    def test_contrastive_can_be_disabled(self):
        model = self._build_model(use_contrastive_loss=False)
        outputs = model(torch.randn(4, 4, 4096))
        self.assertNotIn("contrastive_embeddings", outputs)
        losses = compute_training_losses(
            outputs,
            torch.tensor([0, 1, 2, 0]),
            aux_loss_coef=0.01,
            contrastive_loss_coef=0.0,
            contrastive_criterion=MultiViewContrastiveLoss(),
            use_contrastive_loss=False,
        )
        self.assertEqual(losses["contrastive_loss"].ndim, 0)
        self.assertFalse(torch.isnan(losses["total_loss"]))

    def test_batch_size_one_contrastive_is_finite(self):
        model = self._build_model()
        outputs = model(torch.randn(1, 4, 4096))
        losses = compute_training_losses(
            outputs,
            torch.tensor([1]),
            aux_loss_coef=0.01,
            contrastive_loss_coef=0.1,
            contrastive_criterion=MultiViewContrastiveLoss(),
            use_contrastive_loss=True,
        )
        self.assertFalse(torch.isnan(losses["contrastive_loss"]))
        self.assertFalse(torch.isnan(losses["total_loss"]))


class TestConventionalMoE(MoEModelTestMixin, unittest.TestCase):
    model_cls = ConventionalMoENLIModel
    model_kwargs = {
        "num_routed_experts": 4,
        "num_shared_experts": 0,
        "routed_top_k": 2,
    }
    expected_top_k = 2
    has_shared_expert = False


class TestFineGrainedMoE(MoEModelTestMixin, unittest.TestCase):
    model_cls = FineGrainedMoENLIModel
    model_kwargs = {
        "num_routed_experts": 8,
        "num_shared_experts": 0,
        "routed_top_k": 4,
    }
    expected_top_k = 4
    has_shared_expert = False


class TestDeepSeekMoE(MoEModelTestMixin, unittest.TestCase):
    model_cls = DeepSeekMoENLIModel
    model_kwargs = {
        "num_routed_experts": 8,
        "num_shared_experts": 1,
        "routed_top_k": 3,
    }
    expected_top_k = 3
    has_shared_expert = True


class TestModelFactory(unittest.TestCase):
    def test_zero_contrastive_coef_disables_projection_head(self):
        model = build_model(
            {
                "model_name": "conventional_moe",
                "input_dim": 4096,
                "num_modes": 4,
                "hidden_dim": 64,
                "num_routed_experts": 4,
                "num_shared_experts": 0,
                "routed_top_k": 2,
                "expert_ffn_dim": 128,
                "num_labels": 3,
                "dropout": 0.0,
                "use_contrastive_loss": True,
                "contrastive_hidden_dim": 32,
                "contrastive_dim": 16,
                "contrastive_loss_coef": 0.0,
            }
        )
        outputs = model(torch.randn(2, 4, 4096))
        self.assertIsNone(model.contrastive_head)
        self.assertNotIn("contrastive_embeddings", outputs)


def _has_gradient(module: torch.nn.Module) -> bool:
    for param in module.parameters():
        if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0:
            return True
    return False


if __name__ == "__main__":
    unittest.main()
