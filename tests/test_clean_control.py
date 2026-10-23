"""Clean control, fixed poisoning plan, and train/evaluate-only plumbing."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from attack import build_poison_plan, validate_poison_rate
from run_experiments import _summary
from train import METRIC_PROTOCOL_VERSION, TRAINING_PROTOCOL_VERSION, train, training_loss


class CleanControlTests(unittest.TestCase):
    def test_existing_pipeline_verification(self):
        from verify_pipeline import verify
        verify(torch.device("cpu"))

    def test_zero_rate_never_poisons_any_sample(self):
        for size in (1, 17, 256):
            plan = build_poison_plan(size, 0.0, 2, seed=0)
            self.assertEqual(plan.n_poisoned, 0)
            self.assertFalse(plan.mask.any())

    def test_positive_rate_that_selects_nothing_is_rejected_not_silently_clean(self):
        with self.assertRaises(ValueError):
            build_poison_plan(8, 0.01, 2, seed=0)

    def test_out_of_range_rates_are_rejected(self):
        for rate in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                validate_poison_rate(rate)

    def test_clean_control_uses_only_pose_loss_for_every_attack_mode(self):
        for mode in ("latent", "ftrojan"):
            p = torch.ones(2, 17, 3, requires_grad=True)
            z = torch.full((2, 2), 100.0, requires_grad=True)
            criterion = Mock(side_effect=AssertionError("Attack criterion must not run"))
            loss, _ = training_loss(
                mode, criterion, p, z, torch.zeros_like(p), torch.eye(2),
                torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long),
                torch.full_like(p, 999.0), clean_only=True,
            )
            self.assertEqual(loss.item(), 1.0)
            loss.backward()
            self.assertIsNotNone(p.grad)
            self.assertIsNone(z.grad)
            criterion.assert_not_called()

    def test_clean_control_rejects_poisoned_batches(self):
        with self.assertRaises(ValueError):
            training_loss("latent", Mock(), None, None, None, None,
                          torch.tensor([True]), None, None, clean_only=True)

    def test_summary_does_not_round_asr_or_confidence_interval(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _summary([{"binary_fall_asr": 99.9999912345, "asr_ci95_low": 99.99520978325664}], "test")
        self.assertIn("99.9999912345", output.getvalue())
        self.assertIn("99.99520978325664", output.getvalue())

    def test_summary_shows_missing_measurements_as_nan_never_zero(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _summary([{"experiment_name": "x", "mode": "latent"}], "test")
        self.assertIn("nan", output.getvalue())

    def test_clean_epoch_and_evaluation_only_checkpoint_reuse(self):
        from attack import get_semantic_standing_pose
        from torch.utils.data import DataLoader, TensorDataset

        upright = get_semantic_standing_pose()
        x, y = torch.zeros(3, 3, 114, 10), upright.unsqueeze(0).repeat(3, 1, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        instances = []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pose = torch.nn.Parameter(upright.clone())
                self.training_inputs = []
                instances.append(self)

            def forward(self, inputs):
                if self.training:
                    self.training_inputs.append(inputs.detach().clone())
                return self.pose.unsqueeze(0).expand(len(inputs), -1, -1), self.pose[0, :2].expand(len(inputs), -1)

        with tempfile.TemporaryDirectory() as folder:
            cfg = {
                "experiment_name": "clean_control_test", "device": "cpu", "seed": 0,
                "model": {"backbone": "hpeli", "latent_dim": 2},
                "attack": {"mode": "latent", "poison_rate": 0.0},
                "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "pin_memory": False,
                             "output_dir": folder, "save_dir": folder},
                "evaluation": {"num_workers": 0, "pin_memory": False},
            }
            with (patch("train.load_mmfi_data", return_value=(loader, loader, 1.0, {"Xte": x, "Yte": y})),
                  patch("train.build_backbone", side_effect=lambda *args, **kwargs: Model()),
                  patch("train._build_criterion", return_value=Mock(side_effect=AssertionError("Attack loss ran"))),
                  patch("train.plot_latent_distribution"), patch("train.plot_skeleton_demo")):
                result = train(cfg)
                self.assertEqual(result["is_clean_control"], 1)
                self.assertEqual(result["n_poisoned"], 0)
                self.assertEqual(result["training_protocol_version"], TRAINING_PROTOCOL_VERSION)
                self.assertEqual(result["latent_applicable"], 0)
                self.assertTrue(instances[0].training_inputs)
                self.assertTrue(all(not batch.any() for batch in instances[0].training_inputs))
                cfg["evaluation"]["fall_criteria"] = {"max_bone_relative_error": 0.25}
                reevaluated = train(cfg, evaluate_only=True)
                self.assertEqual(instances[1].training_inputs, [])
                self.assertEqual(reevaluated["training_protocol_version"], TRAINING_PROTOCOL_VERSION)
                self.assertTrue((Path(folder) / f"experiment_summary_{METRIC_PROTOCOL_VERSION}.csv").exists())


if __name__ == "__main__":
    unittest.main()
