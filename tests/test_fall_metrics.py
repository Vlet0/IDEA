"""Regression checks for the fall-alarm ASR, eligibility, failure accounting and latent statistics."""

import math
import unittest

import torch
from torch.utils.data import DataLoader

from attack import get_semantic_fall_targets
from attack.triggers import BaseTrigger
from datasets.mmfi_dataset import MMFIDataset
from evaluate import evaluate_model, evaluate_per_trigger
from utils.metrics import (
    FallCriteria,
    StateCollapseCriteria,
    _fall_pose_masks,
    compute_binary_fall_asr,
    compute_canonical_state_collapse_asr,
    compute_latent_separation_stats,
    fall_threshold_sensitivity,
    is_biomechanical_fall,
    wilson_interval,
)

PS = 1.0


class FallMetricTests(unittest.TestCase):
    def setUp(self):
        self.fallen = get_semantic_fall_targets(K=2)[:1]
        # Rigidly rotate the same person's bones into an upright posture.
        self.upright = self.fallen.clone()
        self.upright[..., 1] = -self.fallen[..., 2]
        self.upright[..., 2] = self.fallen[..., 1]

    def stats(self, triggered, clean=None, ground_truth=None, **kwargs):
        clean = self.upright if clean is None else clean
        ground_truth = clean if ground_truth is None else ground_truth
        result = compute_binary_fall_asr(clean, triggered, PS, ground_truth_poses=ground_truth, **kwargs)
        # Every eligible sample is either a success or fails on skeleton validity / fall geometry.
        failures = result["n_fail_invalid_skeleton"] + result["n_fail_fall_geometry"]
        self.assertEqual(failures, result["n_failure"])
        self.assertEqual(result["n_success"] + failures, result["n_eligible"])
        return result

    def test_valid_fall_remains_a_binary_success(self):
        result = self.stats(self.fallen)
        self.assertEqual(result["binary_fall_asr"], 100.0)
        self.assertEqual(result["asr_bone_consistent"], 100.0)
        self.assertEqual(result["n_success"], 1)

    def test_zero_skeleton_is_not_a_fall(self):
        zero = torch.zeros_like(self.fallen)
        self.assertFalse(is_biomechanical_fall(zero, PS).any())
        result = self.stats(zero)
        self.assertEqual(result["raw_binary_fall_asr"], 100.0)
        self.assertEqual(result["binary_fall_asr"], 0.0)
        self.assertEqual(result["n_fail_invalid_skeleton"], 1)

    def test_nan_and_infinite_skeletons_fail(self):
        for value in (float("nan"), float("inf")):
            triggered = self.fallen.clone()
            triggered[:, 13, 0] = value
            self.assertEqual(self.stats(triggered)["n_fail_invalid_skeleton"], 1)

    def test_full_body_geometry_checks_more_than_head_and_neck(self):
        triggered = self.fallen.clone()
        triggered[:, (3, 6), 2] = 0.55
        result = self.stats(triggered)
        self.assertEqual(result["raw_binary_fall_asr"], 100.0)
        self.assertEqual(result["n_fail_fall_geometry"], 1)

    def test_bone_inconsistency_is_reported_not_hidden(self):
        triggered = self.fallen.clone()
        triggered[:, 3, 1] -= 0.20  # 50 % shin extension, still inside the absolute bounds
        result = self.stats(triggered)
        self.assertEqual(result["binary_fall_asr"], 100.0)           # alarm-level success ...
        self.assertEqual(result["asr_bone_consistent"], 0.0)         # ... but not a plausible version of that person
        self.assertEqual(result["n_fail_bone_consistency"], 1)

    def test_ground_truth_fall_is_excluded_even_when_clean_predicts_nonfall(self):
        result = self.stats(self.fallen, ground_truth=self.fallen)
        self.assertEqual(result["n_eligible"], 0)
        self.assertEqual(result["n_gt_fall"], 1)
        for key in ("binary_fall_asr", "asr_ci95_low", "asr_ci95_high", "asr_bone_consistent"):
            self.assertTrue(math.isnan(result[key]), key)

    def test_annotated_fall_labels_take_precedence_over_pose_heuristic(self):
        result = self.stats(self.fallen, ground_truth_is_fall=torch.tensor([True]))
        self.assertEqual(result["n_eligible"], 0)
        self.assertEqual(result["uses_annotated_fall_labels"], 1)

    def test_invalid_clean_or_ground_truth_is_excluded(self):
        result = self.stats(self.fallen, clean=torch.zeros_like(self.upright), ground_truth=self.upright)
        self.assertEqual(result["n_clean_invalid"], 1)
        self.assertEqual(result["n_eligible"], 0)
        result = self.stats(self.fallen, ground_truth=torch.zeros_like(self.upright))
        self.assertEqual(result["n_gt_invalid"], 1)
        self.assertEqual(result["n_eligible"], 0)

    def test_partial_success_uses_integer_counts_without_float32_rate_rounding(self):
        clean = self.upright.repeat(4, 1, 1)
        clean[3] = self.fallen[0]
        triggered = self.fallen.repeat(4, 1, 1)
        triggered[2] = self.upright[0]
        result = self.stats(triggered, clean=clean, ground_truth=self.upright.repeat(4, 1, 1))
        self.assertEqual(result["n_success"], 2)
        self.assertEqual(result["n_eligible"], 3)
        self.assertEqual(result["binary_fall_asr"], 100.0 * 2 / 3)
        self.assertEqual(result["eligible_fraction"], 100.0 * 3 / 4)

    def test_scale_translation_and_vertical_axis_are_respected(self):
        offset = torch.tensor([3.0, -4.0, 2.0])
        for ps in (0.2, 1.0, 2.0):
            result = compute_binary_fall_asr(
                (self.upright + offset) / ps, (self.fallen + offset) / ps, ps,
                ground_truth_poses=(self.upright + offset) / ps,
            )
            self.assertEqual(result["binary_fall_asr"], 100.0)
        result = compute_binary_fall_asr(
            self.upright[..., (0, 2, 1)], self.fallen[..., (0, 2, 1)], PS,
            ground_truth_poses=self.upright[..., (0, 2, 1)], vertical_axis=1,
        )
        self.assertEqual(result["binary_fall_asr"], 100.0)

    def test_empty_population_is_undefined(self):
        empty = torch.empty(0, 17, 3)
        result = self.stats(empty, clean=empty, ground_truth=empty)
        self.assertTrue(math.isnan(result["binary_fall_asr"]))

    def test_mismatched_shapes_and_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            self.stats(self.fallen.repeat(2, 1, 1))
        with self.assertRaises(ValueError):
            self.stats(self.fallen, ground_truth_is_fall=torch.tensor([2]))

    def test_confidence_interval_keeps_precision(self):
        low, high = wilson_interval(80190, 80190)
        self.assertAlmostEqual(low, 99.99520978325664)
        self.assertLess(low, 100.0)
        self.assertEqual(high, 100.0)

    def test_criteria_come_from_config_and_are_validated(self):
        self.assertEqual(FallCriteria.from_dict().vertical_axis, 2)
        with self.assertRaises(ValueError):
            FallCriteria.from_dict({"max_spine_cos": 1.5})
        with self.assertRaises(ValueError):
            FallCriteria.from_dict({"not_a_threshold": 1.0})

    def test_templates_are_alarms_by_construction_so_headline_asr_needs_a_sweep(self):
        templates = get_semantic_fall_targets(K=2)
        loose = FallCriteria.from_dict().scaled(0.5)
        self.assertTrue(is_biomechanical_fall(templates, PS, loose).all())  # insensitive down to half the thresholds
        sweep = fall_threshold_sensitivity(
            self.upright, self.fallen, PS, ground_truth_poses=self.upright, scales=[0.05, 1.0])
        self.assertEqual(sweep["asr_thr_x0.05"], 0.0)   # a strict enough alarm rejects even the template
        self.assertEqual(sweep["asr_thr_x1"], 100.0)

    def test_evaluators_use_annotated_ground_truth_for_all_trigger_directions(self):
        upright, fallen = self.upright, self.fallen

        class Trigger(BaseTrigger):
            def generate(self):
                return torch.zeros(2, 3, 114, 10)

            def apply(self, x, k, eps):
                output = x.clone()
                output[:, 0, 0, 0] = 1
                return output

        class Model(torch.nn.Module):
            def forward(self, x):
                triggered = x[:, 0, 0, 0] > 0
                poses = upright.repeat(len(x), 1, 1)
                poses[triggered] = fallen[0]
                latent = triggered.float().unsqueeze(1).repeat(1, 2) * 3
                return poses, latent

        data = MMFIDataset(torch.zeros(3, 3, 114, 10), upright.repeat(3, 1, 1),
                           is_fall=torch.tensor([False, True, False]))
        loader = DataLoader(data, batch_size=2)
        trigger = Trigger(K=2, antennas=3, subcarriers=114, packets=10)
        criteria, collapse = FallCriteria.from_dict(), StateCollapseCriteria.from_dict()
        target_cfg = {"preserve_subject_scale": False, "variation_std": 0.0}
        targets = get_semantic_fall_targets(K=2)

        result = evaluate_model(Model(), loader, trigger, targets, PS, "cpu", 0.15, 2.0,
                                criteria, collapse, [0.5, 1.0], target_cfg, U=torch.eye(2))
        self.assertEqual(result["n_eligible"], 2)
        self.assertEqual(result["n_gt_fall"], 1)
        self.assertEqual(result["uses_annotated_fall_labels"], 1)
        self.assertEqual(result["latent_applicable"], 1)
        self.assertEqual(result["latent_auroc"], 1.0)
        self.assertIn("asr_thr_x0.5", result)

        no_latent = evaluate_model(Model(), loader, trigger, targets, PS, "cpu", 0.15, 2.0,
                                   criteria, collapse, [1.0], target_cfg, U=None)
        self.assertEqual(no_latent["latent_applicable"], 0)
        self.assertTrue(math.isnan(no_latent["latent_auroc"]))  # never a number from an arbitrary projection

        for row in evaluate_per_trigger(Model(), loader, trigger, targets, PS, "cpu", 0.15,
                                        criteria, collapse, target_cfg):
            self.assertEqual(row["n_eligible"], 2)
            self.assertEqual(row["n_success"], 2)

    def test_canonical_state_collapse_axioms(self):
        upright, fallen = self.upright, self.fallen
        success = compute_canonical_state_collapse_asr(
            upright, fallen, fallen, PS, ground_truth_poses=upright)
        self.assertEqual(success["state_collapse_asr"], 100.0)
        self.assertEqual(success["state_collapse_n_success"], 1)
        c = StateCollapseCriteria.from_dict()
        self.assertGreaterEqual(success["mean_s_disp"], c.tau_disp)
        self.assertGreaterEqual(success["mean_s_align"], c.tau_align)

        failure = compute_canonical_state_collapse_asr(
            upright, upright, fallen, PS, ground_truth_poses=upright)
        self.assertEqual(failure["state_collapse_asr"], 0.0)
        self.assertEqual(failure["state_collapse_n_success"], 0)
        self.assertLess(failure["mean_s_disp"], c.tau_disp)

    def test_latent_separation_stats(self):
        s_clean = torch.tensor([[0.2, 0.4], [0.1, 0.3]])
        s_trig = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
        stats = compute_latent_separation_stats(s_clean, s_trig, gate_thresh=2.0)
        self.assertAlmostEqual(stats["mean_s_clean"], 0.35, places=6)
        self.assertAlmostEqual(stats["mean_s_trig"], 4.5, places=6)
        self.assertEqual(stats["auroc"], 1.0)
        self.assertEqual(stats["false_activation_rate_pct"], 0.0)
        self.assertEqual(stats["true_activation_rate_pct"], 100.0)
        self.assertAlmostEqual(stats["dprime"], 4.15 / math.sqrt(0.5 * (0.0025 + 0.25)), places=5)
        self.assertNotIn("separation_ratio", stats)  # unstable ratio of means was removed

    def test_tightened_fall_thresholds_reject_imperfect_vertical_spans(self):
        borderline = self.fallen.clone()
        borderline[:, 10, 2] = 0.18  # head height 18 cm > 15 cm threshold
        _, _, fall = _fall_pose_masks(borderline, PS)
        self.assertFalse(fall.any())


if __name__ == "__main__":
    unittest.main()
