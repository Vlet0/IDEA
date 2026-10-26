"""Checks for the binary-fall-v6 rigor additions: cluster CIs, bone-check calibration,
defenses, target-mode validation and the hadamard_erm ablation arm."""

import math
import unittest

import numpy as np
import torch

from attack import (MODE_TO_TRIGGER, PoisonedERMLoss, build_trigger, compute_target_poses,
                    get_semantic_fall_targets)
from train import _build_criterion, experiment_fingerprint
from utils.config import resolve_config
from utils.defenses import (packet_smoothing, remove_static_component, spectral_signature_audit,
                            subcarrier_smoothing)
from utils.logger import read_csv_to_dict_list, save_dict_to_csv
from utils.metrics import cluster_ratio_ci, compute_binary_fall_asr, wilson_interval

DIMS = dict(antennas=3, subcarriers=114, packets=10)


class ClusterCITests(unittest.TestCase):
    @staticmethod
    def ci(success, clusters, proportion=True, scale=100.0):
        return cluster_ratio_ci(success, np.ones_like(success), clusters, scale=scale,
                                proportion=proportion, level=0.95)

    def test_correlated_frames_widen_the_interval(self):
        # 10 subjects x 1000 identical frames: 7 subjects always succeed, 3 never do.
        success = np.repeat([1.0] * 7 + [0.0] * 3, 1000)
        r = self.ci(success, np.repeat(np.arange(10), 1000))
        w_low, w_high = wilson_interval(int(success.sum()), len(success))
        self.assertLess(r["low"], 70.0)
        self.assertGreater(r["high"], 70.0)
        self.assertGreater(r["high"] - r["low"], 5 * (w_high - w_low))
        self.assertEqual((r["min"], r["max"], r["n_clusters"]), (0.0, 100.0, 10))

    def test_saturated_proportion_never_gets_a_zero_width_interval(self):
        # Every one of 10 subjects at 100 %: the honest bound is Wilson at n_eff = 10.
        r = self.ci(np.ones(10 * 297), np.repeat(np.arange(10), 297))
        self.assertAlmostEqual(r["low"], 72.25, places=1)
        self.assertEqual(r["high"], 100.0)
        # The interval is never narrower than frame-level Wilson.
        one_failure = np.ones(80190)
        one_failure[0] = 0.0
        r = self.ci(one_failure, np.repeat(np.arange(10), 8019))
        w_low, w_high = wilson_interval(80189, 80190)
        self.assertLess(r["low"], w_low)

    def test_continuous_values_and_degenerate_inputs(self):
        values = np.concatenate([np.full(100, 10.0), np.full(100, 14.0)])
        r = self.ci(values, np.repeat([0, 1], 100), proportion=False, scale=1.0)
        self.assertLess(r["low"], 12.0)
        self.assertGreater(r["high"], 12.0)
        r = self.ci(np.ones(50), np.zeros(50))
        self.assertTrue(math.isnan(r["low"]) and math.isnan(r["high"]))
        with self.assertRaises(ValueError):
            cluster_ratio_ci(np.ones(3), np.ones(3), np.zeros(2), scale=1.0, proportion=True, level=0.95)


class BoneCalibrationTests(unittest.TestCase):
    def test_clean_bone_rate_is_reported_and_masks_are_consistent(self):
        fallen = get_semantic_fall_targets(K=1)
        upright = fallen.clone()
        upright[..., 1], upright[..., 2] = -fallen[..., 2], fallen[..., 1]
        clean = torch.cat([upright, upright * 1.6])  # second clean prediction: bones 60 % too long
        gt = torch.cat([upright, upright])
        trig = torch.cat([fallen, fallen])
        stats = compute_binary_fall_asr(clean, trig, 1.0, ground_truth_poses=gt, return_masks=True)
        masks = stats.pop("masks")
        self.assertEqual(stats["n_eligible"], 2)
        self.assertEqual(stats["clean_bone_consistent_rate"], 50.0)
        self.assertEqual(int(masks["success"].sum()), stats["n_success"])
        self.assertEqual(int(masks["success_bone"].sum()), round(stats["asr_bone_consistent"] * 2 / 100))
        self.assertNotIn("masks", compute_binary_fall_asr(clean, trig, 1.0, ground_truth_poses=gt))

    def test_sequence_level_asr_requires_min_frames_per_sequence(self):
        # 2 sequences of 5 frames each (seq_ids 0 and 1).  All frames are eligible.
        # Sequence 0: all 5 frames triggered → falls; seq 1: only 2 frames → falls.
        fallen = get_semantic_fall_targets(K=1)[0]   # (17, 3) — drop the K dimension
        upright = fallen.clone()
        upright[..., 1], upright[..., 2] = -fallen[..., 2].clone(), fallen[..., 1].clone()
        N = 10
        clean = upright.unsqueeze(0).expand(N, -1, -1)
        gt = upright.unsqueeze(0).expand(N, -1, -1)
        trig = torch.cat([fallen.unsqueeze(0).expand(5, -1, -1),   # seq 0: 5 falls
                          fallen.unsqueeze(0).expand(2, -1, -1),   # seq 1: 2 falls
                          upright.unsqueeze(0).expand(3, -1, -1)]) # seq 1: 3 non-falls
        seq_ids = [0] * 5 + [1] * 5

        # seq_min_frames=3: seq 0 (5≥3) succeeds, seq 1 (2<3) does not → ASR = 50%
        s3 = compute_binary_fall_asr(clean, trig, 1.0, ground_truth_poses=gt,
                                     seq_ids=seq_ids, seq_min_frames=3)
        self.assertEqual(s3["n_eligible"], 2)
        self.assertEqual(s3["n_success"], 1)
        self.assertAlmostEqual(s3["binary_fall_asr"], 50.0)
        self.assertEqual(s3["seq_min_frames"], 3)
        # Frame-level diagnostic must be present
        self.assertIn("frame_asr", s3)
        self.assertIn("frame_n_eligible", s3)

        # seq_min_frames=6: neither sequence has 6 falls → ASR = 0%
        s6 = compute_binary_fall_asr(clean, trig, 1.0, ground_truth_poses=gt,
                                     seq_ids=seq_ids, seq_min_frames=6)
        self.assertEqual(s6["binary_fall_asr"], 0.0)
        self.assertEqual(s6["n_success"], 0)


class DefenseTests(unittest.TestCase):
    def setUp(self):
        self.trigger = build_trigger("hadamard", K=2, **DIMS)
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(8, *DIMS.values(), generator=g)

    def test_smoothing_cancels_hadamard_rows(self):
        pattern = self.trigger.patterns
        interior = slice(2, 112)  # away from the replicate-padded edges
        self.assertLess(subcarrier_smoothing(pattern[0:1], 2)[..., interior, :].abs().max().item(), 1e-6)
        self.assertLess(subcarrier_smoothing(pattern, 4)[..., interior, :].abs().max().item(), 1e-6)
        self.assertTrue(torch.equal(subcarrier_smoothing(self.x, 1), self.x))
        self.assertEqual(subcarrier_smoothing(self.x, 3).shape, self.x.shape)

    def test_packet_smoothing_acts_on_the_packet_axis(self):
        constant_over_packets = self.trigger.patterns
        self.assertTrue(torch.allclose(packet_smoothing(constant_over_packets, 3), constant_over_packets))
        self.assertEqual(packet_smoothing(self.x, 2).shape, self.x.shape)

    def test_static_removal_erases_packet_constant_trigger(self):
        # The subcarrier (packet-constant) trigger is additive and constant over packets, so
        # subtracting the per-subcarrier packet mean (the static-removal mechanism) cancels it
        # exactly wherever the frame was not clipped to [0, 1]; only a small clipping residual
        # survives.  (remove_static_component additionally min-max renormalises, which rescales
        # that residual; the mechanism is tested here directly.)
        x = 0.1 + 0.8 * self.x  # mid-range so eps*pattern never clips
        triggered = self.trigger.apply(x, 0, 0.15)
        dyn = lambda t: t - t.mean(dim=3, keepdim=True)
        before = (triggered - x).abs().max()
        after = (dyn(triggered) - dyn(x)).abs().max()
        self.assertGreater(before.item(), 1e-4)
        self.assertLess(after.item(), 1e-6)

    def test_spectral_signature_flags_a_planted_cluster(self):
        g = torch.Generator().manual_seed(0)
        feats = torch.randn(2000, 16, generator=g)
        mask = torch.zeros(2000, dtype=torch.bool)
        mask[:40] = True
        feats[mask, 0] += 20.0
        stats = spectral_signature_audit(feats, mask, n_components=1, removal_factor=1.5)
        self.assertGreater(stats["auroc"], 0.99)
        self.assertEqual(stats["recall"], 100.0)
        none = spectral_signature_audit(feats, torch.zeros(2000, dtype=torch.bool), n_components=1,
                                        removal_factor=1.5)
        self.assertTrue(math.isnan(none["auroc"]))


class CsvAppendTests(unittest.TestCase):
    def test_append_with_new_columns_keeps_values_aligned(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rows.csv")
            save_dict_to_csv([{"a": 1, "b": 2}], path)
            save_dict_to_csv([{"c": 3, "a": 10, "b": 20}], path)
            save_dict_to_csv([{"b": 200, "a": 100}], path)
            rows = read_csv_to_dict_list(path)
        self.assertEqual([(r["a"], r["b"]) for r in rows], [(1, 2), (10, 20), (100, 200)])
        self.assertEqual(rows[1]["c"], 3)


class ProtocolTests(unittest.TestCase):
    def test_target_mode_typo_is_rejected(self):
        targets = get_semantic_fall_targets(K=2)
        y = targets[:1].clone()
        with self.assertRaises(ValueError):
            compute_target_poses(y, torch.zeros(1, dtype=torch.long), targets, "per-bone", 0.0)

    def test_preserve_subject_scale_rejects_integer_aliases(self):
        targets = get_semantic_fall_targets(K=1)
        y = torch.randn(4, 17, 3) * 0.3
        kk = torch.zeros(4, dtype=torch.long)
        for bad in (0, 1):  # must not alias False/True
            with self.assertRaises(ValueError):
                compute_target_poses(y, kk, targets, bad, 0.0)

    def test_facing_alignment_is_rigid_and_points_the_fall_forward(self):
        from utils.skeleton import BONES, NECK, PELVIS, facing_direction, horizontal_axes
        targets = get_semantic_fall_targets(K=1, ps=0.29)
        torch.manual_seed(0)
        y = torch.randn(8, 17, 3) * 0.3
        kk = torch.zeros(8, dtype=torch.long)
        base = compute_target_poses(y, kk, targets, "per_bone", 0.0)
        aligned = compute_target_poses(y, kk, targets, "per_bone", 0.0, align_facing=True, vertical_axis=2)
        a0, a1 = horizontal_axes(2)
        facing = facing_direction(y, 2)
        torso = aligned[:, NECK] - aligned[:, PELVIS]
        torso[:, 2] = 0.0
        torso = torso / torso.norm(dim=-1, keepdim=True)
        cos = torso[:, a0] * facing[:, a0] + torso[:, a1] * facing[:, a1]
        self.assertGreater(float(cos.min()), 0.999)  # torso points along facing
        bl = lambda P: torch.stack([(P[:, a] - P[:, b]).norm(dim=-1) for a, b in BONES], 1)
        self.assertLess(float((bl(aligned) - bl(base)).abs().max()), 1e-5)  # rigid: bones preserved

    def test_hadamard_erm_ablation_arm(self):
        self.assertEqual(MODE_TO_TRIGGER["hadamard_erm"], "hadamard")
        atk = resolve_config({})["attack"]
        self.assertIsInstance(_build_criterion("hadamard_erm", atk, atk["margin"], atk["clean_penalty_threshold"]),
                              PoisonedERMLoss)

    def test_evaluation_settings_do_not_change_the_training_fingerprint(self):
        base = resolve_config({})
        changed = resolve_config({"evaluation": {"cluster_ci": {"level": 0.9}},
                                  "training": {"num_workers": 0}})
        self.assertEqual(experiment_fingerprint(base), experiment_fingerprint(changed))
        target = resolve_config({"attack": {"target": {"preserve_subject_scale": False}}})
        self.assertNotEqual(experiment_fingerprint(base), experiment_fingerprint(target))


if __name__ == "__main__":
    unittest.main()
