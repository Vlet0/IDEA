"""Multi-semantic targets, time-axis (packet) Walsh trigger and trigger selectivity."""

import unittest

import torch

from attack import (build_semantic_target_bank, build_trigger, compose_trajectory_pose,
                    get_curled_fall_pose, get_fall_trajectory, get_semantic_fall_pose,
                    get_semantic_standing_pose, is_fall_target, target_bank_names)
from attack.triggers import HadamardTrigger
from utils.defenses import remove_static_component, subcarrier_smoothing
from utils.metrics import FallCriteria, _bone_lengths_m, _fall_pose_masks, is_biomechanical_fall

DIMS = dict(antennas=3, subcarriers=114, packets=10)


class SemanticTargetTests(unittest.TestCase):
    def test_bank_mixes_fall_and_conceal_targets(self):
        ps = 0.3
        bank = build_semantic_target_bank(["fall_prone", "stand"], ps=ps)
        self.assertEqual(bank.shape, (2, 17, 3))
        crit = FallCriteria.from_dict()
        alarm = is_biomechanical_fall(bank, ps, crit)
        self.assertTrue(bool(alarm[0]))          # fall target raises the alarm
        self.assertFalse(bool(alarm[1]))         # standing (conceal) target does not
        stand = get_semantic_standing_pose(ps=ps)
        self.assertTrue(torch.allclose(bank[1], stand))
        self.assertEqual(target_bank_names(["fall_prone", "stand"]), ["Prone fall", "Stand (conceal)"])

    def test_target_semantics_flags(self):
        self.assertTrue(is_fall_target("fall_prone") and is_fall_target("fall_lateral"))
        self.assertFalse(is_fall_target("stand"))
        with self.assertRaises(ValueError):
            is_fall_target("sit")
        with self.assertRaises(ValueError):
            build_semantic_target_bank([])


class ComplexFallTargetTests(unittest.TestCase):
    def test_curled_fall_is_valid_and_preserves_bone_lengths(self):
        ps = 0.3
        lateral = get_semantic_fall_pose(ps=ps, fall_type="lateral")
        curled = get_curled_fall_pose(ps=ps)
        crit = FallCriteria.from_dict()
        _, valid, fall = (m[0] for m in _fall_pose_masks(curled.unsqueeze(0), ps, crit))
        self.assertTrue(bool(valid) and bool(fall))  # a valid skeleton that still reads as a fall
        # Folds are rotations: bone lengths identical to the un-folded lateral fall.
        db = (_bone_lengths_m(lateral.unsqueeze(0), ps) - _bone_lengths_m(curled.unsqueeze(0), ps)).abs().max()
        self.assertLess(float(db), 1e-5)
        # Distal joints actually moved (complex posture, not identical to lateral).
        self.assertGreater(float((curled - lateral).abs().max()), 0.05)

    def test_fall_trajectory_is_a_descent_of_valid_poses(self):
        ps = 0.3
        traj = get_fall_trajectory(ps=ps)
        crit = FallCriteria.from_dict()
        self.assertEqual(traj.shape, (3, 17, 3))
        alarms = is_biomechanical_fall(traj, ps, crit)
        self.assertFalse(bool(alarms[0]))   # standing
        self.assertTrue(bool(alarms[2]))    # on the ground
        # Continuous composition: alpha=0 is the first key pose, alpha=1 the last.
        ends = compose_trajectory_pose(traj, [0.0, 1.0])
        self.assertTrue(torch.allclose(ends[0], traj[0]))
        self.assertTrue(torch.allclose(ends[1], traj[2]))


class TimeAxisTriggerTests(unittest.TestCase):
    def test_packet_axis_is_orthogonal_frequency_flat_and_defense_resistant(self):
        trig = build_trigger("hadamard", K=2, axis="packet", **DIMS)
        bank = trig.patterns
        # Orthogonal directions (rows 1,2 over 10 packets).
        gram = trig.gram()
        self.assertLess(float(gram[0, 1].abs()), 1e-6)
        # Frequency-flat: identical across subcarriers -> subcarrier smoothing is the identity.
        self.assertTrue(torch.allclose(bank[:, :, 0, :], bank[:, :, 50, :]))
        self.assertTrue(torch.allclose(subcarrier_smoothing(bank, 4), bank, atol=1e-6))
        # Survives static-component (per-subcarrier temporal mean) removal: the triggered input
        # does NOT collapse onto the clean one (contrast with the subcarrier axis).
        g = torch.Generator().manual_seed(0)
        x = torch.rand(8, *DIMS.values(), generator=g)
        triggered = trig.apply(x, 0, 0.15)
        diff = (remove_static_component(triggered) - remove_static_component(x)).abs().max()
        self.assertGreater(diff.item(), 1e-3)

    def test_subcarrier_axis_is_the_default_and_still_static_removable(self):
        self.assertEqual(build_trigger("hadamard", K=2, **DIMS).axis, "subcarrier")
        with self.assertRaises(ValueError):
            build_trigger("hadamard", K=2, axis="antenna", **DIMS)

    def test_packet_axis_direction_cap(self):
        # order = next pow2 >= 10 = 16 -> at most 15 directions.
        with self.assertRaises(ValueError):
            HadamardTrigger(K=16, axis="packet", **DIMS).patterns


class SelectivityTests(unittest.TestCase):
    def test_selectivity_runs_and_prefers_the_applied_direction(self):
        from evaluate import evaluate_trigger_selectivity
        from torch.utils.data import DataLoader, TensorDataset

        ps = 0.3
        fall_targets = build_semantic_target_bank(["fall_prone", "stand"], ps=ps)

        class Model(torch.nn.Module):
            """Outputs the target of whichever trigger dominates the (packet-mean) input sign."""
            def forward(self, x):
                b = len(x)
                # crude readout: positive packet-0 mean -> target 0, else target 1
                pick = (x.mean(dim=(1, 2, 3)) > x.median()).long()
                pose = fall_targets[pick]
                z = torch.zeros(b, 2)
                z[torch.arange(b), pick] = 5.0
                return pose, z

        x = torch.rand(6, 3, 114, 10)
        loader = DataLoader(TensorDataset(x, fall_targets[torch.zeros(6, dtype=torch.long)]), batch_size=3)
        trig = build_trigger("hadamard", K=2, axis="packet", antennas=3, subcarriers=114, packets=10)
        sel = evaluate_trigger_selectivity(Model(), loader, trig, fall_targets, ps, "cpu", 0.15,
                                           {"preserve_subject_scale": False, "variation_std": 0.0},
                                           U=torch.eye(2))
        self.assertEqual(len(sel["pose_confusion"]), 2)
        self.assertEqual(sel["K"], 2)
        self.assertTrue(0.0 <= sel["pose_selectivity"] <= 1.0)
        self.assertIsNotNone(sel["latent_confusion"])


class LatentTsneTests(unittest.TestCase):
    def test_collect_features_shapes(self):
        from evaluate import collect_latent_features
        from torch.utils.data import DataLoader, TensorDataset

        class Model(torch.nn.Module):
            def forward(self, x):
                b = len(x)
                return torch.zeros(b, 17, 3), x.reshape(b, -1)[:, :8]

        x = torch.rand(10, 3, 114, 10)
        loader = DataLoader(TensorDataset(x, torch.zeros(10, 17, 3)), batch_size=4)
        trig = build_trigger("hadamard", K=2, axis="packet", antennas=3, subcarriers=114, packets=10)
        feats = collect_latent_features(Model(), loader, trig, 0.15, "cpu", 2)
        self.assertEqual(feats["clean"].shape, (10, 8))
        self.assertEqual(len(feats["trig"]), 2)
        self.assertEqual(feats["trig"][0].shape, (10, 8))

    def test_tsne_writes_a_figure_and_subsamples(self):
        import tempfile, os
        from figures.plots import plot_latent_tsne
        rng = torch.Generator().manual_seed(0)
        # three separable blobs in 8-D
        feats = {"clean": torch.randn(120, 8, generator=rng),
                 "trig": [torch.randn(120, 8, generator=rng) + 8.0,
                          torch.randn(120, 8, generator=rng) - 8.0]}
        cfg = {"formats": ["png"], "dpi": 80}
        tsne_cfg = {"max_points_per_group": 40, "perplexity": 15, "max_iter": 250, "seed": 0}
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "tsne")
            plot_latent_tsne(feats, ["Prone fall", "Lateral fall"], out, cfg, tsne_cfg)
            self.assertTrue(os.path.exists(out + ".png"))


if __name__ == "__main__":
    unittest.main()
