"""Trigger banks: shared budget contract, geometry, injection correctness, EOT, no dead knobs."""

import math
import unittest

import torch

from attack import MODE_TO_TRIGGER, build_trigger, build_trigger_from_config, trigger_for_mode
from attack.triggers.hadamard_trigger import _sign_changes
from utils.config import resolve_config
from utils.stealth import trigger_report

DIMS = dict(antennas=3, subcarriers=114, packets=10)


def make(family: str, K: int = 2, **kw):
    return build_trigger(family, K=K, **DIMS, **kw)


class TriggerTests(unittest.TestCase):
    def test_every_family_shares_the_unit_l2_per_antenna_budget(self):
        for family in MODE_TO_TRIGGER.values():
            bank = make(family).patterns
            self.assertEqual(tuple(bank.shape), (2, 3, 114, 10))
            self.assertTrue(torch.isfinite(bank).all())
            norms = torch.linalg.vector_norm(bank, dim=(2, 3))
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4), family)

    def test_hadamard_directions_are_orthogonal_and_identical_across_antennas(self):
        trigger = make("hadamard")
        self.assertLess(abs(float(trigger.gram()[0, 1])), 1e-6)
        bank = trigger.patterns
        self.assertTrue(torch.equal(bank[:, 0], bank[:, 1]) and torch.equal(bank[:, 0], bank[:, 2]))

    def test_hadamard_order_is_derived_from_the_axis_count(self):
        self.assertEqual(tuple(build_trigger("hadamard", K=3, antennas=2, subcarriers=200, packets=4).patterns.shape),
                         (3, 2, 200, 4))
        with self.assertRaises(ValueError):
            build_trigger("hadamard", K=128, **DIMS).patterns

    def test_packet_axis_is_low_sequency_zero_mean_and_frequency_flat(self):
        trig = make("hadamard", K=1, axis="packet")
        bank = trig.patterns  # (1,3,114,10)
        # Flat over subcarriers: every subcarrier row is identical.
        self.assertTrue(torch.allclose(bank[0, 0], bank[0, 0, 0:1].expand_as(bank[0, 0])))
        # Zero-mean over packets -> survives static-component removal.
        self.assertLess(float(bank.mean(dim=3).abs().max()), 1e-6)
        # Low sequency: the single packet code changes sign at most once.
        code = bank[0, 0, 0]
        self.assertLessEqual(_sign_changes(torch.sign(code)), 1)

    def test_apply_is_in_range_and_int_tensor_index_agree(self):
        torch.manual_seed(0)
        x = torch.rand(4, 3, 114, 10)
        for family in MODE_TO_TRIGGER.values():
            trigger = make(family)
            by_int = trigger.apply(x, 1, 0.15)
            by_tensor = trigger.apply(x, torch.ones(4, dtype=torch.long), 0.15)
            self.assertTrue(torch.allclose(by_int, by_tensor), family)
            self.assertGreaterEqual(float(by_int.min()), 0.0)
            self.assertLessEqual(float(by_int.max()), 1.0)
            self.assertFalse(torch.allclose(by_int, x))

    def test_zero_epsilon_is_identity_for_in_range_input(self):
        x = torch.rand(2, 3, 114, 10)
        self.assertTrue(torch.allclose(make("hadamard").apply(x, 0, 0.0), x))

    def test_perturbation_is_decoupled_from_frame_content(self):
        # Two different in-range frames, small eps so no clipping: the injected delta is identical.
        trig = make("hadamard", K=1)
        a = torch.full((1, 3, 114, 10), 0.4)
        b = torch.full((1, 3, 114, 10), 0.6)
        da = trig.apply(a, 0, 0.01) - a
        db = trig.apply(b, 0, 0.01) - b
        self.assertTrue(torch.allclose(da, db, atol=1e-7))

    def test_eot_reduces_to_apply_and_stays_in_range(self):
        torch.manual_seed(0)
        x = torch.rand(4, 3, 114, 10)
        trig = make("hadamard", K=2, axis="packet")
        g = torch.Generator().manual_seed(0)
        same = trig.apply_eot(x, 1, 0.15, generator=g, gain_range=(1.0, 1.0), max_roll=0)
        self.assertTrue(torch.allclose(same, trig.apply(x, 1, 0.15)))
        out = trig.apply_eot(x, 1, 0.15, generator=g, gain_range=(0.7, 1.3), max_roll=3)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertFalse(torch.allclose(out, x))

    def test_ftrojan_positions_are_validated(self):
        with self.assertRaises(ValueError):
            build_trigger("ftrojan", K=2, **DIMS, pos_list_high=[(0, 0)], pos_list_mid=[(1, 1), (2, 2)])
        with self.assertRaises(ValueError):
            build_trigger("ftrojan", K=1, **DIMS, pos_list_high=[(999, 0)], pos_list_mid=[(1, 1)])

    def test_normalised_away_knobs_are_gone_instead_of_silently_ignored(self):
        with self.assertRaises(TypeError):
            build_trigger("ftrojan", K=2, **DIMS, magnitude=5.0)

    def test_bad_inputs_are_rejected(self):
        trigger = make("hadamard")
        with self.assertRaises(ValueError):
            trigger.apply(torch.rand(2, 3, 100, 10), 0, 0.15)  # wrong subcarrier count
        with self.assertRaises(ValueError):
            trigger.apply(torch.rand(2, 3, 114, 10), 0, -0.1)
        with self.assertRaises(ValueError):
            trigger.apply(torch.rand(2, 3, 114, 10), torch.zeros(5, dtype=torch.long), 0.15)

    def test_trigger_report_is_measured_and_finite(self):
        torch.manual_seed(0)
        x = torch.rand(8, 3, 114, 10)
        report = trigger_report(make("hadamard"), x, 0.15)
        for key in ("trigger_l2_rel_pct", "trigger_linf", "trigger_psnr_db", "trigger_l2_rel_pct_k0", "trigger_linf_k1"):
            self.assertTrue(math.isfinite(report[key]), key)
        self.assertGreater(report["trigger_l2_rel_pct"], 0.0)
        self.assertLess(report["trigger_max_abs_cos"], 1e-6)

    def test_registry_and_config_builder(self):
        self.assertEqual(trigger_for_mode("latent"), "hadamard")
        with self.assertRaises(ValueError):
            trigger_for_mode("badnets")
        cfg = resolve_config({})
        trigger = build_trigger_from_config(cfg)
        self.assertEqual((trigger.antennas, trigger.subcarriers, trigger.packets), tuple(cfg["data"]["input_shape"]))
        for mode, family in MODE_TO_TRIGGER.items():
            cfg = resolve_config({"attack": {"mode": mode}})
            self.assertEqual(type(build_trigger_from_config(cfg)).__name__.lower(), family + "trigger")
        with self.assertRaises(ValueError):
            build_trigger_from_config(resolve_config({"attack": {"mode": "latent", "trigger_type": "ftrojan"}}))


if __name__ == "__main__":
    unittest.main()
