"""Adaptive-defense ablations added 2026-11-03 (EXPLORATORY -- see module docstrings in
attack/triggers/hadamard_trigger.py, attack/losses.py and attack/latent_attack.py for what is and
is not validated about each knob).  Every check here is an input-tensor / synthetic-latent
property on CPU; none of them claims anything about ASR on a trained victim -- that requires a
real training + evaluation run with the MM-Fi dataset.

Covers:
  * HadamardTrigger(axis="subband"/"subcarrier", profile=..., dc_fraction=...) stays
    backward-compatible at the defaults, keeps the unit-norm / orthogonality contract, rejects
    the knobs on axis="packet", and the two new profiles/mixes behave as documented:
      - profile="bump" retains more L2 energy than "walsh" under wide subcarrier_smoothing.
      - dc_fraction < 1 leaves a growing, non-zero residual after remove_static_component on a
        realistic-scale frame, while dc_fraction=1 leaves (numerically) none.
  * SubbandTrigger exposes the same two knobs (it mirrors HadamardTrigger's subband branch but is
    not reachable from a normal "latent"-mode config; see its module docstring).
  * LatentBackdoorLoss's optional L_norm_match term: an exact no-op at the default weight, and at
    a positive weight it only sends gradient into the poisoned latents (the clean side is
    detached).
  * make_U_from_latents' pc_rank_start: reproduces the original top-K PCs at the default, selects
    the next PCs when offset, keeps U orthonormal, and rejects an out-of-range offset.
"""

import math
import unittest

import torch

from attack.latent_attack import make_U_from_latents
from attack.losses import LatentBackdoorLoss
from attack.triggers.hadamard_trigger import HadamardTrigger
from attack.triggers.subband_trigger import SubbandTrigger
from utils.defenses import remove_static_component, subcarrier_smoothing

DIMS = dict(antennas=3, subcarriers=114, packets=10)


def _energy_ratio(bank: torch.Tensor, width: int) -> float:
    """L2 energy retained by ``subcarrier_smoothing`` at ``width``, over the whole pattern."""
    smoothed = subcarrier_smoothing(bank, width)
    return float(torch.linalg.vector_norm(smoothed[0, 0]) / torch.linalg.vector_norm(bank[0, 0]).clamp_min(1e-12))


class HadamardAdaptiveTriggerTests(unittest.TestCase):
    def test_defaults_are_unchanged_from_the_original_subband_construction(self):
        torch.manual_seed(0)
        default = HadamardTrigger(K=2, **DIMS, axis="subband").patterns
        explicit = HadamardTrigger(K=2, **DIMS, axis="subband", profile="walsh", dc_fraction=1.0).patterns
        self.assertTrue(torch.allclose(default, explicit, atol=1e-6))

    def test_bump_profile_keeps_the_unit_norm_and_orthogonality_contract(self):
        for axis in ("subband", "subcarrier"):
            trig = HadamardTrigger(K=2, **DIMS, axis=axis, profile="bump")
            bank = trig.patterns  # raises internally if the contract is violated
            self.assertEqual(tuple(bank.shape), (2, 3, 114, 10))
            self.assertLess(abs(float(trig.gram()[0, 1])), 1e-4, axis)

    def test_bump_profile_retains_more_energy_than_walsh_under_wide_smoothing(self):
        walsh = HadamardTrigger(K=2, **DIMS, axis="subband", profile="walsh").patterns
        bump = HadamardTrigger(K=2, **DIMS, axis="subband", profile="bump").patterns
        for width in (4, 6, 8):
            r_walsh, r_bump = _energy_ratio(walsh, width), _energy_ratio(bump, width)
            self.assertGreater(r_bump, r_walsh, f"width={width}")
            self.assertLessEqual(r_bump, 1.0 + 1e-6)

    def test_packet_axis_rejects_non_default_profile_or_dc_fraction(self):
        with self.assertRaises(ValueError):
            HadamardTrigger(K=1, **DIMS, axis="packet", profile="bump")
        with self.assertRaises(ValueError):
            HadamardTrigger(K=1, **DIMS, axis="packet", dc_fraction=0.5)

    def test_invalid_profile_or_dc_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            HadamardTrigger(K=1, **DIMS, axis="subband", profile="not_a_profile")
        with self.assertRaises(ValueError):
            HadamardTrigger(K=1, **DIMS, axis="subband", dc_fraction=1.5)

    def test_dc_fraction_one_reproduces_constant_over_packets(self):
        trig = HadamardTrigger(K=1, **DIMS, axis="subband", dc_fraction=1.0).patterns[0, 0]
        # Every packet column equal -> zero variance along the packet axis.
        self.assertLess(float(trig.var(dim=-1).max()), 1e-10)

    def test_dc_fraction_below_one_survives_static_component_removal(self):
        torch.manual_seed(1)
        eps = 0.35
        # Realistic-scale clean CSI-like frame (static removal's min-max renormalisation is only
        # meaningful at this scale, not on a bare unit-norm trigger pattern in isolation).
        x = (0.5 + 0.05 * torch.randn(3, 114, 10)).clamp(0.0, 1.0)
        clean_after = remove_static_component(x.unsqueeze(0))[0]

        residuals = []
        for dc_fraction in (1.0, 0.9, 0.7, 0.5):
            trig = HadamardTrigger(K=1, **DIMS, axis="subband", dc_fraction=dc_fraction).patterns[0]
            x_trig = (x + eps * trig).clamp(0.0, 1.0)
            trig_after = remove_static_component(x_trig.unsqueeze(0))[0]
            residuals.append(float(torch.linalg.vector_norm(trig_after - clean_after)))

        self.assertLess(residuals[0], 1e-4)          # dc_fraction=1: removed exactly (up to fp noise)
        for a, b in zip(residuals, residuals[1:]):   # residual grows monotonically as dc_fraction drops
            self.assertGreater(b, a)

    def test_subband_trigger_class_exposes_the_same_knobs(self):
        # SubbandTrigger is not reachable from a normal "latent"-mode config (see its module
        # docstring) but should behave identically to HadamardTrigger(axis="subband", ...).
        hadamard_bank = HadamardTrigger(K=2, **DIMS, axis="subband", profile="bump", dc_fraction=0.8).patterns
        subband_bank = SubbandTrigger(K=2, **DIMS, profile="bump", dc_fraction=0.8).patterns
        self.assertTrue(torch.allclose(hadamard_bank, subband_bank, atol=1e-6))


class NormMatchLossTests(unittest.TestCase):
    def _make(self, lam_norm_match: float) -> LatentBackdoorLoss:
        return LatentBackdoorLoss(lam=0.0, margin=1.0, clean_thresh=0.0, lam_ortho=0.0,
                                  lam_trig_pose=0.0, lam_bone=0.0, lam_norm_match=lam_norm_match)

    def _batch(self, clean_norm: float, trig_norm: float):
        torch.manual_seed(0)
        m, J = 8, 17
        z_clean_dir = torch.nn.functional.normalize(torch.randn(3, m), dim=-1) * clean_norm
        z_trig_dir = torch.nn.functional.normalize(torch.randn(3, m), dim=-1) * trig_norm
        z = torch.cat([z_clean_dir, z_trig_dir], dim=0).requires_grad_(True)
        p = torch.zeros(6, J, 3)
        y = torch.zeros(6, J, 3)
        target_poses = torch.zeros(6, J, 3)
        U = torch.eye(m)[:, :1]
        pm_mask = torch.tensor([False] * 3 + [True] * 3)
        kk_indices = torch.zeros(6, dtype=torch.long)
        return p, z, y, U, pm_mask, kk_indices, target_poses

    def test_zero_weight_is_an_exact_no_op(self):
        crit = self._make(lam_norm_match=0.0)
        batch = self._batch(clean_norm=1.0, trig_norm=5.0)
        total, info = crit(*batch)
        self.assertEqual(info["loss_norm_match"], 0.0)
        self.assertEqual(total.item(), 0.0)

    def test_positive_weight_penalises_the_norm_gap_and_is_zero_when_matched(self):
        crit = self._make(lam_norm_match=1.0)
        _, info_gap = crit(*self._batch(clean_norm=1.0, trig_norm=5.0))
        _, info_matched = crit(*self._batch(clean_norm=2.0, trig_norm=2.0))
        self.assertGreater(info_gap["loss_norm_match"], 0.0)
        self.assertAlmostEqual(info_matched["loss_norm_match"], 0.0, places=5)

    def test_gradient_only_reaches_the_poisoned_latents(self):
        crit = self._make(lam_norm_match=1.0)
        p, z, y, U, pm_mask, kk_indices, target_poses = self._batch(clean_norm=1.0, trig_norm=5.0)
        total, _ = crit(p, z, y, U, pm_mask, kk_indices, target_poses)
        total.backward()
        self.assertTrue(torch.all(z.grad[~pm_mask] == 0.0))   # clean side detached
        self.assertFalse(torch.all(z.grad[pm_mask] == 0.0))   # poisoned side pulled


class PCRankStartTests(unittest.TestCase):
    def _synthetic_latents(self, m=10, n=400, seed=0):
        """Latents with a clear, known, decreasing-variance orthogonal direction ranking."""
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
        scales = torch.tensor([10.0, 6.0, 3.0, 1.0] + [0.3] * (m - 4))
        coeffs = torch.randn(n, m, generator=g) * scales
        return coeffs @ Q.T, Q  # (n, m), (m, m) ground-truth orthonormal directions

    def test_default_offset_matches_the_top_pc(self):
        Z, Q = self._synthetic_latents()
        U, margin, clean_thresh = make_U_from_latents(Z, K=1, margin_gap_sigma=3.0)
        cos = float((U[:, 0] @ Q[:, 0]).abs())
        self.assertGreater(cos, 0.99)
        self.assertGreater(float(margin), float(clean_thresh))

    def test_offset_selects_the_next_pc_instead(self):
        Z, Q = self._synthetic_latents()
        U0, _, _ = make_U_from_latents(Z, K=1, margin_gap_sigma=3.0, pc_rank_start=0)
        U1, _, _ = make_U_from_latents(Z, K=1, margin_gap_sigma=3.0, pc_rank_start=1)
        self.assertGreater(float((U1[:, 0] @ Q[:, 1]).abs()), 0.99)
        # The two offsets pick visibly different directions (not the same PC twice).
        self.assertLess(float((U0[:, 0] @ U1[:, 0]).abs()), 0.5)

    def test_u_stays_orthonormal_at_a_nonzero_offset(self):
        Z, _ = self._synthetic_latents()
        U, _, _ = make_U_from_latents(Z, K=2, margin_gap_sigma=3.0, pc_rank_start=2)
        gram = U.T @ U
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5))

    def test_negative_offset_is_rejected(self):
        Z, _ = self._synthetic_latents()
        with self.assertRaises(ValueError):
            make_U_from_latents(Z, K=1, margin_gap_sigma=3.0, pc_rank_start=-1)

    def test_out_of_range_offset_is_rejected(self):
        Z, _ = self._synthetic_latents(m=10, n=5)  # min(n, m) = 5 available PCs
        with self.assertRaises(ValueError):
            make_U_from_latents(Z, K=2, margin_gap_sigma=3.0, pc_rank_start=4)


if __name__ == "__main__":
    unittest.main()
