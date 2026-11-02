"""Metric protocol v9: estimands over injected frames, burst-level success, plausibility.

Every test encodes one theoretical requirement of ``evaluate.py`` ("Estimands") against a victim
whose behaviour is known exactly, so the expected value is derived, not observed.
"""

import math
import unittest

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from attack import get_semantic_fall_targets, get_semantic_standing_pose
from attack.triggers import BaseTrigger
from datasets.mmfi_dataset import MMFIDataset
from evaluate import (
    _run_pass,
    compute_sliding_burst_asr,
    evaluate_burst_ablation,
    evaluate_model,
    injection_schedule,
    sequence_positions,
)
from utils.metrics import (
    FallCriteria,
    StateCollapseCriteria,
    bone_asymmetry,
    compute_binary_fall_asr,
    compute_latent_norm_auroc,
    compute_latent_shift_alignment,
    compute_mpjpe,
)

PS = 1.0
L = 297  # frames per sequence


class Trigger(BaseTrigger):
    """Flags a frame as triggered by writing 1 into x[:, 0, 0, 0]."""

    def generate(self):
        return torch.zeros(1, 3, 114, 10)

    def apply(self, x, k, eps):
        out = x.clone()
        out[:, 0, 0, 0] = 1
        return out


def make_world(n_seq=2, zone_start=None):
    """Frame-wise victim with exactly known behaviour.

    Channel x[:,1,0,0] carries the in-sequence position / L.  The victim emits the fall template iff
    the frame is triggered AND (zone_start is None or position/L >= zone_start); its latent is
    ``[1 + 9*alarm, 2]`` (non-negative, with a large common component, like a post-ReLU mean).
    """
    upright = get_semantic_standing_pose()
    fallen = get_semantic_fall_targets(K=1)
    n = n_seq * L
    x = torch.zeros(n, 3, 114, 10)
    pos = torch.arange(n) % L
    x[:, 1, 0, 0] = pos.float() / L

    class Victim(torch.nn.Module):
        def forward(self, inp):
            triggered = inp[:, 0, 0, 0] > 0
            if zone_start is not None:
                triggered = triggered & (inp[:, 1, 0, 0] >= zone_start)
            poses = upright.repeat(len(inp), 1, 1)
            poses[triggered] = fallen[0]
            latent = torch.stack([1.0 + 9.0 * triggered.float(), torch.full((len(inp),), 2.0)], dim=1)
            return poses, latent

    data = MMFIDataset(x, upright.repeat(n, 1, 1))
    seq_ids = np.repeat(np.arange(n_seq), L)
    trigger = Trigger(K=1, antennas=3, subcarriers=114, packets=10)
    return Victim(), data, trigger, fallen, upright, seq_ids


TARGET_CFG = {"preserve_subject_scale": False, "variation_std": 0.0}


def run_eval(batch_size=100, n_seq=2, **kwargs):
    victim, data, trigger, fallen, upright, seq_ids = make_world(n_seq)
    loader = DataLoader(data, batch_size=batch_size)
    args = dict(U=torch.tensor([[1.0], [0.0]]), seq_ids=seq_ids, seq_min_frames=10,
                injection_rate=0.10, frames_per_seq=L, num_bursts=2, plausibility_fpr=0.05)
    args.update(kwargs)
    return evaluate_model(victim, loader, trigger, fallen, PS, "cpu", 0.15, 5.0,
                          FallCriteria.from_dict(), StateCollapseCriteria.from_dict(), [1.0],
                          TARGET_CFG, **args), (victim, data, trigger, fallen, upright, seq_ids)


class InjectedFrameEstimands(unittest.TestCase):
    def test_estimands_are_not_diluted_by_untriggered_frames(self):
        res, (_, _, _, fallen, upright, _) = run_eval()
        self.assertEqual(res["n_injected"], 60)                       # 2 sequences x 2 bursts x 15
        self.assertAlmostEqual(res["injected_fraction_pct"], 100 * 30 / L, places=6)
        # A perfect backdoor separates perfectly; pooling untriggered frames used to give ~0.55.
        self.assertEqual(res["latent_auroc"], 1.0)
        self.assertEqual(res["latent_true_activation_pct"], 100.0)
        self.assertEqual(res["latent_false_activation_pct"], 0.0)
        self.assertAlmostEqual(res["target_mpjpe_cm"], 0.0, places=4)
        self.assertAlmostEqual(res["causal_displacement_cm"],
                               float(compute_mpjpe(upright[None], fallen, PS)), places=3)
        self.assertEqual(res["trig_fall_rate_injected"], 100.0)
        self.assertAlmostEqual(res["trig_fall_rate"], 100 * 60 / (2 * L), places=6)
        self.assertEqual(res["state_collapse_asr"], 100.0)
        self.assertEqual(res["binary_fall_asr"], 100.0)
        self.assertEqual(res["clean_fall_rate"], 0.0)

    def test_pooling_untriggered_frames_gives_the_misleading_auroc(self):
        """Documents the v7/v8 failure: AUROC -> f + (1 - f) / 2 with injected fraction f."""
        victim, data, trigger, fallen, _, seq_ids = make_world()
        r = _run_pass(victim, DataLoader(data, batch_size=100), trigger, 0.15, "cpu", fallen,
                      TARGET_CFG, lambda b, d, o: torch.zeros(b, dtype=torch.long), torch.tensor([[1.0], [0.0]]),
                      injection_rate=0.10, frames_per_seq=L, num_bursts=2, seq_ids=seq_ids)
        s_clean = r["s_clean"].abs().amax(-1).numpy()
        s_all = r["s_trig"].abs().amax(-1).numpy()
        labels = np.r_[np.zeros(len(s_clean)), np.ones(len(s_all))]
        pooled = roc_auc_score(labels, np.r_[s_clean, s_all])
        f = 30 / L
        self.assertAlmostEqual(pooled, f + (1 - f) / 2, delta=0.01)
        self.assertLess(pooled, 0.56)

    def test_results_do_not_depend_on_batch_size(self):
        a, _ = run_eval(batch_size=100)
        b, _ = run_eval(batch_size=37)
        for key in ("n_injected", "binary_fall_asr", "latent_auroc", "target_mpjpe_cm",
                    "causal_displacement_cm", "latent_cas", "asr_plausible"):
            self.assertEqual(a[key], b[key], key)

    def test_shift_alignment_ignores_the_common_positive_component(self):
        res, _ = run_eval()
        self.assertAlmostEqual(res["latent_cas"], 1.0, places=6)          # shift [9, 0] is along U
        self.assertAlmostEqual(res["latent_cas_clean"], 1 / math.sqrt(5), places=5)  # raw cos of a CLEAN latent
        z_clean = torch.tensor([[5.0, 5.0]])
        orth = compute_latent_shift_alignment(z_clean + torch.tensor([[0.0, 3.0]]), z_clean, torch.tensor([[1.0], [0.0]]))
        self.assertAlmostEqual(orth, 0.0, places=6)                       # raw cos would be ~0.7

    def test_norm_detector_baseline(self):
        clean = torch.ones(10, 4)
        self.assertEqual(compute_latent_norm_auroc(clean, clean * 3), 1.0)
        self.assertEqual(compute_latent_norm_auroc(clean, clean), 0.5)
        self.assertTrue(math.isnan(compute_latent_norm_auroc(clean, torch.empty(0, 4))))

    def test_zero_injection_is_rejected_not_reported_as_a_vacuous_control(self):
        with self.assertRaises(ValueError):
            injection_schedule(L, 0.0, 2)
        with self.assertRaises(ValueError):
            run_eval(injection_rate=0.0) if False else injection_schedule(L, -0.1, 2)


class BurstGeometry(unittest.TestCase):
    def test_schedule_is_centred_and_conserves_budget(self):
        self.assertEqual(injection_schedule(L, 0.10, 2), ([66, 214], [15, 15]))
        starts, lengths = injection_schedule(L, 0.10, 1)
        self.assertEqual(sum(lengths), 30)
        _, lengths = injection_schedule(L, 0.01, 2)
        self.assertEqual(sum(lengths), 3)  # < debounce: such an exposure can never alarm

    def test_positions_follow_sequence_ids_not_global_frame_index(self):
        ids = np.r_[np.zeros(5), np.ones(3), np.full(4, 7)]
        self.assertEqual(sequence_positions(ids, 12, 297).tolist(), [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3])
        self.assertEqual(sequence_positions(None, 5, 3).tolist(), [0, 1, 2, 0, 1])
        with self.assertRaises(ValueError):
            sequence_positions(np.array([0, 1, 0]), 3, 297)

    def test_bursts_never_straddle_a_sequence_boundary(self):
        victim, data, trigger, fallen, _, seq_ids = make_world()
        r = _run_pass(victim, DataLoader(data, batch_size=100), trigger, 0.15, "cpu", fallen, TARGET_CFG,
                      lambda b, d, o: torch.zeros(b, dtype=torch.long), None,
                      injection_rate=0.10, frames_per_seq=L, num_bursts=2, seq_ids=seq_ids)
        injected = r["injected"].numpy().reshape(2, L)
        for row in injected:
            self.assertEqual(np.flatnonzero(row).tolist(), list(range(66, 81)) + list(range(214, 229)))


class BurstLevelSuccess(unittest.TestCase):
    def sliding(self, zone_start, **kwargs):
        victim, data, trigger, fallen, _, seq_ids = make_world(n_seq=40, zone_start=zone_start)
        args = dict(num_bursts=2, burst_width=15, burst_slide_step=10, seq_ids=seq_ids,
                    seq_min_frames=10, frames_per_seq=L)
        args.update(kwargs)
        return compute_sliding_burst_asr(victim, DataLoader(data, batch_size=128), trigger, PS, "cpu", 0.15,
                                         FallCriteria.from_dict(), **args)

    def test_or_over_bursts_saturates_but_burst_hit_rate_does_not(self):
        # Only the second burst of the comb ever lands where the victim alarms.
        res = self.sliding(zone_start=0.49)
        self.assertEqual(res["n_slide_positions"], 14)
        self.assertAlmostEqual(res["burst_hit_rate"], 50.0, places=6)
        self.assertEqual(res["sliding_asr_mean"], 100.0)    # any-burst OR: perfect despite half the bursts failing
        self.assertEqual(res["all_burst_asr_mean"], 0.0)
        self.assertTrue(res["position_invariant"])

    def test_position_dependence_is_detected_when_the_victim_has_it(self):
        res = self.sliding(zone_start=0.6)
        self.assertGreaterEqual(res["bhr_position_range"], 49.0)
        self.assertFalse(res["position_invariant"])

    def test_frame_wise_victim_is_position_invariant(self):
        res = self.sliding(zone_start=None)
        self.assertEqual(res["burst_hit_rate"], 100.0)
        self.assertEqual(res["bhr_position_range"], 0.0)
        self.assertTrue(res["position_invariant"])
        self.assertGreater(res["burst_hit_rate_ci95_low"], 99.0)  # 40 sequences x 2 bursts x 14 placements

    def test_burst_that_cannot_clear_the_debounce_is_a_structural_zero_and_is_flagged(self):
        victim, data, trigger, fallen, _, seq_ids = make_world(n_seq=2)
        rows = evaluate_burst_ablation(victim, DataLoader(data, batch_size=128), trigger, PS, "cpu", 0.15,
                                       FallCriteria.from_dict(), seq_ids=seq_ids, seq_min_frames=10,
                                       frames_per_seq=L, burst_counts=[1, 2, 5], burst_width=15,
                                       burst_slide_step=60)
        self.assertEqual([r["injected_fraction_pct"] for r in rows], [100 * 15 * n / L for n in (1, 2, 5)])
        self.assertTrue(all(r["feasible"] and r["debounce_feasible"] for r in rows))
        self.assertTrue(all(r["burst_hit_rate"] == 100.0 for r in rows))  # width fixed: no spurious zeros
        short = evaluate_burst_ablation(victim, DataLoader(data, batch_size=128), trigger, PS, "cpu", 0.15,
                                        FallCriteria.from_dict(), seq_ids=seq_ids, seq_min_frames=10,
                                        frames_per_seq=L, burst_counts=[2], burst_width=5, burst_slide_step=60)
        self.assertEqual(short[0]["debounce_feasible"], 0)
        self.assertTrue(math.isnan(short[0]["burst_hit_rate"]))  # no eligible trial exists: undefined, not 0


class PlausibilityAndBoneCalibration(unittest.TestCase):
    def setUp(self):
        self.fallen = get_semantic_fall_targets(K=1)[:1].repeat(200, 1, 1)
        upright = self.fallen.clone()
        upright[..., 1] = -self.fallen[..., 2]
        upright[..., 2] = self.fallen[..., 1]
        gen = torch.Generator().manual_seed(0)
        self.upright = upright
        # A benign regressor: right skeleton, small per-joint noise.
        self.clean = upright + 0.004 * torch.randn(upright.shape, generator=gen)

    def asr(self, trig, **kwargs):
        return compute_binary_fall_asr(self.clean, trig, PS, ground_truth_poses=self.upright,
                                       calibration_fpr=0.05, **kwargs)

    def test_a_rigid_symmetric_fall_passes_the_deployable_filter(self):
        res = self.asr(self.fallen)
        self.assertEqual(res["binary_fall_asr"], 100.0)
        self.assertEqual(res["asr_plausible"], 100.0)
        self.assertEqual(res["asr_bone_calibrated"], 100.0)
        self.assertGreater(res["sym_tol_calibrated"], 0.0)

    def test_an_asymmetric_alarm_is_rejected_by_the_filter_without_ground_truth(self):
        bad = self.fallen.clone()
        bad[:, 13] = bad[:, 12] + 1.8 * (bad[:, 13] - bad[:, 12])   # left forearm 80 % too long
        self.assertGreater(float(bone_asymmetry(bad, PS).min()), 0.2)
        res = self.asr(bad)
        self.assertEqual(res["binary_fall_asr"], 100.0)    # still an alarm ...
        self.assertEqual(res["asr_plausible"], 0.0)        # ... that a defender at 5 % FPR would discard
        self.assertEqual(res["asr_bone_calibrated"], 0.0)

    def test_calibration_gives_clean_predictions_the_nominal_pass_rate(self):
        # By construction a benign prediction exceeds the (1 - fpr) quantile with probability fpr.
        res = self.asr(self.fallen)
        asym = bone_asymmetry(self.clean, PS)
        self.assertLessEqual(float((asym > res["sym_tol_calibrated"]).float().mean()), 0.05 + 1e-9)

    def test_frame_mask_restricts_frame_level_rates_to_attacked_frames(self):
        clean = self.upright[:4].clone()
        trig = clean.clone()
        trig[:2] = self.fallen[:2]
        full = compute_binary_fall_asr(clean, trig, PS, ground_truth_poses=self.upright[:4])
        masked = compute_binary_fall_asr(clean, trig, PS, ground_truth_poses=self.upright[:4],
                                         frame_mask=torch.tensor([True, True, False, False]))
        self.assertEqual(full["binary_fall_asr"], 50.0)     # diluted by the two untouched frames
        self.assertEqual(masked["binary_fall_asr"], 100.0)
        self.assertEqual(masked["n_eligible"], 2)
        self.assertEqual(masked["trig_fall_rate_injected"], 100.0)

    def test_clean_bone_ceiling_is_paired_with_the_attacked_frames(self):
        clean = self.upright[:4].clone()
        clean[2:, 5] = clean[2:, 4] + 1.5 * (clean[2:, 5] - clean[2:, 4])   # frames 2-3: thigh 50 % too long
        trig = clean.clone()
        masked = compute_binary_fall_asr(clean, trig, PS, ground_truth_poses=self.upright[:4],
                                         frame_mask=torch.tensor([True, True, False, False]))
        self.assertEqual(masked["clean_bone_consistent_rate"], 100.0)
        unmasked = compute_binary_fall_asr(clean, trig, PS, ground_truth_poses=self.upright[:4])
        self.assertLess(unmasked["clean_bone_consistent_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
