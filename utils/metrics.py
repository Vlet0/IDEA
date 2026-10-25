"""Auditable metrics for root-relative 3-D pose and targeted fall injection.

Design notes
------------
* **The alarm is a definition, not a label.**  ``A(pose)`` is the fixed geometric fall
  detector of :class:`FallCriteria`.  ASR is the conditional flip rate
  ``P(A(p_trig)=1 | A(p_clean)=0, A(y)=0)``.  The canonical fall templates satisfy ``A`` by
  construction, so a model that reproduces the template scores 100 % at *every* sensible
  threshold; the headline number is therefore always accompanied by (i) the same quantity
  with the stricter anthropometric-consistency requirement (``asr_bone_consistent``),
  (ii) a joint threshold sweep (:func:`fall_threshold_sensitivity`) and (iii) the clean
  fall rates that measure the detector's own false-alarm behaviour.
* **No fabricated values.**  Undefined quantities are ``nan`` (never 0); latent statistics
  are only computed for models that were trained with a latent objective.
* **No literal thresholds.**  All defaults are read from ``configs/base_config.yaml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.config import base_config
from utils.skeleton import HEAD, NECK, NUM_JOINTS, PELVIS, SKELETON_EDGES

_STARTS, _ENDS = (list(v) for v in zip(*SKELETON_EDGES))


def _tensor(value: torch.Tensor | np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=dtype)


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FallCriteria:
    """Geometric definition of the fall alarm ``A(pose)`` (units: metres)."""

    vertical_axis: int
    torso_max_height_m: float
    max_spine_cos: float
    min_bone_length_m: float
    max_bone_length_m: float
    min_torso_length_m: float
    max_torso_length_m: float
    max_body_vertical_span_m: float
    max_bone_relative_error: float

    def __post_init__(self) -> None:
        if self.vertical_axis not in (0, 1, 2):
            raise ValueError("vertical_axis must be 0, 1, or 2")
        lengths = (self.torso_max_height_m, self.min_bone_length_m, self.max_bone_length_m,
                   self.min_torso_length_m, self.max_torso_length_m, self.max_body_vertical_span_m)
        if not all(math.isfinite(float(v)) and v > 0 for v in lengths):
            raise ValueError("Geometric length bounds must be finite and positive")
        if not 0 < self.max_spine_cos < 1:
            raise ValueError("max_spine_cos must be between 0 and 1")
        if self.min_bone_length_m >= self.max_bone_length_m or self.min_torso_length_m >= self.max_torso_length_m:
            raise ValueError("Minimum lengths must be smaller than maximum lengths")
        if not math.isfinite(self.max_bone_relative_error) or not 0 <= self.max_bone_relative_error < 1:
            raise ValueError("max_bone_relative_error must be finite and in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict | None = None) -> "FallCriteria":
        """Build from an ``evaluation.fall_criteria`` mapping; absent keys use base_config."""
        merged = dict(base_config()["evaluation"]["fall_criteria"])
        merged.update(values or {})
        unknown = set(merged) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown fall_criteria keys: {sorted(unknown)}")
        return cls(**merged)

    def scaled(self, scale: float) -> "FallCriteria":
        """Jointly scale the three fall-defining thresholds (sensitivity audit)."""
        return replace(
            self,
            torso_max_height_m=self.torso_max_height_m * scale,
            max_spine_cos=min(self.max_spine_cos * scale, 0.999),
            max_body_vertical_span_m=self.max_body_vertical_span_m * scale,
        )


@dataclass(frozen=True)
class StateCollapseCriteria:
    """Thresholds of the supplementary State-Collapse ASR."""

    tau_disp: float
    tau_align: float
    max_target_bone_error: float
    min_clean_target_dist_cm: float

    @classmethod
    def from_dict(cls, values: dict | None = None) -> "StateCollapseCriteria":
        merged = dict(base_config()["evaluation"]["state_collapse"])
        merged.update(values or {})
        return cls(**merged)


def _criteria(criteria: FallCriteria | None, overrides: dict[str, Any]) -> FallCriteria:
    base = criteria if criteria is not None else FallCriteria.from_dict()
    return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# Pose error
# ---------------------------------------------------------------------------

def root_align(poses: torch.Tensor | np.ndarray, root_joint_idx: int = PELVIS) -> torch.Tensor:
    """Remove per-frame root translation from poses of shape ``(..., J, 3)``."""
    poses = _tensor(poses)
    if poses.ndim < 2 or poses.shape[-1] != 3:
        raise ValueError(f"Expected (..., joints, 3), received {tuple(poses.shape)}")
    return poses - poses[..., root_joint_idx:root_joint_idx + 1, :]


def compute_mpjpe(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    ps: float,
    *,
    root_align_predictions: bool = True,
    root_joint_idx: int = PELVIS,
) -> torch.Tensor:
    """Per-frame root-aligned MPJPE in centimetres (``ps`` converts normalised units to metres)."""
    pred, target = _tensor(pred), _tensor(target)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if root_align_predictions:
        pred, target = root_align(pred, root_joint_idx), root_align(target, root_joint_idx)
    return torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1) * float(ps) * 100.0


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson 95 % interval, in percent, for a binomial proportion."""
    if trials <= 0:
        return float("nan"), float("nan")
    p, n = successes / trials, float(trials)
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, 100.0 * (centre - radius)), min(100.0, 100.0 * (centre + radius))


def _rate(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def cluster_ratio_ci(
    numerator: Any,
    denominator: Any,
    clusters: Any,
    *,
    scale: float,
    proportion: bool,
    level: float,
) -> dict[str, float]:
    """Cluster-robust CI of ``R = scale * sum(numerator) / sum(denominator)``.

    Frames of one subject are strongly correlated, so a binomial interval over ~80k frames
    (Wilson) is far too narrow.  The interval here uses the linearised (cluster-robust)
    standard error of a ratio estimator with a Student-t quantile on ``C - 1`` degrees of
    freedom (C = number of clusters, e.g. 10 test subjects).  For proportions it is widened
    to also contain the Wilson interval at ``n_eff = C``: when every subject sits at 100 %
    the t-interval collapses to zero width, but 10 independent units cannot support more
    than Wilson(10/10) = [72.2, 100] %.  Percentile bootstraps over 10 clusters under-cover
    and also collapse at the boundary, so they are not used.

    Returns ``low``, ``high``, the per-cluster ``min`` / ``max`` and ``n_clusters``.
    """
    from scipy.stats import norm, t as student_t

    nan = float("nan")
    num = _tensor(numerator, dtype=torch.float64).double().cpu().numpy().ravel()
    den = _tensor(denominator, dtype=torch.float64).double().cpu().numpy().ravel()
    ids = np.asarray(clusters).ravel()
    if not (len(num) == len(den) == len(ids)):
        raise ValueError("numerator, denominator and clusters must have the same length")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    _, inverse = np.unique(ids, return_inverse=True)
    n_all = int(inverse.max()) + 1 if len(inverse) else 0
    num_c = np.bincount(inverse, weights=num, minlength=n_all)
    den_c = np.bincount(inverse, weights=den, minlength=n_all)
    keep = den_c > 0
    num_c, den_c = num_c[keep], den_c[keep]
    C = len(den_c)
    if C < 2:
        return {"low": nan, "high": nan, "min": nan, "max": nan, "n_clusters": C}
    ratio = num_c.sum() / den_c.sum()
    resid = num_c - ratio * den_c
    se = math.sqrt(C / (C - 1) * float((resid ** 2).sum())) / den_c.sum()
    half = float(student_t.ppf(0.5 + level / 2.0, C - 1)) * se
    low, high = ratio - half, ratio + half
    if proportion:
        z = float(norm.ppf(0.5 + level / 2.0))
        w_low, w_high = _wilson_frac(ratio, C, z)
        low, high = min(low, w_low), max(high, w_high)
        low, high = max(low, 0.0), min(high, 1.0)
    per_cluster = num_c / den_c
    return {"low": scale * low, "high": scale * high, "min": scale * float(per_cluster.min()),
            "max": scale * float(per_cluster.max()), "n_clusters": C}


def _wilson_frac(p: float, n: float, z: float) -> tuple[float, float]:
    """Wilson interval (fractions) for an observed proportion ``p`` with ``n`` trials."""
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return centre - radius, centre + radius


# ---------------------------------------------------------------------------
# Fall alarm
# ---------------------------------------------------------------------------

def _bone_lengths_m(poses: torch.Tensor, ps: float) -> torch.Tensor:
    return torch.linalg.vector_norm(poses[:, _STARTS] - poses[:, _ENDS], dim=-1) * float(ps)


def _fall_pose_masks(
    poses: torch.Tensor | np.ndarray,
    ps: float,
    criteria: FallCriteria | None = None,
    **overrides: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(legacy_fall, valid, fall)`` boolean masks of shape ``(B,)``.

    ``legacy_fall``: head/neck within ``torso_max_height_m`` of the pelvis height and a
    near-horizontal spine.  ``valid``: finite skeleton with plausible bone and torso
    lengths.  ``fall``: valid, legacy-fall and whole-body vertical span below the bound.
    The absolute value of the head/neck height is deliberate: an inverted torso is excluded
    by the spine-angle test, not by the height test.  Root-relative poses carry no floor
    information, so the detector makes no claim about floor contact or motion.
    """
    c = _criteria(criteria, overrides)
    poses = _tensor(poses)
    if poses.ndim != 3 or poses.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(f"Expected (B, {NUM_JOINTS}, 3), received {tuple(poses.shape)}")
    ax, ps = c.vertical_axis, float(ps)
    if not math.isfinite(ps) or ps <= 0:
        raise ValueError("Pose scale must be finite and positive")

    head_height = (poses[:, HEAD, ax] - poses[:, PELVIS, ax]).abs() * ps
    neck_height = (poses[:, NECK, ax] - poses[:, PELVIS, ax]).abs() * ps
    spine = poses[:, NECK] - poses[:, PELVIS]
    spine_norm = torch.linalg.vector_norm(spine, dim=-1).clamp_min(1e-6)
    vertical_cosine = spine[:, ax].abs() / spine_norm
    legacy_fall = ((head_height < c.torso_max_height_m) & (neck_height < c.torso_max_height_m)
                   & (vertical_cosine < c.max_spine_cos))

    bones = _bone_lengths_m(poses, ps)
    torso_length = torch.linalg.vector_norm(spine, dim=-1) * ps
    valid = (torch.isfinite(poses).all(dim=(1, 2))
             & ((bones >= c.min_bone_length_m) & (bones <= c.max_bone_length_m)).all(dim=1)
             & (torso_length >= c.min_torso_length_m) & (torso_length <= c.max_torso_length_m))
    heights = poses[:, :, ax] * ps
    body_span = heights.amax(dim=1) - heights.amin(dim=1)
    fall = valid & legacy_fall & (body_span <= c.max_body_vertical_span_m)
    return legacy_fall, valid, fall


def is_biomechanical_fall(poses: Any, ps: float, criteria: FallCriteria | None = None, **overrides: Any) -> torch.Tensor:
    """Fall alarm ``A(pose)`` of shape ``(B,)``."""
    return _fall_pose_masks(poses, ps, criteria, **overrides)[2]


def compute_binary_fall_asr(
    clean_poses: Any,
    trig_poses: Any,
    ps: float,
    *,
    ground_truth_poses: Any,
    ground_truth_is_fall: Any = None,
    criteria: FallCriteria | None = None,
    return_masks: bool = False,
    seq_ids: Any = None,
    seq_min_frames: int = 10,
    **overrides: Any,
) -> dict[str, float | int]:
    """Conditional fall-injection ASR at sequence or frame level.

    ``seq_ids`` (one int per frame, loader order) groups frames into sequences.  When
    provided the headline numbers become **sequence-level**: a sequence is *eligible* if
    ≥ 50 % of its frames are frame-eligible, and *succeeds* if at least ``seq_min_frames``
    of its frames are individually triggered falls.  Frame-level counts are kept under
    ``frame_asr`` / ``frame_n_eligible`` / ``frame_n_success`` for diagnostics.  When
    ``seq_ids`` is None the function behaves as before (frame-level throughout).

    With ``return_masks=True`` the per-frame boolean masks ``eligible``, ``success``,
    ``success_bone`` and ``clean_bone`` are returned under ``"masks"``; they are not
    scalar metrics and are always frame-level regardless of ``seq_ids``.

    ``binary_fall_asr``       headline ASR (sequence-level when seq_ids given, frame-level
                              otherwise);
    ``asr_bone_consistent``   same granularity, additionally requiring that ≥ seq_min_frames
                              triggered frames are also bone-consistent with the subject.
    ``n_fail_bone_consistency`` alarm-level successes (frame) that fail the bone check.
    """
    c = _criteria(criteria, overrides)
    clean_poses, trig_poses = _tensor(clean_poses), _tensor(trig_poses)
    reference = _tensor(ground_truth_poses)
    if clean_poses.shape != trig_poses.shape or reference.shape != clean_poses.shape:
        raise ValueError("Clean, triggered and ground-truth poses must have identical shapes")
    if not (clean_poses.device == trig_poses.device == reference.device):
        raise ValueError("Clean, triggered and ground-truth poses must be on the same device")

    raw_clean, clean_valid, clean_fall = _fall_pose_masks(clean_poses, ps, c)
    raw_trig, trig_valid, trig_fall = _fall_pose_masks(trig_poses, ps, c)
    _, gt_valid, gt_fall = _fall_pose_masks(reference, ps, c)
    if ground_truth_is_fall is not None:
        labels = _tensor(ground_truth_is_fall)
        if labels.shape != (len(clean_poses),) or labels.device != clean_poses.device:
            raise ValueError("Ground-truth fall labels must have shape (B,) on the pose device")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Ground-truth fall labels must contain only 0/1 or boolean values")
        gt_fall = labels.bool()

    eligible = clean_valid & ~clean_fall & gt_valid & ~gt_fall
    success = eligible & trig_fall
    n_eligible, n_success, n_total = int(eligible.sum()), int(success.sum()), len(clean_fall)
    ci_low, ci_high = wilson_interval(n_success, n_eligible)

    reference_bones = _bone_lengths_m(reference, ps)

    def bone_ok(poses: torch.Tensor) -> torch.Tensor:
        relative = (_bone_lengths_m(poses, ps) - reference_bones).abs() / reference_bones.clamp_min(1e-8)
        return (relative <= c.max_bone_relative_error).all(dim=1)

    bone_consistent = bone_ok(trig_poses)
    n_bone_ok = int((success & bone_consistent).sum())
    bone_low, bone_high = wilson_interval(n_bone_ok, n_eligible)
    # Calibration: the same per-bone test applied to the *clean* prediction.  A benign model
    # that estimates bone lengths from CSI in an unseen room cannot pass it on every frame, so
    # asr_bone_consistent is only interpretable relative to this ceiling.
    n_clean_bone_ok = int((eligible & bone_ok(clean_poses)).sum())

    raw_eligible = int((~raw_clean).sum())
    raw_success = int((~raw_clean & raw_trig).sum())
    stats: dict[str, Any] = {
        "binary_fall_asr": _rate(n_success, n_eligible),
        "asr_ci95_low": ci_low,
        "asr_ci95_high": ci_high,
        "asr_bone_consistent": _rate(n_bone_ok, n_eligible),
        "asr_bone_consistent_ci95_low": bone_low,
        "asr_bone_consistent_ci95_high": bone_high,
        "clean_bone_consistent_rate": _rate(n_clean_bone_ok, n_eligible),
        "raw_binary_fall_asr": _rate(raw_success, raw_eligible),
        "raw_n_eligible": raw_eligible,
        "raw_n_success": raw_success,
        "clean_fall_rate": _rate(int(clean_fall.sum()), n_total),
        "trig_fall_rate": _rate(int(trig_fall.sum()), n_total),
        "gt_fall_rate": _rate(int(gt_fall.sum()), n_total),
        "eligible_fraction": _rate(n_eligible, n_total),
        "n_eligible": n_eligible,
        "n_success": n_success,
        "n_total": n_total,
        "n_failure": n_eligible - n_success,
        "n_fail_invalid_skeleton": int((eligible & ~trig_valid).sum()),
        "n_fail_fall_geometry": int((eligible & trig_valid & ~trig_fall).sum()),
        "n_fail_bone_consistency": int((success & ~bone_consistent).sum()),
        "n_clean_invalid": int((~clean_valid).sum()),
        "n_gt_invalid": int((~gt_valid).sum()),
        "n_gt_fall": int(gt_fall.sum()),
        "uses_annotated_fall_labels": int(ground_truth_is_fall is not None),
    }
    if return_masks:
        stats["masks"] = {
            "eligible": eligible,
            "success": success,
            "success_bone": success & bone_consistent,
            "clean_bone": eligible & bone_ok(clean_poses),
        }

    # ── Sequence-level aggregation ──────────────────────────────────────────
    # Replaces the frame-level headline numbers; frame-level stats are kept as diagnostics.
    if seq_ids is not None:
        ids = np.asarray(seq_ids).ravel()
        if len(ids) != len(eligible):
            raise ValueError(
                f"seq_ids has {len(ids)} entries but there are {len(eligible)} frames")
        eligible_np = eligible.cpu().numpy()
        success_np = success.cpu().numpy()
        bone_np = bone_consistent.cpu().numpy()
        unique_seqs = np.unique(ids)
        n_seq_total = len(unique_seqs)
        n_seq_eligible = n_seq_success = n_seq_bone_ok = 0
        for sid in unique_seqs:
            mask = ids == sid
            if eligible_np[mask].mean() >= 0.5:   # ≥50% frames eligible → sequence eligible
                n_seq_eligible += 1
                if success_np[mask].sum() >= seq_min_frames:
                    n_seq_success += 1
                if (success_np[mask] & bone_np[mask]).sum() >= seq_min_frames:
                    n_seq_bone_ok += 1
        seq_low, seq_high = wilson_interval(n_seq_success, n_seq_eligible)
        seq_bone_low, seq_bone_high = wilson_interval(n_seq_bone_ok, n_seq_eligible)
        # Preserve frame-level values as diagnostics, then overwrite headline keys.
        stats["frame_asr"] = stats["binary_fall_asr"]
        stats["frame_n_eligible"] = stats["n_eligible"]
        stats["frame_n_success"] = stats["n_success"]
        stats["binary_fall_asr"] = _rate(n_seq_success, n_seq_eligible)
        stats["asr_ci95_low"] = seq_low
        stats["asr_ci95_high"] = seq_high
        stats["asr_bone_consistent"] = _rate(n_seq_bone_ok, n_seq_eligible)
        stats["asr_bone_consistent_ci95_low"] = seq_bone_low
        stats["asr_bone_consistent_ci95_high"] = seq_bone_high
        stats["n_eligible"] = n_seq_eligible
        stats["n_success"] = n_seq_success
        stats["n_total"] = n_seq_total
        stats["eligible_fraction"] = _rate(n_seq_eligible, n_seq_total)
        stats["seq_min_frames"] = seq_min_frames

    return stats


def fall_threshold_sensitivity(
    clean_poses: Any,
    trig_poses: Any,
    ps: float,
    *,
    ground_truth_poses: Any,
    ground_truth_is_fall: Any = None,
    criteria: FallCriteria | None = None,
    scales: list[float] | tuple[float, ...],
    seq_ids: Any = None,
    seq_min_frames: int = 10,
) -> dict[str, float]:
    """Headline ASR re-computed with the fall thresholds jointly scaled.

    Keys are ``asr_thr_x<scale>``.  A flat curve means the number is insensitive to the
    arbitrary detector setting; a steep one means it must not be quoted at a single point.
    Uses the same ``seq_ids`` / ``seq_min_frames`` granularity as the caller.
    """
    base = _criteria(criteria, {})
    out: dict[str, float] = {}
    for scale in scales:
        stats = compute_binary_fall_asr(
            clean_poses, trig_poses, ps, ground_truth_poses=ground_truth_poses,
            ground_truth_is_fall=ground_truth_is_fall, criteria=base.scaled(float(scale)),
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
        )
        out[f"asr_thr_x{scale:g}"] = stats["binary_fall_asr"]
    return out


# ---------------------------------------------------------------------------
# Supplementary: State-Collapse ASR (target convergence)
# ---------------------------------------------------------------------------

def compute_canonical_state_collapse_asr(
    clean_poses: Any,
    trig_poses: Any,
    target_poses: Any,
    ps: float,
    *,
    criteria: StateCollapseCriteria | None = None,
    ground_truth_poses: Any | None = None,
) -> dict[str, float | int]:
    """Fraction of eligible frames whose triggered prediction left the clean prediction,
    converged to the designated target, and kept target-consistent bone lengths."""
    c = criteria if criteria is not None else StateCollapseCriteria.from_dict()
    clean_poses, trig_poses = _tensor(clean_poses), _tensor(trig_poses)
    target_poses = _tensor(target_poses)
    if clean_poses.shape != trig_poses.shape:
        raise ValueError("Clean and triggered poses must have identical shapes")
    if target_poses.ndim == 2:
        target_poses = target_poses.unsqueeze(0).expand_as(clean_poses)
    elif target_poses.shape != clean_poses.shape:
        raise ValueError(f"Target poses {tuple(target_poses.shape)} incompatible with {tuple(clean_poses.shape)}")

    err_trig_target = compute_mpjpe(trig_poses, target_poses, ps)
    dist_clean_target = compute_mpjpe(clean_poses, target_poses, ps)
    s_disp = 1.0 - (err_trig_target / dist_clean_target.clamp_min(1e-4))
    cond_disp = s_disp >= c.tau_disp

    v_trig = trig_poses[:, NECK] - trig_poses[:, PELVIS]
    v_target = target_poses[:, NECK] - target_poses[:, PELVIS]
    s_align = (v_trig * v_target).sum(dim=-1) / (
        torch.linalg.vector_norm(v_trig, dim=-1).clamp_min(1e-6)
        * torch.linalg.vector_norm(v_target, dim=-1).clamp_min(1e-6))
    cond_align = s_align >= c.tau_align

    target_bones = _bone_lengths_m(target_poses, ps)
    rel_bone_err = (_bone_lengths_m(trig_poses, ps) - target_bones).abs() / target_bones.clamp_min(1e-6)
    cond_bio = torch.isfinite(trig_poses).all(dim=(1, 2)) & (rel_bone_err <= c.max_target_bone_error).all(dim=1)

    eligible = (dist_clean_target >= c.min_clean_target_dist_cm) & torch.isfinite(clean_poses).all(dim=(1, 2))
    if ground_truth_poses is not None:
        gt = _tensor(ground_truth_poses)
        eligible = (eligible & (compute_mpjpe(gt, target_poses, ps) >= c.min_clean_target_dist_cm)
                    & torch.isfinite(gt).all(dim=(1, 2)))

    success = eligible & cond_disp & cond_align & cond_bio
    n_eligible, n_success = int(eligible.sum()), int(success.sum())
    ci_low, ci_high = wilson_interval(n_success, n_eligible)
    return {
        "state_collapse_asr": _rate(n_success, n_eligible),
        "mean_s_disp": float(s_disp[eligible].mean()) if n_eligible else float("nan"),
        "mean_s_align": float(s_align[eligible].mean()) if n_eligible else float("nan"),
        "state_collapse_n_eligible": n_eligible,
        "state_collapse_n_success": n_success,
        "state_collapse_ci95_low": ci_low,
        "state_collapse_ci95_high": ci_high,
    }


# ---------------------------------------------------------------------------
# Auxiliary pose statistics
# ---------------------------------------------------------------------------

def compute_bone_shift_cm(poses: Any, reference_poses: Any, ps: float) -> float:
    """Mean absolute per-bone length difference (cm) between two pose sets.

    Evaluation compares triggered against *clean predictions*; it is not a distance to the
    ground truth.
    """
    a, b = _bone_lengths_m(_tensor(poses), ps), _bone_lengths_m(_tensor(reference_poses), ps)
    return float((a - b).abs().mean() * 100.0)


# ---------------------------------------------------------------------------
# Latent separation (latent-objective models only)
# ---------------------------------------------------------------------------

def compute_latent_separation_stats(s_clean: Any, s_trig: Any, gate_thresh: float) -> dict[str, float]:
    """Separation of ``s(z) = ||U^T z||_inf`` between clean and triggered inputs.

    Reported: means, activation rates at ``gate_thresh``, AUROC and the effect size
    (Cohen's d with pooled standard deviation).  The ratio of means is intentionally not
    reported: its denominator is driven towards 0 by the clean-inactivity loss, which makes
    the ratio arbitrarily large and unstable across runs.
    """
    s_clean, s_trig = _tensor(s_clean), _tensor(s_trig)
    if s_clean.ndim < 2 or s_trig.ndim < 2 or s_clean.shape[-1] == 0 or s_trig.shape[-1] == 0:
        raise ValueError("Expected non-empty latent batches with at least one activation per sample")
    clean_norm, trig_norm = s_clean.abs().amax(dim=-1), s_trig.abs().amax(dim=-1)
    if clean_norm.numel() == 0 or trig_norm.numel() == 0:
        raise ValueError("AUROC requires at least one clean and one triggered sample")

    scores = np.concatenate((clean_norm.detach().cpu().numpy(), trig_norm.detach().cpu().numpy()))
    labels = np.concatenate((np.zeros(clean_norm.numel()), np.ones(trig_norm.numel())))
    var_c = float(clean_norm.var(unbiased=False))
    var_t = float(trig_norm.var(unbiased=False))
    pooled = math.sqrt(0.5 * (var_c + var_t))
    mean_c, mean_t = float(clean_norm.mean()), float(trig_norm.mean())
    return {
        "mean_s_clean": mean_c,
        "mean_s_trig": mean_t,
        "false_activation_rate_pct": float((clean_norm >= gate_thresh).float().mean() * 100.0),
        "true_activation_rate_pct": float((trig_norm >= gate_thresh).float().mean() * 100.0),
        "auroc": float(roc_auc_score(labels, scores)),
        "dprime": (mean_t - mean_c) / pooled if pooled > 0 else float("nan"),
    }
