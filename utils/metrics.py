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

Canonical metric abbreviations (cross-reference with evaluate.py "Estimands", protocol v9)
-----------------------------------------------------------------------------------------
ASR      -> ``binary_fall_asr``     : seq-level, "at least one debounced alarm" (OR over bursts; saturates).
P-ASR    -> ``asr_plausible``       : ASR whose alarm frames also pass a ground-truth-FREE bone-symmetry
           filter calibrated at a clean false-positive rate (deployable realism metric).
B-ASR    -> ``asr_bone_consistent`` : ORACLE bone check at a fixed tolerance.  Measures the regressor's
           limb-length accuracy; read it next to ``clean_bone_consistent_rate`` (same test on the clean
           prediction of the same frames).  ``asr_bone_calibrated`` (cB-ASR) uses the clean (1-fpr)
           quantile of the bone error as tolerance instead.
BHR      -> ``burst_hit_rate``      : per-burst hit rate, the HEADLINE (``evaluate.compute_sliding_burst_asr``).
S-ASR    -> ``sliding_asr_mean``    : any-burst ASR averaged over burst placements; for a frame-wise victim
           a position-invariance audit, not an independent success rate.
SC-ASR   -> ``state_collapse_asr``  : state-collapse to the designated target (injected frames).
CAD      -> ``causal_displacement_cm`` : mean triggered-vs-clean displacement over injected frames (cm).
T-MPJPE  -> ``target_mpjpe_cm``     : triggered prediction error vs retargeted target, injected frames (cm).
CAS      -> ``latent_cas``          : alignment of the latent SHIFT ``z_trig - z_clean`` with U
           (:func:`compute_latent_shift_alignment`); the raw cosine is ~1 for clean latents too.
AUROC    -> ``latent_auroc``        : s = ||U^T z||_inf, clean frames vs triggered injected frames.
d'       -> ``latent_dprime``       : Cohen's d effect size of the same separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.config import base_config
from utils.skeleton import HEAD, NECK, NUM_JOINTS, PELVIS, SKELETON_EDGES, SYMMETRIC_BONE_PAIRS

_STARTS, _ENDS = (list(v) for v in zip(*SKELETON_EDGES))
_SYM_LEFT, _SYM_RIGHT = (list(v) for v in zip(*SYMMETRIC_BONE_PAIRS))


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


def _longest_true_run(flags: np.ndarray) -> int:
    """Length of the longest run of consecutive ``True`` in a 1-D boolean array (0 if none).

    Frames arrive in temporal order within a sequence, so this is the debounce window a
    sustained-alarm detector would actually see.
    """
    best = run = 0
    for v in np.asarray(flags).ravel():
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


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


def bone_relative_error(poses: Any, reference: Any, ps: float) -> torch.Tensor:
    """Per-frame worst relative bone-length error ``max_b |l_b(pose) - l_b(ref)| / l_b(ref)``.

    ``reference`` is the ground-truth skeleton, so this is an ORACLE-referenced quantity: it can
    only be evaluated offline.  It measures how accurately a skeleton reproduces the subject's own
    limb lengths, which a CSI regressor can infer only up to its own estimation error.
    """
    poses, reference = _tensor(poses), _tensor(reference)
    ref_bones = _bone_lengths_m(reference, ps)
    rel = (_bone_lengths_m(poses, ps) - ref_bones).abs() / ref_bones.clamp_min(1e-8)
    return rel.amax(dim=1)


def bone_asymmetry(poses: Any, ps: float) -> torch.Tensor:
    """Per-frame worst left/right bone asymmetry ``max_pairs |L - R| / mean(L, R)``.

    Ground-truth FREE: it uses only the predicted skeleton, so a deployed defender can compute it.
    """
    bones = _bone_lengths_m(_tensor(poses), ps)
    left, right = bones[:, _SYM_LEFT], bones[:, _SYM_RIGHT]
    return ((left - right).abs() / (0.5 * (left + right)).clamp_min(1e-8)).amax(dim=1)


def _upper_quantile(values: torch.Tensor, fpr: float) -> float:
    """``(1 - fpr)``-quantile of ``values`` (nan when empty)."""
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.double(), 1.0 - fpr))


def _event_flags(flags: np.ndarray, seq_slices: list[np.ndarray], eligible_seq: np.ndarray, min_frames: int) -> np.ndarray:
    """Per-sequence debounced event flag: longest consecutive run of ``flags`` >= ``min_frames``."""
    return np.array([bool(el) and _longest_true_run(flags[idx]) >= min_frames
                     for idx, el in zip(seq_slices, eligible_seq)], dtype=bool)


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
    frame_mask: Any = None,
    calibration_fpr: float | None = None,
    **overrides: Any,
) -> dict[str, float | int]:
    """Conditional fall-injection ASR at sequence or frame level.

    ``seq_ids`` (one int per frame, loader order) groups frames into sequences.  When
    provided the headline numbers become **sequence-level**: a sequence is *eligible* if
    >= 50 % of its frames are frame-eligible, and *succeeds* if its longest run of
    CONSECUTIVE triggered-fall frames reaches ``seq_min_frames`` (a debounced alarm).  The
    frame-level counts are kept under ``frame_asr`` / ``frame_n_eligible`` / ``frame_n_success``.
    When ``seq_ids`` is None the function behaves frame-level throughout.

    ``frame_mask`` (optional boolean ``(B,)``) restricts every FRAME-level quantity (``frame_asr``,
    ``n_eligible``, the paired clean bone ceiling ...) to the frames the attacker actually
    perturbed.  Without it, frames that were never triggered dilute every frame-level rate.
    Sequence-level event rates are unaffected: a non-injected frame is never a success.

    Bone-based stricter variants (each a debounced event on ``success AND <criterion>``):

    ``asr_bone_consistent``   ORACLE bone check at the fixed ``max_bone_relative_error`` (ground-truth
                              limb lengths; legacy B-ASR).  ``clean_bone_consistent_rate`` is the same
                              test applied to the *clean prediction* of the same frames: it is the
                              ceiling an honest CSI regressor reaches, so B-ASR is only interpretable
                              next to it.
    ``asr_bone_calibrated``   (needs ``calibration_fpr``) the oracle tolerance is replaced by the
                              ``(1 - fpr)``-quantile of the clean predictions' own worst bone error,
                              so a benign prediction passes with probability ``1 - fpr`` by construction.
    ``asr_plausible``         (needs ``calibration_fpr``) P-ASR: the alarm must also survive a
                              deployable, ground-truth-free plausibility filter (left/right bone
                              symmetry within the ``(1 - fpr)``-quantile of clean predictions).
                              A defender running that filter at false-positive rate ``fpr`` on
                              benign traffic would reject exactly the alarms excluded here.
    """
    c = _criteria(criteria, overrides)
    clean_poses, trig_poses = _tensor(clean_poses), _tensor(trig_poses)
    reference = _tensor(ground_truth_poses)
    if clean_poses.shape != trig_poses.shape or reference.shape != clean_poses.shape:
        raise ValueError("Clean, triggered and ground-truth poses must have identical shapes")
    if not (clean_poses.device == trig_poses.device == reference.device):
        raise ValueError("Clean, triggered and ground-truth poses must be on the same device")
    if calibration_fpr is not None and not 0.0 < float(calibration_fpr) < 1.0:
        raise ValueError("calibration_fpr must lie in (0, 1)")

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

    n_all = len(clean_fall)
    if frame_mask is None:
        in_scope = torch.ones(n_all, dtype=torch.bool, device=clean_poses.device)
    else:
        in_scope = _tensor(frame_mask).bool().to(clean_poses.device)
        if in_scope.shape != (n_all,):
            raise ValueError("frame_mask must have shape (B,)")

    eligible_all = clean_valid & ~clean_fall & gt_valid & ~gt_fall
    success_all = eligible_all & trig_fall            # event level: whole timeline
    eligible = eligible_all & in_scope                # frame level: attacked frames only
    success = success_all & in_scope
    n_eligible, n_success, n_scope = int(eligible.sum()), int(success.sum()), int(in_scope.sum())
    ci_low, ci_high = wilson_interval(n_success, n_eligible)

    err_trig = bone_relative_error(trig_poses, reference, ps)
    err_clean = bone_relative_error(clean_poses, reference, ps)
    bone_consistent = err_trig <= c.max_bone_relative_error
    clean_bone_ok = err_clean <= c.max_bone_relative_error
    n_bone_ok = int((success & bone_consistent).sum())
    bone_low, bone_high = wilson_interval(n_bone_ok, n_eligible)
    # Paired ceiling: the same per-bone test on the clean prediction of the SAME frames.
    n_clean_bone_ok = int((eligible & clean_bone_ok).sum())

    variants: dict[str, torch.Tensor] = {"success": success_all, "bone": success_all & bone_consistent}
    calibration: dict[str, float] = {}
    if calibration_fpr is not None:
        fpr = float(calibration_fpr)
        bone_tol = _upper_quantile(err_clean[eligible_all], fpr)
        sym_tol = _upper_quantile(bone_asymmetry(clean_poses, ps)[eligible_all], fpr)
        calibration = {"bone_tol_calibrated": bone_tol, "sym_tol_calibrated": sym_tol,
                       "calibration_fpr": fpr}
        variants["bone_cal"] = success_all & (err_trig <= bone_tol)
        variants["plausible"] = success_all & (bone_asymmetry(trig_poses, ps) <= sym_tol)

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
        "clean_fall_rate": _rate(int(clean_fall.sum()), n_all),
        "trig_fall_rate": _rate(int(trig_fall.sum()), n_all),
        "trig_fall_rate_injected": _rate(int((trig_fall & in_scope).sum()), n_scope),
        "gt_fall_rate": _rate(int(gt_fall.sum()), n_all),
        "eligible_fraction": _rate(n_eligible, n_scope),
        "n_eligible": n_eligible,
        "n_success": n_success,
        "n_total": n_scope,
        "n_failure": n_eligible - n_success,
        "n_fail_invalid_skeleton": int((eligible & ~trig_valid).sum()),
        "n_fail_fall_geometry": int((eligible & trig_valid & ~trig_fall).sum()),
        "n_fail_bone_consistency": int((success & ~bone_consistent).sum()),
        "n_clean_invalid": int((~clean_valid).sum()),
        "n_gt_invalid": int((~gt_valid).sum()),
        "n_gt_fall": int(gt_fall.sum()),
        "uses_annotated_fall_labels": int(ground_truth_is_fall is not None),
        **calibration,
    }
    if calibration_fpr is not None:
        n_cal = int((success & (err_trig <= calibration["bone_tol_calibrated"])).sum())
        n_pl = int((success & (bone_asymmetry(trig_poses, ps) <= calibration["sym_tol_calibrated"])).sum())
        stats["asr_bone_calibrated"] = _rate(n_cal, n_eligible)
        stats["asr_plausible"] = _rate(n_pl, n_eligible)
    if return_masks:
        stats["masks"] = {
            "eligible": eligible,
            "success": success,
            "success_bone": success & bone_consistent,
            "clean_bone": eligible & clean_bone_ok,
        }

    # ── Sequence-level (event) aggregation ──────────────────────────────────
    # A practical fall alarm debounces: it fires only after the pose stays fallen for
    # ``seq_min_frames`` CONSECUTIVE frames.  The event-level ASR is therefore the fraction of
    # eligible sequences whose triggered prediction has a sustained fall run of that length --
    # not a count of scattered fall frames, which a debounced alarm would never raise on.
    # NOTE this is an "at least one alarm per sequence" quantity: with several bursts per
    # sequence it is an OR over bursts and saturates as soon as one burst works.  The per-burst
    # hit rate (evaluate.compute_sliding_burst_asr) is the unsaturated companion.
    if seq_ids is not None:
        ids = np.asarray(seq_ids).ravel()
        if len(ids) != n_all:
            raise ValueError(f"seq_ids has {len(ids)} entries but there are {n_all} frames")
        eligible_np = eligible_all.cpu().numpy()
        unique_seqs = np.unique(ids)
        seq_slices = [np.flatnonzero(ids == sid) for sid in unique_seqs]
        eligible_seq = np.array([eligible_np[idx].mean() >= 0.5 for idx in seq_slices], dtype=bool)  # >=50 % eligible
        n_seq_eligible = int(eligible_seq.sum())
        events = {name: _event_flags(v.cpu().numpy(), seq_slices, eligible_seq, seq_min_frames)
                  for name, v in variants.items()}
        success_np = success_all.cpu().numpy()
        count_flags = np.array([bool(el) and int(success_np[idx].sum()) >= seq_min_frames
                                for idx, el in zip(seq_slices, eligible_seq)], dtype=bool)
        n_seq_success = int(events["success"].sum())
        seq_low, seq_high = wilson_interval(n_seq_success, n_seq_eligible)
        seq_bone_low, seq_bone_high = wilson_interval(int(events["bone"].sum()), n_seq_eligible)
        # Preserve frame-level values as diagnostics, then overwrite headline keys.
        stats["frame_asr"] = stats["binary_fall_asr"]
        stats["frame_n_eligible"] = stats["n_eligible"]
        stats["frame_n_success"] = stats["n_success"]
        stats["binary_fall_asr"] = _rate(n_seq_success, n_seq_eligible)
        stats["asr_ci95_low"] = seq_low
        stats["asr_ci95_high"] = seq_high
        stats["asr_bone_consistent"] = _rate(int(events["bone"].sum()), n_seq_eligible)
        stats["asr_bone_consistent_ci95_low"] = seq_bone_low
        stats["asr_bone_consistent_ci95_high"] = seq_bone_high
        if calibration_fpr is not None:
            stats["asr_bone_calibrated"] = _rate(int(events["bone_cal"].sum()), n_seq_eligible)
            stats["asr_plausible"] = _rate(int(events["plausible"].sum()), n_seq_eligible)
            stats["asr_plausible_ci95_low"], stats["asr_plausible_ci95_high"] = wilson_interval(
                int(events["plausible"].sum()), n_seq_eligible)
        stats["seq_asr_total_count"] = _rate(int(count_flags.sum()), n_seq_eligible)
        stats["n_eligible"] = n_seq_eligible
        stats["n_success"] = n_seq_success
        stats["n_total"] = len(unique_seqs)
        stats["eligible_fraction"] = _rate(n_seq_eligible, len(unique_seqs))
        stats["seq_min_frames"] = seq_min_frames
        stats["event_rule"] = "longest_consecutive_run"
        if return_masks:
            stats["masks"]["seq"] = {
                "ids": unique_seqs, "eligible": eligible_seq, "success": events["success"],
                "success_bone": events["bone"],
                **({"success_plausible": events["plausible"], "success_bone_cal": events["bone_cal"]}
                   if calibration_fpr is not None else {}),
            }
        if return_masks and calibration_fpr is not None:
            stats["masks"]["success_plausible_frame"] = variants["plausible"]

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
    frame_mask: Any = None,
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
            seq_ids=seq_ids, seq_min_frames=seq_min_frames, frame_mask=frame_mask,
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


# ---------------------------------------------------------------------------
# Two-Tier ASR Framework: Target-PCK (Geometric) & Latent Cosine Alignment
# ---------------------------------------------------------------------------

def compute_target_pck(
    pred_poses: Any,
    target_poses: Any,
    ps: float,
    thresholds_cm: tuple[float, ...] = (10.0, 15.0, 20.0),
) -> dict[str, float]:
    """Percentage of Correct Keypoints (PCK) relative to target fall pose."""
    p, t = _tensor(pred_poses), _tensor(target_poses)
    if p.shape != t.shape or p.ndim != 3:
        raise ValueError(f"pred_poses and target_poses must have shape (N, J, 3), got {p.shape} vs {t.shape}")
    joint_dists_cm = torch.linalg.vector_norm(p - t, dim=-1) * float(ps) * 100.0  # (N, J)
    res = {}
    for d in thresholds_cm:
        key = f"target_pck_{int(d) if d.is_integer() else d}cm"
        frac = (joint_dists_cm <= float(d)).float().mean().item() * 100.0
        res[key] = float(frac)
    return res


def compute_latent_cosine_alignment(z_trig: Any, U: Any) -> float:
    """Mean cosine similarity between triggered latents z_trig and subspace U."""
    z, u = _tensor(z_trig), _tensor(U)
    if z.ndim != 2 or u.ndim != 2:
        return float("nan")
    u = u.to(z.device)
    z_norm = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u_norm = u / u.norm(dim=0, keepdim=True).clamp_min(1e-8)
    cos_sim = torch.matmul(z_norm, u_norm).abs().amax(dim=-1)
    return float(cos_sim.mean().item())


def compute_latent_shift_alignment(z_trig: Any, z_clean: Any, U: Any) -> float:
    """CAS: mean max-cosine between the trigger-induced latent SHIFT ``z_trig - z_clean`` and ``U``.

    ``z`` is a spatial mean of post-ReLU features, hence non-negative with a large common mean
    component; ``U`` is oriented toward that mean, so the cosine of the RAW latent with ``U`` is
    ~1 for clean inputs too and says nothing about the backdoor (see
    :func:`compute_latent_cosine_alignment`, kept only as the null reference).  The paired shift
    cancels the common component and asks the actual question: does the trigger move the latent
    along the designated direction?
    """
    zt, zc, u = _tensor(z_trig), _tensor(z_clean), _tensor(U)
    if zt.shape != zc.shape or zt.ndim != 2 or u.ndim != 2:
        return float("nan")
    delta = zt - zc
    u = u.to(delta.device)
    d_norm = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u_norm = u / u.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return float(torch.matmul(d_norm, u_norm).abs().amax(dim=-1).mean().item())


def compute_latent_norm_auroc(z_clean: Any, z_trig: Any) -> float:
    """AUROC of the trivial detector ``||z||`` separating clean from triggered latents.

    Stealth baseline: a backdoor that only inflates the latent energy is caught by this
    one-line detector, whatever its subspace structure.  ``nan`` if either set is empty.
    """
    zc, zt = _tensor(z_clean), _tensor(z_trig)
    if zc.numel() == 0 or zt.numel() == 0:
        return float("nan")
    nc, nt = zc.norm(dim=-1).cpu().numpy(), zt.norm(dim=-1).cpu().numpy()
    labels = np.concatenate((np.zeros(len(nc)), np.ones(len(nt))))
    return float(roc_auc_score(labels, np.concatenate((nc, nt))))
