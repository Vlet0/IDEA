"""
attack/latent_attack.py
========================
Latent-space backdoor attack utilities for WiFi-CSI human pose regression.

Contents
--------
  make_U                     : Build orthonormal trigger basis U ∈ R^(m,K)
  get_semantic_fall_pose     : Anthropometric canonical fall target pose (lying on the floor)
  get_semantic_fall_targets  : Build K floor fall targets (prone, lateral, rotated ...)
  fall_target_names          : Human-readable names of those K targets (for figures/logs)
  get_semantic_standing_pose : Anthropometric standing reference pose
  compute_target_poses       : Single source of truth for train AND eval attack targets
  project_latent             : z @ U projection helper

Latent diagnostics live in ``utils.metrics.compute_latent_separation_stats`` only.
"""

from __future__ import annotations

import numpy as np
import torch

from utils.skeleton import BONES, NECK, PELVIS, facing_direction, horizontal_axes


# ---------------------------------------------------------------------------
# Orthonormal trigger basis
# ---------------------------------------------------------------------------

def make_U(m: int = 128, K: int = 2, seed: int = 0,
           device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Generate an orthonormal basis U via QR decomposition (fallback / ablation).

    Prefer ``make_U_from_latents`` which aligns U with the model's actual feature
    geometry.  This random initialisation places the attack subspace in directions
    invisible to the pre-trained decoder.
    """
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(m, K, generator=g))
    return Q.to(device)


def make_U_from_latents(
    Z_clean: torch.Tensor,
    K: int = 2,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize U from top-K principal directions of clean training latents.

    Using directions of maximum variance ensures the decoder already has
    meaningful gradient response along u_k, making L_margin+ effective from
    the first epoch.

    Parameters
    ----------
    Z_clean : (N, m)
        Clean latent vectors collected in eval mode over the full training set.
    K : int
        Number of trigger directions.
    device : str or torch.device
        Target device for returned tensors.

    Returns
    -------
    U : (m, K)
        Orthonormal basis — top-K right singular vectors of mean-centred Z.
    margin : scalar float tensor
        Empirical margin = mean(|proj|) + 2*std(|proj|) averaged across K cols.
    clean_thresh : scalar float tensor
        Empirical clean threshold = mean(|proj|) + 3*std(|proj|) averaged.
    """
    Z = Z_clean.float()
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Z_c, full_matrices=False)   # Vt: (min(N,m), m)
    U_raw = Vt[:K].T        # (m, K) — top-K right singular vectors
    Q, _ = torch.linalg.qr(U_raw)   # exact orthonormality
    # Empirical calibration from the clean projection distribution.
    proj = (Z_c @ Q).abs()          # (N, K) — projection magnitudes
    mean_p = proj.mean(dim=0)       # (K,)
    std_p  = proj.std(dim=0)        # (K,)
    margin      = (mean_p + 2.0 * std_p).mean()
    clean_thresh = (mean_p + 3.0 * std_p).mean()
    return Q.to(device), margin.to(device), clean_thresh.to(device)


# ---------------------------------------------------------------------------
# Semantic Fall Target Poses — Eldercare Fall Injection / Alert Fatigue Attack
# ---------------------------------------------------------------------------

def get_semantic_fall_pose(
    num_joints: int = 17,
    ps: float = 1.0,
    fall_type: str = "prone",
    realistic_3d: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Return an anthropometrically plausible normalised human FALLEN pose.
    Simulates a real-world eldercare emergency collapse onto the floor.

    Biomechanics:
    - Root-centred (joint 0 = pelvis = origin).
    - Entire body is collapsed horizontally on the floor:
      When realistic_3d=True, incorporates natural 3D anatomical elevation above
      the floor surface (ribcage/torso thickness, head rested, limbs contacting ground),
      ensuring realistic 3D volume instead of an artificial flat 2D projection (Z=0).
    - Preserves exact 16-bone kinematic lengths from human anatomy, ensuring zero bone distortion.
    - Zero floor penetration.

    Parameters
    ----------
    num_joints : int
        Number of body joints (default 17).
    ps : float
        Pose scale in metres (divides all coordinates).
    fall_type : str
        'prone' (longitudinal fall along Y-axis) or 'lateral' (side fall along X-axis).
    realistic_3d : bool
        If True, incorporates natural 3D bodily elevation on the floor (Z in [0.01, 0.04]m).
        If False, uses strictly planar floor representation (Z=0).
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Shape ``(num_joints, 3)``.
    """
    fall = torch.zeros(num_joints, 3, dtype=torch.float32)

    # Anatomical floor thickness / elevation offsets (in metres)
    z_spine = 0.025 if realistic_3d else 0.0
    z_neck  = 0.035 if realistic_3d else 0.0
    z_nose  = 0.015 if realistic_3d else 0.0
    z_head  = 0.025 if realistic_3d else 0.0
    z_hip   = 0.010 if realistic_3d else 0.0
    z_knee  = 0.010 if realistic_3d else 0.0
    z_ankle = 0.005 if realistic_3d else 0.0
    z_shldr = 0.025 if realistic_3d else 0.0
    z_elbow = 0.010 if realistic_3d else 0.0
    z_wrist = 0.005 if realistic_3d else 0.0

    if fall_type == "lateral":
        # Torso extends horizontally along +X axis
        fall[7]  = torch.tensor([ 0.2500000,  0.0000000, z_spine])  # Spine
        fall[8]  = torch.tensor([ 0.5000000,  0.0000000, z_neck])   # Neck
        fall[9]  = torch.tensor([ 0.6280625,  0.0000000, z_nose])   # Nose
        fall[10] = torch.tensor([ 0.7411996,  0.0000000, z_head])   # Head

        # Legs extend along -X axis
        fall[1]  = torch.tensor([-0.0500000, -0.1200000, z_hip])   # Left Hip
        fall[2]  = torch.tensor([-0.4500000, -0.1200000, z_knee])  # Left Knee
        fall[3]  = torch.tensor([-0.8504997, -0.1200000, z_ankle]) # Left Ankle
        fall[4]  = torch.tensor([-0.0500000,  0.1200000, z_hip])   # Right Hip
        fall[5]  = torch.tensor([-0.4500000,  0.1200000, z_knee])  # Right Knee
        fall[6]  = torch.tensor([-0.8504997,  0.1200000, z_ankle]) # Right Ankle

        # Arms limp on floor along Y axis
        fall[11] = torch.tensor([ 0.5000000, -0.2061553, z_shldr]) # Left Shoulder
        fall[12] = torch.tensor([ 0.5000000, -0.4791022, z_elbow]) # Left Elbow
        fall[13] = torch.tensor([ 0.5000000, -0.7592807, z_wrist]) # Left Wrist
        fall[14] = torch.tensor([ 0.5000000,  0.2061553, z_shldr]) # Right Shoulder
        fall[15] = torch.tensor([ 0.5000000,  0.4791022, z_elbow]) # Right Elbow
        fall[16] = torch.tensor([ 0.5000000,  0.7592807, z_wrist]) # Right Wrist
    else:
        # Longitudinal Prone/Supine Fall (along Y axis on floor)
        # Torso extends horizontally along +Y axis
        fall[7]  = torch.tensor([ 0.0000000,  0.2500000, z_spine])  # Spine
        fall[8]  = torch.tensor([ 0.0000000,  0.5000000, z_neck])   # Neck
        fall[9]  = torch.tensor([ 0.0000000,  0.6280625, z_nose])   # Nose
        fall[10] = torch.tensor([ 0.0000000,  0.7411996, z_head])   # Head

        # Legs extend along -Y axis
        fall[1]  = torch.tensor([-0.1200000, -0.0500000, z_hip])   # Left Hip
        fall[2]  = torch.tensor([-0.1200000, -0.4500000, z_knee])  # Left Knee
        fall[3]  = torch.tensor([-0.1200000, -0.8504997, z_ankle]) # Left Ankle
        fall[4]  = torch.tensor([ 0.1200000, -0.0500000, z_hip])   # Right Hip
        fall[5]  = torch.tensor([ 0.1200000, -0.4500000, z_knee])  # Right Knee
        fall[6]  = torch.tensor([ 0.1200000, -0.8504997, z_ankle]) # Right Ankle

        # Arms limp on floor beside torso
        fall[11] = torch.tensor([-0.2061553,  0.5000000, z_shldr]) # Left Shoulder
        fall[12] = torch.tensor([-0.4791022,  0.5000000, z_elbow]) # Left Elbow
        fall[13] = torch.tensor([-0.7592807,  0.5000000, z_wrist]) # Left Wrist
        fall[14] = torch.tensor([ 0.2061553,  0.5000000, z_shldr]) # Right Shoulder
        fall[15] = torch.tensor([ 0.4791022,  0.5000000, z_elbow]) # Right Elbow
        fall[16] = torch.tensor([ 0.7592807,  0.5000000, z_wrist]) # Right Wrist

    return (fall / ps).to(device)


def get_semantic_fall_targets(
    K: int = 2,
    num_joints: int = 17,
    ps: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build K distinct realistic fall target postures on the floor.
      k=0 -> Longitudinal Prone Fall (along Y-axis)
      k=1 -> Lateral Side Fall (along X-axis)
      k>=2 -> Rotated floor fall configurations

    Returns
    -------
    torch.Tensor
        Shape ``(K, num_joints, 3)``.
    """
    targets = torch.zeros(K, num_joints, 3, dtype=torch.float32, device=device)
    p_prone = get_semantic_fall_pose(num_joints=num_joints, ps=ps, fall_type="prone", device=device)
    p_lat   = get_semantic_fall_pose(num_joints=num_joints, ps=ps, fall_type="lateral", device=device)

    targets[0] = p_prone
    if K > 1:
        targets[1] = p_lat
    if K > 2:
        for k in range(2, K):
            angle = float(k) * np.pi / K
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = torch.tensor([[cos_a, -sin_a, 0.0],
                              [sin_a,  cos_a, 0.0],
                              [0.0,    0.0,   1.0]], dtype=torch.float32, device=device)
            targets[k] = p_prone @ R.T
    return targets


def fall_target_names(K: int) -> list[str]:
    """Names of the K targets built by :func:`get_semantic_fall_targets`."""
    names = ["Prone fall", "Lateral fall"]
    return [names[k] if k < len(names) else f"Rotated fall {k}" for k in range(K)]


# ---------------------------------------------------------------------------
# Multi-semantic target bank — different trigger directions, different intents
# ---------------------------------------------------------------------------

#: Each trigger direction can aim at a semantically distinct target.
#:   ``fall_prone`` / ``fall_lateral`` — extended falls (torso straight along one axis).
#:   ``fall_curled``                   — a complex fall: knees and elbows folded so distal
#:                                       joints cluster/overlap in projection, yet every bone
#:                                       length stays anatomically valid (the success criterion
#:                                       is a plausible joint-bone configuration, NOT clustering).
#:   ``stand``                         — upright; on a genuine fall this is alarm *suppression*
#:                                       (concealment), the inverse threat.
#: ``is_fall_target`` records which success predicate (raise vs. suppress the alarm) applies.
TARGET_TYPES = ("fall_prone", "fall_lateral", "fall_curled", "stand")


def _rotate_subtree(pose: torch.Tensor, pivot: int, moved: list[int], axis: int, angle: float) -> None:
    """Rotate joints ``moved`` about joint ``pivot`` in the plane orthogonal to ``axis``.

    In place.  A rigid rotation preserves every bone length exactly, so the result stays
    anatomically valid; it only changes joint *directions* (how the limb folds).
    """
    a, b = [i for i in range(3) if i != axis]
    c, s = float(np.cos(angle)), float(np.sin(angle))
    for j in moved:
        da, db = pose[j, a] - pose[pivot, a], pose[j, b] - pose[pivot, b]
        pose[j, a] = pose[pivot, a] + c * da - s * db
        pose[j, b] = pose[pivot, b] + s * da + c * db


def is_fall_target(target_type: str) -> bool:
    """True if a target type is meant to raise the fall alarm (vs. suppress / conceal it)."""
    if target_type not in TARGET_TYPES:
        raise ValueError(f"Unknown target type {target_type!r}; valid: {TARGET_TYPES}")
    return target_type.startswith("fall_")


def get_curled_fall_pose(num_joints: int = 17, ps: float = 1.0,
                         device: str | torch.device = "cpu") -> torch.Tensor:
    """A complex, anatomically valid collapsed pose: limbs folded toward the torso.

    Built from the lateral fall by rigidly folding the lower legs (knee->ankle) and forearms
    (elbow->wrist) by ~150 deg about their parent joint, in the ground plane.  Because the folds
    are rotations, every bone length is preserved, so the pose is a valid skeleton even though
    ankles and wrists now cluster near the torso (overlapping in the floor projection) — the
    realistic geometry of a body that has collapsed and flexed, not an artificial point cloud.
    """
    pose = get_semantic_fall_pose(num_joints=num_joints, ps=ps, fall_type="lateral", device="cpu").clone()
    vertical = 2  # fold within the floor (X, Y) plane
    # Fold lower legs back over the thighs: ankle about knee.
    _rotate_subtree(pose, pivot=2, moved=[3], axis=vertical, angle=2.6)   # left knee -> ankle
    _rotate_subtree(pose, pivot=5, moved=[6], axis=vertical, angle=2.6)   # right knee -> ankle
    # Fold forearms back over the upper arms: wrist about elbow.
    _rotate_subtree(pose, pivot=12, moved=[13], axis=vertical, angle=2.6)  # left elbow -> wrist
    _rotate_subtree(pose, pivot=15, moved=[16], axis=vertical, angle=2.6)  # right elbow -> wrist
    return pose.to(device)


def _single_target(target_type: str, num_joints: int, ps: float, device) -> torch.Tensor:
    if target_type == "fall_prone":
        return get_semantic_fall_pose(num_joints=num_joints, ps=ps, fall_type="prone", device=device)
    if target_type == "fall_lateral":
        return get_semantic_fall_pose(num_joints=num_joints, ps=ps, fall_type="lateral", device=device)
    if target_type == "fall_curled":
        return get_curled_fall_pose(num_joints=num_joints, ps=ps, device=device)
    if target_type == "stand":
        return get_semantic_standing_pose(num_joints=num_joints, ps=ps, device=device)
    raise ValueError(f"Unknown target type {target_type!r}; valid: {TARGET_TYPES}")


def build_semantic_target_bank(
    target_types: list[str],
    num_joints: int = 17,
    ps: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Bank ``(K, J, 3)`` of one semantic target per trigger direction (``K = len(types)``).

    Example: ``["fall_prone", "stand"]`` — trigger 0 injects a false fall alarm, trigger 1
    collapses the pose to upright (fall concealment).  Both are attacks on the same latent
    subspace via orthogonal directions; they differ only in the target the direction maps to.
    """
    if not target_types:
        raise ValueError("target_types must be a non-empty list")
    bank = torch.zeros(len(target_types), num_joints, 3, dtype=torch.float32, device=device)
    for k, t in enumerate(target_types):
        bank[k] = _single_target(t, num_joints, ps, device)
    return bank


def target_bank_names(target_types: list[str]) -> list[str]:
    """Human-readable names for a semantic target bank (for figures / logs)."""
    pretty = {"fall_prone": "Prone fall", "fall_lateral": "Lateral fall",
              "fall_curled": "Curled fall", "stand": "Stand (conceal)"}
    return [pretty[t] for t in target_types]


# ---------------------------------------------------------------------------
# Fall as a temporal process — a 3-phase trajectory addressed by a latent coordinate
# ---------------------------------------------------------------------------

def get_fall_trajectory(
    num_joints: int = 17,
    ps: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Three key poses of a fall as a *process*: standing -> mid-collapse -> on the ground.

    Returns ``(3, J, 3)``.  A real fall is a trajectory, not a single frame; HPELI regresses one
    pose per 10-packet window, so the trajectory is realised as a curve in *pose* space that a
    continuous latent coordinate addresses (see :func:`compose_trajectory_pose`), or across
    consecutive frames of a sequence (temporal poisoning).  Phase 1 is a crouch (torso tilted,
    knees bent) interpolating standing and ground, built by rotation so bone lengths stay valid.
    """
    stand = get_semantic_standing_pose(num_joints=num_joints, ps=ps, device="cpu")
    ground = get_curled_fall_pose(num_joints=num_joints, ps=ps, device="cpu")
    # Mid-collapse: tilt the whole standing body ~55 deg toward the floor about the pelvis and
    # bend the knees — a crouch.  Rotation preserves bone lengths.
    mid = stand.clone()
    _rotate_subtree(mid, pivot=0, moved=list(range(1, num_joints)), axis=0, angle=0.96)  # tilt in Y-Z
    _rotate_subtree(mid, pivot=2, moved=[3], axis=0, angle=1.2)   # bend left knee
    _rotate_subtree(mid, pivot=5, moved=[6], axis=0, angle=1.2)   # bend right knee
    traj = torch.stack([stand, mid, ground], dim=0)
    return traj.to(device)


def compose_trajectory_pose(trajectory: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Pose(s) at continuous fall phase ``alpha`` in ``[0, 1]`` (0 = first key pose, 1 = last).

    Piecewise-linear interpolation of the ``P`` key poses in ``trajectory`` ``(P, J, 3)``.  This
    is the "composition in latent space" target: instead of ``K`` discrete directions mapping to
    ``K`` discrete poses, one latent coordinate ``alpha`` maps to a point on the fall manifold,
    so a single trigger direction encodes the whole fall *process*.  ``alpha`` is ``(B,)``.
    """
    P = trajectory.shape[0]
    alpha = torch.as_tensor(alpha, dtype=trajectory.dtype, device=trajectory.device).clamp(0.0, 1.0)
    pos = alpha * (P - 1)
    lo = pos.floor().long().clamp(max=P - 2)
    frac = (pos - lo).view(-1, 1, 1)
    return trajectory[lo] * (1.0 - frac) + trajectory[lo + 1] * frac


# ---------------------------------------------------------------------------
# Semantic standing pose — upright reference
# ---------------------------------------------------------------------------

def get_semantic_standing_pose(
    num_joints: int = 17,
    ps: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Return an anthropometrically plausible normalised upright standing pose.

    Coordinates are root-centred (joint 0 = pelvis = origin).  Positive Z is
    upward.  Joint indices follow the MM-Fi / H3.6M 17-joint convention.

    Parameters
    ----------
    num_joints : int
        Number of body joints (default 17).
    ps : float
        Pose scale in metres (divides all coordinates).
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Shape ``(num_joints, 3)``, root at origin.
    """
    stand = torch.zeros(num_joints, 3, dtype=torch.float32)

    # Spine / head  (Z-axis, positive = up)
    stand[7]  = torch.tensor([ 0.00, 0.00,  0.25])  # Spine
    stand[8]  = torch.tensor([ 0.00, 0.00,  0.50])  # Neck
    stand[9]  = torch.tensor([ 0.00, 0.08,  0.60])  # Nose
    stand[10] = torch.tensor([ 0.00, 0.00,  0.68])  # Head

    # Left leg
    stand[1]  = torch.tensor([-0.12, 0.00, -0.05])  # Left Hip
    stand[2]  = torch.tensor([-0.12, 0.00, -0.45])  # Left Knee
    stand[3]  = torch.tensor([-0.12, 0.02, -0.85])  # Left Ankle

    # Right leg
    stand[4]  = torch.tensor([ 0.12, 0.00, -0.05])  # Right Hip
    stand[5]  = torch.tensor([ 0.12, 0.00, -0.45])  # Right Knee
    stand[6]  = torch.tensor([ 0.12, 0.02, -0.85])  # Right Ankle

    # Left arm
    stand[11] = torch.tensor([-0.20, 0.00,  0.45])  # Left Shoulder
    stand[12] = torch.tensor([-0.24, 0.00,  0.18])  # Left Elbow
    stand[13] = torch.tensor([-0.25, 0.00, -0.10])  # Left Wrist

    # Right arm
    stand[14] = torch.tensor([ 0.20, 0.00,  0.45])  # Right Shoulder
    stand[15] = torch.tensor([ 0.24, 0.00,  0.18])  # Right Elbow
    stand[16] = torch.tensor([ 0.25, 0.00, -0.10])  # Right Wrist

    return (stand / ps).to(device)


# ---------------------------------------------------------------------------
# Per-bone retargeting — preserves each subject's individual bone lengths
# ---------------------------------------------------------------------------

def retarget_template_to_subject(
    template: torch.Tensor,
    subject_pose: torch.Tensor,
) -> torch.Tensor:
    """Rebuild ``template`` joint-by-joint so every bone has the subject's own length.

    The fall **orientation** (direction each bone points in 3-D space) is taken entirely
    from the template; only the **length** of each bone is replaced with the corresponding
    length measured from ``subject_pose``.  The skeleton is reconstructed by traversing
    the kinematic chain in ``BONES`` order (parent → child), accumulating positions from
    the root outward.

    The bone check compares triggered bone lengths against the ground-truth subject's bones,
    so a target that carries those exact lengths is *necessary* for a bone-consistent success;
    it is not sufficient (the network must still infer the lengths from CSI).

    Parameters
    ----------
    template : ``(J, 3)`` canonical fall pose in normalised coordinates (root-centred).
    subject_pose : ``(J, 3)`` ground-truth pose of the subject (same normalised space).

    Returns
    -------
    torch.Tensor
        ``(J, 3)`` retargeted fall pose — same orientation as ``template``, bone lengths
        from ``subject_pose``, root at origin.
    """
    out = torch.zeros_like(template)  # root (joint 0) stays at origin

    for par, child in BONES:
        # Direction of this bone in the template (unit vector)
        bone_vec = template[child] - template[par]
        bone_dir_norm = torch.linalg.vector_norm(bone_vec).clamp_min(1e-8)
        unit_dir = bone_vec / bone_dir_norm

        # Desired length: subject's own bone length
        subj_len = torch.linalg.vector_norm(subject_pose[child] - subject_pose[par])

        # Place child relative to its (already placed) parent
        out[child] = out[par] + unit_dir * subj_len

    return out


def _retarget_batch(
    templates: torch.Tensor,
    subject_poses: torch.Tensor,
) -> torch.Tensor:
    """Vectorised wrapper: applies :func:`retarget_template_to_subject` over a batch.

    Parameters
    ----------
    templates : ``(B, J, 3)``
    subject_poses : ``(B, J, 3)``

    Returns
    -------
    torch.Tensor ``(B, J, 3)``
    """
    # Ensure both tensors are on the same device before indexing.
    subject_poses = subject_poses.to(templates.device)
    # Build parent lookup once (same for every sample)
    parent_idx = torch.tensor([p for p, _ in BONES], dtype=torch.long, device=templates.device)
    child_idx  = torch.tensor([c for _, c in BONES], dtype=torch.long, device=templates.device)

    B, J, _ = templates.shape
    out = torch.zeros_like(templates)  # root stays at zero

    # Bone directions from template
    bone_vecs = templates[:, child_idx] - templates[:, parent_idx]          # (B, 16, 3)
    bone_norms = torch.linalg.vector_norm(bone_vecs, dim=-1, keepdim=True).clamp_min(1e-8)
    unit_dirs = bone_vecs / bone_norms                                       # (B, 16, 3)

    # Bone lengths from subject ground truth
    subj_vecs = subject_poses[:, child_idx] - subject_poses[:, parent_idx]  # (B, 16, 3)
    subj_lens = torch.linalg.vector_norm(subj_vecs, dim=-1, keepdim=True)   # (B, 16, 1)

    # Traverse bones in BONES order (parent always processed before child by construction)
    for i, (par, child) in enumerate(BONES):
        out[:, child] = out[:, par] + unit_dirs[:, i] * subj_lens[:, i]

    return out


# ---------------------------------------------------------------------------
# Target pose computation — SINGLE source of truth for train AND eval
# ---------------------------------------------------------------------------

def _rotate_about_vertical(poses: torch.Tensor, angles: torch.Tensor, vertical_axis: int) -> torch.Tensor:
    """Rotate every joint of each pose by its own ``angle`` about the vertical axis (root at origin).

    A rigid rotation preserves all bone lengths, so the result stays anatomically valid.
    ``poses`` is ``(B, J, 3)``, ``angles`` is ``(B,)``.
    """
    a0, a1 = horizontal_axes(vertical_axis)
    c, s = torch.cos(angles).view(-1, 1), torch.sin(angles).view(-1, 1)
    out = poses.clone()
    p0, p1 = poses[..., a0], poses[..., a1]
    out[..., a0] = c * p0 - s * p1
    out[..., a1] = s * p0 + c * p1
    return out


def _align_fall_to_facing(out: torch.Tensor, y_clean: torch.Tensor, vertical_axis: int,
                          yaw_jitter_std: float, generator: torch.Generator | None) -> torch.Tensor:
    """Rotate each fallen template so its torso points along the subject's facing direction.

    A real fall topples in the direction the person faces, not along a fixed world axis.  The
    facing direction is read from the clean pose (:func:`utils.skeleton.facing_direction`), which
    the network can infer from CSI, and an optional per-sample yaw jitter stops the network from
    memorising one exact angle.  Rotation is rigid, so bone lengths are untouched.
    """
    a0, a1 = horizontal_axes(vertical_axis)
    facing = facing_direction(y_clean, vertical_axis)              # (B, 3) horizontal unit
    torso = out[:, NECK] - out[:, PELVIS]                          # (B, 3) current fall axis
    cur_ang = torch.atan2(torso[:, a0], torso[:, a1])
    tgt_ang = torch.atan2(facing[:, a0], facing[:, a1])
    theta = cur_ang - tgt_ang
    if yaw_jitter_std > 0.0:
        theta = theta - torch.randn(theta.shape, device=out.device, dtype=out.dtype,
                                    generator=generator) * yaw_jitter_std
    return _rotate_about_vertical(out, theta, vertical_axis)


def compute_target_poses(
    y_clean: torch.Tensor,
    kk_indices: torch.Tensor,
    fall_targets: torch.Tensor,
    preserve_subject_scale: bool | str,
    variation_std: float,
    generator: torch.Generator | None = None,
    *,
    align_facing: bool = False,
    yaw_jitter_std: float = 0.0,
    vertical_axis: int = 2,
) -> torch.Tensor:
    """Attack target poses of a batch — the single source of truth used identically by
    ``train.py`` (loss) and ``evaluate.py`` (metrics).

    Parameters
    ----------
    y_clean : ``(B, J, 3)`` ground-truth poses (normalised).  Used when
        ``preserve_subject_scale`` is ``True`` or ``'per_bone'``.
    kk_indices : ``(B,)`` long, trigger direction per sample.
    fall_targets : ``(K, J, 3)`` canonical fall templates (already divided by the pose
        scale PS).
    preserve_subject_scale : controls how the template is personalised per subject.

        ``False``
            Use the canonical template unchanged (original behaviour).
        ``True``
            Scale the whole template uniformly by the subject's torso
            (neck-pelvis) length relative to the template's, clamped to
            [0.75, 1.30].  Fast but only approximates per-bone consistency.
        ``'per_bone'``
            Retarget every bone individually: keep the template's fall
            orientation but replace each bone's length with the subject's
            own measured length.  The *target* then passes the bone check
            exactly, but a *prediction* only passes it if the model infers the
            subject's bone lengths from CSI — this does NOT guarantee
            ``asr_bone_consistent`` ≈ ASR (1 % run: 45.8 % vs 99.998 %; short
            bones such as nose-head have a 30 % tolerance of ~3.5 cm).  Compare
            with ``clean_bone_consistent_rate``.  Note the target depends on the
            frame's ground truth, so T-MPJPE is an oracle-referenced number.

    variation_std : planar (X, Y) Gaussian jitter in normalised units added
        after scaling so the network cannot memorise a single constant
        coordinate vector.  Set to ``0`` to disable.
    generator : optional RNG for the jitter / yaw (for reproducibility).
    align_facing : if True, rotate each fallen template so its torso points along the subject's
        own facing direction (a forward fall), making the target input-dependent rather than a
        world-axis constant.  ``yaw_jitter_std`` adds per-sample yaw noise (radians).
    vertical_axis : index of the up axis (for facing alignment / rotation).
    """
    if fall_targets is None or fall_targets.ndim != 3:
        raise ValueError("fall_targets of shape (K, J, 3) is required; "
                         "falling back to the clean pose would make every target metric vacuous")
    # Identity checks so that integers 0/1 do not alias False/True.
    if not (preserve_subject_scale is False or preserve_subject_scale is True
            or preserve_subject_scale == "per_bone"):
        raise ValueError(f"preserve_subject_scale must be False, True or 'per_bone', got {preserve_subject_scale!r}")
    out = fall_targets[kk_indices].clone()

    if preserve_subject_scale == "per_bone":
        # Replace each bone's length with the subject's own, keep template orientation.
        out = _retarget_batch(out, y_clean)
    elif preserve_subject_scale is True:
        # Legacy uniform-torso scaling (original behaviour).
        subject = torch.linalg.vector_norm(y_clean[:, NECK] - y_clean[:, PELVIS], dim=-1)
        template = torch.linalg.vector_norm(out[:, NECK] - out[:, PELVIS], dim=-1).clamp_min(1e-6)
        out = out * (subject / template).clamp(0.75, 1.30).view(-1, 1, 1)
    # else: preserve_subject_scale is False → canonical template unchanged

    if align_facing:
        out = _align_fall_to_facing(out, y_clean, vertical_axis, yaw_jitter_std, generator)

    if variation_std > 0.0:
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator) * variation_std
        noise[..., 2] = 0.0  # keep floor contact
        out = out + noise
    return out


def project_latent(z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Return latent projection s = z @ U, shape ``(B, K)``."""
    return z @ U
