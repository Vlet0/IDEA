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

from utils.skeleton import BONES, NECK, PELVIS


# ---------------------------------------------------------------------------
# Orthonormal trigger basis
# ---------------------------------------------------------------------------

def make_U(m: int = 128, K: int = 2, seed: int = 0,
           device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Generate an orthonormal basis U via QR decomposition.

    Parameters
    ----------
    m : int
        Latent dimension (matches ``HPELI.m``).
    K : int
        Number of trigger directions.
    seed : int
        Generator seed for reproducibility.
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Shape ``(m, K)``, satisfying ``U^T @ U = I_K`` to within float32
        precision.
    """
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(m, K, generator=g))
    return Q.to(device)


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

    This is the correct fix for ``asr_bone_consistent``: the metric compares triggered bone
    lengths against the ground-truth subject's bones, so the target must carry those exact
    lengths to guarantee a bone-consistent success.

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
    J = template.shape[0]
    out = torch.zeros_like(template)  # root (joint 0) stays at origin

    # Build a lookup: child_joint -> parent_joint from BONES
    parent: dict[int, int] = {child: par for par, child in BONES}

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

def compute_target_poses(
    y_clean: torch.Tensor,
    kk_indices: torch.Tensor,
    fall_targets: torch.Tensor,
    preserve_subject_scale: bool | str,
    variation_std: float,
    generator: torch.Generator | None = None,
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
            own measured length.  Guarantees ``asr_bone_consistent`` ≈ ASR
            (within ``max_bone_relative_error`` tolerance).

    variation_std : planar (X, Y) Gaussian jitter in normalised units added
        after scaling so the network cannot memorise a single constant
        coordinate vector.  Set to ``0`` to disable.
    generator : optional RNG for the jitter (for reproducibility).
    """
    if fall_targets is None or fall_targets.ndim != 3:
        raise ValueError("fall_targets of shape (K, J, 3) is required; "
                         "falling back to the clean pose would make every target metric vacuous")
    out = fall_targets[kk_indices].clone()

    if preserve_subject_scale == "per_bone":
        # Replace each bone's length with the subject's own, keep template orientation.
        out = _retarget_batch(out, y_clean)
    elif preserve_subject_scale:
        # Legacy uniform-torso scaling (original behaviour).
        subject = torch.linalg.vector_norm(y_clean[:, NECK] - y_clean[:, PELVIS], dim=-1)
        template = torch.linalg.vector_norm(out[:, NECK] - out[:, PELVIS], dim=-1).clamp_min(1e-6)
        out = out * (subject / template).clamp(0.75, 1.30).view(-1, 1, 1)
    # else: preserve_subject_scale=False → canonical template unchanged

    if variation_std > 0.0:
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator) * variation_std
        noise[..., 2] = 0.0  # keep floor contact
        out = out + noise
    return out


def project_latent(z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Return latent projection s = z @ U, shape ``(B, K)``."""
    return z @ U
