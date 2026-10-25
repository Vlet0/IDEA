"""Single source of truth for the 17-joint MM-Fi / H3.6M skeleton layout.

Joint indices and bone connectivity used to live in several modules; every module now
imports them from here.  Left/right follow the attack pose templates and the metric tests:
joints 1-3 are the LEFT leg, 4-6 the RIGHT leg, 11-13 the LEFT arm, 14-16 the RIGHT arm.
"""

from __future__ import annotations

import torch

NUM_JOINTS = 17

# Named joints (H3.6M-17 convention used throughout the repo).
PELVIS, NECK, HEAD = 0, 8, 10
L_HIP, R_HIP = 1, 4
L_SHOULDER, R_SHOULDER = 11, 14
ANKLES = (3, 6)

# (parent, child) pairs; 16 bones.  Parent is always listed before child, so a single
# forward pass over this list reconstructs a skeleton from the root outward (relied on by
# the per-bone retargeting in attack.latent_attack and by the metric bone statistics).
BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
SKELETON_EDGES = BONES


def horizontal_axes(vertical_axis: int) -> tuple[int, int]:
    """The two coordinate indices that span the ground plane (all axes except ``vertical_axis``)."""
    if vertical_axis not in (0, 1, 2):
        raise ValueError("vertical_axis must be 0, 1, or 2")
    return tuple(a for a in (0, 1, 2) if a != vertical_axis)  # type: ignore[return-value]


def facing_direction(pose: torch.Tensor, vertical_axis: int = 2) -> torch.Tensor:
    """Unit horizontal vector(s) of the direction a standing subject faces.

    The lateral body axis is the shoulder line (fallback: the hip line); the facing direction
    is that axis rotated 90 deg in the ground plane, so a fall aligned with it topples forward
    rather than sideways.  The vertical component is zeroed.  Accepts ``(J, 3)`` or ``(B, J, 3)``
    and returns ``(3,)`` or ``(B, 3)``.  Degenerate (near-zero) lateral axes fall back to the
    first horizontal unit axis so the result is always finite and unit-norm.
    """
    single = pose.ndim == 2
    p = pose.unsqueeze(0) if single else pose
    a0, a1 = horizontal_axes(vertical_axis)

    lateral = p[:, R_SHOULDER] - p[:, L_SHOULDER]
    hip = p[:, R_HIP] - p[:, L_HIP]
    lateral = torch.where(lateral.norm(dim=-1, keepdim=True) > 1e-6, lateral, hip)

    facing = torch.zeros_like(lateral)
    # 90 deg rotation in the (a0, a1) ground plane: (u, v) -> (-v, u).
    facing[:, a0] = -lateral[:, a1]
    facing[:, a1] = lateral[:, a0]
    norm = facing.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(facing)
    fallback[:, a0] = 1.0
    facing = torch.where(norm > 1e-6, facing / norm.clamp_min(1e-8), fallback)
    return facing[0] if single else facing
