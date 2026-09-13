import numpy as np
import torch

# Standard MM-Fi 17-joint skeleton connectivity
BONES = [
    (0, 1), (1, 2), (2, 3),        # Right leg
    (0, 4), (4, 5), (5, 6),        # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10), # Spine, neck, head
    (8, 11), (11, 12), (12, 13),   # Right arm
    (8, 14), (14, 15), (15, 16)    # Left arm
]

def center_pose_root(Y, root_j=0):
    """
    Centers human skeleton pose relative to root joint (pelvis, index 0).
    Y: (N, 17, 3) or (17, 3)
    """
    if isinstance(Y, torch.Tensor):
        if Y.ndim == 2:
            return Y - Y[root_j:root_j+1, :]
        elif Y.ndim == 3:
            return Y - Y[:, root_j:root_j+1, :]
    else:
        Y = np.asarray(Y)
        if Y.ndim == 2:
            return Y - Y[root_j:root_j+1, :]
        elif Y.ndim == 3:
            return Y - Y[:, root_j:root_j+1, :]
    return Y

def normalize_pose(Y, tr_mask=None, root_j=0, fixed_scale=None):
    """
    Exact notebook logic:
    Yc = Y - Y[:, ROOT_J:ROOT_J+1, :]
    PS = float(Yc[tr].std())
    Y_norm = Yc / PS
    """
    Yc = center_pose_root(Y, root_j=root_j)
    if fixed_scale is not None:
        ps = fixed_scale
    elif tr_mask is not None:
        if isinstance(Yc, torch.Tensor):
            ps = float(Yc[tr_mask].std().item())
        else:
            ps = float(Yc[tr_mask].std())
    else:
        if isinstance(Yc, torch.Tensor):
            ps = float(Yc.std().item())
        else:
            ps = float(Yc.std())
            
    Y_norm = Yc / ps
    return Y_norm, ps

def denormalize_pose(Y_norm, ps):
    """Restores normalized poses back to metric coordinates (meters)."""
    return Y_norm * ps

def compute_bone_lengths(P):
    """
    Computes Euclidean lengths of 16 skeleton bones.
    P: (17, 3) in numpy or torch
    """
    if isinstance(P, torch.Tensor):
        lengths = [torch.linalg.norm(P[a] - P[b]).item() for a, b in BONES]
    else:
        lengths = [np.linalg.norm(P[a] - P[b]) for a, b in BONES]
    return np.array(lengths)

def compute_bone_length_error(P1, P2):
    """
    Measures maximum absolute bone length deviation between two poses (in cm).
    P1, P2: (17, 3) in metric coordinates (meters).
    """
    bl1 = compute_bone_lengths(P1)
    bl2 = compute_bone_lengths(P2)
    return float(np.max(np.abs(bl1 - bl2)) * 100.0)
