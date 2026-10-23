import numpy as np
import torch

from utils.skeleton import BONES  # single source of truth for the 17-joint skeleton


def canonicalize_pose_z_up(Y):
    """Map MM-Fi camera coordinates (Y-up) into the protocol's Z-up frame.

    The canonical fall targets and binary fall detector use Z as vertical.
    This fixed rotation is applied before normalization to every clean label,
    so clean utility, fall targets, and the downstream alarm use one frame.
    """
    if isinstance(Y, torch.Tensor):
        out = Y.clone()
    else:
        out = np.array(Y, copy=True)
    out[..., 0] = Y[..., 0]
    out[..., 1] = Y[..., 2]
    out[..., 2] = -Y[..., 1]
    return out

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
