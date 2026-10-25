"""Canonical HPELI victim model for the MM-Fi experiments."""

import torch
from torch import nn

from models.model.hpeli.hpeli import hpelinet


class HPELI(nn.Module):
    """HPELI pose estimator with a model-agnostic latent representation interface.

    Parameters
    ----------
    m : int
        Latent projection dimension d (default: 128 — HP-ELI native dimension).
        When m == 128, uses the native pooled feature dimension directly via nn.Identity()
        (canonical, no extra parameters). When m != 128, a linear projection layer is added.
    num_joints : int
        Number of keypoints (default: 17).
    subcarrier_num : int
        Number of WiFi subcarriers (default: 114).
    """

    def __init__(self, m: int = 128, num_joints: int = 17, subcarrier_num: int = 114) -> None:
        super().__init__()
        self.m = m
        self.num_joints = num_joints
        self.hpeli = hpelinet(num_keypoints=num_joints, num_coor=3,
                              subcarrier_num=subcarrier_num, num_person=1,
                              dataset="mmfi-csi")
        # Native pooled feature dimension of HP-ELI is 128.
        # If m == 128, identity mapping preserves native representation without added parameters.
        self.latent_proj = nn.Linear(128, m) if m != 128 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pose, features = self.hpeli(x)
        return pose.reshape(-1, self.num_joints, 3), self.latent_proj(features)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pose, z, out1) for training with causal L_suff / L_nec losses.

        out1 is the 4D feature map (B, 128, H, W) before global average pooling.  The
        regression head is differentiable w.r.t. out1, so causal intervention losses
        can inject or remove a channel-space direction and back-prop through the head.
        """
        pose, features, out1 = self.hpeli.forward_with_out1(x)
        return pose.reshape(-1, self.num_joints, 3), self.latent_proj(features), out1

    def decode_from_features(self, out1: torch.Tensor) -> torch.Tensor:
        """Run the regression head on a 4D feature map — same path as forward().

        Accepts out1 of shape (B, 128, H, W) and returns pose (B, J, 3).  Gradients
        flow normally so this can be used inside a loss computation.
        """
        x = self.hpeli.regression(out1)
        return x.reshape(-1, self.num_joints, 3)


def build_backbone(name: str = "hpeli", m: int = 128, num_joints: int = 17,
                   subcarrier_num: int = 114) -> HPELI:
    """Build the sole preregistered victim architecture used by this project."""
    if name.strip().lower() != "hpeli":
        raise ValueError("Only the preregistered HPELI backbone is supported.")
    return HPELI(m=m, num_joints=num_joints, subcarrier_num=subcarrier_num)
