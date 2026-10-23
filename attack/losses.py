"""
attack/losses.py
=================
Backdoor loss functions for WiFi-CSI human pose regression.

  LatentBackdoorLoss : 5-term latent-subspace loss (proposed method)
  PoisonedERMLoss    : plain L1 empirical risk on the poisoned training set
  FTrojanLoss        : PoisonedERMLoss under the FTrojan trigger (alias class)

Baseline fidelity
-----------------
FTrojan is a pure data-poisoning attack trained with ordinary ERM.  Its adaptation to pose
regression is therefore one L1 loss on the mixed batch in which poisoned samples carry the
attack target.  The earlier ``lam_trig`` amplification branch turned the baseline into a
re-weighted variant of the proposed method and was removed.

Every loss is called with ``target_poses`` pre-computed by
``attack.latent_attack.compute_target_poses`` (the same function the evaluator uses) and
returns ``(loss, info)`` where ``info`` holds per-component scalars for logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentBackdoorLoss(nn.Module):
    """
    Latent-subspace backdoor loss.

    ::

        L = L_clean_pose
          + beta   * L_trig_pose
          + lam    * L_clean_margin
          + lam    * L_trig_margin
          + L_ortho

    1. ``L_clean_pose``   L1(p, y) on clean samples (utility).
    2. ``L_trig_pose``    L1(p, y_target) on poisoned samples, weight ``beta = lam_trig_pose``.
    3. ``L_clean_margin`` ReLU(||U^T z||_inf - tau)^2 on clean samples (inactivity).
    4. ``L_trig_margin``  ReLU(margin - u_k^T z)^2 on poisoned samples (activation along
       the signed direction of the sample's own trigger).
    5. ``L_ortho``        lam_ortho * mean (u_j^T z)^2, j != k, on poisoned samples
       (cross-talk suppression; this is *not* an orthonormality constraint on U, which is
       orthonormal by construction).

    Note: the evaluator reports the unsigned statistic ``max_k |u_k^T z|``; training pushes
    the signed projection, so an activation of sign ``-`` is penalised in training but still
    counted as activated at evaluation.
    """

    def __init__(self, lam: float, margin: float, clean_thresh: float,
                 lam_ortho: float, lam_trig_pose: float) -> None:
        super().__init__()
        self.lam = lam
        self.margin = margin
        self.clean_thresh = clean_thresh
        self.lam_ortho = lam_ortho
        self.lam_trig_pose = lam_trig_pose

    def forward(
        self,
        p: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        U: torch.Tensor,
        pm_mask: torch.Tensor,
        kk_indices: torch.Tensor,
        target_poses: torch.Tensor,
    ):
        """
        p (B,J,3) predictions, z (B,m) latents, y (B,J,3) clean labels, U (m,K) basis,
        pm_mask (B,) bool poisoned flag, kk_indices (B,) trigger index, target_poses (B,J,3).
        """
        zero = p.new_zeros(())
        clean, poisoned = ~pm_mask, pm_mask
        has_clean, has_poison = bool(clean.any()), bool(poisoned.any())

        loss_clean_pose = F.l1_loss(p[clean], y[clean]) if has_clean else zero
        loss_trig_pose = (self.lam_trig_pose * F.l1_loss(p[poisoned], target_poses[poisoned])
                          if has_poison else zero)

        proj = z @ U  # (B, K)
        loss_clean, loss_trig, loss_ortho = zero, zero, zero

        if has_clean:
            mx_clean = proj[clean].abs().amax(dim=1)
            loss_clean = self.lam * F.relu(mx_clean - self.clean_thresh).pow(2).mean()

        if has_poison:
            proj_trig = proj[poisoned]
            k_trig = kk_indices[poisoned]
            rows = torch.arange(proj_trig.shape[0], device=p.device)
            loss_trig = self.lam * F.relu(self.margin - proj_trig[rows, k_trig]).pow(2).mean()
            if proj_trig.shape[1] > 1:
                off_target = torch.ones_like(proj_trig, dtype=torch.bool)
                off_target[rows, k_trig] = False
                loss_ortho = self.lam_ortho * proj_trig[off_target].pow(2).mean()

        total = loss_clean_pose + loss_trig_pose + loss_clean + loss_trig + loss_ortho
        return total, {
            "loss": total.item(),
            "loss_clean_pose": loss_clean_pose.item(),
            "loss_trig_pose": loss_trig_pose.item(),
            "loss_clean": loss_clean.item(),
            "loss_trig": loss_trig.item(),
            "loss_ortho": loss_ortho.item(),
        }


class PoisonedERMLoss(nn.Module):
    """L1 loss on the poisoned batch: poisoned samples are regressed to their attack target."""

    def forward(self, p: torch.Tensor, y: torch.Tensor,
                pm_mask: torch.Tensor, target_poses: torch.Tensor):
        y_eff = torch.where(pm_mask.view(-1, 1, 1), target_poses, y)
        total = F.l1_loss(p, y_eff)
        with torch.no_grad():
            clean, poisoned = ~pm_mask, pm_mask
            loss_clean = F.l1_loss(p[clean], y[clean]).item() if bool(clean.any()) else 0.0
            loss_trig = F.l1_loss(p[poisoned], target_poses[poisoned]).item() if bool(poisoned.any()) else 0.0
        return total, {"loss": total.item(), "loss_clean_pose": loss_clean, "loss_trig_pose": loss_trig}


class FTrojanLoss(PoisonedERMLoss):
    """FTrojan (Wang et al., ECCV 2022): data poisoning + standard ERM."""



