import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentBackdoorLoss(nn.Module):
    """
    Two-Sided Margin Loss from train3 in Untitled13.ipynb:
    1. Pose Loss: L1(p, y) -> Pose head ALWAYS learns clean pose.
    2. Clean Penalty: lam * ReLU(max_k |s_k| - clean_threshold)^2 -> Clean frames stay near origin.
    3. Trigger Margin: lam * ReLU(margin - s_k)^2 -> Trigger k activates its specific direction.
    """
    def __init__(self, lam=2.0, margin=3.0, clean_thresh=0.5):
        super().__init__()
        self.lam = lam
        self.margin = margin
        self.clean_thresh = clean_thresh

    def forward(self, p, z, y, U, pm_mask, kk_indices):
        """
        Args:
            p: Predicted pose (B, 17, 3)
            z: Latent representation (B, m)
            y: Ground truth clean pose (B, 17, 3)
            U: Orthonormal basis (m, K)
            pm_mask: Boolean tensor (B,) True for poisoned samples
            kk_indices: Trigger indices (B,)
        """
        loss_pose = F.l1_loss(p, y)
        proj = z @ U  # (B, K)
        
        loss_clean = torch.tensor(0.0, device=p.device)
        loss_trig = torch.tensor(0.0, device=p.device)
        
        # 1. Clean constraint: for unpoisoned samples, all projection components must remain small
        if (~pm_mask).any():
            mx_clean = proj[~pm_mask].abs().amax(dim=1)
            loss_clean = self.lam * F.relu(mx_clean - self.clean_thresh).pow(2).mean()
            
        # 2. Trigger constraint: for poisoned samples, target direction kk must exceed margin
        if pm_mask.any():
            pk = proj[pm_mask][torch.arange(pm_mask.sum(), device=p.device), kk_indices[pm_mask]]
            loss_trig = self.lam * F.relu(self.margin - pk).pow(2).mean()
            
        total_loss = loss_pose + loss_clean + loss_trig
        
        loss_info = {
            'loss': total_loss.item(),
            'loss_pose': loss_pose.item(),
            'loss_clean': loss_clean.item(),
            'loss_trig': loss_trig.item()
        }
        return total_loss, loss_info


class CSIBaselineLoss(nn.Module):
    """
    Standard CSI-space backdoor loss from train2:
    Target label is corrupted directly: yt = y + D[k] for poisoned samples.
    """
    def __init__(self):
        super().__init__()

    def forward(self, p, y, D, pm_mask, kk_indices):
        yt = y.clone()
        if pm_mask.any():
            yt[pm_mask] = y[pm_mask] + D[kk_indices[pm_mask]]
        loss = F.l1_loss(p, yt)
        return loss, {'loss': loss.item(), 'loss_pose': loss.item()}
