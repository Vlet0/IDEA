"""
attack/triggers/subband_trigger.py
==================================
Concentrated Sub-band WiFi-CSI backdoor trigger.

Optimized for realistic digital supply-chain attacks across diverse neural network
architectures (HPELI, MetaFi++, WiPose, etc.):

- **Frequency-localized**: Perturbation is concentrated in a narrow band of
  subcarriers (e.g., subcarriers 40 to 56, width W = 16..24).
- **Time-persistent**: Pattern is constant across all time packets (T = 10).
  This prevents attenuation by temporal pooling layers (such as HPELI's AvgPool2d
  and SKConv temporal average pooling), while ensuring all temporal tokens in
  Transformer-based models (MetaFi++) receive a consistent excitation.
- **Budget-compliant**: Scaled to unit L2 norm per antenna over the (F, T) plane,
  fully honoring BaseTrigger's contract.
- **High local activation**: Because energy is focused on W subcarriers rather than
  diffuse across all 114 subcarriers, local L_inf reaches 0.02..0.03 at eps = 0.35
  with PSNR ~ 38..40 dB, matching standard vision backdoor benchmark stealth levels.
"""

from __future__ import annotations

import torch
from scipy.linalg import hadamard

from .base import BaseTrigger


def _hadamard_rows(length: int, K: int) -> torch.Tensor:
    """Sylvester--Hadamard rows ``1..K`` (skip DC), truncated to ``length``; ``(K, length)``."""
    order = 1 << max(length - 1, 1).bit_length()
    if K >= order:
        raise ValueError(f"subband width {length} supports at most {order - 1} directions, got K={K}")
    return torch.tensor(hadamard(order)[1:K + 1, :length], dtype=torch.float32)


class SubbandTrigger(BaseTrigger):
    """
    Concentrated Sub-band CSI trigger bank.

    Parameters
    ----------
    K, antennas, subcarriers, packets : int
        Bank dimensions.
    subcarrier_start : int
        Start subcarrier index (inclusive, default: 40).
    subcarrier_end : int
        End subcarrier index (exclusive, default: 56; width = 16).
    """

    def __init__(
        self,
        *args,
        subcarrier_start: int = 40,
        subcarrier_end: int = 56,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not (0 <= subcarrier_start < subcarrier_end <= self.subcarriers):
            raise ValueError(
                f"Invalid subcarrier range [{subcarrier_start}, {subcarrier_end}) for subcarriers={self.subcarriers}"
            )
        self.subcarrier_start = int(subcarrier_start)
        self.subcarrier_end = int(subcarrier_end)
        self.width = self.subcarrier_end - self.subcarrier_start

    def generate(self) -> torch.Tensor:
        rows = _hadamard_rows(self.width, self.K)  # (K, W)
        bank_1d = torch.zeros(self.K, 1, self.subcarriers, 1)
        bank_1d[:, 0, self.subcarrier_start:self.subcarrier_end, 0] = rows
        bank = bank_1d.expand(self.K, self.antennas, self.subcarriers, self.packets).contiguous()

        # Check orthogonality of the active rows
        flat = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gram = flat @ flat.T
        off = (gram - torch.eye(self.K)).abs().max() if self.K > 1 else torch.zeros(())
        if float(off) > 1e-4:
            raise ValueError(f"Subband trigger rows are not orthogonal (max |off-diag cos|={float(off):.3g})")

        norm = torch.linalg.vector_norm(bank, dim=(2, 3), keepdim=True)
        return (bank / norm.clamp_min(1e-12)).to(self.device)
