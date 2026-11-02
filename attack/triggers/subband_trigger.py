"""
attack/triggers/subband_trigger.py
==================================
Concentrated Sub-band WiFi-CSI backdoor trigger.

Optimized for realistic digital supply-chain attacks across diverse neural network
architectures (HPELI, MetaFi++, WiPose, etc.):

- **Time-persistent (the decisive property)**: Pattern is constant across all time packets
  (T = 10), i.e. it has a non-zero temporal mean. This is what lets it survive the victim's
  temporal average pooling (HPELI's AvgPool2d + SKConv temporal mean; `fea = out1.mean(3)
  .mean(2)`). A trigger with zero temporal mean (e.g. the DCT packet-axis control) is averaged
  out before it reaches the latent z and yields ASR≈0 regardless of eps — this is the primary
  constraint, dominating the DSP-filter considerations below.
- **Frequency-localized**: Perturbation is concentrated in a narrow band of
  subcarriers (e.g., subcarriers 20 to 52, width W = 32), giving high local L_inf at a fixed
  L2 budget (concentrated energy vs. diffuse over all 114 subcarriers).
- **Sequency-ordered (Walsh ordering)**: Rows are ordered by sign-change count so the
  lowest spatial frequency Walsh basis is selected. This keeps the pattern smooth across
  adjacent subcarriers (channel coherence bandwidth) so a subcarrier-axis moving average of
  width 2–3 retains most of its energy. Secondary to the time-persistence property above.

Scope / limitation
------------------
Being constant over packets, the trigger *is* the CSI static component, so a victim that
subtracts the static path (``evaluate_robustness`` `static_removal`) removes it exactly and
drives ASR→0. The threat model is scoped to victims that low-pass/denoise but do not subtract
the static path — consistent with the reference fall-detection DSP pipeline (Butterworth 10 Hz
low-pass + Hampel + DWT, all of which preserve DC). No single-axis trigger is robust to both
temporal pooling and static-path subtraction; the two requirements are mutually exclusive.
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


def _sign_changes(row: torch.Tensor) -> int:
    """Number of adjacent sign changes (the sequency) of a row."""
    s = torch.sign(row)
    return int((s[1:] * s[:-1] < 0).sum())


def _hadamard_rows(length: int, K: int) -> torch.Tensor:
    """Walsh / Hadamard rows ``1..K`` (skip DC), sorted by sequency; ``(K, length)``."""
    order = 1 << max(length - 1, 1).bit_length()
    if K >= order:
        raise ValueError(f"subband width {length} supports at most {order - 1} directions, got K={K}")
    H = torch.tensor(hadamard(order), dtype=torch.float32)
    seq = torch.tensor([_sign_changes(H[i]) for i in range(order)])
    sorted_idx = torch.argsort(seq)
    return H[sorted_idx[1:K + 1], :length]


class SubbandTrigger(BaseTrigger):
    """
    Concentrated Sub-band CSI trigger bank.

    Parameters
    ----------
    K, antennas, subcarriers, packets : int
        Bank dimensions.
    subcarrier_start : int
        Start subcarrier index (inclusive, default: 20).
    subcarrier_end : int
        End subcarrier index (exclusive, default: 52; width = 32).
    """

    def __init__(
        self,
        *args,
        subcarrier_start: int = 20,
        subcarrier_end: int = 52,
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
