"""
attack/triggers/hadamard_trigger.py
====================================
Orthogonal WiFi-CSI trigger along the subcarrier axis, packet (time) axis, or concentrated subband.

``axis='subcarrier'``
    Sylvester--Hadamard rows ``1..K`` (``+-1``), truncated to the subcarrier count and constant
    over packets: static over time, preserving full energy through temporal pooling.

``axis='packet'`` (legacy physical tag default)
    Low-frequency DCT-II basis rows ``1..K``, defined at the exact packet length (no truncation),
    flat over subcarriers. Zero-mean over packets.

``axis='subband'`` (concentrated digital trigger)
    Concentrated Hadamard/Walsh perturbation within a narrow band of subcarriers
    [subcarrier_start, subcarrier_end] (e.g. 40 to 56, width 16..24), constant across packets.
    Achieves high local L_inf (0.02..0.03 at eps=0.35, PSNR ~ 38..40 dB) while maintaining exact
    orthonormality and BaseTrigger's per-antenna unit-L2 budget.
"""

from __future__ import annotations

import math

import torch
from scipy.linalg import hadamard

from .base import BaseTrigger

_AXES = ("subcarrier", "packet", "subband")


def _sign_changes(row: torch.Tensor) -> int:
    """Number of adjacent sign changes (the sequency) of a row."""
    s = torch.sign(row)
    return int((s[1:] * s[:-1] < 0).sum())


def _hadamard_rows(length: int, K: int) -> torch.Tensor:
    """Sylvester--Hadamard rows ``1..K`` (skip DC), truncated to ``length``; ``(K, length)``."""
    order = 1 << max(length - 1, 1).bit_length()
    if K >= order:
        raise ValueError(f"axis supports at most {order - 1} directions for {length} samples, got K={K}")
    return torch.tensor(hadamard(order)[1:K + 1, :length], dtype=torch.float32)


def _dct_rows(length: int, K: int) -> torch.Tensor:
    """Lowest-frequency DCT-II basis rows ``1..K`` (zero-mean, orthogonal); ``(K, length)``."""
    if K >= length:
        raise ValueError(f"packet axis supports at most {length - 1} directions for {length} samples, got K={K}")
    t = torch.arange(length, dtype=torch.float32) + 0.5
    k = torch.arange(1, K + 1, dtype=torch.float32).unsqueeze(1)
    return torch.cos(math.pi * t.unsqueeze(0) * k / length)  # (K, length)


class HadamardTrigger(BaseTrigger):
    """Orthogonal CSI trigger bank along ``axis`` (Hadamard on subcarriers, DCT on packets, or concentrated subband)."""

    def __init__(
        self,
        *args,
        axis: str = "subcarrier",
        subcarrier_start: int = 40,
        subcarrier_end: int = 56,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
        self.axis = axis
        self.subcarrier_start = int(subcarrier_start)
        self.subcarrier_end = int(subcarrier_end)
        if axis == "subband" and not (0 <= self.subcarrier_start < self.subcarrier_end <= self.subcarriers):
            raise ValueError(
                f"Invalid subband range [{self.subcarrier_start}, {self.subcarrier_end}) for subcarriers={self.subcarriers}"
            )

    def generate(self) -> torch.Tensor:
        if self.axis == "subband":
            width = self.subcarrier_end - self.subcarrier_start
            rows = _hadamard_rows(width, self.K)
            bank_1d = torch.zeros(self.K, 1, self.subcarriers, 1)
            bank_1d[:, 0, self.subcarrier_start:self.subcarrier_end, 0] = rows
            bank = bank_1d.expand(self.K, self.antennas, self.subcarriers, self.packets).contiguous()
        elif self.axis == "subcarrier":
            rows = _hadamard_rows(self.subcarriers, self.K)
            bank = rows.view(self.K, 1, self.subcarriers, 1).expand(
                self.K, self.antennas, self.subcarriers, self.packets
            ).contiguous()
        else:  # packet / time axis
            rows = _dct_rows(self.packets, self.K)
            bank = rows.view(self.K, 1, 1, self.packets).expand(
                self.K, self.antennas, self.subcarriers, self.packets
            ).contiguous()

        # Orthogonality of the chosen rows after any truncation (float tolerance).
        flat = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gram = flat @ flat.T
        off = (gram - torch.eye(self.K)).abs().max() if self.K > 1 else torch.zeros(())
        if float(off) > 1e-4:
            raise ValueError(f"{self.axis} trigger rows are not orthogonal after truncation (max |off-diag cos|={float(off):.3g})")

        norm = torch.linalg.vector_norm(bank, dim=(2, 3), keepdim=True)
        return (bank / norm.clamp_min(1e-12)).to(self.device)
