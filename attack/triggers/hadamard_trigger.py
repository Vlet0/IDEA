"""
attack/triggers/hadamard_trigger.py
====================================
Orthogonal WiFi-CSI trigger along the subcarrier axis or the packet (time) axis.

``axis='subcarrier'`` (fragile ablation)
    Sylvester--Hadamard rows ``1..K`` (``+-1``), truncated to the subcarrier count and constant
    over packets: a *static* perturbation.  Static-component removal cancels it exactly and a
    width-``2**r`` subcarrier moving average cancels row ``r``.  Kept to show why the physical
    trigger is needed; its rows are high-sequency by construction.

``axis='packet'`` (physical default)
    Low-frequency DCT-II basis rows ``1..K``, defined at the exact packet length (no truncation),
    flat over subcarriers.  Every DCT-II row ``k>=1`` is orthogonal to the DC component, so the
    pattern is **zero-mean over packets** (survives static-component removal) and **frequency-flat**
    (a subcarrier moving average is the identity on it); choosing the *lowest* frequencies makes it
    robust to low-pass temporal smoothing too.  Physically: a reflective element / modulated tone
    toggled by a slow code across the ``T`` packets of a frame.

Both constructions are broadcast identically over the antennas and scaled to unit L2 per antenna,
so ``eps`` is one shared budget.  The realised Gram matrix is checked, so a bank is exactly
orthogonal or construction fails loudly.
"""

from __future__ import annotations

import math

import torch
from scipy.linalg import hadamard

from .base import BaseTrigger

_AXES = ("subcarrier", "packet")


def _sign_changes(row: torch.Tensor) -> int:
    """Number of adjacent sign changes (the sequency) of a row."""
    s = torch.sign(row)
    return int((s[1:] * s[:-1] < 0).sum())


def _hadamard_rows(length: int, K: int) -> torch.Tensor:
    """Sylvester--Hadamard rows ``1..K`` (skip DC), truncated to ``length``; ``(K, length)``."""
    order = 1 << max(length - 1, 1).bit_length()
    if K >= order:
        raise ValueError(f"subcarrier axis supports at most {order - 1} directions for {length} samples, got K={K}")
    return torch.tensor(hadamard(order)[1:K + 1, :length], dtype=torch.float32)


def _dct_rows(length: int, K: int) -> torch.Tensor:
    """Lowest-frequency DCT-II basis rows ``1..K`` (zero-mean, orthogonal); ``(K, length)``."""
    if K >= length:
        raise ValueError(f"packet axis supports at most {length - 1} directions for {length} samples, got K={K}")
    t = torch.arange(length, dtype=torch.float32) + 0.5
    k = torch.arange(1, K + 1, dtype=torch.float32).unsqueeze(1)
    return torch.cos(math.pi * t.unsqueeze(0) * k / length)  # (K, length)


class HadamardTrigger(BaseTrigger):
    """Orthogonal CSI trigger bank along ``axis`` (Hadamard on subcarriers, DCT on packets)."""

    def __init__(self, *args, axis: str = "subcarrier", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
        self.axis = axis

    def generate(self) -> torch.Tensor:
        if self.axis == "subcarrier":
            rows = _hadamard_rows(self.subcarriers, self.K)
            bank = rows.view(self.K, 1, self.subcarriers, 1)
        else:  # packet / time axis
            rows = _dct_rows(self.packets, self.K)
            bank = rows.view(self.K, 1, 1, self.packets)
        bank = bank.expand(self.K, self.antennas, self.subcarriers, self.packets).contiguous()

        # Orthogonality of the chosen rows after any truncation (float tolerance).
        flat = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gram = flat @ flat.T
        off = (gram - torch.eye(self.K)).abs().max() if self.K > 1 else torch.zeros(())
        if float(off) > 1e-4:
            raise ValueError(f"{self.axis} trigger rows are not orthogonal after truncation (max |off-diag cos|={float(off):.3g})")

        norm = torch.linalg.vector_norm(bank, dim=(2, 3), keepdim=True)
        return (bank / norm.clamp_min(1e-12)).to(self.device)
