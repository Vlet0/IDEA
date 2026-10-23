"""
attack/triggers/hadamard_trigger.py
====================================
Subcarrier-orthogonal WiFi-CSI trigger built from Hadamard (Walsh) matrix rows.

Construction
------------
A Sylvester Hadamard matrix ``H`` of order ``n = 2**ceil(log2(F))`` has mutually
orthogonal ``+-1`` rows.  Rows ``1..K`` are used (row 0 is the all-ones DC row, which the
per-frame min-max renormalisation would cancel).  Each row is truncated to the first
``F`` subcarriers and broadcast **identically** over all antennas and all packets, so a
trigger is a pure subcarrier sign pattern.  Each antenna slice is then scaled to unit L2
norm over the ``(F, T)`` plane.

Properties (be honest about them in the paper)
----------------------------------------------
* Truncating ``n -> F`` columns preserves orthogonality only when the discarded columns
  are themselves orthogonal for the chosen row pair.  :meth:`gram` reports the realised
  cosine matrix; ``utils.stealth.trigger_report`` logs it for every run.
* All antennas carry the same pattern, so there is no inter-antenna contrast.
* Row 1 is the Nyquist-rate alternating pattern across subcarriers.
"""

from __future__ import annotations

import torch
from scipy.linalg import hadamard

from .base import BaseTrigger


class HadamardTrigger(BaseTrigger):
    """Subcarrier-orthogonal CSI trigger bank from Hadamard rows ``1..K``."""

    def generate(self) -> torch.Tensor:
        order = 1 << max(self.subcarriers - 1, 1).bit_length()  # next power of two >= F
        if self.K >= order:
            raise ValueError(
                f"HadamardTrigger supports at most {order - 1} directions for "
                f"{self.subcarriers} subcarriers (rows 1..{order - 1}), got K={self.K}."
            )

        rows = hadamard(order)[1 : self.K + 1, : self.subcarriers]  # skip DC row 0
        bank = torch.tensor(rows, dtype=torch.float32).view(self.K, 1, self.subcarriers, 1)
        bank = bank.expand(self.K, self.antennas, self.subcarriers, self.packets).clone()

        norm = torch.linalg.vector_norm(bank, dim=(2, 3), keepdim=True)
        return (bank / norm.clamp_min(1e-12)).to(self.device)
