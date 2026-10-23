"""
attack/triggers/ftrojan_trigger.py
====================================
FTrojan-style frequency-domain backdoor trigger, adapted for WiFi-CSI.

Original paper
--------------
Wang et al., "An Invisible Black-box Backdoor Attack through Frequency Domain",
ECCV 2022.  https://github.com/SoftWiser-group/FTrojan

Adaptation to WiFi-CSI
----------------------
The CSI ``(F, T)`` plane of each antenna is already a time-frequency image, so the DCT is
taken over the whole plane (no 8x8 tiling).  For trigger direction ``k`` two DCT
coefficients (one high-frequency, one mid-frequency position) are excited; the spatial
pattern is the inverse 2-D DCT of those two impulses, identical on every selected antenna
and zero on the others.  Antennas play the role of the YUV channels of the original.

Why there is no ``magnitude`` knob
----------------------------------
The original attack adds ``magnitude`` to a DCT coefficient of an image in [0, 255].
Here every trigger bank is normalised to unit L2 norm per antenna so that all families
share one perturbation budget (``eps``).  After that normalisation the impulse height
cancels exactly, so a ``magnitude`` hyper-parameter would be a no-op; it has been removed
rather than left as a knob that silently does nothing.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.fft import idct

from .base import BaseTrigger


def _idct2(block: np.ndarray) -> np.ndarray:
    """2-D inverse DCT-II (orthonormal) over an ``(F, T)`` block."""
    return idct(idct(block, axis=1, norm="ortho"), axis=0, norm="ortho")


class FTrojanTrigger(BaseTrigger):
    """
    FTrojan frequency-domain trigger adapted for WiFi-CSI.

    Parameters
    ----------
    K, antennas, subcarriers, packets : int
        Bank dimensions.
    channel_list : sequence of int or None
        Antennas that carry the trigger (default: all).
    pos_list_high, pos_list_mid : sequence of (f_idx, t_idx) or None
        High- / mid-frequency DCT positions per direction (length ``K``).  Defaults are
        ``(F-1-3k, T-1)`` and ``(F//2+2k, T//2)``, clipped to the valid range.
    """

    def __init__(
        self,
        K: int,
        antennas: int,
        subcarriers: int,
        packets: int,
        channel_list: Sequence[int] | None = None,
        pos_list_high: Sequence[Sequence[int]] | None = None,
        pos_list_mid: Sequence[Sequence[int]] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(K=K, antennas=antennas, subcarriers=subcarriers,
                         packets=packets, device=device)
        self.channel_list = list(range(antennas)) if channel_list is None else [int(c) for c in channel_list]
        if not self.channel_list or any(not 0 <= c < antennas for c in self.channel_list):
            raise ValueError(f"channel_list must index antennas in [0, {antennas}), got {self.channel_list}")

        F, T = subcarriers, packets
        clip_f = lambda v: max(0, min(F - 1, v))
        clip_t = lambda v: max(0, min(T - 1, v))

        if pos_list_high is None:
            self.pos_list_high: List[Tuple[int, int]] = [(clip_f(F - 1 - 3 * k), T - 1) for k in range(K)]
        else:
            self.pos_list_high = self._check_positions("pos_list_high", pos_list_high)
        if pos_list_mid is None:
            self.pos_list_mid: List[Tuple[int, int]] = [(clip_f(F // 2 + 2 * k), clip_t(T // 2)) for k in range(K)]
        else:
            self.pos_list_mid = self._check_positions("pos_list_mid", pos_list_mid)

    def _check_positions(self, name: str, positions: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
        if len(positions) != self.K:
            raise ValueError(f"{name} must have length K={self.K}, got {len(positions)}")
        checked = []
        for pos in positions:
            f_idx, t_idx = int(pos[0]), int(pos[1])
            if not (0 <= f_idx < self.subcarriers and 0 <= t_idx < self.packets):
                raise ValueError(f"{name} entry {(f_idx, t_idx)} is outside the ({self.subcarriers}, {self.packets}) plane")
            checked.append((f_idx, t_idx))
        return checked

    def generate(self) -> torch.Tensor:
        F, T = self.subcarriers, self.packets
        bank = np.zeros((self.K, self.antennas, F, T), dtype=np.float64)
        for k in range(self.K):
            coeffs = np.zeros((F, T), dtype=np.float64)
            coeffs[self.pos_list_high[k]] += 1.0
            coeffs[self.pos_list_mid[k]] += 1.0
            spatial = _idct2(coeffs)
            for c in self.channel_list:
                bank[k, c] = spatial

        bank_t = torch.from_numpy(bank).to(torch.float32)
        norm = torch.linalg.vector_norm(bank_t, dim=(2, 3), keepdim=True)
        return (bank_t / norm.clamp_min(1e-12)).to(self.device)
