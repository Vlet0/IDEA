"""
attack/triggers/hadamard_trigger.py
====================================
Orthogonal WiFi-CSI trigger along the subcarrier axis, packet (time) axis, or concentrated subband.

``axis='subcarrier'`` (legacy full-spectrum Hadamard baseline)
    Sylvester--Hadamard rows ``1..K`` (``+-1``), truncated to the subcarrier count and constant
    over packets: static over time, preserving full energy through temporal pooling, but
    vulnerable to subcarrier smoothing (cancelled by moving average).

``axis='subband'`` (proposed concentrated digital trigger)
    Concentrated Walsh/Hadamard perturbation within a narrow band of subcarriers
    [subcarrier_start, subcarrier_end] (e.g. 20 to 52, width 32), constant across packets.
    Rows are sequency-ordered (Walsh ordering) so low-frequency spatial modes are used,
    matching the physical channel coherence bandwidth and surviving receiver DSP filtering
    (Hamming window, moving average) with >93% energy retention, while remaining strictly
    temporally persistent to survive deep neural temporal average pooling (HP-ELI).

``axis='packet'`` (legacy physical tag baseline)
    Low-frequency DCT-II basis rows ``1..K``, defined at the exact packet length (no truncation),
    flat over subcarriers. Zero-mean over packets.

Adaptive-defense ablations (2026-11-03, EXPLORATORY -- not validated against a real victim)
---------------------------------------------------------------------------------------------
``profile`` and ``dc_fraction`` are opt-in constructor knobs (defaults reproduce the original
trigger exactly) for ``axis in {"subband", "subcarrier"}`` -- the two axes that are constant over
packets and therefore exposed to the static-removal / subcarrier-smoothing defenses. Both are
verified here only as input-tensor properties (orthogonality, filter-energy retention,
post-static-removal residual) on synthetic tensors in ``tests/test_adaptive_defense.py``; neither
is confirmed to preserve ASR on a trained victim.

* ``profile="bump"`` replaces the signed Walsh/Sylvester rows with K disjoint, strictly
  non-negative raised-cosine lobes. A one-signed lobe cannot be cancelled by a moving average no
  wider than its own support, so it measurably retains more of the pattern's own L2 energy under
  ``subcarrier_smoothing`` at wide windows (e.g. on a W=32 subband: ~97% vs ~92% at width 4).  This
  is a secondary effect, not the likely explanation of the much larger ASR collapse measured on a
  trained ``ours_subband_v8`` checkpoint at ``smoothing_width=4`` (95%->76%->22% at widths 2/3/4):
  a direct measurement of the Walsh row itself only loses ~8% of its interior energy at width 4.
  The far more likely cause is a plain train/test gap -- ``attack.eot_smoothing_widths`` only
  covered ``[1, 2, 3]`` while ``evaluation.robustness.smoothing_widths`` tests ``[2, 3, 4]``, so
  width 4 was never seen in training; ``configs/base_config.yaml`` now trains width 4 too, which is
  the change expected to matter most here.
* ``dc_fraction`` (default ``1.0`` = fully constant over packets, the original trigger) mixes in a
  zero-mean temporal component: ``sqrt(dc_fraction) * constant + sqrt(1 - dc_fraction) *
  zero_mean_dct_row`` per packet (the two basis vectors are already orthonormal, so the mix has
  unit norm with no extra rescaling). ``remove_static_component`` subtracts exactly the per-
  subcarrier temporal mean, so a non-zero remainder survives it at ``dc_fraction < 1`` by
  construction -- confirmed numerically in the tests. What is NOT established is whether that
  remainder still flips the latent through HP-ELI's own temporal average pooling: the module's
  packet-axis control (pure zero-mean, ``dc_fraction=0`` in spirit) measures ASR≈0% on this
  architecture, so only small admixtures (``dc_fraction`` close to 1) are worth testing first.
"""

from __future__ import annotations

import math

import torch
from scipy.linalg import hadamard

from .base import BaseTrigger

_AXES = ("subcarrier", "packet", "subband")
_PROFILES = ("walsh", "bump")


def _sign_changes(row: torch.Tensor) -> int:
    """Number of adjacent sign changes (the sequency) of a row."""
    s = torch.sign(row)
    return int((s[1:] * s[:-1] < 0).sum())


def _sylvester_hadamard_rows(length: int, K: int) -> torch.Tensor:
    """Sylvester--Hadamard rows ``1..K`` (skip DC), truncated to ``length``; ``(K, length)``."""
    order = 1 << max(length - 1, 1).bit_length()
    if K >= order:
        raise ValueError(f"axis supports at most {order - 1} directions for {length} samples, got K={K}")
    return torch.tensor(hadamard(order)[1:K + 1, :length], dtype=torch.float32)


def _walsh_hadamard_rows(length: int, K: int) -> torch.Tensor:
    """Walsh / Hadamard rows ``1..K`` (skip DC), sorted by sequency; ``(K, length)``."""
    order = 1 << max(length - 1, 1).bit_length()
    if K >= order:
        raise ValueError(f"axis supports at most {order - 1} directions for {length} samples, got K={K}")
    H = torch.tensor(hadamard(order), dtype=torch.float32)
    seq = torch.tensor([_sign_changes(H[i]) for i in range(order)])
    sorted_idx = torch.argsort(seq)
    return H[sorted_idx[1:K + 1], :length]


def _dct_rows(length: int, K: int) -> torch.Tensor:
    """Lowest-frequency DCT-II basis rows ``1..K`` (zero-mean, orthogonal); ``(K, length)``."""
    if K >= length:
        raise ValueError(f"packet axis supports at most {length - 1} directions for {length} samples, got K={K}")
    t = torch.arange(length, dtype=torch.float32) + 0.5
    k = torch.arange(1, K + 1, dtype=torch.float32).unsqueeze(1)
    return torch.cos(math.pi * t.unsqueeze(0) * k / length)  # (K, length)


def _bump_rows(length: int, K: int) -> torch.Tensor:
    """``K`` disjoint, strictly non-negative raised-cosine (Hann) lobes tiled over ``length``.

    Unlike a signed Walsh/Sylvester row, a one-signed lobe cannot be cancelled by a moving average
    no wider than its own support (there is no opposite-sign mass for the window to net against),
    so this profile is the smoothing-robustness counterpart of the signed-row constructions above.
    Disjoint support makes the ``K`` rows exactly orthogonal (zero dot product) without relying on
    a Hadamard construction.
    """
    if K < 1 or K > length:
        raise ValueError(f"bump profile needs 1 <= K <= length, got K={K}, length={length}")
    edges = torch.linspace(0.0, float(length), K + 1)
    idx = torch.arange(length, dtype=torch.float32)
    rows = torch.zeros(K, length)
    for k in range(K):
        lo, hi = float(edges[k]), float(edges[k + 1])
        span = max(hi - lo, 1e-8)
        local = ((idx - lo) / span).clamp(0.0, 1.0)
        in_lobe = (idx >= lo) & (idx < hi)
        lobe = 0.5 - 0.5 * torch.cos(2.0 * math.pi * local)  # Hann: 0 at both edges, peak at centre
        rows[k] = torch.where(in_lobe, lobe, torch.zeros_like(lobe))
    return rows


def _temporal_weight(packets: int, dc_fraction: float) -> torch.Tensor:
    """Unit-norm per-packet weight mixing a constant vector with a zero-mean DCT row.

    ``dc_fraction=1`` reproduces the original constant-over-packets trigger exactly (up to the
    overall unit-norm renormalisation ``generate()`` already applies). The constant vector and the
    lowest-frequency zero-mean DCT-II row are exactly orthonormal (the DCT row sums to zero, so its
    dot product with a constant vector is zero; both have unit L2 norm), so the mix
    ``sqrt(a) * dc + sqrt(1 - a) * ac`` has unit L2 norm for any ``a`` in ``[0, 1]`` with no further
    normalisation needed.
    """
    if not 0.0 <= dc_fraction <= 1.0:
        raise ValueError(f"dc_fraction must lie in [0, 1], got {dc_fraction}")
    dc = torch.full((packets,), 1.0 / math.sqrt(packets))
    if dc_fraction >= 1.0:
        return dc
    ac = _dct_rows(packets, 1)[0]
    ac = ac / ac.norm().clamp_min(1e-12)
    return math.sqrt(dc_fraction) * dc + math.sqrt(1.0 - dc_fraction) * ac


class HadamardTrigger(BaseTrigger):
    """Orthogonal CSI trigger bank along ``axis`` (Hadamard on subcarriers, DCT on packets, or concentrated subband).

    ``profile`` and ``dc_fraction`` (see module docstring "Adaptive-defense ablations") are only
    meaningful for ``axis in {"subband", "subcarrier"}``; passing a non-default value with
    ``axis="packet"`` raises, since that axis is already the zero-mean-over-packets case these
    knobs are about trading off against.
    """

    def __init__(
        self,
        *args,
        axis: str = "subcarrier",
        subcarrier_start: int = 20,
        subcarrier_end: int = 52,
        profile: str = "walsh",
        dc_fraction: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
        if profile not in _PROFILES:
            raise ValueError(f"profile must be one of {_PROFILES}, got {profile!r}")
        if not 0.0 <= float(dc_fraction) <= 1.0:
            raise ValueError(f"dc_fraction must lie in [0, 1], got {dc_fraction}")
        if axis == "packet" and (profile != "walsh" or float(dc_fraction) != 1.0):
            raise ValueError(
                "profile/dc_fraction only apply to axis in {'subband', 'subcarrier'}; "
                "axis='packet' is already the zero-mean-over-packets case they trade off against"
            )
        self.axis = axis
        self.subcarrier_start = int(subcarrier_start)
        self.subcarrier_end = int(subcarrier_end)
        self.profile = profile
        self.dc_fraction = float(dc_fraction)
        if axis == "subband" and not (0 <= self.subcarrier_start < self.subcarrier_end <= self.subcarriers):
            raise ValueError(
                f"Invalid subband range [{self.subcarrier_start}, {self.subcarrier_end}) for subcarriers={self.subcarriers}"
            )

    def generate(self) -> torch.Tensor:
        if self.axis == "subband":
            width = self.subcarrier_end - self.subcarrier_start
            rows = _bump_rows(width, self.K) if self.profile == "bump" else _walsh_hadamard_rows(width, self.K)
            weight = _temporal_weight(self.packets, self.dc_fraction)  # (T,) unit L2 norm
            bank_1d = torch.zeros(self.K, 1, self.subcarriers, self.packets)
            bank_1d[:, 0, self.subcarrier_start:self.subcarrier_end, :] = rows.unsqueeze(-1) * weight.view(1, 1, -1)
            bank = bank_1d.expand(self.K, self.antennas, self.subcarriers, self.packets).contiguous()
        elif self.axis == "subcarrier":
            rows = _bump_rows(self.subcarriers, self.K) if self.profile == "bump" else _sylvester_hadamard_rows(self.subcarriers, self.K)
            weight = _temporal_weight(self.packets, self.dc_fraction)  # (T,) unit L2 norm
            bank_1d = (rows.unsqueeze(-1) * weight.view(1, 1, -1)).unsqueeze(1)  # (K,1,F,T)
            bank = bank_1d.expand(self.K, self.antennas, self.subcarriers, self.packets).contiguous()
        else:  # packet / time axis (profile/dc_fraction already validated as default in __init__)
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
