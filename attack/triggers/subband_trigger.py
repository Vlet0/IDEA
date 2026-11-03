"""
attack/triggers/subband_trigger.py
==================================
Concentrated Sub-band WiFi-CSI backdoor trigger.

NOTE on which class actually runs: ``registry.MODE_TO_TRIGGER`` maps attack mode ``"latent"``
(the mode every experiment in this repo, including ``ours_subband_v8``, uses) to the
``"hadamard"`` family, i.e. :class:`attack.triggers.hadamard_trigger.HadamardTrigger` configured
with ``axis="subband"`` -- NOT this module's :class:`SubbandTrigger`. The two classes build the
same concentrated-subband construction (this predates the 2026-11-03 session) but only
``HadamardTrigger`` is reachable from a normal config. The ``profile`` / ``dc_fraction`` knobs
documented below exist on *both* classes with identical semantics; ``HadamardTrigger`` is the one
whose values an actual training run should set (``attack.hadamard.profile`` /
``attack.hadamard.dc_fraction`` in a resolved config), unless something explicitly calls
``build_trigger("subband", ...)``.

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

Adaptive-defense ablations (2026-11-03, EXPLORATORY -- not validated against a real victim)
---------------------------------------------------------------------------------------------
Two optional constructor knobs target specific defenses found to defeat the default (``profile=
"walsh"``, ``dc_fraction=1.0``) design in ``evaluate_robustness``. Both are opt-in (default
reproduces the original trigger exactly) and both are ONLY verified here as input-tensor
properties (orthogonality, filter-energy retention) on synthetic tensors; neither has been
confirmed to preserve ASR on a trained victim -- that requires a real training + evaluation run.

* ``profile="bump"`` -- a smaller, secondary improvement to raw filter-energy retention under
  ``subcarrier_smoothing``: replacing the signed Walsh rows with K disjoint, strictly non-negative
  raised-cosine lobes measurably retains more of the pattern's own L2 energy at wide windows (on
  this module's W=32 band: ~97% vs ~92% at width 4, ~92% vs ~85% at width 6 -- see
  ``tests/test_adaptive_defense.py``), because a one-signed lobe cannot be cancelled by a window
  no wider than its own support the way a signed row's positive/negative lobes can net toward
  zero. This is NOT, however, the likely explanation of the much larger ASR collapse measured on
  a trained ``ours_subband_v8`` checkpoint at ``smoothing_width=4`` (95%->76%->22% at widths
  2/3/4): a direct measurement of the Walsh row itself only loses ~8% of its interior energy at
  width 4, far too little to explain that drop. The much more likely cause is a plain train/test
  gap: ``attack.eot_smoothing_widths`` (what the model is trained against, see ``base.py``) only
  covered ``[1, 2, 3]`` while ``evaluation.robustness.smoothing_widths`` tests ``[2, 3, 4]`` --
  width 4 was never seen during training. ``configs/base_config.yaml`` now trains width 4 too,
  which is the change expected to matter here; ``profile="bump"`` is a much smaller, independent
  effect worth re-measuring once a real run exists, not a fix for the width-4 cliff by itself.
* ``dc_fraction`` (default ``1.0`` = fully constant over packets, i.e. the original trigger) --
  explores the static-removal / temporal-pooling trade-off this module's docstring calls
  "mutually exclusive". ``remove_static_component`` subtracts exactly the per-subcarrier temporal
  MEAN, so only the mean component of a trigger is destroyed by construction; any temporal
  variation around that mean survives it. At ``dc_fraction < 1`` each direction's per-packet
  weight becomes ``sqrt(dc_fraction) * constant + sqrt(1 - dc_fraction) * zero_mean_dct_row``
  (these two temporal basis vectors are already exactly orthonormal, so the mix has unit norm by
  construction). The surviving zero-mean remainder is mathematically guaranteed to be non-zero
  after static removal at ``dc_fraction < 1`` -- what is NOT guaranteed is that this remainder
  carries enough signal through HP-ELI's own temporal average pooling to still flip the latent;
  the module docstring's "Scope / limitation" note records that a pure zero-mean packet-axis
  trigger measured ASR≈0% on this architecture, so ``dc_fraction`` close to 1 (a small AC
  admixture) is the regime worth testing first, not a large one.
"""

from __future__ import annotations

import torch
from scipy.linalg import hadamard

from .base import BaseTrigger
from .hadamard_trigger import _PROFILES, _bump_rows, _temporal_weight


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
    profile : str
        ``"walsh"`` (default, proposed) -- sequency-ordered signed Hadamard rows. ``"bump"``
        (exploratory, see module docstring) -- disjoint non-negative raised-cosine lobes, a
        smoothing-robustness ablation untested against a trained victim.
    dc_fraction : float
        ``1.0`` (default, proposed) -- trigger constant over packets. ``< 1.0`` (exploratory, see
        module docstring) mixes in a zero-mean temporal component so a non-zero remainder survives
        ``remove_static_component``; whether that remainder still flips the latent through the
        victim's own temporal pooling is untested against a trained victim.
    """

    def __init__(
        self,
        *args,
        subcarrier_start: int = 20,
        subcarrier_end: int = 52,
        profile: str = "walsh",
        dc_fraction: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not (0 <= subcarrier_start < subcarrier_end <= self.subcarriers):
            raise ValueError(
                f"Invalid subcarrier range [{subcarrier_start}, {subcarrier_end}) for subcarriers={self.subcarriers}"
            )
        if profile not in _PROFILES:
            raise ValueError(f"profile must be one of {_PROFILES}, got {profile!r}")
        if not 0.0 <= float(dc_fraction) <= 1.0:
            raise ValueError(f"dc_fraction must lie in [0, 1], got {dc_fraction}")
        self.subcarrier_start = int(subcarrier_start)
        self.subcarrier_end = int(subcarrier_end)
        self.width = self.subcarrier_end - self.subcarrier_start
        self.profile = profile
        self.dc_fraction = float(dc_fraction)

    def generate(self) -> torch.Tensor:
        rows = _bump_rows(self.width, self.K) if self.profile == "bump" else _hadamard_rows(self.width, self.K)
        weight = _temporal_weight(self.packets, self.dc_fraction)  # (T,) unit L2 norm
        bank_1d = torch.zeros(self.K, 1, self.subcarriers, self.packets)
        bank_1d[:, 0, self.subcarrier_start:self.subcarrier_end, :] = rows.unsqueeze(-1) * weight.view(1, 1, -1)
        bank = bank_1d.expand(self.K, self.antennas, self.subcarriers, self.packets).contiguous()

        # Check orthogonality of the active (frequency-axis) rows; the shared temporal weight
        # multiplies every direction identically, so it cannot introduce cross-talk between them.
        flat = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gram = flat @ flat.T
        off = (gram - torch.eye(self.K)).abs().max() if self.K > 1 else torch.zeros(())
        if float(off) > 1e-4:
            raise ValueError(f"Subband trigger rows are not orthogonal (max |off-diag cos|={float(off):.3g})")

        norm = torch.linalg.vector_norm(bank, dim=(2, 3), keepdim=True)
        return (bank / norm.clamp_min(1e-12)).to(self.device)
