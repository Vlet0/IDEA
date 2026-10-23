"""
attack/triggers/base.py
=======================
Abstract base class for all WiFi-CSI backdoor trigger generators.

Every concrete trigger must implement :meth:`BaseTrigger.generate`, returning a
bank of shape ``(K, C, F, T)``.  ``train.py`` / ``evaluate.py`` are trigger-agnostic:
they only call ``trigger.apply(x, k, eps)``.

Injection model (identical for every trigger family)
----------------------------------------------------
::

    x_pert = x + eps * pattern_k * max(x)          # pattern_k has unit L2 norm per antenna
    x_out  = (x_pert - min) / (max - min + RENORM_EPS)   # per-frame min-max, MM-Fi standard

Because each antenna slice of ``pattern_k`` has unit L2 norm over the ``F x T`` plane,
``eps * max(x)`` is the **per-antenna L2 budget** of the injected perturbation for all
families.  It is *not* a per-element amplitude: the per-element L_inf is roughly
``eps / sqrt(F*T)`` for a flat pattern.  Families therefore share the same L2 budget but
may differ in L_inf; ``utils.stealth.trigger_report`` measures both.
"""

from __future__ import annotations

import abc

import torch


class BaseTrigger(abc.ABC):
    """
    Abstract base class for all CSI trigger generators.

    Attributes
    ----------
    K : int
        Number of trigger directions (>= 1).
    antennas, subcarriers, packets : int
        Trigger tensor dimensions ``(C, F, T)``; derive them from ``data.input_shape``.
    device : torch.device
        Device where the trigger bank lives.
    """

    #: Numerical guard of the min-max renormalisation (constant frames).
    RENORM_EPS: float = 1e-8
    #: Tolerance when checking the unit-norm contract of a generated bank.
    NORM_TOL: float = 1e-4

    def __init__(
        self,
        K: int,
        antennas: int,
        subcarriers: int,
        packets: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = int(K)
        self.antennas = int(antennas)
        self.subcarriers = int(subcarriers)
        self.packets = int(packets)
        self.device = torch.device(device)
        self._trig: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def generate(self) -> torch.Tensor:
        """Build the trigger bank, shape ``(K, antennas, subcarriers, packets)``.

        Each antenna slice must have L2 norm <= 1 (exactly 1 for active antennas) so
        that ``eps`` has the same meaning for every trigger family.
        """

    # ------------------------------------------------------------------
    # Bank access and diagnostics
    # ------------------------------------------------------------------

    @property
    def patterns(self) -> torch.Tensor:
        """Lazily generated and validated trigger bank, shape ``(K, C, F, T)``."""
        if self._trig is None:
            bank = self.generate()
            expected = (self.K, self.antennas, self.subcarriers, self.packets)
            if tuple(bank.shape) != expected:
                raise ValueError(f"{type(self).__name__}.generate() returned {tuple(bank.shape)}, expected {expected}")
            if not torch.isfinite(bank).all():
                raise ValueError(f"{type(self).__name__} produced a non-finite trigger bank")
            norms = torch.linalg.vector_norm(bank, dim=(2, 3))
            if (norms > 1.0 + self.NORM_TOL).any():
                raise ValueError(f"{type(self).__name__} violates the unit-L2-per-antenna contract")
            self._trig = bank.to(self.device)
        return self._trig

    def gram(self) -> torch.Tensor:
        """Cosine-similarity matrix ``(K, K)`` between the flattened trigger patterns."""
        flat = self.patterns.reshape(self.K, -1)
        flat = flat / torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
        return flat @ flat.T

    def to(self, device: str | torch.device) -> "BaseTrigger":
        """Move the trigger bank to *device* in place and return self."""
        self.device = torch.device(device)
        if self._trig is not None:
            self._trig = self._trig.to(self.device)
        return self

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def apply(
        self,
        x: torch.Tensor,
        k: int | torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Inject trigger *k* into batch *x* and re-normalise to [0, 1].

        Parameters
        ----------
        x : torch.Tensor
            CSI batch ``(B, C, F, T)`` with values in ``[0, 1]``.
        k : int or LongTensor of shape (B,)
            Trigger index; a tensor assigns a (possibly different) trigger per sample.
        eps : float
            Per-antenna L2 budget relative to ``max(x)`` (see module docstring).
        """
        if not eps >= 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.antennas, self.subcarriers, self.packets):
            raise ValueError(
                f"Expected CSI batch (B, {self.antennas}, {self.subcarriers}, {self.packets}), "
                f"received {tuple(x.shape)}"
            )
        bank = self.patterns
        if bank.device != x.device or bank.dtype != x.dtype:
            bank = bank.to(device=x.device, dtype=x.dtype)
        if isinstance(k, torch.Tensor):
            if k.ndim != 1 or len(k) != len(x):
                raise ValueError("A tensor k must have shape (B,)")
            k = k.to(device=bank.device, dtype=torch.long)

        pattern = bank[k]  # (C,F,T) for int k, (B,C,F,T) for tensor k
        scale = x.amax(dim=(1, 2, 3), keepdim=True)
        x_pert = x + eps * pattern * scale

        mn = x_pert.amin(dim=(1, 2, 3), keepdim=True)
        mx = x_pert.amax(dim=(1, 2, 3), keepdim=True)
        return (x_pert - mn) / (mx - mn + self.RENORM_EPS)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(K={self.K}, antennas={self.antennas}, "
            f"subcarriers={self.subcarriers}, packets={self.packets}, device={self.device})"
        )
