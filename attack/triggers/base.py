"""
attack/triggers/base.py
=======================
Abstract base class for all WiFi-CSI backdoor trigger generators.

Every concrete trigger implements :meth:`BaseTrigger.generate`, returning a bank of shape
``(K, C, F, T)`` whose every antenna slice has unit L2 norm over the ``(F, T)`` plane, so
``eps`` means the same thing for every family.  ``train.py`` / ``evaluate.py`` are
trigger-agnostic: they call :meth:`apply` (deterministic, test time) or :meth:`apply_eot`
(randomised, training time).

Injection model (v7)
--------------------
::

    x_trig = clip(x + eps * pattern_k, 0, 1)

``x`` is the CSI frame *after* the shared per-frame min-max preprocessing (values in [0, 1]);
``pattern_k`` has unit L2 per antenna, so ``eps`` is a genuine per-antenna L2 budget and the
realised perturbation is **decoupled from the frame content**.  The clip (not a per-frame
min-max renormalisation) keeps the output in range while leaving in-range values untouched,
so ``eps`` is the true budget that :mod:`utils.stealth` measures.  The earlier model added
``eps * pattern * max(x)`` and then re-min-maxed the whole frame, which rescaled the clean
content and made ``eps`` not equal to the realised budget; that coupling is removed.

Expectation-over-transformations (EOT)
--------------------------------------
Over the air the attacker does not control the received gain or the exact packet alignment,
so :meth:`apply_eot` injects the pattern with a random per-sample gain and a random circular
shift along the packet axis.  Training with these variations yields a backdoor that does not
depend on a single exact pattern amplitude or phase.  (Robustness to the *victim's*
preprocessing -- subcarrier smoothing, static-component removal, noise -- is applied at the
batch level in the training step, identically to clean and poisoned samples, so the transform
itself cannot become a trigger cue.)  The test-time trigger is always the clean :meth:`apply`.
"""

from __future__ import annotations

import abc

import torch


class BaseTrigger(abc.ABC):
    """Abstract base class for all CSI trigger generators."""

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

        Each antenna slice must have L2 norm <= 1 (exactly 1 for active antennas).
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

    def _validate(self, x: torch.Tensor, k: int | torch.Tensor, eps: float) -> torch.Tensor:
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
        return bank

    def apply(self, x: torch.Tensor, k: int | torch.Tensor, eps: float) -> torch.Tensor:
        """Deterministic injection ``clip(x + eps * pattern_k, 0, 1)`` (the test-time trigger).

        ``x`` is a CSI batch ``(B, C, F, T)`` already in ``[0, 1]``; ``k`` is an int or a
        ``(B,)`` LongTensor; ``eps`` is the per-antenna L2 budget.
        """
        bank = self._validate(x, k, eps)
        kk = k.to(device=bank.device, dtype=torch.long) if isinstance(k, torch.Tensor) else k
        pattern = bank[kk]  # (C,F,T) for int k, (B,C,F,T) for tensor k
        return (x + eps * pattern).clamp(0.0, 1.0)

    def apply_eot(
        self,
        x: torch.Tensor,
        k: int | torch.Tensor,
        eps: float,
        *,
        generator: torch.Generator | None = None,
        gain_range: tuple[float, float] = (0.7, 1.3),
        max_roll: int = 0,
    ) -> torch.Tensor:
        """Randomised injection for training: per-sample gain and a circular packet shift.

        ``gain_range`` multiplies ``eps`` per sample (unknown over-the-air strength); ``max_roll``
        circularly shifts the pattern along the packet axis by a random amount in
        ``[-max_roll, max_roll]`` (unknown frame alignment).  Falls back to :meth:`apply` when
        both are disabled.  The shift is drawn once per call; the gain is per sample.
        """
        bank = self._validate(x, k, eps)
        kk = k.to(device=bank.device, dtype=torch.long) if isinstance(k, torch.Tensor) else k
        pattern = bank[kk]
        if pattern.ndim == 3:
            pattern = pattern.unsqueeze(0).expand(len(x), -1, -1, -1)
        if max_roll > 0:
            # torch.randint only supports CPU generators; use torch.rand which works on any device
            gen_device = generator.device if generator is not None else torch.device('cpu')
            r = torch.rand(1, generator=generator, device=gen_device).item()
            shift = int(r * (2 * max_roll + 1)) - max_roll
            if shift:
                pattern = torch.roll(pattern, shifts=shift, dims=-1)
        lo, hi = gain_range
        gain = torch.rand(len(x), 1, 1, 1, generator=generator, device=x.device, dtype=x.dtype) * (hi - lo) + lo
        return (x + eps * gain * pattern).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(K={self.K}, antennas={self.antennas}, "
            f"subcarriers={self.subcarriers}, packets={self.packets}, device={self.device})"
        )
