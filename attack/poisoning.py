"""
attack/poisoning.py
===================
Fixed, seeded data poisoning of the training set.

Threat model: the attacker controls a fraction ``poison_rate`` of the *stored* training
data.  The poisoned subset, the trigger index of every poisoned sample and the triggered
CSI are therefore fixed once, before training.  (Re-drawing a Bernoulli mask per batch and
epoch, as an earlier protocol did, exposes the model to triggers on roughly
``1 - (1 - rate)**epochs`` of the distinct samples — about 40 % at rate 1 % and 50 epochs —
so the nominal poison rate understated the attacker's real influence.)

Labels are never stored modified: the attack target of a poisoned sample is a pure function
of ``(y, k)`` computed by ``attack.latent_attack.compute_target_poses`` in the loss, which is
exactly what the evaluator uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .triggers import BaseTrigger


@dataclass(frozen=True)
class PoisonPlan:
    """Which training samples are poisoned and with which trigger direction."""

    mask: torch.Tensor  # (N,) bool
    kk: torch.Tensor    # (N,) long; meaningful where ``mask`` is True

    @property
    def n_poisoned(self) -> int:
        return int(self.mask.sum())

    def per_trigger_counts(self, K: int) -> list[int]:
        return [int(((self.kk == k) & self.mask).sum()) for k in range(K)]


def validate_poison_rate(rate: float) -> float:
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("poison_rate must be finite and in [0, 1]")
    return float(rate)


def build_poison_plan(n_samples: int, rate: float, K: int, seed: int) -> PoisonPlan:
    """Choose ``round(rate * n_samples)`` distinct samples, trigger directions balanced.

    A positive rate that rounds to zero poisoned samples is rejected instead of silently
    producing a clean run.
    """
    rate = validate_poison_rate(rate)
    n_poison = int(round(rate * n_samples))
    if rate > 0.0 and n_poison == 0:
        raise ValueError(f"poison_rate={rate} selects 0 of {n_samples} samples")
    generator = torch.Generator().manual_seed(int(seed))
    chosen = torch.randperm(n_samples, generator=generator)[:n_poison]
    mask = torch.zeros(n_samples, dtype=torch.bool)
    kk = torch.zeros(n_samples, dtype=torch.long)
    mask[chosen] = True
    kk[chosen] = torch.arange(n_poison) % K
    return PoisonPlan(mask=mask, kk=kk)


def dataset_tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(X, Y)`` tensors of an MMFIDataset or TensorDataset."""
    if hasattr(dataset, "X") and hasattr(dataset, "Y"):
        return dataset.X, dataset.Y
    if hasattr(dataset, "tensors"):
        return dataset.tensors[0], dataset.tensors[1]
    raise TypeError(f"Cannot extract (X, Y) tensors from {type(dataset).__name__}")


class PoisonedDataset(Dataset):
    """Training set whose poisoned samples carry a statically injected trigger.

    Items are ``(x, y, is_poisoned, k)``.  Only the poisoned CSI frames are materialised
    (a few thousand), so memory stays equal to the clean dataset.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, plan: PoisonPlan,
                 trigger: BaseTrigger, eps: float) -> None:
        if not (len(X) == len(Y) == len(plan.mask)):
            raise ValueError("X, Y and the poison plan must have the same length")
        self.X, self.Y, self.plan = X, Y, plan
        poisoned_idx = torch.nonzero(plan.mask, as_tuple=False).flatten()
        self._slot = torch.full((len(X),), -1, dtype=torch.long)
        self._slot[poisoned_idx] = torch.arange(len(poisoned_idx))
        frames = torch.empty((len(poisoned_idx),) + tuple(X.shape[1:]), dtype=X.dtype)
        for k in range(trigger.K):
            sel = torch.nonzero(plan.mask & (plan.kk == k), as_tuple=False).flatten()
            if len(sel):
                frames[self._slot[sel]] = trigger.apply(X[sel], k, eps).to(X.dtype).cpu()
        self._poisoned_frames = frames

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        slot = int(self._slot[idx])
        x = self._poisoned_frames[slot] if slot >= 0 else self.X[idx]
        return x, self.Y[idx], self.plan.mask[idx], self.plan.kk[idx]
