"""Canonical trigger application entry point."""

import torch

from .base import BaseTrigger


def apply_trigger(x: torch.Tensor, k: int | torch.Tensor, trigger: BaseTrigger, eps: float) -> torch.Tensor:
    """Apply a typed trigger to a CSI batch (``eps`` is mandatory; it is an experimental setting)."""
    if not isinstance(trigger, BaseTrigger):
        raise TypeError("trigger must be a BaseTrigger instance")
    return trigger.apply(x, k, eps)
