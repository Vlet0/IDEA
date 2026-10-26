"""
attack/triggers/registry.py
============================
Factory functions that build a trigger object from a name or from a run configuration.

``MODE_TO_TRIGGER`` is the single canonical mapping from attack mode to trigger family;
``train.py``, ``run_experiments.py`` and ``evaluate_checkpoint.py`` all import it from here.

  'hadamard' -> HadamardTrigger  (subcarrier/packet/subband orthogonal rows)
  'subband'  -> SubbandTrigger   (concentrated frequency-domain subband)
  'ftrojan'  -> FTrojanTrigger   (2-D DCT impulse pair)
"""

from __future__ import annotations

import torch

from .base import BaseTrigger
from .ftrojan_trigger import FTrojanTrigger
from .hadamard_trigger import HadamardTrigger
from .subband_trigger import SubbandTrigger

_REGISTRY: dict[str, type[BaseTrigger]] = {
    "hadamard": HadamardTrigger,
    "subband": SubbandTrigger,
    "ftrojan": FTrojanTrigger,
}

#: attack mode -> trigger family (the only permitted pairing).
#: ``hadamard_erm`` is an ablation arm of the proposed method: the same Hadamard trigger
#: trained with plain poisoned ERM (no latent terms); without it, an ASR of the proposed
#: method cannot be attributed to the latent loss rather than to the trigger.  This is not a
#: full factorial design: the FTrojan-trigger + latent-loss cell does not exist.
MODE_TO_TRIGGER: dict[str, str] = {
    "latent": "hadamard",
    "hadamard_erm": "hadamard",
    "ftrojan": "ftrojan",
}


def build_trigger(
    trigger_type: str,
    K: int,
    antennas: int,
    subcarriers: int,
    packets: int,
    device: str | torch.device = "cpu",
    **kwargs,
) -> BaseTrigger:
    """Instantiate a trigger by family name; extra kwargs go to the constructor."""
    key = trigger_type.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown trigger_type '{trigger_type}'. Valid options: {sorted(_REGISTRY)}")
    return _REGISTRY[key](K=K, antennas=antennas, subcarriers=subcarriers,
                          packets=packets, device=device, **kwargs)


def trigger_for_mode(mode: str) -> str:
    """Return the canonical trigger family of an attack mode."""
    key = mode.lower().strip()
    if key not in MODE_TO_TRIGGER:
        raise ValueError(f"Unknown attack mode '{mode}'. Valid modes: {sorted(MODE_TO_TRIGGER)}")
    return MODE_TO_TRIGGER[key]


def build_trigger_from_config(config: dict, device: str | torch.device = "cpu") -> BaseTrigger:
    """Build the trigger of a *resolved* run config.

    Tensor dimensions come from ``data.input_shape``; family-specific constructor
    arguments come from ``attack.<family>``.  A ``trigger_type`` that contradicts the
    mode is rejected (cross-mode substitutions are not part of the protocol).
    """
    attack = config["attack"]
    family = trigger_for_mode(attack["mode"])
    requested = attack.get("trigger_type")
    if requested is not None and requested != family:
        raise ValueError(f"Mode '{attack['mode']}' requires trigger '{family}', not '{requested}'.")
    antennas, subcarriers, packets = config["data"]["input_shape"]
    return build_trigger(
        family, K=attack["K"], antennas=antennas, subcarriers=subcarriers,
        packets=packets, device=device, **(attack.get(family) or {}),
    )


def list_triggers() -> list[str]:
    """Return the registered trigger family names."""
    return sorted(_REGISTRY)
