"""Configuration loading with ``configs/base_config.yaml`` as the single source of defaults.

The code base contains no literal fallback values for experimental settings:
every run configuration is merged on top of the base configuration, so a key that
is missing from an experiment file resolves to the value documented in
``base_config.yaml`` and nowhere else.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base_config.yaml"


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update ``base_dict`` in place with ``update_dict``."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = copy.deepcopy(value)
    return base_dict


def load_config(config_path: str | os.PathLike) -> dict:
    """Load a YAML file, following a ``defaults:`` pointer to a parent file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "defaults" in config:
        base_path = str(config.pop("defaults"))
        if not os.path.isabs(base_path):
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(config_path)), base_path),
                os.path.join(os.getcwd(), base_path),
                str(REPO_ROOT / base_path),
            ]
            base_path = next((c for c in candidates if os.path.exists(c)), base_path)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Defaults config not found: {base_path}")
        config = deep_update(load_config(base_path), config)
    return config


@lru_cache(maxsize=1)
def _base_config_cached() -> str:
    # Cache the raw text so each caller receives an independent deep copy.
    return BASE_CONFIG_PATH.read_text(encoding="utf-8")


def base_config() -> dict:
    """Return a fresh copy of ``configs/base_config.yaml``."""
    return yaml.safe_load(_base_config_cached())


def resolve_config(config: dict | None) -> dict:
    """Merge ``config`` on top of the base configuration (a new dict is returned)."""
    return deep_update(base_config(), copy.deepcopy(config or {}))


def require(config: dict, dotted_key: str) -> Any:
    """Fetch ``a.b.c`` from a resolved config, failing loudly when absent."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing required config key '{dotted_key}' (not found at '{part}')")
        node = node[part]
    return node
