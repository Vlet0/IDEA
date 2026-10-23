#!/usr/bin/env python3
"""Audit the fall alarm A(pose) on MM-Fi ground truth and write annotated ``is_fall`` labels.

What the labels are
-------------------
MM-Fi contains 27 daily-living / exercise actions and **no falls**.  The annotated truth is
therefore "no frame is a fall" (all zeros) unless ``--fall_actions`` names action codes AND the
cache stores a per-frame ``action`` array (the stock cache does not).  The labels never come from
the geometric detector, so they are independent of it.

What the audit is
-----------------
Because the true fall rate is zero, every frame on which the alarm ``A`` fires is a *false
positive of the detector on real data*.  The audit reports that false-alarm rate (overall, per
room and, when available, per action) at the configured thresholds and at the jointly scaled
thresholds of ``evaluation.sensitivity_scales`` — the empirical specificity of the alarm that the
ASR depends on.  Poses are canonicalised to the protocol's Z-up frame first (raw MM-Fi labels are
Y-up), exactly like the data loader.

Usage::

    python scripts/generate_fall_labels.py --cache_path ../MMFI/Compress/mmfi_cache_all_actions.npz
    python scripts/generate_fall_labels.py --cache_path cache.npz --embed_npz   # also rewrite the cache
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from datasets.transforms import canonicalize_pose_z_up, center_pose_root  # noqa: E402
from utils.config import load_config, resolve_config  # noqa: E402
from utils.metrics import FallCriteria, _fall_pose_masks  # noqa: E402


def audit_and_generate_fall_labels(
    cache_path: str,
    criteria: FallCriteria,
    sensitivity_scales: list[float],
    output_npy: str | None = None,
    embed_npz: bool = False,
    fall_actions: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    print(f"[FallLabels] Loading cache from: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as d:
        missing = {"X", "Y", "env"}.difference(d.files)
        if missing:
            raise ValueError(f"Cache missing required arrays: {missing}")
        cache_data = {k: d[k] for k in d.files}
    Y, env = cache_data["Y"], cache_data["env"]
    N = len(Y)

    poses = torch.as_tensor(center_pose_root(canonicalize_pose_z_up(Y)), dtype=torch.float32)
    print(f"[FallLabels] Auditing {N} ground-truth frames (Z-up metres, criteria: {criteria})")

    # Annotated truth: independent of the detector.
    labels = np.zeros(N, dtype=np.int32)
    action_key = next((k for k in ("actions", "action") if k in cache_data), None)
    if fall_actions:
        if action_key is None:
            raise ValueError("--fall_actions was given but the cache has no per-frame 'action' array")
        labels[np.isin(cache_data[action_key], fall_actions)] = 1
    print(f"[FallLabels] Annotated falls: {int(labels.sum())}/{N} "
          f"({'from --fall_actions' if fall_actions else 'MM-Fi has no fall action'})")

    stats: dict = {"n_total": N, "n_annotated_fall": int(labels.sum())}
    groups = {"all": np.ones(N, dtype=bool)}
    groups.update({f"room {r}": env == r for r in sorted(set(env.tolist()))})
    if action_key is not None:
        groups.update({f"action {a}": cache_data[action_key] == a for a in sorted(set(cache_data[action_key].tolist()))})

    for scale in [1.0] + [s for s in sensitivity_scales if s != 1.0]:
        c = criteria.scaled(scale)
        _, valid, alarm = _fall_pose_masks(poses, 1.0, c)
        false_alarm = alarm.numpy() & (labels == 0)
        print(f"[FallLabels] thresholds x{scale:g}: invalid skeletons {int((~valid).sum())} | "
              f"alarm on non-fall GT (false-alarm rate) {100.0 * false_alarm.mean():.4f} %")
        if scale == 1.0:
            stats.update(n_invalid=int((~valid).sum()), n_alarm=int(alarm.sum()),
                         false_alarm_rate_pct=100.0 * float(false_alarm.mean()))
            for name, member in groups.items():
                if name != "all" and member.any():
                    print(f"    {name:<12s} {100.0 * false_alarm[member].mean():8.4f} % of {int(member.sum())}")
        stats[f"false_alarm_rate_pct_x{scale:g}"] = 100.0 * float(false_alarm.mean())

    output_npy = output_npy or cache_path.replace(".npz", "_is_fall.npy")
    np.save(output_npy, labels)
    print(f"[FallLabels] Saved {N} binary labels to: {output_npy}")

    if embed_npz:
        cache_data["is_fall"] = labels
        temp_path = f"{cache_path}.tmp.{os.getpid()}.npz"
        try:
            np.savez_compressed(temp_path, **cache_data)
            os.replace(temp_path, cache_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        print(f"[FallLabels] Embedded 'is_fall' into: {cache_path}")
    return labels, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(repo_root, "configs", "base_config.yaml"))
    parser.add_argument("--cache_path", default=None, help="MM-Fi npz cache (default: data.cache_path of the config)")
    parser.add_argument("--output_npy", default=None, help="Output .npy (default: <cache>_is_fall.npy)")
    parser.add_argument("--fall_actions", nargs="*", default=None,
                        help="Action codes that are genuine falls (needs a per-frame 'action' array; MM-Fi has none)")
    parser.add_argument("--embed_npz", action="store_true",
                        help="Also rewrite the cache with an 'is_fall' array (atomic, compressed)")
    args = parser.parse_args()

    config = resolve_config(load_config(args.config))
    audit_and_generate_fall_labels(
        cache_path=args.cache_path or config["data"]["cache_path"],
        criteria=FallCriteria.from_dict(config["evaluation"]["fall_criteria"]),
        sensitivity_scales=config["evaluation"]["sensitivity_scales"],
        output_npy=args.output_npy,
        embed_npz=args.embed_npz,
        fall_actions=args.fall_actions,
    )


if __name__ == "__main__":
    main()
