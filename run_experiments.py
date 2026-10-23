"""Reproducible experiment runner for the preregistered HPELI protocol.

The backbone, epoch count and every attack/evaluation setting come from the base config
passed with ``--config``; the suite specs under ``configs/experiments/`` only list the swept
values (and optional ``default_params`` that are applied to every run of that suite).
"""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import yaml

from attack import MODE_TO_TRIGGER, trigger_for_mode
from train import (METRIC_PROTOCOL_VERSION, TRAINING_PROTOCOL_VERSION,
                   experiment_fingerprint, load_config, resolve_config, train)
from utils.logger import read_csv_to_dict_list, save_dict_to_csv


def _config(base: dict, mode: str, name: str, spec: dict | None = None) -> dict:
    """Isolated run configuration; trigger identity is derived from the mode."""
    cfg = copy.deepcopy(resolve_config(base))
    cfg["attack"].pop("trigger_type", None)
    cfg["attack"]["mode"] = mode.lower().strip()
    trigger_for_mode(cfg["attack"]["mode"])  # validates the mode
    cfg["experiment_name"] = name
    defaults = (spec or {}).get("default_params", {})
    if "poison_rate" in defaults:
        cfg["attack"]["poison_rate"] = float(defaults["poison_rate"])
    if "epsilon" in defaults:
        cfg["attack"]["epsilon"] = float(defaults["epsilon"])
    if "seed" in defaults:
        cfg["seed"] = int(defaults["seed"])
    return cfg


def _path(base: dict, filename: str) -> Path:
    path = Path(resolve_config(base)["training"]["output_dir"]) / "experiments" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _valid(base: dict, path: Path, required: tuple[str, ...]) -> list[dict]:
    """Cached rows produced by the current protocol, backbone and epoch budget."""
    cfg = resolve_config(base)
    rows = read_csv_to_dict_list(str(path))
    return [r for r in rows if all(k in r for k in required)
            and r.get("protocol_version") == METRIC_PROTOCOL_VERSION
            and r.get("training_protocol_version") == TRAINING_PROTOCOL_VERSION
            and int(r.get("training_epochs", -1)) == cfg["training"]["epochs"]
            and str(r.get("backbone", "")).lower() == cfg["model"]["backbone"]
            and r.get("trigger_type", "") == MODE_TO_TRIGGER.get(str(r.get("mode", "")).lower())]


def _save(rows: list[dict], path: Path) -> None:
    save_dict_to_csv(rows, str(path), mode="w")


def _f(row: dict, key: str) -> float:
    """Row value as float; ``nan`` (never 0) when the measurement is absent."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _summary(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("experiment | mode | trigger | clean MPJPE (cm) | T-MPJPE (cm) | ASR alarm [95% CI] | "
          "ASR bone-consistent | clean fall rate (%)")
    for r in rows:
        mode = "clean control" if int(_f(r, "is_clean_control") == 1) else r.get("mode", "-")
        print(f"{r.get('experiment_name', '-')} | {mode} | {r.get('trigger_type', '-')} | "
              f"{_f(r, 'clean_mpjpe_cm'):.2f} | {_f(r, 'target_mpjpe_cm'):.2f} | "
              f"{_f(r, 'binary_fall_asr')} [{_f(r, 'asr_ci95_low')}, {_f(r, 'asr_ci95_high')}] | "
              f"{_f(r, 'asr_bone_consistent')} | {_f(r, 'clean_fall_rate')}")


def _load_spec(spec_path: str) -> dict:
    with open(spec_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _done(rows: list[dict], cfg: dict, **match) -> bool:
    fingerprint = experiment_fingerprint(cfg)
    return any(r.get("config_id") == fingerprint and all(str(r.get(k)) == str(v) for k, v in match.items())
               for r in rows)


def run_main_benchmark(base: dict) -> None:
    path = _path(base, "main_benchmark_results.csv")
    rows = _valid(base, path, ("mode",))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    control = _config(base, "latent", f"benchmark_clean_control_{backbone}")
    control["attack"]["poison_rate"] = 0.0
    control["training"]["output_dir"] = str(out / "experiments" / "main_benchmark" / "clean_control")
    if not _done(rows, control):
        rows.append(train(control))
        _save(rows, path)
    for mode in MODE_TO_TRIGGER:
        cfg = _config(base, mode, f"benchmark_{mode}_{backbone}")
        cfg["training"]["output_dir"] = str(out / "experiments" / "main_benchmark" / mode)
        if not _done(rows, cfg):
            rows.append(train(cfg))
            _save(rows, path)
    _summary(rows, "Table 1: benchmark")


def run_loro(base: dict, spec_path: str) -> None:
    spec = _load_spec(spec_path)
    path = _path(base, "leave_one_room_out_results.csv")
    rows = _valid(base, path, ("split", "mode"))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    for split in spec["splits"]:
        for mode in spec["modes"]:
            cfg = _config(base, mode, f"loro_{split['name']}_{mode}_{backbone}", spec)
            cfg["data"]["train_rooms"], cfg["data"]["test_rooms"] = split["train_rooms"], split["test_rooms"]
            cfg["training"]["output_dir"] = str(out / "experiments" / "leave_one_out" / split["name"] / mode)
            if _done(rows, cfg, split=split["name"]):
                continue
            result = train(cfg)
            result.update({"split": split["name"], "test_room": split["test_rooms"][0]})
            rows.append(result)
            _save(rows, path)
    _summary(rows, "Leave-one-room-out evaluation")


def run_seeds(base: dict, spec_path: str) -> None:
    spec = _load_spec(spec_path)
    path = _path(base, "seed_stability_results.csv")
    rows = _valid(base, path, ("seed", "mode"))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    for mode in spec["modes"]:
        for seed in spec["seeds"]:
            cfg = _config(base, mode, f"seed_{seed}_{mode}_{backbone}", spec)
            cfg["seed"] = int(seed)
            cfg["training"]["output_dir"] = str(out / "experiments" / "seeds" / f"{mode}_seed{seed}")
            if _done(rows, cfg, seed=int(seed)):
                continue
            rows.append(train(cfg))
            _save(rows, path)
    _summary(rows, "Seed stability evaluation")
    for mode in spec["modes"]:
        values = [_f(r, "binary_fall_asr") for r in rows if str(r.get("mode", "")).lower() == mode]
        values = [v for v in values if not math.isnan(v)]
        if values:
            std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            print(f"{mode}: ASR {float(np.mean(values))} ± {std}% (n={len(values)} seeds)")


def run_sweep(base: dict, spec_path: str) -> None:
    spec = _load_spec(spec_path)
    path = _path(base, "poison_rate_sweep_results.csv")
    rows = _valid(base, path, ("poison_rate", "mode"))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    for rate in spec["poison_rates"]:
        cfg = _config(base, "latent", f"sweep_pr_{rate:g}_{backbone}", spec)
        cfg["attack"]["poison_rate"] = float(rate)
        cfg["training"]["output_dir"] = str(out / "experiments" / "sweeps" / f"poison_rate_{rate:g}")
        if _done(rows, cfg):
            continue
        rows.append(train(cfg))
        _save(rows, path)
    _summary(rows, "Latent-method poison-rate sensitivity")


def run_loss_ablation(base: dict, spec_path: str) -> None:
    spec = _load_spec(spec_path)
    path = _path(base, "loss_weight_ablation_results.csv")
    rows = _valid(base, path, ("lam_trig_pose", "mode"))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    for w in spec["loss_weights"]:
        cfg = _config(base, "latent", f"loss_ablation_lam_{w:g}_{backbone}", spec)
        cfg["attack"]["lam_trig_pose"] = float(w)
        cfg["training"]["output_dir"] = str(out / "experiments" / "loss_ablation" / f"lam_{w:g}")
        if _done(rows, cfg):
            continue
        rows.append(train(cfg))
        _save(rows, path)
    _summary(rows, "Triggered-pose loss weight (lam_trig_pose) sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--suite", default="all",
                        choices=["all", "main_benchmark", "leave_one_out", "seeds", "sweeps", "loss_ablation"])
    args = parser.parse_args()
    base = load_config(args.config)
    if args.suite in ("all", "main_benchmark"):
        run_main_benchmark(base)
    if args.suite in ("all", "leave_one_out"):
        run_loro(base, "configs/experiments/leave_one_room_out.yaml")
    if args.suite in ("all", "seeds"):
        run_seeds(base, "configs/experiments/seed_stability.yaml")
    if args.suite in ("all", "sweeps"):
        run_sweep(base, "configs/experiments/hyperparam_sweeps.yaml")
    if args.suite in ("all", "loss_ablation"):
        run_loss_ablation(base, "configs/experiments/loss_weight_ablation.yaml")


if __name__ == "__main__":
    main()
