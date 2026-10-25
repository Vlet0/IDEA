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
from train import (METRIC_PROTOCOL_VERSION, TRAINING_PROTOCOL_VERSION, evaluation_fingerprint,
                   experiment_fingerprint, load_config, resolve_config, train)
from utils.config import deep_update
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
    target = str(cfg["attack"]["target"]["preserve_subject_scale"])
    eval_id = evaluation_fingerprint(base)
    valid = [r for r in rows if all(k in r for k in required)
            and r.get("protocol_version") == METRIC_PROTOCOL_VERSION
            # Rows scored under other evaluation settings are re-scored (the checkpoint is
            # reused because the training fingerprint excludes `evaluation`).
            and str(r.get("eval_id")) == eval_id
            # Rows trained against a different attack-target definition are not comparable.
            and r.get("preserve_subject_scale") == target
            and r.get("training_protocol_version") == TRAINING_PROTOCOL_VERSION
            and int(r.get("training_epochs", -1)) == cfg["training"]["epochs"]
            and str(r.get("backbone", "")).lower() == cfg["model"]["backbone"]
            and r.get("trigger_type", "") == MODE_TO_TRIGGER.get(str(r.get("mode", "")).lower())]
    if len(valid) < len(rows):
        # _save rewrites the file with valid rows only: keep the superseded rows on disk.
        archive = path.with_name(f"{path.stem}.superseded{path.suffix}")
        save_dict_to_csv([r for r in rows if r not in valid], str(archive))
        if valid:  # archived once, not on every call
            save_dict_to_csv(valid, str(path), mode="w")
        else:
            path.unlink()
        print(f"[archive] {len(rows) - len(valid)} superseded row(s) of {path.name} -> {archive.name}")
    return valid


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
    # Every method over the same rates: the rate at which ASR collapses is the only part of
    # this curve that can separate methods (all of them sit at ~100 % from 0.5 % upwards).
    for mode in spec["modes"]:
        for rate in spec["poison_rates"]:
            cfg = _config(base, mode, f"sweep_pr_{rate:g}_{mode}_{backbone}", spec)
            cfg["attack"]["poison_rate"] = float(rate)
            cfg["training"]["output_dir"] = str(out / "experiments" / "sweeps" / f"{mode}_poison_rate_{rate:g}")
            if _done(rows, cfg):
                continue
            rows.append(train(cfg))
            _save(rows, path)
    _summary(rows, "Poison-rate sensitivity")


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


def run_component_ablation(base: dict, spec_path: str) -> None:
    """Remove one ingredient of the proposed loss at a time (same trigger, data and budget)."""
    spec = _load_spec(spec_path)
    path = _path(base, "loss_component_ablation_results.csv")
    rows = _valid(base, path, ("variant", "mode"))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    allowed = set(resolve_config(base)["attack"])
    for variant in spec["variants"]:
        unknown = set(variant.get("attack", {})) - allowed
        if unknown:
            raise ValueError(f"Variant {variant['name']}: unknown attack keys {sorted(unknown)}")
        for seed in spec["seeds"]:
            name = f"component_{variant['name']}_seed{seed}_{backbone}"
            cfg = _config(base, variant["mode"], name, spec)
            cfg["seed"] = int(seed)
            cfg["attack"].update(variant.get("attack", {}))
            cfg["training"]["output_dir"] = str(out / "experiments" / "component_ablation" / f"{variant['name']}_seed{seed}")
            if _done(rows, cfg, variant=variant["name"]):
                continue
            result = train(cfg)
            result["variant"] = variant["name"]
            rows.append(result)
            _save(rows, path)
    _summary(rows, "Loss-component ablation")


def run_attack_variants(base: dict, spec_path: str) -> None:
    """Latent-subspace attack design variants (same loss, data and budget).

    Covers the semantically-mixed multi-trigger (e.g. false-alarm + concealment) and the
    time-axis (packet) Walsh trigger.  Each variant deep-merges its ``attack`` overrides so
    nested keys (``attack.target.target_types``, ``attack.hadamard.axis``) are updated, not
    replaced.  Reports trigger selectivity so routing to the correct target is auditable.
    """
    spec = _load_spec(spec_path)
    path = _path(base, "attack_variant_results.csv")
    rows = _valid(base, path, ("variant", "mode"))
    out = Path(resolve_config(base)["training"]["output_dir"])
    backbone = resolve_config(base)["model"]["backbone"]
    for variant in spec["variants"]:
        for seed in spec["seeds"]:
            cfg = _config(base, variant.get("mode", "latent"), f"variant_{variant['name']}_seed{seed}_{backbone}", spec)
            cfg["attack"] = deep_update(cfg["attack"], variant.get("attack", {}))
            cfg["seed"] = int(seed)
            cfg["training"]["output_dir"] = str(out / "experiments" / "attack_variants" / f"{variant['name']}_seed{seed}")
            if _done(rows, cfg, variant=variant["name"]):
                continue
            result = train(cfg)
            result["variant"] = variant["name"]
            rows.append(result)
            _save(rows, path)
    _summary(rows, "Latent-subspace attack variants (see pose_selectivity / latent_selectivity columns)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--suite", default="all",
                        choices=["all", "main_benchmark", "leave_one_out", "seeds", "sweeps", "loss_ablation",
                                 "component_ablation", "attack_variants"])
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
    if args.suite in ("all", "component_ablation"):
        run_component_ablation(base, "configs/experiments/loss_component_ablation.yaml")
    if args.suite in ("all", "attack_variants"):
        run_attack_variants(base, "configs/experiments/attack_variants.yaml")


if __name__ == "__main__":
    main()
