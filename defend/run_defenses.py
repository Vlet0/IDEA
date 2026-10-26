"""
defend/run_defenses.py
======================
Run every defender-side audit on a trained checkpoint and write the result CSVs.

    python -m defend.run_defenses --checkpoint checkpoints/ours_latent_model.pt \
        --device cuda --output_dir outputs/defenses --defender_frames 2000 --eval_frames 8000

Threat model (defender): owns the weights and a small clean dataset (``--defender_frames`` clean
training frames the attacker did not poison — a genuine clean set, standard in Liu et al. 2018),
does NOT know the trigger, the poisoned subset, or the target.  ASR is measured on a held-out test
subset (``--eval_frames``).  Reads the trigger / poison mask only to *grade* detectors, never as a
defense input.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from attack import (PoisonedDataset, build_poison_plan, build_trigger_from_config,
                    dataset_tensors)
from datasets.mmfi_dataset import load_mmfi_data
from defend.detection import feature_space_report
from defend.model_repair import fine_pruning_curve, fine_tuning_curve
from evaluate import collect_latent_features, evaluate_model
from models.backbone import build_backbone
from utils.config import resolve_config
from utils.defenses import extract_features
from utils.logger import save_dict_to_csv
from utils.metrics import FallCriteria, StateCollapseCriteria


def _measure_fn(net, loader, trigger, fall_targets, PS, device, atk, eval_cfg, target_cfg, U):
    """Return ``measure(net) -> {binary_fall_asr, clean_mpjpe_cm}`` over ``loader`` (round-robin k)."""
    criteria = FallCriteria.from_dict(eval_cfg["fall_criteria"])
    collapse = StateCollapseCriteria.from_dict(eval_cfg["state_collapse"])

    @torch.no_grad()
    def measure(model):
        res = evaluate_model(
            model, loader, trigger, fall_targets, PS, device, atk["epsilon"], atk["gate_threshold"],
            criteria, collapse, eval_cfg["sensitivity_scales"], target_cfg,
            U=U,  # latent diagnostics are cheap and ignored by the caller when None
        )
        return {"binary_fall_asr": res["binary_fall_asr"], "clean_mpjpe_cm": res["clean_mpjpe_cm"]}

    return measure


def run(ckpt_path: str, device: torch.device, output_dir: str,
        defender_frames: int, eval_frames: int, seed: int = 0) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = resolve_config(saved["config"])
    atk, eval_cfg = config["attack"], config["evaluation"]
    atk.update(saved.get("calibration") or {})  # calibrated thresholds the weights were trained with
    mode = atk["mode"]
    latent_active = mode == "latent" and atk["poison_rate"] > 0
    target_cfg = {k: v for k, v in atk["target"].items() if k != "target_types"}

    train_loader, test_loader, PS, meta = load_mmfi_data(config)
    trigger = build_trigger_from_config(config, device=device)
    U = saved["U"].to(device) if latent_active else None
    fall_targets = saved["fall_targets"].to(device)

    net = build_backbone(config["model"]["backbone"], m=config["model"]["latent_dim"],
                         num_joints=config["model"]["num_joints"],
                         subcarrier_num=config["data"]["input_shape"][1]).to(device)
    net.load_state_dict(saved["net_state_dict"])
    net.eval()

    # ── Defender clean data: clean (un-poisoned) training frames, standard clean-set assumption ──
    X_train, Y_train = dataset_tensors(train_loader.dataset)
    poison_seed = config["seed"] if atk["poison_seed"] is None else atk["poison_seed"]
    plan = build_poison_plan(len(X_train), atk["poison_rate"], atk["K"], poison_seed)
    clean_idx = torch.nonzero(~plan.mask, as_tuple=False).flatten()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(clean_idx), size=min(defender_frames, len(clean_idx)), replace=False)
    defender_idx = clean_idx[torch.as_tensor(np.sort(pick))]
    defender_loader = DataLoader(
        Subset(train_loader.dataset, defender_idx.tolist()),
        batch_size=eval_cfg["batch_size"], shuffle=True, num_workers=0)

    # ── ASR measurement subset of the test set (held out from the defender data) ──
    n_test = len(test_loader.dataset)
    eval_idx = np.sort(rng.choice(n_test, size=min(eval_frames, n_test), replace=False))
    eval_loader = DataLoader(Subset(test_loader.dataset, eval_idx.tolist()),
                             batch_size=eval_cfg["batch_size"], shuffle=False, num_workers=0)
    measure = _measure_fn(net, eval_loader, trigger, fall_targets, PS, device, atk, eval_cfg, target_cfg, U)

    summary: dict = {"checkpoint": os.path.basename(ckpt_path), "mode": mode,
                     "poison_rate": atk["poison_rate"], "epsilon": atk["epsilon"]}
    baseline = measure(net)
    summary["asr_before"] = baseline["binary_fall_asr"]
    summary["clean_mpjpe_before"] = baseline["clean_mpjpe_cm"]
    print(f"[defend] baseline ASR {baseline['binary_fall_asr']:.2f} % | "
          f"clean MPJPE {baseline['clean_mpjpe_cm']:.2f} cm")

    # ── 1. Feature-space detectability (training detection + test-time separability) ──
    audit_set = PoisonedDataset(X_train, Y_train, plan, trigger, atk["epsilon"], use_eot=False)
    train_feats = extract_features(net, audit_set, device, eval_cfg["batch_size"])
    feats = collect_latent_features(net, eval_loader, trigger, atk["epsilon"], device, atk["K"])
    detect = feature_space_report(
        train_feats, plan.mask, feats["clean"], feats["trig"][0],
        spectral_components=tuple(eval_cfg["defenses"]["spectral"]["n_components"]),
        removal_factor=float(eval_cfg["defenses"]["spectral"]["removal_factor"]), seed=seed)
    save_dict_to_csv([{**summary, **detect}], os.path.join(output_dir, "feature_space_detection.csv"), mode="w")
    summary.update({f"detect_{k}": v for k, v in detect.items()})
    print("[defend] feature-space: " + ", ".join(f"{k}={v:.3f}" for k, v in detect.items()))

    # ── 2. Fine-Pruning curve ──
    prune_rows = fine_pruning_curve(net, defender_loader, measure, device, finetune_epochs=0)
    save_dict_to_csv([{**summary_row, **r} for summary_row, r in
                      [({"checkpoint": summary["checkpoint"]}, row) for row in prune_rows]],
                     os.path.join(output_dir, "fine_pruning.csv"), mode="w")
    worst = min(prune_rows, key=lambda r: r["binary_fall_asr"])
    print(f"[defend] fine-pruning: ASR floor {worst['binary_fall_asr']:.2f} % at "
          f"prune {worst['fraction']:.0%} (clean MPJPE {worst['clean_mpjpe_cm']:.2f} cm)")

    # ── 3. Clean Fine-Tuning curve (on a deep copy) ──
    ft_rows = fine_tuning_curve(net, defender_loader, measure, device, epochs=10)
    save_dict_to_csv([{"checkpoint": summary["checkpoint"], **r} for r in ft_rows],
                     os.path.join(output_dir, "fine_tuning.csv"), mode="w")
    print(f"[defend] fine-tuning: ASR {ft_rows[0]['binary_fall_asr']:.2f} % -> "
          f"{ft_rows[-1]['binary_fall_asr']:.2f} % over {len(ft_rows) - 1} clean epochs "
          f"(clean MPJPE {ft_rows[-1]['clean_mpjpe_cm']:.2f} cm)")

    summary["asr_fine_pruning_floor"] = worst["binary_fall_asr"]
    summary["asr_fine_tuning_final"] = ft_rows[-1]["binary_fall_asr"]
    save_dict_to_csv([summary], os.path.join(output_dir, "defense_summary.csv"), mode="w")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="outputs/defenses")
    p.add_argument("--defender_frames", type=int, default=2000,
                   help="Clean training frames available to the defender (Fine-Pruning / Fine-Tuning set).")
    p.add_argument("--eval_frames", type=int, default=8000, help="Held-out test frames used to measure ASR.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.checkpoint, torch.device(args.device), args.output_dir,
        args.defender_frames, args.eval_frames, args.seed)


if __name__ == "__main__":
    main()
