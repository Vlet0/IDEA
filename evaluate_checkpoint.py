"""
evaluate_checkpoint.py
======================
Stand-alone evaluation of saved checkpoints WITHOUT retraining.  The run configuration,
trigger basis and fall targets are restored from the checkpoint; evaluation settings
(fall criteria, sensitivity scales, ...) come from the checkpoint's config merged on
``configs/base_config.yaml``, optionally overridden by ``--config``.

Usage:
------
python evaluate_checkpoint.py --checkpoint checkpoints/ours_model.pt
python evaluate_checkpoint.py --checkpoint_dir checkpoints/ --device cuda --output eval_summary.csv
"""

import argparse
import glob
import os

import torch

from attack import build_trigger_from_config, trigger_for_mode
from datasets.mmfi_dataset import load_mmfi_data
from evaluate import evaluate_model
from models.backbone import build_backbone
from utils.config import deep_update, load_config, resolve_config
from utils.logger import save_dict_to_csv
from utils.metrics import FallCriteria, StateCollapseCriteria


def evaluate_single_checkpoint(ckpt_path: str, device: torch.device, output_csv: str | None = None,
                               override_config: dict | None = None) -> dict:
    print(f"\n{'=' * 56}\nLoading checkpoint: {ckpt_path}\n{'=' * 56}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(saved, dict) or "net_state_dict" not in saved or not saved.get("config"):
        raise ValueError(f"Invalid checkpoint {ckpt_path}: expected a dict with 'net_state_dict' and 'config'.")

    config = resolve_config(saved["config"])
    if override_config:
        config = deep_update(config, override_config)
    atk, eval_cfg = config["attack"], config["evaluation"]
    mode = atk["mode"]
    clean_control = atk["poison_rate"] == 0
    latent_active = mode == "latent" and not clean_control

    print("[1/3] Loading test dataset...")
    _, test_loader, PS, _ = load_mmfi_data(config)
    print(f"Loaded test split with {len(test_loader.dataset)} frames. PS = {PS:.4f} m")

    trigger = build_trigger_from_config(config, device=device)
    U = saved["U"].to(device)
    fall_targets = saved["fall_targets"].to(device)

    print(f"[2/3] Initialising {config['model']['backbone'].upper()} and loading weights...")
    net = build_backbone(config["model"]["backbone"], m=config["model"]["latent_dim"],
                         num_joints=config["model"]["num_joints"],
                         subcarrier_num=config["data"]["input_shape"][1]).to(device)
    net.load_state_dict(saved["net_state_dict"])
    net.eval()

    print("[3/3] Evaluating (no retraining)...")
    res = evaluate_model(
        net, test_loader, trigger, fall_targets, PS, device, atk["epsilon"], atk["gate_threshold"],
        FallCriteria.from_dict(eval_cfg["fall_criteria"]),
        StateCollapseCriteria.from_dict(eval_cfg["state_collapse"]),
        eval_cfg["sensitivity_scales"], dict(atk["target"]),
        U=U if latent_active else None,
    )

    print("\n---------------- EVALUATION REPORT ----------------")
    print(f"Checkpoint             : {os.path.basename(ckpt_path)}")
    print(f"Attack mode            : {mode.upper()} (trigger: {trigger_for_mode(mode)}, poison rate {atk['poison_rate']})")
    print(f"Clean MPJPE            : {res['clean_mpjpe_cm']:6.2f} ± {res['clean_mpjpe_std']:.2f} cm")
    print(f"Distance to target     : {res['target_mpjpe_cm']:6.4f} ± {res['target_mpjpe_std_cm']:.4f} cm")
    print(f"ASR (fall alarm)       : {res['binary_fall_asr']:.4f} % (95% CI [{res['asr_ci95_low']:.4f}, {res['asr_ci95_high']:.4f}])")
    print(f"ASR (bone-consistent)  : {res['asr_bone_consistent']:.4f} %")
    print(f"Eligible / success     : {res['n_eligible']} / {res['n_success']} of {res['n_total']} frames")
    print(f"Fall rates clean/trig/GT: {res['clean_fall_rate']:.4f} / {res['trig_fall_rate']:.4f} / {res['gt_fall_rate']:.4f} %")
    print(f"State-collapse ASR     : {res['state_collapse_asr']:.4f} %")
    if latent_active:
        print(f"Latent AUROC / d'      : {res['latent_auroc']:.4f} / {res['latent_dprime']:.2f}")
    else:
        print("Latent statistics      : n/a (not a latent-objective model)")
    print("---------------------------------------------------\n")

    row = {"checkpoint": os.path.basename(ckpt_path), "mode": mode, "poison_rate": atk["poison_rate"],
           **{k: v for k, v in res.items() if v is not None and not isinstance(v, torch.Tensor)}}
    if output_csv:
        save_dict_to_csv([row], output_csv)
        print(f"Appended summary result to: {output_csv}")
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints without retraining.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a single .pt checkpoint file.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing .pt checkpoints.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML whose values override the checkpoint's evaluation settings.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="checkpoint_evaluation_results.csv")
    args = parser.parse_args()
    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Must specify either --checkpoint or --checkpoint_dir")

    override = load_config(args.config) if args.config else None
    device = torch.device(args.device)
    print(f"Using evaluation device: {device}")

    paths = [args.checkpoint] if args.checkpoint else []
    if args.checkpoint_dir:
        found = glob.glob(os.path.join(args.checkpoint_dir, "**", "*.pt"), recursive=True)
        paths.extend(p for p in found if "_epoch_ckpt" not in p)  # skip in-progress checkpoints
    paths = sorted(set(paths))
    print(f"Found {len(paths)} checkpoint(s) to evaluate.")

    failures = 0
    for i, path in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] Evaluating: {path}")
        try:
            evaluate_single_checkpoint(path, device=device, output_csv=args.output, override_config=override)
        except Exception as exc:  # report and continue with the remaining checkpoints
            failures += 1
            print(f"[Error evaluating {path}]: {exc}")
    print(f"\nFinished: {len(paths) - failures}/{len(paths)} checkpoint(s) evaluated.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
