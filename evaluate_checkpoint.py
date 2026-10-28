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

from attack import (PoisonedDataset, build_poison_plan, build_trigger_from_config, dataset_tensors,
                    fall_target_names, target_bank_names, trigger_for_mode)
from datasets.mmfi_dataset import load_mmfi_data
from evaluate import (evaluate_latent_intervention, evaluate_model, evaluate_robustness,
                      spectral_audit)
from models.backbone import build_backbone
from train import METRIC_PROTOCOL_VERSION, _git_commit, evaluation_fingerprint, experiment_fingerprint
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
    # Latent thresholds the weights were trained with (PCA calibration).  Older checkpoints
    # already carry them inside the saved config.
    atk.update(saved.get("calibration") or {})
    mode = atk["mode"]
    clean_control = atk["poison_rate"] == 0
    latent_active = mode == "latent" and not clean_control

    print("[1/3] Loading dataset...")
    train_loader, test_loader, PS, meta = load_mmfi_data(config)
    print(f"Loaded test split with {len(test_loader.dataset)} frames. PS = {PS:.4f} m")

    trigger = build_trigger_from_config(config, device=device)
    U = saved["U"].to(device)
    fall_targets = saved["fall_targets"].to(device)
    # target_types builds the bank (restored from the checkpoint); strip it from the per-sample cfg.
    target_cfg = {k: v for k, v in atk["target"].items() if k != "target_types"}
    _types = atk["target"].get("target_types")
    target_names = target_bank_names(_types) if _types else fall_target_names(atk["K"])

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
        eval_cfg["sensitivity_scales"], target_cfg,
        U=U if latent_active else None,
        clusters=meta.get("te_subject"), cluster_ci=eval_cfg["cluster_ci"],
        seq_ids=meta.get("te_seq_id"), seq_min_frames=int(eval_cfg["seq_min_frames"]),
        poison_rate=atk.get("poison_rate", 1.0),
        frames_per_seq=int(config.get("data", {}).get("expected_frames_per_action", 297)),
        num_bursts=int(eval_cfg.get("num_bursts", 2)),
    )
    robustness = evaluate_robustness(
        net, test_loader, trigger, fall_targets, PS, device, atk["epsilon"],
        FallCriteria.from_dict(eval_cfg["fall_criteria"]), target_cfg, eval_cfg["robustness"],
        seq_ids=meta.get("te_seq_id"), seq_min_frames=int(eval_cfg["seq_min_frames"]),
        poison_rate=atk.get("poison_rate", 1.0),
        frames_per_seq=int(config.get("data", {}).get("expected_frames_per_action", 297)),
        num_bursts=int(eval_cfg.get("num_bursts", 2)),
    )
    # The poisoned training subset is a deterministic function of (N, rate, K, poison seed).
    X_train, Y_train = dataset_tensors(train_loader.dataset)
    poison_seed = config["seed"] if atk["poison_seed"] is None else atk["poison_seed"]
    plan = build_poison_plan(len(X_train), atk["poison_rate"], atk["K"], poison_seed)
    if saved.get("poison_plan_sha") not in (None, plan.fingerprint()):
        raise ValueError("Rebuilt poison plan differs from the one used in training")
    defense = spectral_audit(net, PoisonedDataset(X_train, Y_train, plan, trigger, atk["epsilon"]),
                             plan.mask, device, eval_cfg["batch_size"], eval_cfg["defenses"]["spectral"])
    intervention = None
    if latent_active:
        intervention = evaluate_latent_intervention(
            net, test_loader, trigger, fall_targets, PS, device, atk["epsilon"],
            FallCriteria.from_dict(eval_cfg["fall_criteria"]),
            StateCollapseCriteria.from_dict(eval_cfg["state_collapse"]),
            target_cfg, U, atk["margin"], eval_cfg["intervention"],
        )

    print("\n---------------- EVALUATION REPORT ----------------")
    print(f"Checkpoint             : {os.path.basename(ckpt_path)}")
    print(f"Attack mode            : {mode.upper()} (trigger: {trigger_for_mode(mode)}, poison rate {atk['poison_rate']})")
    print(f"Clean MPJPE            : {res['clean_mpjpe_cm']:6.2f} ± {res['clean_mpjpe_std']:.2f} cm")
    print(f"Distance to target     : {res['target_mpjpe_cm']:6.4f} ± {res['target_mpjpe_std_cm']:.4f} cm")
    print(f"ASR (fall alarm)       : {res['binary_fall_asr']:.4f} % (95% CI [{res['asr_ci95_low']:.4f}, {res['asr_ci95_high']:.4f}])")
    print(f"ASR subject-level CI   : [{res['asr_ci95_low_cluster']:.4f}, {res['asr_ci95_high_cluster']:.4f}]")
    print(f"ASR (bone-consistent)  : {res['asr_bone_consistent']:.4f} % "
          f"(clean predictions pass the same check: {res['clean_bone_consistent_rate']:.4f} %)")
    print(f"Target preservation    : {atk['target']['preserve_subject_scale']}")
    print(f"Eligible / success     : {res['n_eligible']} / {res['n_success']} of {res['n_total']} frames")
    print(f"Fall rates clean/trig/GT: {res['clean_fall_rate']:.4f} / {res['trig_fall_rate']:.4f} / {res['gt_fall_rate']:.4f} %")
    print(f"State-collapse ASR     : {res['state_collapse_asr']:.4f} %")
    if latent_active:
        print(f"Latent AUROC / d'      : {res['latent_auroc']:.4f} / {res['latent_dprime']:.2f}")
    else:
        print("Latent statistics      : n/a (not a latent-objective model)")
    for r in robustness:
        print(f"Robustness {r['condition']}={r['value']:g}: ASR {r['asr']:.2f} % | "
              f"T-MPJPE {r['target_mpjpe_cm']:.2f} cm | clean MPJPE {r['clean_mpjpe_cm']:.2f} cm")
    print("Spectral Signatures    : " + ", ".join(f"{k}={v:.2f}" for k, v in defense.items()))
    if intervention is not None:
        for r in intervention["necessity"]:
            print(f"[causal] k={r['k']} necessity : fall-ASR {r['asr_baseline']:.2f} -> "
                  f"{r['asr_with_uk_removed']:.2f} % with u_k removed")
        for tag in ("sufficiency", "specificity"):
            at1 = [x for x in intervention[tag] if abs(x["scale"] - 1.0) < 1e-9]
            print(f"[causal] {tag} @1 : " + "; ".join(
                f"k{x['k']} fall {x['fall_rate']:.1f}% collapse {x['state_collapse_to_target']:.1f}%" for x in at1))
    print("---------------------------------------------------\n")

    row = {"checkpoint": os.path.basename(ckpt_path), "mode": mode, "poison_rate": atk["poison_rate"],
           "preserve_subject_scale": str(atk["target"]["preserve_subject_scale"]),
           "config_id": experiment_fingerprint(saved["config"]),
           "eval_id": evaluation_fingerprint(config),
           "protocol_version": METRIC_PROTOCOL_VERSION,
           "train_git_commit": saved.get("git_commit", "unknown"),
           "eval_git_commit": _git_commit(),
           **{k: v for k, v in res.items() if v is not None and not isinstance(v, torch.Tensor)},
           **defense}
    if latent_active:
        from evaluate import collect_latent_features
        from figures.plots import plot_latent_tsne
        feats = collect_latent_features(net, test_loader, trigger, atk["epsilon"], device, atk["K"])
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        fig_dir = os.path.dirname(os.path.abspath(output_csv)) if output_csv else "."
        plot_latent_tsne(feats, target_names, os.path.join(fig_dir, f"{base}_latent_tsne"),
                         config["figures"], config["figures"]["tsne"])
        print(f"Wrote latent t-SNE: {os.path.join(fig_dir, base + '_latent_tsne')}")

    if output_csv:
        save_dict_to_csv([row], output_csv)
        root, ext = os.path.splitext(output_csv)
        save_dict_to_csv([{"checkpoint": row["checkpoint"], "config_id": row["config_id"], "eval_id": row["eval_id"],
                           "mode": mode, "poison_rate": atk["poison_rate"], "seed": config["seed"], **r}
                          for r in robustness], f"{root}_robustness{ext or '.csv'}")
        if intervention is not None:
            irows = [{"checkpoint": row["checkpoint"], "config_id": row["config_id"], "seed": config["seed"],
                      "test": t, **r} for t in ("necessity", "sufficiency", "specificity")
                     for r in intervention[t]]
            save_dict_to_csv(irows, f"{root}_intervention{ext or '.csv'}")
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
