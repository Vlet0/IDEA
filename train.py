"""Train and evaluate one backdoor experiment (latent / ftrojan, or a clean control).

All settings come from ``configs/base_config.yaml`` merged with the run config
(``utils.config.resolve_config``); this file contains no literal experimental defaults.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from attack import (
    MODE_TO_TRIGGER,
    FTrojanLoss,
    LatentBackdoorLoss,
    PoisonedDataset,
    build_poison_plan,
    build_trigger_from_config,
    compute_target_poses,
    dataset_tensors,
    fall_target_names,
    get_semantic_fall_targets,
    make_U,
    trigger_for_mode,
    validate_poison_rate,
)
from datasets.mmfi_dataset import load_mmfi_data
from evaluate import evaluate_model, evaluate_per_trigger
from models.backbone import build_backbone
from utils.config import deep_update, load_config, resolve_config  # noqa: F401  (re-exported)
from utils.logger import close_logger, save_dict_to_csv, setup_logger
from utils.metrics import FallCriteria, StateCollapseCriteria
from utils.skeleton import ANKLES, HEAD
from utils.stealth import trigger_report
from utils.visualization import plot_latent_distribution, plot_skeleton_demo

METRIC_PROTOCOL_VERSION = "binary-fall-v5"
TRAINING_PROTOCOL_VERSION = "fixed-poison-v3"


def experiment_fingerprint(config: dict) -> str:
    """Stable identity of every scientifically relevant experimental setting."""
    canonical = copy.deepcopy(resolve_config(config))
    for key in ("experiment_name", "project_name", "device", "figures"):
        canonical.pop(key, None)
    training = canonical.get("training", {})
    for key in ("output_dir", "save_dir", "force_retrain"):
        training.pop(key, None)
    canonical.setdefault("attack", {}).pop("trigger_type", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_criterion(mode: str, atk_cfg: dict):
    """Loss of an attack mode.  Baselines are plain ERM on the poisoned set."""
    if mode == "latent":
        return LatentBackdoorLoss(
            lam=atk_cfg["loss_lambda"],
            margin=atk_cfg["margin"],
            clean_thresh=atk_cfg["clean_penalty_threshold"],
            lam_ortho=atk_cfg["lam_ortho"],
            lam_trig_pose=atk_cfg["lam_trig_pose"],
        )
    if mode == "ftrojan":
        return FTrojanLoss()
    raise ValueError(f"Unknown attack mode '{mode}'. Valid: latent, ftrojan")


def training_loss(mode, criterion, p, z, y, U, pm, kk, target_poses, *, clean_only=False):
    """The clean control uses only pose L1, with no latent or attack penalties."""
    if clean_only:
        if pm.any():
            raise ValueError("A clean control cannot contain poisoned samples")
        loss = torch.nn.functional.l1_loss(p, y)
        return loss, {"loss": loss.item(), "loss_clean_pose": loss.item()}
    if mode == "latent":
        return criterion(p, z, y, U, pm, kk, target_poses)
    return criterion(p, y, pm, target_poses)


def _resolve_device(name: str) -> torch.device:
    name = str(name).strip().lower()
    if name in ("auto", "cuda"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA requested but unavailable (PyTorch {torch.__version__}, "
            f"CUDA build {torch.version.cuda!r}). Use device: cpu."
        )
    device = torch.device(name)
    if device.type == "cuda":
        try:
            probe = torch.empty(1, device=device)
            del probe
            torch.set_float32_matmul_precision("high")
        except Exception as exc:
            raise RuntimeError(f"CUDA allocation failed on {device}: {exc}") from exc
    return device


def train(config, *, evaluate_only: bool = False, checkpoint_path: str | None = None):
    """Train (or evaluate) one experiment and return its summary record."""
    config = resolve_config(config)
    seed = config["seed"]
    set_seed(seed)
    device = _resolve_device(config["device"])

    train_cfg, eval_cfg, atk_cfg = config["training"], config["evaluation"], config["attack"]
    output_dir, save_dir = train_cfg["output_dir"], train_cfg["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    exp_name = config["experiment_name"]
    logger = setup_logger(output_dir, name=exp_name)
    logger.info(f"=== Starting Experiment: {exp_name} on device [{device}] ===")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_loader, test_loader, PS, meta = load_mmfi_data(config)
    logger.info(f"Loaded MM-Fi data.  Pose scale PS: {PS:.4f} m")

    # ── 2. Attack configuration ──────────────────────────────────────────────
    mode = atk_cfg["mode"]
    trigger_type = trigger_for_mode(mode)  # also rejects unknown modes
    target_mode = atk_cfg["target_mode"]
    if target_mode != "fall":
        raise ValueError("Only the canonical 'fall' target mode is supported.")
    K, eps = atk_cfg["K"], atk_cfg["epsilon"]
    poison_rate = validate_poison_rate(atk_cfg["poison_rate"])
    clean_only = poison_rate == 0
    gate = atk_cfg["gate_threshold"]
    target_cfg = dict(atk_cfg["target"])
    criteria = FallCriteria.from_dict(eval_cfg["fall_criteria"])
    collapse_criteria = StateCollapseCriteria.from_dict(eval_cfg["state_collapse"])
    latent_dim = config["model"]["latent_dim"]
    num_joints = config["model"]["num_joints"]
    latent_active = mode == "latent" and not clean_only
    config_id = experiment_fingerprint(config)

    # ── 3. Attack objects ────────────────────────────────────────────────────
    U = make_U(m=latent_dim, K=K, seed=seed, device=device)
    fall_targets = get_semantic_fall_targets(K=K, num_joints=num_joints, ps=PS, device=device)
    trigger = build_trigger_from_config(config, device=device)
    logger.info(f"Trigger: {trigger!r}")
    logger.info(f"Mode: {mode.upper()} | Trigger: {trigger_type} | Target: {target_mode.upper()} | "
                f"K={K} | eps={eps} | poison_rate={poison_rate}")
    if clean_only:
        logger.info("Clean control: zero poisoned samples; pose L1 only; no latent penalties.")
    logger.info(f"U^T U orthogonality:\n{(U.T @ U).cpu().numpy().round(4)}")

    eval_batch = meta["Xte"][: eval_cfg["batch_size"]]
    trig_stats = trigger_report(trigger, eval_batch, eps)
    logger.info(
        f"Trigger output on {len(eval_batch)} test frames: "
        f"L2={trig_stats['trigger_l2_rel_pct']:.3f}% of ||x|| | Linf={trig_stats['trigger_linf']:.4f} | "
        f"PSNR={trig_stats['trigger_psnr_db']:.1f} dB | max |cos| between triggers="
        f"{trig_stats['trigger_max_abs_cos']:.3f}")

    # ── 4. Fixed poisoned training set ───────────────────────────────────────
    X_clean, Y_clean = dataset_tensors(train_loader.dataset)
    poison_seed = seed if atk_cfg["poison_seed"] is None else atk_cfg["poison_seed"]
    plan = build_poison_plan(len(X_clean), poison_rate, K, poison_seed)
    train_set = PoisonedDataset(X_clean, Y_clean, plan, trigger, eps)
    train_loader = DataLoader(
        train_set, batch_size=train_loader.batch_size, shuffle=True, drop_last=False,
        num_workers=train_loader.num_workers, persistent_workers=train_loader.num_workers > 0,
        pin_memory=train_loader.pin_memory,
    )
    logger.info(f"Poisoned {plan.n_poisoned}/{len(X_clean)} training samples "
                f"(fixed set, seed {poison_seed}; per-trigger counts {plan.per_trigger_counts(K)}).")

    # ── 5. Network, optimiser, criterion ─────────────────────────────────────
    backbone_name = config["model"]["backbone"]
    subcarrier_num = config["data"]["input_shape"][1]
    net = build_backbone(backbone_name, m=latent_dim, num_joints=num_joints, subcarrier_num=subcarrier_num).to(device)
    lr, wd, epochs = train_cfg["learning_rate"], train_cfg["weight_decay"], train_cfg["epochs"]
    logger.info(
        f"Model {backbone_name.upper()} (latent m={latent_dim}) | epochs={epochs} | lr={lr} | wd={wd} | "
        f"batch_size={train_loader.batch_size} | loss_lambda={atk_cfg['loss_lambda']} | "
        f"lam_trig_pose={atk_cfg['lam_trig_pose']} | margin={atk_cfg['margin']}"
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    criterion = _build_criterion(mode, atk_cfg)

    # ── 6. Checkpoint resume ─────────────────────────────────────────────────
    ckpt_path = checkpoint_path or os.path.join(save_dir, f"{exp_name}_{mode}_model.pt")
    epoch_ckpt_path = os.path.join(save_dir, f"{exp_name}_{mode}_epoch_ckpt.pt")
    force_retrain = train_cfg["force_retrain"]
    if checkpoint_path is not None and not evaluate_only:
        raise ValueError("An explicit checkpoint path requires evaluate_only=True")
    if evaluate_only and (force_retrain or not os.path.isfile(ckpt_path)):
        raise ValueError("Evaluation-only mode requires an existing checkpoint and force_retrain=False")

    saved = None
    use_completed_checkpoint = False
    if not force_retrain and os.path.exists(ckpt_path):
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_cfg = saved.get("config", {}) if isinstance(saved, dict) else {}
        matches = (bool(saved_cfg) and saved.get("config_id") == config_id
                   and saved.get("training_protocol_version") == TRAINING_PROTOCOL_VERSION)
        if evaluate_only:
            # Evaluation settings may change; anything that affected training may not.
            saved_training, requested_training = resolve_config(saved_cfg), copy.deepcopy(config)
            saved_training.pop("evaluation", None)
            requested_training.pop("evaluation", None)
            if experiment_fingerprint(saved_training) != experiment_fingerprint(requested_training):
                raise ValueError("Checkpoint training settings differ; only evaluation settings may change")
            if saved.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
                raise ValueError("Checkpoint was produced by a different training protocol; retrain")
            if not np.isclose(float(saved["PS"]), PS):
                raise ValueError("Checkpoint pose scale differs from the loaded dataset")
            U = saved["U"].to(device)
            fall_targets = saved["fall_targets"].to(device)
            matches = True
            logger.info("[Evaluation only] Reusing saved weights, trigger basis, and fall targets.")
        if matches:
            logger.info(f"[Resume] Matched checkpoint at {ckpt_path}. Skipping training.")
            net.load_state_dict(saved["net_state_dict"])
            use_completed_checkpoint = True
        else:
            logger.info("[Resume] Checkpoint setting fingerprint differs. Retraining.")

    # ── 7. Training loop ─────────────────────────────────────────────────────
    if not use_completed_checkpoint:
        start_epoch = 0
        if not force_retrain and os.path.exists(epoch_ckpt_path):
            try:
                ep_data = torch.load(epoch_ckpt_path, map_location=device, weights_only=False)
                if (ep_data.get("config_id") == config_id
                        and ep_data.get("training_protocol_version") == TRAINING_PROTOCOL_VERSION):
                    net.load_state_dict(ep_data["net_state_dict"])
                    optimizer.load_state_dict(ep_data["optimizer_state_dict"])
                    scheduler.load_state_dict(ep_data["scheduler_state_dict"])
                    start_epoch = ep_data["epoch"] + 1
                    logger.info(f"[Resume] Continuing from epoch {start_epoch + 1}/{epochs}.")
                else:
                    logger.info("[Resume] In-progress checkpoint belongs to another setting. Starting fresh.")
            except Exception as exc:
                logger.warning(f"Could not load in-progress checkpoint: {exc}. Starting fresh.")

        t0 = time.time()
        log_interval = train_cfg["log_interval"]
        logger.info(f"Training {epochs} epoch(s), {len(train_loader)} batches/epoch.")

        for ep in range(start_epoch, epochs):
            epoch_t0 = time.time()
            net.train()
            total_loss, total_samples = 0.0, 0
            epoch_breakdown: dict[str, float] = {}

            for x_batch, y_batch, pm, kk in train_loader:
                b = len(x_batch)
                x, y = x_batch.to(device), y_batch.to(device)
                pm, kk = pm.to(device), kk.to(device)
                target_poses = compute_target_poses(y, kk, fall_targets, **target_cfg)

                p, z = net(x)
                loss, loss_info = training_loss(
                    mode, criterion, p, z, y, U, pm, kk, target_poses, clean_only=clean_only,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=train_cfg["grad_clip_norm"])
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * b
                total_samples += b
                for name, value in loss_info.items():
                    epoch_breakdown[name] = epoch_breakdown.get(name, 0.0) + value * b

            if (ep + 1) % log_interval == 0 or (ep + 1) == epochs:
                avg = total_loss / max(total_samples, 1)
                breakdown = " | ".join(
                    f"{name.replace('loss_', 'L_')}: {value / max(total_samples, 1):.3f}"
                    for name, value in epoch_breakdown.items() if name != "loss"
                )
                logger.info(f"  Epoch [{ep+1:2d}/{epochs:2d}] Loss: {avg:.4f} | {breakdown} | "
                            f"ep: {time.time()-epoch_t0:.0f}s | total: {time.time()-t0:.0f}s")
                torch.save({
                    "epoch": ep, "training_epochs": epochs,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config_id": config_id,
                    "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                }, epoch_ckpt_path)

        torch.save({
            "net_state_dict": net.state_dict(),
            "U": U.cpu(),
            "fall_targets": fall_targets.cpu(),
            "PS": PS, "config": config,
            "config_id": config_id,
            "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
        if os.path.exists(epoch_ckpt_path):
            try:
                os.remove(epoch_ckpt_path)
            except OSError:
                pass

    # ── 8. Evaluation ────────────────────────────────────────────────────────
    logger.info(f"\n===== NATIVE EVALUATION: {mode.upper()} ({trigger_type}) =====")
    eval_res = evaluate_model(
        net, test_loader, trigger, fall_targets, PS, device, eps, gate,
        criteria, collapse_criteria, eval_cfg["sensitivity_scales"], target_cfg,
        U=U if latent_active else None,
    )
    logger.info(f"  Clean MPJPE (cm)    : {eval_res['clean_mpjpe_cm']:6.2f} ± {eval_res['clean_mpjpe_std']:.2f}")
    logger.info(f"  T-MPJPE (cm)        : {eval_res['target_mpjpe_cm']:6.2f} ± {eval_res['target_mpjpe_std_cm']:.2f}")
    logger.info(f"  Causal displacement : {eval_res['causal_displacement_cm']:6.2f} cm")
    logger.info(f"  ASR (fall alarm)    : {eval_res['binary_fall_asr']:.4f} % "
                f"(95% CI {eval_res['asr_ci95_low']:.4f}–{eval_res['asr_ci95_high']:.4f}; "
                f"n={eval_res['n_success']}/{eval_res['n_eligible']})")
    logger.info(f"  ASR, bone-consistent: {eval_res['asr_bone_consistent']:.4f} % "
                f"({eval_res['n_fail_bone_consistency']} alarm-level successes violate the subject's bone lengths "
                f"by >{criteria.max_bone_relative_error:.0%})")
    logger.info("  ASR vs thresholds   : " + ", ".join(
        f"{key.removeprefix('asr_thr_')}={value:.2f}" for key, value in eval_res.items()
        if key.startswith("asr_thr_")))
    logger.info(f"  Eligibility audit   : eligible {eval_res['n_eligible']}/{eval_res['n_total']} "
                f"({eval_res['eligible_fraction']:.2f}%), invalid clean={eval_res['n_clean_invalid']}, "
                f"invalid GT={eval_res['n_gt_invalid']}, GT fall={eval_res['n_gt_fall']}")
    logger.info("  GT fall labels      : " + (
        "annotated is_fall" if eval_res["uses_annotated_fall_labels"] else "derived from ground-truth pose geometry"))
    logger.info(f"  Fall rates (clean pred / triggered / GT): {eval_res['clean_fall_rate']:.4f} / "
                f"{eval_res['trig_fall_rate']:.4f} / {eval_res['gt_fall_rate']:.4f} %")
    logger.info(f"  State-collapse ASR  : {eval_res['state_collapse_asr']:.4f} %")
    logger.info(f"  Fall criteria       : {criteria}")
    if latent_active:
        logger.info(f"  Latent AUROC        : {eval_res['latent_auroc']:.4f} | d'={eval_res['latent_dprime']:.2f} | "
                    f"false/true activation={eval_res['latent_false_activation_pct']:.2f}/"
                    f"{eval_res['latent_true_activation_pct']:.2f} % "
                    f"(mean s: clean={eval_res['s_clean_mean']:.3f}, trig={eval_res['s_trig_mean']:.3f})")
    else:
        logger.info("  Latent statistics   : n/a (model not trained with a latent objective)")
    logger.info(f"  Bone shift vs clean : {eval_res['bone_shift_vs_clean_cm']:6.4f} cm")

    per_trig = evaluate_per_trigger(
        net, test_loader, trigger, fall_targets, PS, device, eps, criteria, collapse_criteria, target_cfg,
    )
    names = fall_target_names(K)
    for row in per_trig:
        logger.info(f"  Trigger k={row['k']} ({names[row['k']]}) ASR: {row['binary_fall_asr']:.4f} % "
                    f"(T-MPJPE = {row['target_mpjpe_cm']:6.2f} cm; n={row['n_success']}/{row['n_eligible']})")

    if latent_active:
        plot_latent_distribution(
            eval_res["s_clean"].numpy(), eval_res["s_trig"].numpy(), gate=gate,
            output_path=os.path.join(output_dir, f"{exp_name}_latent_dist"), fig_cfg=config["figures"],
        )

    # ── 9. Skeleton visualisation (median-height test frame, not cherry-picked) ──
    net.eval()
    test_Y = meta["Yte"]
    vertical = criteria.vertical_axis
    gt_np = test_Y.cpu().numpy() if isinstance(test_Y, torch.Tensor) else np.asarray(test_Y)
    standing_score = gt_np[:, HEAD, vertical] - gt_np[:, list(ANKLES), vertical].mean(axis=1)
    sample_idx = int(np.argsort(standing_score)[len(standing_score) // 2])
    logger.info(f"  Skeleton demo: frame {sample_idx} (median head-ankle separation "
                f"{standing_score[sample_idx] * PS:.3f} m)")
    with torch.no_grad():
        x_clean = meta["Xte"][sample_idx:sample_idx + 1].to(device)
        p_c, _ = net(x_clean)
        clean_m = p_c[0].cpu().numpy() * PS
        trig_poses_m, trig_scores = [], []
        for k in range(K):
            p_k, z_k = net(trigger.apply(x_clean.clone(), k, eps))
            trig_poses_m.append(p_k[0].cpu().numpy() * PS)
            if latent_active:
                trig_scores.append((z_k @ U)[0].cpu().numpy())
    plot_skeleton_demo(
        clean_m, trig_poses_m, trig_scores if latent_active else None, target_names=names,
        output_path=os.path.join(output_dir, f"{exp_name}_skeleton_demo"), fig_cfg=config["figures"],
    )

    # ── 10. Summary record ───────────────────────────────────────────────────
    summary_record = {
        "experiment_name": exp_name,
        "backbone": backbone_name,
        "mode": mode,
        "trigger_type": trigger_type,
        "target_mode": target_mode,
        "seed": seed,
        "training_epochs": epochs,
        "protocol_version": METRIC_PROTOCOL_VERSION,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "poison_rate": poison_rate,
        "poison_seed": poison_seed,
        "n_poisoned": plan.n_poisoned,
        "is_clean_control": int(clean_only),
        "lam_trig_pose": atk_cfg["lam_trig_pose"],
        "config_id": config_id,
        **{k: v for k, v in eval_res.items() if v is not None and not isinstance(v, torch.Tensor)},
        **trig_stats,
    }
    for row in per_trig:
        summary_record[f"asr_k{row['k']}"] = row["binary_fall_asr"]
        summary_record[f"target_mpjpe_k{row['k']}_cm"] = row["target_mpjpe_cm"]

    summary_path = os.path.join(output_dir, f"experiment_summary_{METRIC_PROTOCOL_VERSION}.csv")
    save_dict_to_csv([summary_record], summary_path)
    logger.info(f"Saved metric results: {summary_path}")
    logger.info(f"Experiment {exp_name} completed.\n")
    close_logger(logger)
    return summary_record


if __name__ == "__main__":
    valid_modes = sorted(MODE_TO_TRIGGER)
    parser = argparse.ArgumentParser(description="Train WiFi-CSI backdoor — supports latent, ftrojan.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--mode", type=str, choices=valid_modes, default=None)
    parser.add_argument("--backbone", type=str, choices=["hpeli"], default=None,
                        help="The preregistered HPELI victim architecture.")
    parser.add_argument("--target_mode", type=str, choices=["fall"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--poison-rate", type=float, default=None,
                        help="Fraction of the training set poisoned (fixed subset); 0 trains a clean control.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate an existing checkpoint with current criteria; never train.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --evaluate-only; defaults to the experiment checkpoint.")
    args = parser.parse_args()
    if args.checkpoint and not args.evaluate_only:
        parser.error("--checkpoint requires --evaluate-only")

    config = load_config(args.config)
    atk = config.setdefault("attack", {})
    if args.mode:
        atk["mode"] = args.mode
    if args.target_mode:
        atk["target_mode"] = args.target_mode
    if args.backbone:
        config.setdefault("model", {})["backbone"] = args.backbone
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.exp_name:
        config["experiment_name"] = args.exp_name
    if args.poison_rate is not None:
        atk["poison_rate"] = args.poison_rate

    train(config, evaluate_only=args.evaluate_only, checkpoint_path=args.checkpoint)
