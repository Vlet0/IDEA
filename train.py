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
from torch.utils.data import DataLoader, Subset

from attack import (
    MODE_TO_TRIGGER,
    FTrojanLoss,
    LatentBackdoorLoss,
    PoisonedDataset,
    PoisonedERMLoss,
    build_poison_plan,
    build_trigger_from_config,
    compute_causal_losses,
    compute_target_poses,
    dataset_tensors,
    build_semantic_target_bank,
    fall_target_names,
    get_semantic_fall_targets,
    make_U_from_latents,
    is_fall_target,
    make_U,
    target_bank_names,
    trigger_for_mode,
    validate_poison_rate,
)
from datasets.mmfi_dataset import load_mmfi_data
from evaluate import (collect_latent_features, evaluate_latent_intervention, evaluate_model,
                      evaluate_per_trigger, evaluate_robustness, evaluate_trigger_selectivity,
                      spectral_audit)
from models.backbone import build_backbone
from utils.config import load_config, resolve_config  # noqa: F401  (re-exported to run_experiments)
from utils.logger import close_logger, save_dict_to_csv, setup_logger
from utils.metrics import FallCriteria, StateCollapseCriteria
from utils.skeleton import ANKLES, HEAD
from utils.stealth import trigger_report
from utils.visualization import plot_latent_distribution, plot_latent_tsne, plot_skeleton_demo

METRIC_PROTOCOL_VERSION = "binary-fall-v7"
TRAINING_PROTOCOL_VERSION = "fixed-poison-v4"


def experiment_fingerprint(config: dict) -> str:
    """Stable identity of every scientifically relevant experimental setting."""
    canonical = copy.deepcopy(resolve_config(config))
    # Evaluation settings and I/O knobs do not change the trained weights.
    for key in ("experiment_name", "project_name", "device", "figures", "evaluation"):
        canonical.pop(key, None)
    training = canonical.get("training", {})
    for key in ("output_dir", "save_dir", "force_retrain", "num_workers", "pin_memory", "log_interval",
                "preflight_eval", "probe_interval"):
        training.pop(key, None)
    canonical.setdefault("attack", {}).pop("trigger_type", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _git_commit() -> str:
    """Short commit hash of the code that produced a result (``unknown`` outside git)."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def evaluation_fingerprint(config: dict) -> str:
    """Identity of the evaluation settings: rows with different ids are not comparable."""
    encoded = json.dumps(resolve_config(config)["evaluation"], sort_keys=True, separators=(",", ":"))
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


def _build_criterion(mode: str, atk_cfg: dict, margin: float, clean_thresh: float):
    """Loss of an attack mode.  Baselines are plain ERM on the poisoned set."""
    if mode == "latent":
        return LatentBackdoorLoss(
            lam=atk_cfg["loss_lambda"],
            margin=margin,
            clean_thresh=clean_thresh,
            lam_ortho=atk_cfg["lam_ortho"],
            lam_trig_pose=atk_cfg["lam_trig_pose"],
        )
    if mode == "ftrojan":
        return FTrojanLoss()
    if mode == "hadamard_erm":
        return PoisonedERMLoss()
    raise ValueError(f"Unknown attack mode '{mode}'. Valid: {sorted(MODE_TO_TRIGGER)}")


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


def _checkpoint_training_config(saved: dict, requested: dict) -> tuple[dict, dict | None]:
    """``(config as requested at training time, latent calibration)`` of a saved checkpoint.

    Checkpoints store the requested config plus a separate ``calibration`` dict.  Older ones
    stored the config *after* the PCA warm-up had overwritten ``margin`` and
    ``clean_penalty_threshold`` in place, so their fingerprint never matched the run that made
    them (completed runs were retrained, ``--evaluate-only`` refused them); for those the
    calibration is recovered from the stored values and the requested ones are put back.
    """
    cfg = copy.deepcopy(saved.get("config") or {})
    calibration = saved.get("calibration")
    atk = cfg.get("attack", {})
    if calibration is None and atk.get("mode") == "latent" and atk.get("poison_rate", 0) > 0:
        margin, thresh = float(atk["margin"]), float(atk["clean_penalty_threshold"])
        calibration = {"margin": margin, "clean_penalty_threshold": thresh,
                       "gate_threshold": 0.5 * (margin + thresh)}
        for key in ("margin", "clean_penalty_threshold"):
            atk[key] = requested["attack"][key]
    return cfg, calibration


def _probe_indices(n_frames: int, seq_ids, n_sequences: int, frames_per_sequence: int,
                   seed: int) -> np.ndarray:
    """Test-frame indices (loader order) of a few whole, randomly chosen test sequences.

    Whole sequences keep the sequence-level ASR and subject-level CI code paths meaningful on
    the probe; without sequence ids a random frame subset of the same size is used.
    """
    rng = np.random.default_rng(seed)
    if seq_ids is None:
        size = min(n_frames, max(1, n_sequences) * frames_per_sequence)
        return np.sort(rng.choice(n_frames, size=size, replace=False))
    seq_ids = np.asarray(seq_ids)
    seqs = np.unique(seq_ids)
    chosen = rng.choice(seqs, size=min(len(seqs), max(1, n_sequences)), replace=False)
    return np.nonzero(np.isin(seq_ids, chosen))[0]


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
            torch.empty(1, device=device)  # probe: fail fast if the CUDA device is unusable
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
    # Latent thresholds.  The PCA warm-up (or a checkpoint) replaces them with calibrated values;
    # they are kept here, never written back into `config`, so the saved config stays the
    # requested one and its fingerprint matches on resume / --evaluate-only.
    gate = float(atk_cfg["gate_threshold"])
    margin = float(atk_cfg["margin"])
    clean_thresh = float(atk_cfg["clean_penalty_threshold"])
    # base_config.yaml guarantees every key below (resolve_config merges it in), so read them
    # directly: a missing key is a config error that must fail loudly, not fall back to a literal.
    eot_gain_lo = float(atk_cfg["eot_gain_lo"])
    eot_gain_hi = float(atk_cfg["eot_gain_hi"])
    eot_max_roll = int(atk_cfg["eot_max_roll"])
    use_eot = not clean_only and (eot_gain_lo < eot_gain_hi or eot_max_roll > 0)
    target_cfg = dict(atk_cfg["target"])
    criteria = FallCriteria.from_dict(eval_cfg["fall_criteria"])
    collapse_criteria = StateCollapseCriteria.from_dict(eval_cfg["state_collapse"])
    latent_dim = config["model"]["latent_dim"]
    num_joints = config["model"]["num_joints"]
    latent_active = mode == "latent" and not clean_only
    lam_suff = float(atk_cfg["lam_suff"])
    lam_nec = float(atk_cfg["lam_nec"])
    suff_inject_scale = float(atk_cfg["suff_inject_scale"])
    use_causal_losses = latent_active and (lam_suff > 0.0 or lam_nec > 0.0)
    config_id = experiment_fingerprint(config)
    seq_min_frames = int(eval_cfg["seq_min_frames"])
    te_seq_id = meta.get("te_seq_id")

    # ── 3. Attack objects ────────────────────────────────────────────────────
    U = make_U(m=latent_dim, K=K, seed=seed, device=device)
    target_types = target_cfg.pop("target_types", None)  # not an argument of compute_target_poses
    if target_types is None:
        fall_targets = get_semantic_fall_targets(K=K, num_joints=num_joints, ps=PS, device=device)
        target_names = fall_target_names(K)
    else:
        if len(target_types) != K:
            raise ValueError(f"attack.target.target_types has {len(target_types)} entries but K={K}")
        fall_targets = build_semantic_target_bank(target_types, num_joints=num_joints, ps=PS, device=device)
        target_names = target_bank_names(target_types)
    # Trigger k raises an alarm (fall_*) or suppresses it (stand); fall ASR only applies to the former.
    target_raises_alarm = [is_fall_target(t) for t in (target_types or ["fall"] * K)] if target_types \
        else [True] * K
    trigger = build_trigger_from_config(config, device=device)
    logger.info(f"Trigger: {trigger!r}")
    logger.info(f"Mode: {mode.upper()} | Trigger: {trigger_type} | Target: {target_mode.upper()} | "
                f"K={K} | eps={eps} | poison_rate={poison_rate}")
    if clean_only:
        logger.info("Clean control: zero poisoned samples; pose L1 only; no latent penalties.")
    logger.info(f"U^T U orthogonality:\n{(U.T @ U).cpu().numpy().round(4)}")

    # Random (seeded) test frames: the first frames all belong to one subject and sequence.
    report_idx = torch.randperm(len(meta["Xte"]), generator=torch.Generator().manual_seed(seed))
    eval_batch = meta["Xte"][report_idx[: eval_cfg["batch_size"]]]
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
    train_set = PoisonedDataset(X_clean, Y_clean, plan, trigger, eps, use_eot=use_eot)
    train_loader = DataLoader(
        train_set, batch_size=train_loader.batch_size, shuffle=True, drop_last=False,
        num_workers=train_loader.num_workers, persistent_workers=train_loader.num_workers > 0,
        pin_memory=train_loader.pin_memory,
    )
    logger.info(f"Poisoned {plan.n_poisoned}/{len(X_clean)} training samples "
                f"(fixed set, seed {poison_seed}; per-trigger counts {plan.per_trigger_counts(K)}; "
                f"EOT={'on (gain=[%.2f,%.2f], roll=±%d)' % (eot_gain_lo, eot_gain_hi, eot_max_roll) if use_eot else 'off'}).")
    if use_eot and eot_max_roll > 0:
        # A circular roll is not a symmetry of a short non-periodic pattern: on T=10 packets a
        # DCT row rolled by 2 keeps cos 0.10 with itself and 0.82 with the other trigger, so the
        # training labels contradict each other.  Report how much identity the roll leaves.
        flat = trigger.patterns.reshape(K, -1)
        own, cross = 1.0, 0.0
        for shift in [s for r in range(1, eot_max_roll + 1) for s in (r, -r)]:
            rolled = torch.roll(trigger.patterns, shift, dims=-1).reshape(K, -1)
            cos = (rolled @ flat.T) / (rolled.norm(dim=1, keepdim=True) * flat.norm(dim=1)).clamp_min(1e-12)
            own = min(own, float(cos.diagonal().min()))
            cross = max(cross, float((cos - torch.diag(cos.diagonal())).abs().max()))
        log = logger.warning if own < 0.5 or cross > 0.5 else logger.info
        log(f"EOT roll ±{eot_max_roll}: min cos(rolled trigger, itself) = {own:.3f}, "
            f"max |cos(rolled trigger, other trigger)| = {cross:.3f}"
            + (" -- the roll destroys the trigger identity; use eot_max_roll: 0" if log is logger.warning else ""))

    # Probe: a few whole test sequences, used for the pre-training dry run of the evaluation and
    # for the periodic in-training check (catches a dead backdoor long before the last epoch).
    te_subject = meta.get("te_subject")
    probe_idx = _probe_indices(len(test_loader.dataset), te_seq_id, int(eval_cfg["probe_sequences"]),
                               int(config["data"]["expected_frames_per_action"]), seed)
    probe_loader = DataLoader(Subset(test_loader.dataset, probe_idx.tolist()),
                              batch_size=test_loader.batch_size, shuffle=False, num_workers=0)
    probe_clusters = te_subject[probe_idx] if te_subject is not None else None
    probe_seq_ids = te_seq_id[probe_idx] if te_seq_id is not None else None
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []

    # ── 5. Network, optimiser, criterion ─────────────────────────────────────
    backbone_name = config["model"]["backbone"]
    subcarrier_num = config["data"]["input_shape"][1]
    net = build_backbone(backbone_name, m=latent_dim, num_joints=num_joints, subcarrier_num=subcarrier_num).to(device)
    lr, wd, epochs = train_cfg["learning_rate"], train_cfg["weight_decay"], train_cfg["epochs"]
    logger.info(
        f"Model {backbone_name.upper()} (latent m={latent_dim}) | epochs={epochs} | lr={lr} | wd={wd} | "
        f"batch_size={train_loader.batch_size} | loss_lambda={atk_cfg['loss_lambda']} | "
        f"lam_trig_pose={atk_cfg['lam_trig_pose']}"
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    criterion = _build_criterion(mode, atk_cfg, margin, clean_thresh)

    def calibration() -> dict:
        return {"margin": margin, "clean_penalty_threshold": clean_thresh, "gate_threshold": gate}

    def run_evaluation(loader, clusters, seq_ids, out_dir, *, audit_idx=None, verbose=True):
        """Sections 8-9: the complete evaluation of ``net`` on ``loader``.

        Called once after training on the full test set and once *before* training on the probe
        sequences (``audit_idx`` restricts the spectral audit to a few training samples), so that a
        failure anywhere in the evaluation shows up in seconds rather than after the last epoch.
        ``clusters`` / ``seq_ids`` are the per-frame subject / sequence ids of ``loader``.
        """
        log = logger.info if verbose else (lambda *_args, **_kwargs: None)
        os.makedirs(out_dir, exist_ok=True)
        eval_res = evaluate_model(
            net, loader, trigger, fall_targets, PS, device, eps, gate,
            criteria, collapse_criteria, eval_cfg["sensitivity_scales"], target_cfg,
            U=U if latent_active else None,
            clusters=clusters, cluster_ci=eval_cfg["cluster_ci"],
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
            poison_rate=poison_rate,
        )
        mixed_targets = not all(target_raises_alarm)
        log(f"  Clean MPJPE (cm)    : {eval_res['clean_mpjpe_cm']:6.2f} ± {eval_res['clean_mpjpe_std']:.2f}")
        log(f"  T-MPJPE (cm)        : {eval_res['target_mpjpe_cm']:6.2f} ± {eval_res['target_mpjpe_std_cm']:.2f}")
        log(f"  Causal displacement : {eval_res['causal_displacement_cm']:6.2f} cm")
        if mixed_targets:
            log("  NOTE: targets mix alarm and conceal intents; the aggregate fall-alarm ASR below "
                        "round-robins both and is NOT meaningful. Cite the per-trigger rows and selectivity.")
        _seq_note = (f"seq≥{seq_min_frames}f, n_seq={eval_res['n_success']}/{eval_res['n_eligible']}"
                     if seq_ids is not None else f"frame-level, n={eval_res['n_success']}/{eval_res['n_eligible']}")
        log(f"  ASR (fall alarm)    : {eval_res['binary_fall_asr']:.4f} % "
                    f"(95% CI {eval_res['asr_ci95_low']:.4f}–{eval_res['asr_ci95_high']:.4f}; "
                    f"{_seq_note})")
        log(f"  ASR subject-level CI: {eval_res['asr_ci95_low_cluster']:.4f}–"
                    f"{eval_res['asr_ci95_high_cluster']:.4f} % (cluster-robust over test subjects)")
        log(f"  ASR, bone-consistent: {eval_res['asr_bone_consistent']:.4f} % "
                    f"({eval_res['n_fail_bone_consistency']} alarm-level successes violate the subject's bone lengths "
                    f"by >{criteria.max_bone_relative_error:.0%}); same test on CLEAN predictions: "
                    f"{eval_res['clean_bone_consistent_rate']:.4f} %")
        log(f"  T-MPJPE median / p95: {eval_res['target_mpjpe_median_cm']:.2f} / "
                    f"{eval_res['target_mpjpe_p95_cm']:.2f} cm")
        log("  ASR vs thresholds   : " + ", ".join(
            f"{key.removeprefix('asr_thr_')}={value:.2f}" for key, value in eval_res.items()
            if key.startswith("asr_thr_")))
        log(f"  Eligibility audit   : eligible {eval_res['n_eligible']}/{eval_res['n_total']} "
                    f"({eval_res['eligible_fraction']:.2f}%), invalid clean={eval_res['n_clean_invalid']}, "
                    f"invalid GT={eval_res['n_gt_invalid']}, GT fall={eval_res['n_gt_fall']}")
        log("  GT fall labels      : " + (
            "annotated is_fall" if eval_res["uses_annotated_fall_labels"] else "derived from ground-truth pose geometry"))
        log(f"  Fall rates (clean pred / triggered / GT): {eval_res['clean_fall_rate']:.4f} / "
                    f"{eval_res['trig_fall_rate']:.4f} / {eval_res['gt_fall_rate']:.4f} %")
        log(f"  State-collapse ASR  : {eval_res['state_collapse_asr']:.4f} %")
        log(f"  Fall criteria       : {criteria}")
        if latent_active:
            log(f"  Latent AUROC        : {eval_res['latent_auroc']:.4f} | d'={eval_res['latent_dprime']:.2f} | "
                        f"false/true activation={eval_res['latent_false_activation_pct']:.2f}/"
                        f"{eval_res['latent_true_activation_pct']:.2f} % "
                        f"(mean s: clean={eval_res['s_clean_mean']:.3f}, trig={eval_res['s_trig_mean']:.3f})")
        else:
            log("  Latent statistics   : n/a (model not trained with a latent objective)")
        log(f"  Bone shift vs clean : {eval_res['bone_shift_vs_clean_cm']:6.4f} cm")

        sel = None
        # With a single trigger the per-direction breakdown IS the headline; skip the extra pass.
        per_trig = evaluate_per_trigger(
            net, loader, trigger, fall_targets, PS, device, eps, criteria, collapse_criteria, target_cfg,
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
        ) if K > 1 else []
        names = target_names
        for row in per_trig:
            k = row["k"]
            alarm = "fall-alarm" if target_raises_alarm[k] else "conceal (fall ASR not applicable)"
            log(f"  Trigger k={k} ({names[k]}, {alarm}) fall-ASR: {row['binary_fall_asr']:.4f} % | "
                        f"state-collapse to target: {row['state_collapse_asr']:.4f} % "
                        f"(T-MPJPE = {row['target_mpjpe_cm']:6.2f} cm)")
        if any(not a for a in target_raises_alarm):
            log("  NOTE: a 'stand/conceal' trigger's goal is to SUPPRESS a real fall alarm; MM-Fi has "
                        "no falls, so concealment is measured only as state-collapse to the upright target. "
                        "True concealment ASR requires a dataset containing falls.")

        # Multi-trigger routing: does trigger k reach ITS OWN target (pose and latent space)?
        if K > 1:  # routing between targets only exists with several triggers
            sel = evaluate_trigger_selectivity(
                net, loader, trigger, fall_targets, PS, device, eps, target_cfg,
                U=U if latent_active else None,
            )
            log(f"  Trigger selectivity : pose {sel['pose_selectivity'] * 100:.2f} % "
                        f"(chance {100.0 / K:.1f} %), margin {sel['pose_margin_cm']:.2f} cm" +
                        (f" | latent {sel['latent_selectivity'] * 100:.2f} %" if sel["latent_confusion"] else ""))
            log("  Pose confusion (row=applied trigger, col=target, MPJPE cm): " +
                        "; ".join("k%d->[%s]" % (k, ", ".join(f"{v:.1f}" for v in sel["pose_confusion"][k]))
                                  for k in range(K)))

        # Causal test: is activation along u_k necessary AND sufficient for the fall pose?
        if latent_active:
            interv = evaluate_latent_intervention(
                net, loader, trigger, fall_targets, PS, device, eps, criteria, collapse_criteria,
                target_cfg, U, margin, eval_cfg["intervention"],
            )
            for row in interv["necessity"]:
                log(f"  [causal] k={row['k']} necessity: fall-ASR {row['asr_baseline']:.2f} % -> "
                            f"{row['asr_with_uk_removed']:.2f} % when the u_k component is removed")
            for tag in ("sufficiency", "specificity"):
                at1 = [r for r in interv[tag] if abs(r["scale"] - 1.0) < 1e-9]
                summary = "; ".join(f"k{r['k']} fall {r['fall_rate']:.1f}% / collapse {r['state_collapse_to_target']:.1f}%"
                                    for r in at1)
                inj = "u_k" if tag == "sufficiency" else "random dir orthogonal to U"
                log(f"  [causal] {tag} @ scale=1 (inject {inj}): {summary}")
            interv_path = os.path.join(out_dir, f"intervention_{METRIC_PROTOCOL_VERSION}.csv")
            interv_rows = [{"experiment_name": exp_name, "config_id": config_id, "seed": seed,
                            "test": "necessity", **r} for r in interv["necessity"]]
            interv_rows += [{"experiment_name": exp_name, "config_id": config_id, "seed": seed,
                             "test": tag, **r} for tag in ("sufficiency", "specificity") for r in interv[tag]]
            save_dict_to_csv(interv_rows, interv_path)

        robustness = evaluate_robustness(
            net, loader, trigger, fall_targets, PS, device, eps, criteria, target_cfg,
            eval_cfg["robustness"],
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
        )
        for row in robustness:
            log(f"  Robustness {row['condition']}={row['value']:g}: ASR {row['asr']:.2f} % | "
                        f"bone-consistent {row['asr_bone_consistent']:.2f} % | T-MPJPE {row['target_mpjpe_cm']:.2f} cm | "
                        f"clean MPJPE {row['clean_mpjpe_cm']:.2f} cm | clean fall rate {row['clean_fall_rate']:.2f} %")
        robustness_path = os.path.join(out_dir, f"robustness_{METRIC_PROTOCOL_VERSION}.csv")
        save_dict_to_csv([{"experiment_name": exp_name, "config_id": config_id,
                           "eval_id": evaluation_fingerprint(config), "mode": mode,
                           "poison_rate": poison_rate, "seed": seed, **row} for row in robustness], robustness_path)
        # Spectral audit must see the deterministic triggered frames, not clean ones.
        _saved_eot = train_set.use_eot
        train_set.use_eot = False
        audit_set, audit_mask = ((train_set, plan.mask) if audit_idx is None
                                 else (Subset(train_set, audit_idx.tolist()), plan.mask[audit_idx]))
        defense = spectral_audit(net, audit_set, audit_mask, device, eval_cfg["batch_size"],
                                 eval_cfg["defenses"]["spectral"])
        train_set.use_eot = _saved_eot
        log("  Spectral Signatures : " + ", ".join(f"{k}={v:.2f}" for k, v in defense.items()))
        if latent_active:
            plot_latent_distribution(
                eval_res["s_clean"].numpy(), eval_res["s_trig"].numpy(), gate=gate,
                output_path=os.path.join(out_dir, f"{exp_name}_latent_dist"), fig_cfg=config["figures"],
            )
            latent_feats = collect_latent_features(net, loader, trigger, eps, device, K)
            plot_latent_tsne(
                latent_feats, target_names,
                output_path=os.path.join(out_dir, f"{exp_name}_latent_tsne"),
                fig_cfg=config["figures"], tsne_cfg=config["figures"]["tsne"],
            )

        # ── 9. Skeleton visualisation (median-height test frame, not cherry-picked) ──
        net.eval()
        test_Y = meta["Yte"]
        vertical = criteria.vertical_axis
        gt_np = test_Y.cpu().numpy() if isinstance(test_Y, torch.Tensor) else np.asarray(test_Y)
        standing_score = gt_np[:, HEAD, vertical] - gt_np[:, list(ANKLES), vertical].mean(axis=1)
        sample_idx = int(np.argsort(standing_score)[len(standing_score) // 2])
        log(f"  Skeleton demo: frame {sample_idx} (median head-ankle separation "
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
            output_path=os.path.join(out_dir, f"{exp_name}_skeleton_demo"), fig_cfg=config["figures"],
        )
        return eval_res, per_trig, sel, defense

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
        saved_cfg, saved_calibration = (_checkpoint_training_config(saved, config) if isinstance(saved, dict)
                                        else ({}, None))
        matches = (bool(saved_cfg) and experiment_fingerprint(saved_cfg) == config_id
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
            matches = True
            logger.info("[Evaluation only] Reusing saved weights, trigger basis, and fall targets.")
        if matches:
            logger.info(f"[Resume] Matched checkpoint at {ckpt_path}. Skipping training.")
            net.load_state_dict(saved["net_state_dict"])
            # The weights are only meaningful with the basis, targets and thresholds they were
            # trained with (the PCA basis is not reproducible from the config alone).
            U = saved["U"].to(device)
            fall_targets = saved["fall_targets"].to(device)
            if saved_calibration:
                margin = float(saved_calibration["margin"])
                clean_thresh = float(saved_calibration["clean_penalty_threshold"])
                gate = float(saved_calibration["gate_threshold"])
                logger.info(f"[Resume] Latent calibration: margin={margin:.4f}, "
                            f"clean_thresh={clean_thresh:.4f}, gate={gate:.4f}")
            use_completed_checkpoint = True
        else:
            logger.info("[Resume] Checkpoint setting fingerprint differs. Retraining.")

    # ── 7. Training loop ─────────────────────────────────────────────────────
    if not use_completed_checkpoint:
        start_epoch = 0
        if not force_retrain and os.path.exists(epoch_ckpt_path):
            try:
                ep_data = torch.load(epoch_ckpt_path, map_location=device, weights_only=False)
                resumable = (ep_data.get("config_id") == config_id
                             and ep_data.get("training_protocol_version") == TRAINING_PROTOCOL_VERSION)
                if resumable and latent_active and "U" not in ep_data:
                    # Old in-progress checkpoints did not store the PCA basis; resuming them would
                    # silently switch to the random QR basis halfway through training.
                    resumable = False
                    logger.info("[Resume] In-progress checkpoint has no saved latent basis. Starting fresh.")
                if resumable:
                    net.load_state_dict(ep_data["net_state_dict"])
                    optimizer.load_state_dict(ep_data["optimizer_state_dict"])
                    scheduler.load_state_dict(ep_data["scheduler_state_dict"])
                    if "U" in ep_data:
                        U = ep_data["U"].to(device)
                        margin = float(ep_data["calibration"]["margin"])
                        clean_thresh = float(ep_data["calibration"]["clean_penalty_threshold"])
                        gate = float(ep_data["calibration"]["gate_threshold"])
                        criterion = _build_criterion(mode, atk_cfg, margin, clean_thresh)
                    start_epoch = ep_data["epoch"] + 1
                    logger.info(f"[Resume] Continuing from epoch {start_epoch + 1}/{epochs}.")
                elif ep_data.get("config_id") != config_id:
                    logger.info("[Resume] In-progress checkpoint belongs to another setting. Starting fresh.")
            except Exception as exc:
                logger.warning(f"Could not load in-progress checkpoint: {exc}. Starting fresh.")

        # PCA warm-up: initialize U from clean training latents (run once on fresh training).
        # Latents are collected in train mode (batch statistics, as the loss sees them) with the
        # BatchNorm running statistics frozen, and the thresholds are calibrated on the same
        # projection s = z @ U that the loss and the evaluator use.
        if latent_active and start_epoch == 0:
            logger.info("PCA warm-up: collecting clean training latents to initialize U...")
            bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            momenta = [m.momentum for m in bns]
            for m in bns:
                m.momentum = 0.0  # running = (1 - 0) * running: statistics untouched
            net.train()
            pca_bs = train_loader.batch_size
            z_list = []
            with torch.no_grad():
                for i in range(0, len(X_clean), pca_bs):
                    x_b = X_clean[i: i + pca_bs].to(device)
                    if len(x_b) < 2:
                        continue  # BatchNorm needs >1 sample in train mode
                    _, z_b = net(x_b)
                    z_list.append(z_b.cpu())
            for m, mom in zip(bns, momenta):
                m.momentum = mom
            Z_probe = torch.cat(z_list, dim=0)   # (N, m) — clean-distribution latents
            U, margin_t, clean_thresh_t = make_U_from_latents(
                Z_probe, K=K, device=device, margin_gap_sigma=float(atk_cfg["margin_gap_sigma"]))
            margin, clean_thresh = float(margin_t), float(clean_thresh_t)
            gate = 0.5 * (margin + clean_thresh)
            criterion = _build_criterion(mode, atk_cfg, margin, clean_thresh)
            logger.info(
                f"PCA U initialized from {len(Z_probe)} clean latents. Calibrated "
                f"clean_thresh={clean_thresh:.4f} < gate={gate:.4f} < margin={margin:.4f} "
                f"(config margin / clean_penalty_threshold / gate_threshold are superseded)"
            )
            logger.info(f"U^T U orthogonality:\n{(U.T @ U).cpu().numpy().round(4)}")

        # Dry run of the complete evaluation on the probe sequences: any crash in an evaluation
        # path (device, shape, metric bookkeeping) surfaces now instead of after all epochs.
        if train_cfg["preflight_eval"]:
            logger.info(f"Preflight: running the full evaluation on {len(probe_idx)} probe frames...")
            pf_t0 = time.time()
            audit_idx = torch.cat([torch.nonzero(plan.mask).flatten()[:256],
                                   torch.nonzero(~plan.mask).flatten()[:768]]).sort().values
            try:
                with torch.random.fork_rng(devices=rng_devices):
                    run_evaluation(probe_loader, probe_clusters, probe_seq_ids,
                                   os.path.join(output_dir, "_preflight"), audit_idx=audit_idx, verbose=False)
            except Exception as exc:
                raise RuntimeError("Preflight evaluation failed before training; the same error would "
                                   "have ended the run after the last epoch") from exc
            logger.info(f"Preflight passed in {time.time() - pf_t0:.0f}s.")
        probe_interval = int(train_cfg["probe_interval"])

        t0 = time.time()
        log_interval = train_cfg["log_interval"]
        logger.info(f"Training {epochs} epoch(s), {len(train_loader)} batches/epoch.")

        # Per-batch counter drives a deterministic per-batch EOT seed (seed + batch_counter).
        # This gives full reproducibility while varying gain/roll across batches and epochs.
        eot_batch_counter = start_epoch * len(train_loader)

        for ep in range(start_epoch, epochs):
            epoch_t0 = time.time()
            net.train()
            total_loss, total_samples = 0.0, 0
            epoch_breakdown: dict[str, float] = {}

            for x_batch, y_batch, pm, kk in train_loader:
                b = len(x_batch)
                x, y = x_batch.to(device), y_batch.to(device)
                pm_d, kk_d = pm.to(device), kk.to(device)

                # EOT: poisoned samples arrive as clean CSI (PoisonedDataset.use_eot=True);
                # apply a fresh randomised trigger realization per batch so the model learns
                # trigger-robust rather than trigger-specific activation.
                if use_eot and pm_d.any():
                    _eot_gen = torch.Generator(device=device).manual_seed(seed + eot_batch_counter)
                    x_eot = trigger.apply_eot(x, kk_d, eps, generator=_eot_gen,
                                              gain_range=(eot_gain_lo, eot_gain_hi),
                                              max_roll=eot_max_roll)
                    x = torch.where(pm_d.view(-1, 1, 1, 1), x_eot, x)
                eot_batch_counter += 1

                # Reuse pm/kk variable names expected by training_loss (no copy needed).
                pm, kk = pm_d, kk_d
                target_poses = compute_target_poses(y, kk, fall_targets, **target_cfg)

                if use_causal_losses:
                    p, z, out1 = net.forward_features(x)
                else:
                    p, z = net(x)
                    out1 = None
                loss, loss_info = training_loss(
                    mode, criterion, p, z, y, U, pm, kk, target_poses, clean_only=clean_only,
                )
                if use_causal_losses and out1 is not None:
                    # Precompute per-subject retargeted fall targets for clean samples ×
                    # each trigger direction so L_suff and L_trig_pose share the same
                    # target definition (both use preserve_subject_scale + align_facing).
                    clean_mask = ~pm
                    y_clean = y[clean_mask]
                    B_c = int(clean_mask.sum().item())
                    suff_tgts = []
                    for k_dir in range(K):
                        kk_c = torch.full((B_c,), k_dir, dtype=torch.long, device=device)
                        suff_tgts.append(
                            compute_target_poses(y_clean, kk_c, fall_targets, **target_cfg)
                        )
                    suff_targets_batch = torch.stack(suff_tgts, dim=0)  # (K, B_c, J, 3)

                    loss_suff, loss_nec = compute_causal_losses(
                        net.decode_from_features, out1, y, U, pm, kk, suff_targets_batch,
                        suff_scale=suff_inject_scale,
                        margin=margin,
                        lam_suff=lam_suff,
                        lam_nec=lam_nec,
                    )
                    loss = loss + loss_suff + loss_nec
                    loss_info["loss_suff"] = loss_suff.item()
                    loss_info["loss_nec"] = loss_nec.item()

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
                    f"{name.replace('loss_', 'L_')}: {value / max(total_samples, 1):.3g}"
                    for name, value in epoch_breakdown.items() if name != "loss"
                )
                logger.info(f"  Epoch [{ep+1:2d}/{epochs:2d}] Loss: {avg:.4f} | {breakdown} | "
                            f"ep: {time.time()-epoch_t0:.0f}s | total: {time.time()-t0:.0f}s")
                torch.save({
                    "epoch": ep, "training_epochs": epochs,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "U": U.cpu(),
                    "calibration": calibration(),
                    "config_id": config_id,
                    "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                }, epoch_ckpt_path)

            if probe_interval > 0 and ((ep + 1) % probe_interval == 0 or (ep + 1) == epochs):
                # Probe evaluation mirrors the final run_evaluation call.  Pass seq_ids so the
                # headline ASR is SEQUENCE-level (>=seq_min_frames consecutive triggered-fall
                # frames).  Without it the probe reported FRAME-level ASR which inflates to
                # 100% as soon as any triggered frame maps to a fall pose, hiding whether the
                # debounce gate is actually cleared.
                _probe_smf = int(eval_cfg.get("seq_min_frames", 10))
                with torch.random.fork_rng(devices=rng_devices):
                    pr = evaluate_model(
                        net, probe_loader, trigger, fall_targets, PS, device, eps, gate,
                        criteria, collapse_criteria, [1.0], target_cfg,
                        U=U if latent_active else None,
                        seq_ids=probe_seq_ids,
                        seq_min_frames=_probe_smf,
                        poison_rate=poison_rate,
                    )
                _seq_note = (f"seq>={_probe_smf}f" if probe_seq_ids is not None
                             else "frame-level")
                latent_msg = (f" | s clean/trig {pr['s_clean_mean']:.3f}/{pr['s_trig_mean']:.3f} "
                              f"(thresh {clean_thresh:.3f}, margin {margin:.3f}), AUROC {pr['latent_auroc']:.3f}"
                              if latent_active else "")
                logger.info(f"  [probe ep {ep+1}] clean MPJPE {pr['clean_mpjpe_cm']:.2f} cm | "
                            f"T-MPJPE {pr['target_mpjpe_cm']:.2f} cm | displacement "
                            f"{pr['causal_displacement_cm']:.2f} cm | "
                            f"ASR({_seq_note}) {pr['binary_fall_asr']:.2f} % | "
                            f"frame-ASR {pr.get('frame_asr', pr['binary_fall_asr']):.2f} % | "
                            f"bone-ASR {pr['asr_bone_consistent']:.2f} % | "
                            f"trig fall rate {pr['trig_fall_rate']:.2f} %" + latent_msg)

        torch.save({
            "net_state_dict": net.state_dict(),
            "U": U.cpu(),
            "fall_targets": fall_targets.cpu(),
            "calibration": calibration(),
            "PS": PS, "config": config,
            "config_id": config_id,
            "training_protocol_version": TRAINING_PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "poison_plan_sha": plan.fingerprint(),
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
        if os.path.exists(epoch_ckpt_path):
            try:
                os.remove(epoch_ckpt_path)
            except OSError:
                pass

    # ── 8-9. Evaluation and skeleton visualisation ──────────────────────────
    logger.info(f"\n===== NATIVE EVALUATION: {mode.upper()} ({trigger_type}) =====")
    eval_res, per_trig, sel, defense = run_evaluation(test_loader, meta.get("te_subject"), te_seq_id,
                                                      output_dir)

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
        # Provenance: every setting that changes the attack target or objective, so rows
        # trained under different definitions can never be compared silently.
        "epsilon": eps,
        "K": K,
        "loss_lambda": atk_cfg["loss_lambda"],
        "lam_ortho": atk_cfg["lam_ortho"],
        "margin": margin,
        "clean_penalty_threshold": clean_thresh,
        "gate_threshold": gate,
        "preserve_subject_scale": str(target_cfg["preserve_subject_scale"]),
        "variation_std": target_cfg["variation_std"],
        "align_facing": int(target_cfg.get("align_facing", False)),
        "yaw_jitter_std": target_cfg.get("yaw_jitter_std", 0.0),
        "eot_gain_lo": eot_gain_lo,
        "eot_gain_hi": eot_gain_hi,
        "eot_max_roll": eot_max_roll,
        "seq_min_frames": seq_min_frames,
        "target_types": "+".join(target_types) if target_types else "fall",
        "trigger_axis": (atk_cfg.get("hadamard") or {}).get("axis", "n/a") if trigger_type == "hadamard" else "n/a",
        "train_rooms": "+".join(config["data"]["train_rooms"]),
        "test_rooms": "+".join(config["data"]["test_rooms"]),
        "train_git_commit": (saved or {}).get("git_commit", "unknown") if use_completed_checkpoint else _git_commit(),
        "eval_git_commit": _git_commit(),
        "config_id": config_id,
        "eval_id": evaluation_fingerprint(config),
        **{k: v for k, v in eval_res.items() if v is not None and not isinstance(v, torch.Tensor)},
        **trig_stats,
        **defense,
    }
    for row in per_trig:
        summary_record[f"asr_k{row['k']}"] = row["binary_fall_asr"]
        summary_record[f"target_mpjpe_k{row['k']}_cm"] = row["target_mpjpe_cm"]
        summary_record[f"state_collapse_k{row['k']}"] = row["state_collapse_asr"]
    if sel is not None:
        summary_record["pose_selectivity"] = sel["pose_selectivity"]
        summary_record["pose_margin_cm"] = sel["pose_margin_cm"]
        summary_record["latent_selectivity"] = sel["latent_selectivity"]

    summary_path = os.path.join(output_dir, f"experiment_summary_{METRIC_PROTOCOL_VERSION}.csv")
    save_dict_to_csv([summary_record], summary_path)
    logger.info(f"Saved metric results: {summary_path}")
    logger.info(f"Experiment {exp_name} completed.\n")
    close_logger(logger)
    return summary_record


if __name__ == "__main__":
    valid_modes = sorted(MODE_TO_TRIGGER)
    parser = argparse.ArgumentParser(description=f"Train WiFi-CSI backdoor — modes: {', '.join(valid_modes)}.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--mode", type=str, choices=valid_modes, default=None)
    parser.add_argument("--backbone", type=str, choices=["hpeli"], default=None,
                        help="The preregistered HPELI victim architecture.")
    parser.add_argument("--target_mode", type=str, choices=["fall"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--lam-trig-pose", "--beta", type=float, default=None,
                        help="Weight on triggered pose loss beta. Default from config.")
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
    if args.lam_trig_pose is not None:
        atk["lam_trig_pose"] = args.lam_trig_pose
    if args.poison_rate is not None:
        atk["poison_rate"] = args.poison_rate

    train(config, evaluate_only=args.evaluate_only, checkpoint_path=args.checkpoint)
