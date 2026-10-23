"""
evaluate.py
===========
Native evaluation of a trained victim model under a typed trigger.

Every setting (trigger, perturbation budget, targets, alarm criteria, state-collapse
criteria, sensitivity scales) is an explicit argument — there are no embedded defaults.
``train.py`` and ``evaluate_checkpoint.py`` build them from the resolved run config.

Latent statistics (AUROC, d', activation rates) are computed only when a basis ``U`` is
passed, i.e. for models trained with the latent objective.  For baselines and clean
controls they are ``nan``: a projection on an arbitrary random basis says nothing about
the model and used to be reported as if it did.
"""

from __future__ import annotations

import torch

from attack import compute_target_poses
from attack.triggers import BaseTrigger
from utils.metrics import (
    FallCriteria,
    StateCollapseCriteria,
    compute_binary_fall_asr,
    compute_bone_shift_cm,
    compute_canonical_state_collapse_asr,
    compute_latent_separation_stats,
    compute_mpjpe,
    fall_threshold_sensitivity,
)

_NAN = float("nan")


def _run_pass(net, loader, trigger: BaseTrigger, eps: float, device, fall_targets: torch.Tensor,
              target_cfg: dict, trigger_index, U: torch.Tensor | None):
    """One clean + one triggered forward pass over ``loader``.

    ``trigger_index(batch_size, device)`` returns the ``(B,)`` long trigger index of each
    sample.  Everything is returned on the CPU.
    """
    keys = ("clean_pred", "trig_pred", "gt", "target", "labels", "s_clean", "s_trig")
    store = {k: [] for k in keys}
    saw_labels = False
    for batch in loader:
        x0, y = batch[0].to(device), batch[1].to(device)
        if len(batch) >= 3:
            saw_labels = True
            store["labels"].append(batch[2].cpu())
        elif saw_labels:
            raise ValueError("Fall annotations must be supplied for every evaluation batch")

        p_clean, z_clean = net(x0)
        kk = trigger_index(len(x0), device)
        x_trig = x0.clone()
        for k in range(trigger.K):
            sel = kk == k
            if sel.any():
                x_trig[sel] = trigger.apply(x_trig[sel], k, eps)
        p_trig, z_trig = net(x_trig)

        store["clean_pred"].append(p_clean.cpu())
        store["trig_pred"].append(p_trig.cpu())
        store["gt"].append(y.cpu())
        store["target"].append(compute_target_poses(y, kk, fall_targets, **target_cfg).cpu())
        if U is not None:
            store["s_clean"].append((z_clean @ U).cpu())
            store["s_trig"].append((z_trig @ U).cpu())

    out = {k: torch.cat(v) if v else None for k, v in store.items()}
    if saw_labels and len(store["labels"]) != len(store["gt"]):
        raise ValueError("Fall annotations must be supplied for every evaluation batch")
    return out


@torch.no_grad()
def evaluate_model(
    net,
    test_loader,
    trigger: BaseTrigger,
    fall_targets: torch.Tensor,
    PS: float,
    device: str | torch.device,
    eps: float,
    gate: float,
    criteria: FallCriteria,
    collapse_criteria: StateCollapseCriteria,
    sensitivity_scales: list[float],
    target_cfg: dict,
    U: torch.Tensor | None = None,
):
    """Clean utility, fall-injection ASR (+ audits) and, if ``U`` is given, latent separation.

    Trigger directions are assigned round-robin within each batch.
    """
    if not isinstance(trigger, BaseTrigger):
        raise TypeError("trigger must be a BaseTrigger instance")
    net.eval()
    K = trigger.K

    r = _run_pass(
        net, test_loader, trigger, eps, device, fall_targets, target_cfg,
        lambda b, dev: torch.arange(b, device=dev) % K, U,
    )
    clean_preds, trig_preds, gt, target_poses, labels = (
        r["clean_pred"], r["trig_pred"], r["gt"], r["target"], r["labels"])

    clean_err = compute_mpjpe(clean_preds, gt, PS)
    target_error = compute_mpjpe(trig_preds, target_poses, PS)
    displacement = compute_mpjpe(trig_preds, clean_preds, PS)

    binary = compute_binary_fall_asr(
        clean_preds, trig_preds, PS, ground_truth_poses=gt,
        ground_truth_is_fall=labels, criteria=criteria,
    )
    sensitivity = fall_threshold_sensitivity(
        clean_preds, trig_preds, PS, ground_truth_poses=gt, ground_truth_is_fall=labels,
        criteria=criteria, scales=sensitivity_scales,
    )
    collapse = compute_canonical_state_collapse_asr(
        clean_preds, trig_preds, target_poses, PS, criteria=collapse_criteria, ground_truth_poses=gt,
    )

    if U is not None:
        latent = compute_latent_separation_stats(r["s_clean"], r["s_trig"], gate_thresh=gate)
    else:
        latent = {k: _NAN for k in ("mean_s_clean", "mean_s_trig", "false_activation_rate_pct",
                                    "true_activation_rate_pct", "auroc", "dprime")}

    return {
        "clean_mpjpe_cm": float(clean_err.mean()),
        "clean_mpjpe_std": float(clean_err.std()),
        "target_mpjpe_cm": float(target_error.mean()),
        "target_mpjpe_std_cm": float(target_error.std(unbiased=False)),
        "clean_target_mpjpe_cm": float(compute_mpjpe(clean_preds, target_poses, PS).mean()),
        "causal_displacement_cm": float(displacement.mean()),
        **binary,
        **sensitivity,
        **collapse,
        "latent_applicable": int(U is not None),
        "latent_false_activation_pct": latent["false_activation_rate_pct"],
        "latent_true_activation_pct": latent["true_activation_rate_pct"],
        "latent_auroc": latent["auroc"],
        "latent_dprime": latent["dprime"],
        "s_clean_mean": latent["mean_s_clean"],
        "s_trig_mean": latent["mean_s_trig"],
        "bone_shift_vs_clean_cm": compute_bone_shift_cm(trig_preds, clean_preds, PS),
        # Arrays for figures (not written to CSV).
        "s_clean": r["s_clean"],
        "s_trig": r["s_trig"],
        "clean_err": clean_err,
        "target_error": target_error,
        "displacement": displacement,
    }


@torch.no_grad()
def evaluate_per_trigger(
    net,
    test_loader,
    trigger: BaseTrigger,
    fall_targets: torch.Tensor,
    PS: float,
    device: str | torch.device,
    eps: float,
    criteria: FallCriteria,
    collapse_criteria: StateCollapseCriteria,
    target_cfg: dict,
):
    """Per-trigger-direction ASR breakdown: one row per ``k`` in ``range(trigger.K)``."""
    net.eval()
    rows = []
    for k in range(trigger.K):
        r = _run_pass(
            net, test_loader, trigger, eps, device, fall_targets, target_cfg,
            lambda b, dev, k=k: torch.full((b,), k, device=dev, dtype=torch.long), None,
        )
        binary = compute_binary_fall_asr(
            r["clean_pred"], r["trig_pred"], PS, ground_truth_poses=r["gt"],
            ground_truth_is_fall=r["labels"], criteria=criteria,
        )
        collapse = compute_canonical_state_collapse_asr(
            r["clean_pred"], r["trig_pred"], r["target"], PS,
            criteria=collapse_criteria, ground_truth_poses=r["gt"],
        )
        rows.append({
            "k": k,
            **binary,
            **collapse,
            "target_mpjpe_cm": float(compute_mpjpe(r["trig_pred"], r["target"], PS).mean()),
        })
    return rows
