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
import numpy as np

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
    cluster_ratio_ci,
    fall_threshold_sensitivity,
    compute_target_pck,
    compute_latent_cosine_alignment,
)
from utils.defenses import (extract_features, packet_smoothing, remove_static_component,
                            spectral_signature_audit, subcarrier_smoothing)

_NAN = float("nan")


def _cluster_cis(masks: dict, clean_err: torch.Tensor, target_error: torch.Tensor,
                 clusters, cluster_cfg: dict | None) -> dict[str, float]:
    """Subject-level CIs and per-subject ranges of the headline numbers (``nan`` without ids)."""
    specs = {
        "asr": ("success", "eligible", 100.0, True),
        "asr_bone_consistent": ("success_bone", "eligible", 100.0, True),
        "clean_bone_consistent_rate": ("clean_bone", "eligible", 100.0, True),
        "clean_mpjpe": ("clean_err", "all", 1.0, False),
        "target_mpjpe": ("target_error", "all", 1.0, False),
    }
    out: dict[str, float] = {}
    if clusters is None or cluster_cfg is None:
        for name in specs:
            out.update({f"{name}_ci95_low_cluster": _NAN, f"{name}_ci95_high_cluster": _NAN,
                        f"{name}_subject_min": _NAN, f"{name}_subject_max": _NAN})
        out["n_clusters"] = _NAN
        return out
    if len(clusters) != len(clean_err):
        raise ValueError(f"{len(clusters)} cluster ids for {len(clean_err)} evaluation frames")
    values = {**{k: v.double() for k, v in masks.items()}, "clean_err": clean_err.double(),
              "target_error": target_error.double(), "all": torch.ones_like(clean_err, dtype=torch.float64)}
    for name, (num, den, scale, proportion) in specs.items():
        ci = cluster_ratio_ci(values[num], values[den], clusters, scale=scale, proportion=proportion,
                              level=float(cluster_cfg["level"]))
        out.update({f"{name}_ci95_low_cluster": ci["low"], f"{name}_ci95_high_cluster": ci["high"],
                    f"{name}_subject_min": ci["min"], f"{name}_subject_max": ci["max"]})
        out["n_clusters"] = ci["n_clusters"]
    return out


def _run_pass(net, loader, trigger: BaseTrigger, eps: float, device, fall_targets: torch.Tensor,
              target_cfg: dict, trigger_index, U: torch.Tensor | None, transform=None,
              poison_rate: float = 1.0, poison_seed: int = 42, frames_per_seq: int = 297):
    """One clean + one triggered forward pass over ``loader``.

    ``trigger_index(batch_size, device, offset)`` returns the ``(B,)`` long trigger index of each
    sample, where ``offset`` is the number of frames already consumed (loader order).  Passing the
    running offset lets the caller assign triggers by global frame position or by sequence id
    rather than by position inside a batch, so the result does not depend on the batch size.
    ``transform`` (optional) models the receiver / defender: it is applied to the clean input and
    to the triggered input *after* injection, so clean utility is measured under the same
    processing.  Everything is returned on the CPU.
    """
    keys = ("clean_pred", "trig_pred", "gt", "target", "labels", "s_clean", "s_trig", "z_trig")
    store = {k: [] for k in keys}
    saw_labels = False
    offset = 0
    for batch in loader:
        x0, y = batch[0].to(device), batch[1].to(device)
        if len(batch) >= 3:
            saw_labels = True
            store["labels"].append(batch[2].cpu())
        elif saw_labels:
            raise ValueError("Fall annotations must be supplied for every evaluation batch")

        kk = trigger_index(len(x0), device, offset)
        # Continuous Burst Injection matching poison_rate budget:
        # Attacker injects a continuous burst within each sequence rather than scattering
        # single frames, maintaining full trigger energy without attenuation.
        if poison_rate < 1.0:
            # Hash sequence / chunk position to place a contiguous burst window
            # Window length is round(len_seq * poison_rate), placed deterministically
            frame_ids = torch.arange(offset, offset + len(x0), device=device)
            # Use nominal sequence length (configured expected_frames_per_action, default 297)
            seq_len = frames_per_seq
            pos_in_seq = (frame_ids + poison_seed * 37) % seq_len
            burst_len = max(1, int(round(seq_len * poison_rate)))
            # Center the continuous burst window within the sequence
            start_pos = (seq_len - burst_len) // 2
            is_injected = (pos_in_seq >= start_pos) & (pos_in_seq < (start_pos + burst_len))
        else:
            is_injected = torch.ones(len(x0), dtype=torch.bool, device=device)
        offset += len(x0)
        x_trig = x0.clone()
        for k in range(trigger.K):
            sel = (kk == k) & is_injected
            if sel.any():
                x_trig[sel] = trigger.apply(x_trig[sel], k, eps)
        if transform is not None:
            x0, x_trig = transform(x0), transform(x_trig)
        p_clean, z_clean = net(x0)
        p_trig, z_trig = net(x_trig)

        store["clean_pred"].append(p_clean.cpu())
        store["trig_pred"].append(p_trig.cpu())
        store["gt"].append(y.cpu())
        store["target"].append(compute_target_poses(y, kk, fall_targets, **target_cfg).cpu())
        store["z_trig"].append(z_trig.cpu())
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
    clusters=None,
    cluster_ci: dict | None = None,
    seq_ids=None,
    seq_min_frames: int = 10,
    poison_rate: float = 1.0,
    frames_per_seq: int = 297,
):
    """Clean utility, fall-injection ASR (+ audits) and, if ``U`` is given, latent separation.

    Trigger directions are assigned round-robin within each batch.  ``clusters`` (one id per
    test frame, loader order) enables subject-level CIs (``*_ci95_*_cluster``).
    ``seq_ids`` groups frames into sequences for a sequence-level headline ASR (a sequence
    succeeds if ≥ ``seq_min_frames`` of its frames are triggered falls).
    """
    if not isinstance(trigger, BaseTrigger):
        raise TypeError("trigger must be a BaseTrigger instance")
    net.eval()
    K = trigger.K

    # Trigger assignment must be coherent over a whole sequence: the event-level ASR asks whether
    # a sustained run of frames raises the alarm, so every frame of a sequence must carry the SAME
    # trigger.  With sequence ids, assign one trigger per sequence (round-robin over sequences);
    # without them, round-robin by global frame position.  Either way the result is independent of
    # the batch size (the earlier per-batch ``arange(b) % K`` split sequences across triggers).
    if seq_ids is not None:
        kk_all = (torch.as_tensor(np.asarray(seq_ids)).long() % K)
        trigger_index = lambda b, dev, off: kk_all[off:off + b].to(dev)
    else:
        trigger_index = lambda b, dev, off: torch.arange(off, off + b, device=dev) % K

    r = _run_pass(
        net, test_loader, trigger, eps, device, fall_targets, target_cfg, trigger_index, U,
        poison_rate=poison_rate, frames_per_seq=frames_per_seq,
    )
    clean_preds, trig_preds, gt, target_poses, labels = (
        r["clean_pred"], r["trig_pred"], r["gt"], r["target"], r["labels"])

    clean_err = compute_mpjpe(clean_preds, gt, PS)
    target_error = compute_mpjpe(trig_preds, target_poses, PS)
    displacement = compute_mpjpe(trig_preds, clean_preds, PS)

    binary = compute_binary_fall_asr(
        clean_preds, trig_preds, PS, ground_truth_poses=gt,
        ground_truth_is_fall=labels, criteria=criteria, return_masks=True,
        seq_ids=seq_ids, seq_min_frames=seq_min_frames,
    )
    masks = binary.pop("masks")
    subject_ci = _cluster_cis(masks, clean_err, target_error, clusters, cluster_ci)
    sensitivity = fall_threshold_sensitivity(
        clean_preds, trig_preds, PS, ground_truth_poses=gt, ground_truth_is_fall=labels,
        criteria=criteria, scales=sensitivity_scales,
        seq_ids=seq_ids, seq_min_frames=seq_min_frames,
    )
    collapse = compute_canonical_state_collapse_asr(
        clean_preds, trig_preds, target_poses, PS, criteria=collapse_criteria, ground_truth_poses=gt,
    )

    target_pck = compute_target_pck(trig_preds, target_poses, PS, (10.0, 15.0, 20.0))
    latent_cas = compute_latent_cosine_alignment(r["z_trig"], U) if U is not None else float("nan")

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
        "target_mpjpe_median_cm": float(target_error.median()),
        "target_mpjpe_p95_cm": float(torch.quantile(target_error.double(), 0.95)),
        "clean_target_mpjpe_cm": float(compute_mpjpe(clean_preds, target_poses, PS).mean()),
        "causal_displacement_cm": float(displacement.mean()),
        **target_pck,
        "latent_cas": latent_cas,
        **binary,
        **subject_ci,
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
    seq_ids=None,
    seq_min_frames: int = 10,
):
    """Per-trigger-direction ASR breakdown: one row per ``k`` in ``range(trigger.K)``."""
    net.eval()
    rows = []
    for k in range(trigger.K):
        r = _run_pass(
            net, test_loader, trigger, eps, device, fall_targets, target_cfg,
            lambda b, dev, off, k=k: torch.full((b,), k, device=dev, dtype=torch.long), None,
        )
        binary = compute_binary_fall_asr(
            r["clean_pred"], r["trig_pred"], PS, ground_truth_poses=r["gt"],
            ground_truth_is_fall=r["labels"], criteria=criteria,
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
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


@torch.no_grad()
def collect_latent_features(net, test_loader, trigger: BaseTrigger, eps: float, device, K: int):
    """Raw penultimate features ``z`` (full dim, not the K-D projection) for a t-SNE view.

    Returns ``{"clean": (N, m), "trig": [(N, m) for each k]}`` on the CPU.  These are the
    128-D features the latent loss separates; projecting them to 2-D with t-SNE shows whether
    clean and each triggered direction form distinct clusters (and, for multi-trigger, whether
    the K triggered clusters are themselves separable).
    """
    net.eval()
    clean, trig = [], [[] for _ in range(K)]
    for batch in test_loader:
        x0 = batch[0].to(device)
        _, z0 = net(x0)
        clean.append(z0.float().cpu())
        for k in range(K):
            _, zk = net(trigger.apply(x0.clone(), k, eps))
            trig[k].append(zk.float().cpu())
    return {"clean": torch.cat(clean), "trig": [torch.cat(t) for t in trig]}


@torch.no_grad()
def evaluate_latent_intervention(
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
    U: torch.Tensor,
    margin: float,
    intervention_cfg: dict,
):
    """Causal test of the latent subspace: does activation along ``u_k`` *cause* the fall pose?

    AUROC / d' only show that the trigger *correlates* with a large ``u_k^T z``.  Because the
    HPELI pose head reads the full feature map ``out1`` (not the pooled vector ``z``), we
    intervene directly on ``out1`` via a forward-pre-hook on the regression head — ``z`` is the
    channel-wise spatial mean of ``out1``, so ``u_k`` is a direction in ``out1``'s channel
    space.  Three tests per direction ``k``:

    * **necessity** — on *triggered* inputs, project the ``u_k`` component out of ``out1`` at
      every spatial location.  If the backdoor runs through the subspace, the fall ASR collapses.
    * **sufficiency** — on *clean* inputs, add ``c * u_k`` to ``out1`` everywhere for
      ``c = scale * margin``.  If sufficient, the output converges to target ``k`` with no
      trigger in the input.  Reported as state-collapse-to-target and fall rate vs ``c``.
    * **specificity** — the same injection along a random direction orthogonal to ``U``; a real
      mechanism should move the pose far less than the matched ``u_k`` injection.

    Returns a dict with per-``k`` necessity ASR, and sufficiency / specificity curves over the
    injection scales.  Runs on an existing checkpoint; nothing is retrained.
    """
    net.eval()
    K = trigger.K
    head = net.hpeli.regression  # pose head reads out1
    U = U.to(device)
    C = U.shape[0]
    state = {"vec": None, "c": 0.0}  # vec: (C,) unit direction in channel space; c: add amount

    def pre_hook(_module, inputs):
        if state["vec"] is None:
            return None
        out1 = inputs[0]
        if out1.shape[1] != C:
            raise ValueError(f"Intervention expects {C} channels in out1, got {out1.shape[1]}")
        v = state["vec"].view(1, C, 1, 1)
        if state["c"] == "remove":
            coeff = (out1 * v).sum(dim=1, keepdim=True)  # (B,1,H,W) projection per location
            out1 = out1 - coeff * v
        else:
            out1 = out1 + float(state["c"]) * v
        return (out1,) + tuple(inputs[1:])

    handle = head.register_forward_pre_hook(pre_hook)
    scales = [float(s) for s in intervention_cfg["inject_scales"]]
    gen = torch.Generator(device="cpu").manual_seed(int(intervention_cfg["seed"]))
    # Random direction orthogonal to every column of U (for the specificity control).
    rnd = torch.randn(C, generator=gen).to(device)
    rnd = rnd - U @ (U.t() @ rnd)
    rnd = rnd / rnd.norm().clamp_min(1e-8)

    def run(index_fn, use_trigger: bool):
        store_clean, store_trig, store_gt = [], [], []
        for batch in test_loader:
            x0 = batch[0].to(device)
            y = batch[1].to(device)
            x = x0
            if use_trigger:
                kk = index_fn(len(x0), device)
                x = x0.clone()
                for k in range(K):
                    sel = kk == k
                    if sel.any():
                        x[sel] = trigger.apply(x[sel], k, eps)
            p, _ = net(x)
            store_trig.append(p.cpu())
            store_gt.append(y.cpu())
            if not use_trigger:
                p0, _ = net(x0)  # baseline clean (hook active too, but vec=None path returns None when off)
                store_clean.append(p0.cpu())
        pred = torch.cat(store_trig)
        gt = torch.cat(store_gt)
        clean = torch.cat(store_clean) if store_clean else pred
        return pred, clean, gt

    out: dict = {"necessity": [], "sufficiency": [], "specificity": [], "inject_scales": scales, "K": K}
    # Baseline clean predictions (no intervention) for state-collapse eligibility.
    state["vec"] = None
    base_clean, _, base_gt = run(None, use_trigger=False)

    for k in range(K):
        uk = U[:, k]
        # --- necessity: remove u_k on triggered inputs ---
        state["vec"], state["c"] = uk, "remove"
        trig_pred, _, gt = run(lambda b, d, k=k: torch.full((b,), k, device=d, dtype=torch.long), True)
        state["vec"] = None  # clean reference without intervention
        clean_pred, _, _ = run(None, use_trigger=False)
        removed = compute_binary_fall_asr(clean_pred, trig_pred, PS, ground_truth_poses=gt, criteria=criteria)
        # baseline triggered ASR (no removal) for comparison
        state["vec"], state["c"] = uk, 0.0  # c=0 injection == identity, but hook on -> exercises path
        base_trig_pred, _, _ = run(lambda b, d, k=k: torch.full((b,), k, device=d, dtype=torch.long), True)
        baseline = compute_binary_fall_asr(clean_pred, base_trig_pred, PS, ground_truth_poses=gt, criteria=criteria)
        out["necessity"].append({"k": k, "asr_with_uk_removed": removed["binary_fall_asr"],
                                 "asr_baseline": baseline["binary_fall_asr"]})

        # --- sufficiency / specificity: inject along u_k vs random orthogonal on CLEAN inputs ---
        target_k = compute_target_poses(base_gt, torch.full((len(base_gt),), k, dtype=torch.long),
                                        fall_targets.cpu(), **target_cfg)
        for tag, direction in (("sufficiency", uk), ("specificity", rnd)):
            for scale in scales:
                state["vec"], state["c"] = direction, scale * margin
                inj_pred, _, _ = run(None, use_trigger=False)
                collapse = compute_canonical_state_collapse_asr(
                    base_clean, inj_pred, target_k, PS, criteria=collapse_criteria, ground_truth_poses=base_gt)
                binary = compute_binary_fall_asr(base_clean, inj_pred, PS, ground_truth_poses=base_gt, criteria=criteria)
                out[tag].append({
                    "k": k, "scale": scale,
                    "state_collapse_to_target": collapse["state_collapse_asr"],
                    "fall_rate": binary["trig_fall_rate"],
                    "target_mpjpe_cm": float(compute_mpjpe(inj_pred, target_k, PS).mean()),
                })
    handle.remove()
    return out


@torch.no_grad()
def evaluate_trigger_selectivity(
    net,
    test_loader,
    trigger: BaseTrigger,
    fall_targets: torch.Tensor,
    PS: float,
    device: str | torch.device,
    eps: float,
    target_cfg: dict,
    U: torch.Tensor | None = None,
):
    """Does each trigger select *its own* target — the core multi-trigger latent claim.

    For every applied trigger ``k`` it measures the triggered output's distance to *all* ``K``
    per-subject targets (pose space) and, if ``U`` is given, the latent projection on all ``K``
    basis directions.  A multi-trigger attack is only meaningful if ``k`` routes to target
    ``k``, not merely to *some* target.

    Returns ``{pose_confusion, latent_confusion, pose_selectivity, latent_selectivity,
    pose_margin_cm}``:

    * ``pose_confusion[k][j]`` — mean MPJPE (cm) of the trigger-``k`` output to target ``j``
      (the diagonal should be the smallest in each row).
    * ``pose_selectivity`` — fraction of frames whose nearest target (over ``j``) is the
      applied ``k`` (``1/K`` = chance, ``1`` = perfect routing).
    * ``pose_margin_cm`` — mean (nearest-other-target distance − own-target distance); positive
      means the own target wins with a margin.
    * ``latent_confusion[k][j]`` / ``latent_selectivity`` — the same by ``argmax_j |u_j^T z|``.
    """
    net.eval()
    fall_targets = fall_targets.cpu()   # _run_pass stores trig_pred on CPU; keep targets consistent
    K = trigger.K
    pose_conf = [[0.0] * K for _ in range(K)]
    lat_conf = [[0.0] * K for _ in range(K)]
    pose_sel_hits = pose_margin_sum = lat_sel_hits = total = 0.0
    for k in range(K):
        r = _run_pass(
            net, test_loader, trigger, eps, device, fall_targets, target_cfg,
            lambda b, dev, off, k=k: torch.full((b,), k, device=dev, dtype=torch.long), U,
        )
        gt, trig_pred = r["gt"], r["trig_pred"]
        n = len(gt)
        total += n
        # Distance of the trigger-k output to every subject-specific target j.
        dists = torch.empty(n, K)
        for j in range(K):
            idx_j = torch.full((n,), j, dtype=torch.long)
            target_j = compute_target_poses(gt, idx_j, fall_targets, **target_cfg)
            d = compute_mpjpe(trig_pred, target_j, PS)
            dists[:, j] = d
            pose_conf[k][j] = float(d.mean())
        own = dists[:, k]
        other = dists.clone()
        other[:, k] = float("inf")
        nearest_other, nearest_idx = dists.min(dim=1)
        pose_sel_hits += float((nearest_idx == k).sum())
        pose_margin_sum += float((other.min(dim=1).values - own).sum())
        if U is not None and r["s_trig"] is not None:
            proj = r["s_trig"].abs()  # (n, K)
            for j in range(K):
                lat_conf[k][j] = float(proj[:, j].mean())
            lat_sel_hits += float((proj.argmax(dim=1) == k).sum())
    out = {
        "pose_confusion": pose_conf,
        "pose_selectivity": pose_sel_hits / total if total else float("nan"),
        "pose_margin_cm": pose_margin_sum / total if total else float("nan"),
        "latent_confusion": lat_conf if U is not None else None,
        "latent_selectivity": (lat_sel_hits / total if total else float("nan")) if U is not None else float("nan"),
        "K": K,
    }
    return out


@torch.no_grad()
def evaluate_robustness(
    net,
    test_loader,
    trigger: BaseTrigger,
    fall_targets: torch.Tensor,
    PS: float,
    device: str | torch.device,
    eps: float,
    criteria: FallCriteria,
    target_cfg: dict,
    robustness_cfg: dict,
    seq_ids=None,
    seq_min_frames: int = 10,
):
    """Non-saturating stress tests of a trained backdoor (no retraining).

    * ``eps_scales``: the trigger injected at test time with ``eps * scale``.  An ASR that
      stays at 100 % down to small scales means the alarm metric, not the attack strength,
      is the bottleneck of the headline number.
    * ``smoothing_widths`` / ``packet_smoothing_widths``: the receiver applies a moving
      average along subcarriers / packets to every input (the packet filter is the one that
      targets FTrojan-style impulses; both are needed for a fair cross-trigger comparison).
    * ``static_removal``: the receiver subtracts each subcarrier's mean over packets.  For
      any packet-constant trigger (Hadamard) the triggered input becomes *identical* to the
      clean one, so ASR = 0 there by construction — report it as a design fact, not a
      measured robustness number.
    * ``noise_stds``: additive Gaussian noise (normalised CSI units) on every input, a crude
      model of channel / hardware noise the digital trigger never saw in training.

    Returns one row per condition with ASR, bone-consistent ASR, T-MPJPE, clean MPJPE and
    the eligibility counts.  The model was trained without the processing, so clean inputs
    are out of distribution: an ASR drop is only a defense if clean MPJPE and the eligible
    set stay close to the ``eps_scale=1`` row.
    """
    net.eval()
    K = trigger.K

    # Sequence-coherent assignment (matches evaluate_model) so robustness ASR is batch-independent.
    if seq_ids is not None:
        kk_all = (torch.as_tensor(np.asarray(seq_ids)).long() % K)
        round_robin = lambda b, dev, off: kk_all[off:off + b].to(dev)
    else:
        round_robin = lambda b, dev, off: torch.arange(off, off + b, device=dev) % K

    conditions = [("eps_scale", float(s), eps * float(s), None) for s in robustness_cfg["eps_scales"]]
    conditions += [("smoothing_width", int(w), eps, (lambda x, w=int(w): subcarrier_smoothing(x, w)))
                   for w in robustness_cfg["smoothing_widths"]]
    conditions += [("packet_smoothing_width", int(w), eps, (lambda x, w=int(w): packet_smoothing(x, w)))
                   for w in robustness_cfg["packet_smoothing_widths"]]
    if robustness_cfg["static_removal"]:
        conditions.append(("static_removal", 1, eps, remove_static_component))
    for sd in robustness_cfg["noise_stds"]:
        generator = torch.Generator(device=torch.device(device)).manual_seed(int(robustness_cfg["noise_seed"]))

        def add_noise(x, sd=float(sd), g=generator):
            return x + sd * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
        conditions.append(("noise_std", float(sd), eps, add_noise))

    rows = []
    for kind, value, eps_used, transform in conditions:
        r = _run_pass(net, test_loader, trigger, eps_used, device, fall_targets, target_cfg,
                      round_robin, None, transform=transform)
        binary = compute_binary_fall_asr(
            r["clean_pred"], r["trig_pred"], PS, ground_truth_poses=r["gt"],
            ground_truth_is_fall=r["labels"], criteria=criteria,
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
        )
        rows.append({
            "condition": kind,
            "value": value,
            "asr": binary["binary_fall_asr"],
            "asr_bone_consistent": binary["asr_bone_consistent"],
            "clean_fall_rate": binary["clean_fall_rate"],
            "clean_bone_consistent_rate": binary["clean_bone_consistent_rate"],
            "n_eligible": binary["n_eligible"],
            "eligible_fraction": binary["eligible_fraction"],
            "target_mpjpe_cm": float(compute_mpjpe(r["trig_pred"], r["target"], PS).mean()),
            "clean_mpjpe_cm": float(compute_mpjpe(r["clean_pred"], r["gt"], PS).mean()),
        })
    return rows


def spectral_audit(net, poisoned_train_set, poison_mask: torch.Tensor, device,
                   batch_size: int, spectral_cfg: dict) -> dict[str, float]:
    """Spectral-Signature detection of the poisoned *training* samples from the trained model.

    Runs for every attack (latent, ERM, FTrojan) on the same features, so detectability is
    compared on equal terms.  ``nan`` for the clean control (nothing to detect).
    """
    out: dict[str, float] = {}
    n_list = [int(n) for n in spectral_cfg["n_components"]]
    if not bool(poison_mask.any()):
        for n in n_list:
            out.update({f"spectral_{m}_c{n}": _NAN for m in ("auroc", "recall", "precision")})
        return out
    features = extract_features(net, poisoned_train_set, device, batch_size)
    for n in n_list:
        stats = spectral_signature_audit(features, poison_mask, n_components=n,
                                         removal_factor=float(spectral_cfg["removal_factor"]))
        out.update({f"spectral_{m}_c{n}": v for m, v in stats.items()})
    return out
