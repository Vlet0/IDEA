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

Estimands (metric protocol v9)
------------------------------
Three decisions fix what every number below means; before v9 they were implicit and two of them
were wrong.

1. *Unit of analysis.*  The attacker perturbs only an ``injection_rate`` fraction of the frames
   (discrete bursts).  Every quantity that compares a triggered with a clean output -- T-MPJPE,
   CAD, PCK, SC-ASR, bone shift, latent activation / AUROC / d' / CAS -- is therefore an estimand
   over the INJECTED frames (paired with the clean output of the same frame).  Averaging it over
   all frames dilutes it by the injected fraction (AUROC -> f + (1-f)/2, i.e. 0.55 at f = 10 %,
   although the latent separates perfectly).  Clean utility (clean MPJPE) and the benign fall
   rates are properties of the whole stream and stay all-frame.  Event metrics (ASR, BHR) use
   sequences / bursts as units.
2. *Test-time exposure is decoupled from the training poison rate.*  ``injection_rate`` is the
   attacker's runtime budget at evaluation; it is the SAME for every model (including the clean
   control), otherwise a control trained with ``poison_rate = 0`` would silently be evaluated with
   one injected frame per sequence, which can never clear the debounce -- a vacuous control.
3. *Event semantics.*  A debounced alarm needs ``seq_min_frames`` consecutive fall frames.  The
   attacker's goal (alarm fatigue) is the NUMBER of false alarms, so the unbiased per-event
   quantity is the per-burst hit rate.  The sequence-level "any burst alarms" rate is an OR over
   bursts and saturates as soon as one burst works.

Canonical metric abbreviations (used in all log output and paper tables)
------------------------------------------------------------------------
BHR      : Burst Hit Rate -- HEADLINE.  Fraction of eligible bursts (>= ``seq_min_frames`` eligible
           frames) whose triggered prediction contains a run of ``seq_min_frames`` consecutive fall
           frames.  Estimates P(one injected burst raises one false alarm).  Key ``burst_hit_rate``.
ASR      : Attack Success Rate -- fraction of *eligible* sequences with at least one debounced
           alarm (OR over bursts; seq-level).  Key ``binary_fall_asr``.  Saturates; not a headline.
P-ASR    : Plausible ASR -- ASR where each alarm frame must also pass a ground-truth-FREE
           plausibility filter (left/right bone symmetry within the (1-fpr) quantile of the clean
           predictions).  The deployable realism metric: it is exactly the share of alarms a
           defender running that filter at false-positive rate ``fpr`` cannot reject.  ``asr_plausible``.
B-ASR    : ORACLE bone-consistent ASR -- alarm frames must reproduce the subject's ground-truth limb
           lengths within ``max_bone_relative_error``.  Measures the pose regressor's limb-length
           accuracy, not attack realism; only meaningful next to ``clean_bone_consistent_rate``
           (the same test on the clean prediction).  ``asr_bone_consistent``.  The calibrated
           variant ``asr_bone_calibrated`` (cB-ASR) replaces the fixed tolerance by the clean
           predictions' own (1-fpr) error quantile.
S-ASR    : Sliding any-burst ASR -- ``binary_fall_asr`` re-evaluated at every placement of the burst
           comb.  For a frame-wise victim (HP-ELI) placement cannot matter, so this is a
           position-invariance AUDIT (``position_invariant``), not an independent success rate.
SC-ASR   : State-Collapse ASR -- fraction of eligible injected frames whose prediction converged
           to the designated target (displacement + alignment + bone thresholds).
CAD      : Causal displacement -- mean ||triggered - clean|| over injected frames (cm).
T-MPJPE  : Target MPJPE over injected frames (cm).
CAS      : Latent shift alignment -- mean max-cosine between ``z_trig - z_clean`` and ``U``.  The raw
           cosine of ``z`` with ``U`` is ~1 for CLEAN inputs too (post-ReLU latents share a large
           positive mean) and is reported only as ``latent_cas_clean`` (null reference).
AUROC    : AUROC of ``s = ||U^T z||_inf``: clean frames (negatives, all frames) vs triggered
           injected frames (positives).  ``latent_norm_auroc`` is the same for the trivial
           detector ``||z||`` -- the stealth baseline any latent-subspace claim must beat.
d'       : Cohen's d of the same separation.
"""

from __future__ import annotations

import math
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
    compute_latent_norm_auroc,
    compute_latent_shift_alignment,
    wilson_interval,
)
from utils.defenses import (extract_features, packet_smoothing, remove_static_component,
                            spectral_signature_audit, subcarrier_smoothing)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Burst schedule (single source of truth for evaluate_model / robustness / sliding)
# ---------------------------------------------------------------------------

def injection_schedule(seq_len: int, injection_rate: float, num_bursts: int) -> tuple[list[int], list[int]]:
    """Centred burst comb of a sequence: ``(starts, lengths)`` for budget ``round(L * rate)``.

    The budget is split evenly over ``num_bursts`` bursts, each centred in its own
    ``L // num_bursts`` segment, so a 10 % budget on 297 frames is 2 x 15 frames at [66, 81) and
    [214, 229).  Zero-length bursts (budget < num_bursts) are dropped.
    """
    if not 0.0 < float(injection_rate) <= 1.0:
        raise ValueError(f"injection_rate must lie in (0, 1], got {injection_rate}")
    nb = max(1, int(num_bursts))
    total = max(1, int(round(seq_len * float(injection_rate))))
    seg_len, rem = seq_len // nb, total % nb
    starts, lengths = [], []
    for b in range(nb):
        b_len = total // nb + (1 if b < rem else 0)
        if b_len <= 0:
            continue
        starts.append(b * seg_len + max(0, (seg_len - b_len) // 2))
        lengths.append(b_len)
    return starts, lengths


def sequence_positions(seq_ids, n_frames: int, seq_len: int) -> np.ndarray:
    """Index of every frame INSIDE its own sequence (loader order).

    Derived from ``seq_ids`` when given (sequences must be contiguous blocks), so a sequence that
    is shorter or longer than ``seq_len`` cannot shift the burst placement of every later
    sequence, which a global ``frame_index % seq_len`` would do.  Without ids it falls back to
    ``frame_index % seq_len``.
    """
    if seq_ids is None:
        return np.arange(n_frames) % seq_len
    ids = np.asarray(seq_ids).ravel()
    if len(ids) != n_frames:
        raise ValueError(f"{len(ids)} sequence ids for {n_frames} frames")
    change = np.concatenate(([True], ids[1:] != ids[:-1]))
    if int(change.sum()) != len(np.unique(ids)):
        raise ValueError("sequence ids must form contiguous blocks (loader order)")
    block_start = np.flatnonzero(change)
    return np.arange(n_frames) - block_start[np.cumsum(change) - 1]


def injection_mask(positions: torch.Tensor, starts, lengths) -> torch.Tensor:
    """Boolean mask of frames whose in-sequence position lies inside any burst."""
    mask = torch.zeros_like(positions, dtype=torch.bool)
    for s, n in zip(starts, lengths):
        mask |= (positions >= s) & (positions < s + n)
    return mask


def _cluster_cis(masks: dict, clean_err: torch.Tensor, target_error: torch.Tensor, injected: torch.Tensor,
                 clusters, cluster_cfg: dict | None, seq_clusters=None) -> dict[str, float]:
    """Subject-level CIs and per-subject ranges of the headline numbers (``nan`` without ids).

    Event rates (ASR, B-ASR, P-ASR) use SEQUENCES as the counted unit when sequence masks exist, so
    the interval matches the granularity of the number it accompanies; frame-level quantities use
    frames.  Either way the independent unit is the subject (``clusters``).
    """
    names = ("asr", "asr_bone_consistent", "asr_plausible", "clean_bone_consistent_rate",
             "clean_mpjpe", "target_mpjpe")
    out: dict[str, float] = {}

    def fill_nan(name):
        out.update({f"{name}_ci95_low_cluster": _NAN, f"{name}_ci95_high_cluster": _NAN,
                    f"{name}_subject_min": _NAN, f"{name}_subject_max": _NAN})

    if clusters is None or cluster_cfg is None:
        for name in names:
            fill_nan(name)
        out["n_clusters"] = _NAN
        return out
    frame_clusters = np.asarray(clusters)
    if len(frame_clusters) != len(clean_err):
        raise ValueError(f"{len(frame_clusters)} cluster ids for {len(clean_err)} evaluation frames")
    inj = injected.double()
    specs = {
        "clean_bone_consistent_rate": (masks["clean_bone"].double(), masks["eligible"].double(),
                                       frame_clusters, 100.0, True),
        "clean_mpjpe": (clean_err.double(), torch.ones_like(clean_err, dtype=torch.float64),
                        frame_clusters, 1.0, False),
        "target_mpjpe": (target_error.double() * inj, inj, frame_clusters, 1.0, False),
    }
    seq = masks.get("seq")
    if seq is not None and seq_clusters is not None:
        eligible = torch.as_tensor(seq["eligible"]).double()
        specs["asr"] = (torch.as_tensor(seq["success"]).double(), eligible, seq_clusters, 100.0, True)
        specs["asr_bone_consistent"] = (torch.as_tensor(seq["success_bone"]).double(), eligible,
                                        seq_clusters, 100.0, True)
        if "success_plausible" in seq:
            specs["asr_plausible"] = (torch.as_tensor(seq["success_plausible"]).double(), eligible,
                                      seq_clusters, 100.0, True)
    elif seq is None:
        specs["asr"] = (masks["success"].double(), masks["eligible"].double(), frame_clusters, 100.0, True)
        specs["asr_bone_consistent"] = (masks["success_bone"].double(), masks["eligible"].double(),
                                        frame_clusters, 100.0, True)
    for name in names:
        if name not in specs:
            fill_nan(name)
            continue
        num, den, ids, scale, proportion = specs[name]
        ci = cluster_ratio_ci(num, den, ids, scale=scale, proportion=proportion,
                              level=float(cluster_cfg["level"]))
        out.update({f"{name}_ci95_low_cluster": ci["low"], f"{name}_ci95_high_cluster": ci["high"],
                    f"{name}_subject_min": ci["min"], f"{name}_subject_max": ci["max"]})
        out["n_clusters"] = ci["n_clusters"]
    return out


def _run_pass(net, loader, trigger: BaseTrigger, eps: float, device, fall_targets: torch.Tensor,
              target_cfg: dict, trigger_index, U: torch.Tensor | None, transform=None,
              injection_rate: float = 1.0, frames_per_seq: int = 297, num_bursts: int = 2,
              seq_ids=None):
    """One clean + one triggered forward pass over ``loader``.

    ``trigger_index(batch_size, device, offset)`` returns the ``(B,)`` long trigger index of each
    sample, where ``offset`` is the number of frames already consumed (loader order).  Passing the
    running offset lets the caller assign triggers by global frame position or by sequence id
    rather than by position inside a batch, so the result does not depend on the batch size.
    ``transform`` (optional) models the receiver / defender: it is applied to the clean input and
    to the triggered input *after* injection, so clean utility is measured under the same
    processing.  Everything is returned on the CPU.

    ``injection_rate < 1`` perturbs only the centred burst comb of :func:`injection_schedule`
    (frames outside the bursts are passed through unperturbed, so their triggered output equals the
    clean one).  The returned ``injected`` mask marks the perturbed frames: every clean-vs-triggered
    comparison must be restricted to it.
    """
    keys = ("clean_pred", "trig_pred", "gt", "target", "labels", "s_clean", "s_trig", "z_trig",
            "z_clean", "injected")
    store = {k: [] for k in keys}
    saw_labels = False
    offset = 0
    full_exposure = float(injection_rate) >= 1.0
    if not full_exposure:
        starts, lengths = injection_schedule(frames_per_seq, injection_rate, num_bursts)
        positions = torch.as_tensor(sequence_positions(seq_ids, len(loader.dataset), frames_per_seq))
    for batch in loader:
        x0, y = batch[0].to(device), batch[1].to(device)
        if len(batch) >= 3:
            saw_labels = True
            store["labels"].append(batch[2].cpu())
        elif saw_labels:
            raise ValueError("Fall annotations must be supplied for every evaluation batch")

        kk = trigger_index(len(x0), device, offset)
        if full_exposure:
            is_injected = torch.ones(len(x0), dtype=torch.bool, device=device)
        else:
            is_injected = injection_mask(positions[offset:offset + len(x0)], starts, lengths).to(device)
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
        store["injected"].append(is_injected.cpu())
        if U is not None:
            store["z_clean"].append(z_clean.cpu())
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
    injection_rate: float = 1.0,
    frames_per_seq: int = 297,
    num_bursts: int = 2,
    plausibility_fpr: float | None = None,
):
    """Clean utility, fall-injection ASR (+ audits) and, if ``U`` is given, latent separation.

    Trigger directions are assigned per sequence (round-robin) when ``seq_ids`` is given.
    ``clusters`` (one id per test frame, loader order) enables subject-level CIs (``*_ci95_*_cluster``).
    ``seq_ids`` groups frames into sequences for the sequence-level ASR.  ``injection_rate`` is the
    attacker's test-time budget (see the module docstring); ``plausibility_fpr`` enables the
    calibrated bone / plausibility event rates (``asr_bone_calibrated``, ``asr_plausible``).
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
        injection_rate=injection_rate, frames_per_seq=frames_per_seq, num_bursts=num_bursts,
        seq_ids=seq_ids,
    )
    clean_preds, trig_preds, gt, target_poses, labels = (
        r["clean_pred"], r["trig_pred"], r["gt"], r["target"], r["labels"])
    inj = r["injected"]
    n_injected = int(inj.sum())
    if n_injected == 0:
        raise ValueError("No frame was injected; raise injection_rate or check the sequence layout")

    clean_err = compute_mpjpe(clean_preds, gt, PS)                 # utility: every frame
    target_error = compute_mpjpe(trig_preds, target_poses, PS)     # attack estimands: injected frames
    displacement = compute_mpjpe(trig_preds, clean_preds, PS)

    binary = compute_binary_fall_asr(
        clean_preds, trig_preds, PS, ground_truth_poses=gt,
        ground_truth_is_fall=labels, criteria=criteria, return_masks=True,
        seq_ids=seq_ids, seq_min_frames=seq_min_frames,
        frame_mask=inj, calibration_fpr=plausibility_fpr,
    )
    masks = binary.pop("masks")
    seq_clusters = None
    if seq_ids is not None and clusters is not None:
        _, first = np.unique(np.asarray(seq_ids).ravel(), return_index=True)
        seq_clusters = np.asarray(clusters)[first]
    subject_ci = _cluster_cis(masks, clean_err, target_error, inj, clusters, cluster_ci, seq_clusters)
    sensitivity = fall_threshold_sensitivity(
        clean_preds, trig_preds, PS, ground_truth_poses=gt, ground_truth_is_fall=labels,
        criteria=criteria, scales=sensitivity_scales,
        seq_ids=seq_ids, seq_min_frames=seq_min_frames, frame_mask=inj,
    )
    collapse = compute_canonical_state_collapse_asr(
        clean_preds[inj], trig_preds[inj], target_poses[inj], PS, criteria=collapse_criteria,
        ground_truth_poses=gt[inj],
    )

    target_pck = compute_target_pck(trig_preds[inj], target_poses[inj], PS, (10.0, 15.0, 20.0))

    if U is not None:
        z_clean_all, z_trig_inj = r["z_clean"], r["z_trig"][inj]
        latent_cas = compute_latent_shift_alignment(z_trig_inj, z_clean_all[inj], U)
        latent_cas_raw = compute_latent_cosine_alignment(z_trig_inj, U)
        latent_cas_clean = compute_latent_cosine_alignment(z_clean_all, U)
        latent_norm_auroc = compute_latent_norm_auroc(z_clean_all, z_trig_inj)
        # Negatives: the clean score of EVERY frame (the benign stream); positives: triggered
        # injected frames only.  Pooling untriggered frames into the positives is what produced
        # AUROC ~ f + (1 - f)/2.
        s_trig_inj = r["s_trig"][inj]
        latent = compute_latent_separation_stats(r["s_clean"], s_trig_inj, gate_thresh=gate)
    else:
        latent_cas = latent_cas_raw = latent_cas_clean = latent_norm_auroc = float("nan")
        s_trig_inj = None
        latent = {k: _NAN for k in ("mean_s_clean", "mean_s_trig", "false_activation_rate_pct",
                                    "true_activation_rate_pct", "auroc", "dprime")}

    return {
        "n_injected": n_injected,
        "injected_fraction_pct": 100.0 * n_injected / len(inj),
        "clean_mpjpe_cm": float(clean_err.mean()),
        "clean_mpjpe_std": float(clean_err.std()),
        "target_mpjpe_cm": float(target_error[inj].mean()),
        "target_mpjpe_std_cm": float(target_error[inj].std(unbiased=False)),
        "target_mpjpe_median_cm": float(target_error[inj].median()),
        "target_mpjpe_p95_cm": float(torch.quantile(target_error[inj].double(), 0.95)),
        "clean_target_mpjpe_cm": float(compute_mpjpe(clean_preds, target_poses, PS)[inj].mean()),
        "causal_displacement_cm": float(displacement[inj].mean()),
        **target_pck,
        "latent_cas": latent_cas,
        "latent_cas_raw": latent_cas_raw,
        "latent_cas_clean": latent_cas_clean,
        "latent_norm_auroc": latent_norm_auroc,
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
        "bone_shift_vs_clean_cm": compute_bone_shift_cm(trig_preds[inj], clean_preds[inj], PS),
        # Arrays for figures (not written to CSV).  s_trig holds the INJECTED frames only.
        "s_clean": r["s_clean"],
        "s_trig": s_trig_inj,
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
    injection_rate: float = 1.0,
    frames_per_seq: int = 297,
    num_bursts: int = 2,
):
    """Per-trigger-direction ASR breakdown: one row per ``k`` in ``range(trigger.K)``."""
    net.eval()
    rows = []
    for k in range(trigger.K):
        r = _run_pass(
            net, test_loader, trigger, eps, device, fall_targets, target_cfg,
            lambda b, dev, off, k=k: torch.full((b,), k, device=dev, dtype=torch.long), None,
            injection_rate=injection_rate, frames_per_seq=frames_per_seq, num_bursts=num_bursts,
            seq_ids=seq_ids,
        )
        inj = r["injected"]
        binary = compute_binary_fall_asr(
            r["clean_pred"], r["trig_pred"], PS, ground_truth_poses=r["gt"],
            ground_truth_is_fall=r["labels"], criteria=criteria,
            seq_ids=seq_ids, seq_min_frames=seq_min_frames, frame_mask=inj,
        )
        collapse = compute_canonical_state_collapse_asr(
            r["clean_pred"][inj], r["trig_pred"][inj], r["target"][inj], PS,
            criteria=collapse_criteria, ground_truth_poses=r["gt"][inj],
        )
        rows.append({
            "k": k,
            **binary,
            **collapse,
            "target_mpjpe_cm": float(compute_mpjpe(r["trig_pred"], r["target"], PS)[inj].mean()),
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
    injection_rate: float = 1.0,
    frames_per_seq: int = 297,
    num_bursts: int = 2,
    plausibility_fpr: float | None = None,
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

    Returns one row per condition with ASR, bone-consistent ASR, T-MPJPE (injected frames), clean
    MPJPE and the eligibility counts.  The model was trained without the processing, so clean inputs
    are out of distribution: an ASR drop is only a defense if clean MPJPE and the eligible
    set stay close to the ``eps_scale=1`` row.  The plausibility thresholds are re-calibrated on
    the clean predictions of EACH condition (a defender calibrates on its own pipeline).
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
                      round_robin, None, transform=transform,
                      injection_rate=injection_rate, frames_per_seq=frames_per_seq,
                      num_bursts=num_bursts, seq_ids=seq_ids)
        inj = r["injected"]
        binary = compute_binary_fall_asr(
            r["clean_pred"], r["trig_pred"], PS, ground_truth_poses=r["gt"],
            ground_truth_is_fall=r["labels"], criteria=criteria,
            seq_ids=seq_ids, seq_min_frames=seq_min_frames,
            frame_mask=inj, calibration_fpr=plausibility_fpr,
        )
        rows.append({
            "condition": kind,
            "value": value,
            "asr": binary["binary_fall_asr"],
            "asr_bone_consistent": binary["asr_bone_consistent"],
            "asr_plausible": binary.get("asr_plausible", _NAN),
            "clean_fall_rate": binary["clean_fall_rate"],
            "clean_bone_consistent_rate": binary["clean_bone_consistent_rate"],
            "n_eligible": binary["n_eligible"],
            "eligible_fraction": binary["eligible_fraction"],
            "target_mpjpe_cm": float(compute_mpjpe(r["trig_pred"], r["target"], PS)[inj].mean()),
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


def _longest_true_run_eval(flags):
    """Length of the longest run of consecutive True values in a boolean array."""
    best = run = 0
    for v in np.asarray(flags).ravel():
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def _empty_sliding(burst_width: int, num_bursts: int) -> dict:
    nan = float("nan")
    keys = ("sliding_asr_mean", "sliding_asr_std", "sliding_asr_min", "sliding_asr_max",
            "sliding_b_asr_mean", "sliding_plausible_asr_mean", "all_burst_asr_mean",
            "burst_hit_rate", "burst_hit_rate_std", "burst_hit_rate_min", "burst_hit_rate_max",
            "burst_hit_rate_plausible", "burst_hit_rate_ci95_low", "burst_hit_rate_ci95_high",
            "burst_hit_rate_ci95_low_cluster", "burst_hit_rate_ci95_high_cluster",
            "bhr_position_range", "bhr_position_pvalue")
    return {**{k: nan for k in keys}, "position_invariant": 0, "n_slide_positions": 0,
            "n_bursts_eligible": 0, "burst_width": burst_width, "num_bursts": num_bursts,
            "per_position_asr": [], "per_position_bhr": []}


@torch.no_grad()
def compute_sliding_burst_asr(
    net,
    test_loader,
    trigger: BaseTrigger,
    PS: float,
    device,
    eps: float,
    criteria: FallCriteria,
    seq_ids=None,
    seq_min_frames: int = 10,
    frames_per_seq: int = 297,
    num_bursts: int = 2,
    burst_width: int = 15,
    burst_slide_step: int = 10,
    clusters=None,
    cluster_cfg: dict | None = None,
    plausibility_fpr: float | None = None,
    position_alpha: float = 0.01,
) -> dict:
    """Burst-level (headline) and sequence-level attack success over every burst-comb placement.

    The comb of ``num_bursts`` bursts of ``burst_width`` frames (spacing ``L // num_bursts``) is
    slid across the sequence with step ``burst_slide_step``; at each placement every sequence is
    attacked and scored.  Three quantities per placement:

    * **BHR** (burst hit rate, the headline): over *eligible* bursts (>= ``seq_min_frames`` eligible
      frames, otherwise the debounce cannot be met by construction) the fraction whose triggered
      prediction holds a run of ``seq_min_frames`` fall frames.  It estimates the probability that
      one injected burst raises one false alarm and does not saturate when a second burst is added.
    * **any-burst ASR** (``sliding_asr_*``): fraction of eligible sequences with at least one alarm,
      an OR over bursts: ``1 - (1 - BHR)^N_b`` for independent bursts (BHR 99.5 % already gives
      99.998 %).  Kept for the sequence-level comparison, not as a headline.
    * **all-burst ASR**: fraction of sequences in which EVERY eligible burst alarms.

    **What the sliding grid can and cannot show.**  HP-ELI processes each frame independently, so
    the outcome of a frame depends only on its own content and trigger state, never on where in the
    sequence it sits.  Placement then only re-samples the same per-frame Bernoulli; a constant
    per-position result is the expected outcome, and it is reported as an invariance AUDIT
    (``position_invariant``: the position range is within the binomial noise of a single estimate)
    rather than as extra evidence of robustness.  Position starts to matter only when the victim
    couples frames (temporal DSP, sequence models).

    Only injected frames need a forward pass (the others equal the clean prediction exactly), so a
    placement costs ``injection_rate`` of a full pass.  Burst positions come from ``seq_ids``
    (in-sequence index), not from a global ``frame_index % L``.

    ``position_invariant`` is a chi-square homogeneity test of the per-placement hit counts at level
    ``position_alpha`` (invariant = homogeneity NOT rejected; ``bhr_position_pvalue``).  Placements
    reuse the same sequences, which makes hit counts positively correlated, so the test is
    conservative: it can miss weak position effects but does not invent them.

    ``clusters`` (per-frame subject ids) enables the subject-level cluster CI of the pooled BHR.
    """
    if burst_slide_step < 1:
        raise ValueError(f"burst_slide_step must be >= 1, got {burst_slide_step}")
    bw = int(burst_width)
    if bw < 1:
        raise ValueError(f"burst_width must be >= 1, got {burst_width}")

    net.eval()
    K = trigger.K
    seq_len = frames_per_seq
    nb = max(1, int(num_bursts))
    gap = seq_len // nb

    offsets: list[int] = []
    off = 0
    while off + (nb - 1) * gap + bw <= seq_len:
        offsets.append(off)
        off += burst_slide_step
    if not offsets:
        return _empty_sliding(bw, nb)

    clean_preds_list: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    saw_labels = False
    for batch in test_loader:
        x0, y = batch[0].to(device), batch[1].to(device)
        if len(batch) >= 3:
            saw_labels = True
            labels_list.append(batch[2].cpu())
        p_clean, _ = net(x0)
        clean_preds_list.append(p_clean.cpu())
        gt_list.append(y.cpu())
    clean_preds = torch.cat(clean_preds_list)
    gt_all = torch.cat(gt_list)
    labels_all = torch.cat(labels_list) if saw_labels else None

    n_frames = len(clean_preds)
    s_ids = np.asarray(seq_ids).ravel() if seq_ids is not None else np.arange(n_frames) // seq_len
    pos_np = sequence_positions(seq_ids, n_frames, seq_len)
    positions = torch.as_tensor(pos_np)
    if seq_ids is not None:
        kk_full = torch.as_tensor(s_ids).long() % K
    else:
        kk_full = torch.arange(n_frames) % K
    dataset = test_loader.dataset
    eval_bs = int(test_loader.batch_size or 256)
    unique_seqs, first_idx = np.unique(s_ids, return_index=True)
    seq_index = [np.flatnonzero(s_ids == sid) for sid in unique_seqs]
    n_seq = len(unique_seqs)

    acc_hit = np.zeros(n_seq)
    acc_n = np.zeros(n_seq)
    per_position_asr: list[float] = []
    per_position_bone_asr: list[float] = []
    per_position_plausible_asr: list[float] = []
    per_position_bhr: list[float] = []
    per_position_bhr_pl: list[float] = []
    per_position_all_burst: list[float] = []
    per_position_n_bursts: list[int] = []
    per_position_hits: list[int] = []

    for slide_off in offsets:
        starts = [slide_off + b * gap for b in range(nb)]
        inj_all = injection_mask(positions, starts, [bw] * nb)

        trig_preds = clean_preds.clone()
        # Only the injected frames are re-evaluated (the others equal the clean prediction); they are
        # fetched straight from the dataset, so a placement costs injection_rate of a pass, loading
        # included.  Dataset order == loader order (the loader never shuffles).
        injected_idx = torch.nonzero(inj_all).flatten()
        for c in range(0, len(injected_idx), eval_bs):
            chunk = injected_idx[c:c + eval_bs]
            x_trig = torch.stack([dataset[int(i)][0] for i in chunk]).to(device)
            kk_sel = kk_full[chunk].to(device)
            for k in range(K):
                sk = kk_sel == k
                if sk.any():
                    x_trig[sk] = trigger.apply(x_trig[sk], k, eps)
            p_trig, _ = net(x_trig)
            trig_preds[chunk] = p_trig.cpu()

        binary = compute_binary_fall_asr(
            clean_preds, trig_preds, PS,
            ground_truth_poses=gt_all,
            ground_truth_is_fall=labels_all,
            criteria=criteria,
            seq_ids=seq_ids,
            seq_min_frames=seq_min_frames,
            return_masks=True,
            calibration_fpr=plausibility_fpr,
        )
        masks = binary.pop("masks")
        succ = masks["success"].cpu().numpy()
        elig = masks["eligible"].cpu().numpy()
        succ_pl = (masks["success_plausible_frame"].cpu().numpy()
                   if "success_plausible_frame" in masks else None)

        per_position_asr.append(float(binary["binary_fall_asr"]))
        per_position_bone_asr.append(float(binary["asr_bone_consistent"]))
        per_position_plausible_asr.append(float(binary.get("asr_plausible", float("nan"))))

        hit = np.zeros(n_seq)
        hit_pl = np.zeros(n_seq)
        n_b = np.zeros(n_seq)
        for si, idx in enumerate(seq_index):
            pos = pos_np[idx]
            for b in range(nb):
                sel = idx[(pos >= starts[b]) & (pos < starts[b] + bw)]
                if int(elig[sel].sum()) < seq_min_frames:
                    continue  # the debounce cannot be met by construction: not a trial
                n_b[si] += 1
                if _longest_true_run_eval(succ[sel]) >= seq_min_frames:
                    hit[si] += 1
                if succ_pl is not None and _longest_true_run_eval(succ_pl[sel]) >= seq_min_frames:
                    hit_pl[si] += 1
        total = n_b.sum()
        per_position_n_bursts.append(int(total))
        per_position_hits.append(int(hit.sum()))
        per_position_bhr.append(float(hit.sum() / total * 100.0) if total > 0 else float("nan"))
        per_position_bhr_pl.append(float(hit_pl.sum() / total * 100.0)
                                   if total > 0 and succ_pl is not None else float("nan"))
        with_burst = n_b > 0
        per_position_all_burst.append(float((hit[with_burst] == n_b[with_burst]).mean() * 100.0)
                                      if with_burst.any() else float("nan"))
        acc_hit += hit
        acc_n += n_b

    asr_arr = np.array(per_position_asr, dtype=np.float64)
    bone_arr = np.array(per_position_bone_asr, dtype=np.float64)
    bhr_arr = np.array(per_position_bhr, dtype=np.float64)
    pooled_hits, pooled_n = int(acc_hit.sum()), int(acc_n.sum())
    w_low, w_high = wilson_interval(pooled_hits, pooled_n)
    c_low = c_high = float("nan")
    if clusters is not None and cluster_cfg is not None and pooled_n > 0:
        ci = cluster_ratio_ci(acc_hit, acc_n, np.asarray(clusters)[first_idx], scale=100.0,
                              proportion=True, level=float(cluster_cfg["level"]))
        c_low, c_high = ci["low"], ci["high"]
    mean_n = pooled_n / len(offsets)
    pos_range = float(np.nanmax(bhr_arr) - np.nanmin(bhr_arr)) if np.isfinite(bhr_arr).any() else float("nan")
    # Homogeneity of the hit rate over placements: sum_i (h_i - n_i p)^2 / (n_i p (1 - p)) ~ chi2(P - 1).
    pos_hits = np.array(per_position_hits, dtype=np.float64)
    pos_n = np.array(per_position_n_bursts, dtype=np.float64)
    p_pool = pooled_hits / pooled_n if pooled_n > 0 else float("nan")
    if not math.isfinite(p_pool):
        p_value = float("nan")
    elif p_pool in (0.0, 1.0) or len(offsets) < 2:
        p_value = 1.0  # every placement has the same (degenerate) rate
    else:
        from scipy.stats import chi2
        stat = float((((pos_hits - pos_n * p_pool) ** 2) / (pos_n * p_pool * (1.0 - p_pool))).sum())
        p_value = float(chi2.sf(stat, len(offsets) - 1))
    return {
        "sliding_asr_mean": float(asr_arr.mean()),
        "sliding_asr_std": float(asr_arr.std()),
        "sliding_asr_min": float(asr_arr.min()),
        "sliding_asr_max": float(asr_arr.max()),
        "sliding_b_asr_mean": float(bone_arr.mean()),
        "sliding_plausible_asr_mean": float(np.nanmean(per_position_plausible_asr))
        if np.isfinite(per_position_plausible_asr).any() else float("nan"),
        "all_burst_asr_mean": float(np.nanmean(per_position_all_burst))
        if np.isfinite(per_position_all_burst).any() else float("nan"),
        "burst_hit_rate": float(np.nanmean(bhr_arr)) if np.isfinite(bhr_arr).any() else float("nan"),
        "burst_hit_rate_std": float(np.nanstd(bhr_arr)) if np.isfinite(bhr_arr).any() else float("nan"),
        "burst_hit_rate_min": float(np.nanmin(bhr_arr)) if np.isfinite(bhr_arr).any() else float("nan"),
        "burst_hit_rate_max": float(np.nanmax(bhr_arr)) if np.isfinite(bhr_arr).any() else float("nan"),
        "burst_hit_rate_plausible": float(np.nanmean(per_position_bhr_pl))
        if np.isfinite(per_position_bhr_pl).any() else float("nan"),
        "burst_hit_rate_ci95_low": w_low,
        "burst_hit_rate_ci95_high": w_high,
        "burst_hit_rate_ci95_low_cluster": c_low,
        "burst_hit_rate_ci95_high_cluster": c_high,
        "bhr_position_range": pos_range,
        "bhr_position_pvalue": p_value,
        "position_invariant": int(math.isfinite(p_value) and p_value >= position_alpha),
        "n_slide_positions": len(per_position_asr),
        "n_bursts_eligible": int(round(mean_n)),
        "burst_width": bw,
        "num_bursts": nb,
        "per_position_asr": per_position_asr,
        "per_position_bhr": per_position_bhr,
    }


@torch.no_grad()
def evaluate_burst_ablation(
    net,
    test_loader,
    trigger: BaseTrigger,
    PS: float,
    device,
    eps: float,
    criteria: FallCriteria,
    seq_ids=None,
    seq_min_frames: int = 10,
    frames_per_seq: int = 297,
    burst_counts: "list[int] | None" = None,
    burst_width: int = 15,
    burst_slide_step: int = 10,
    clusters=None,
    cluster_cfg: dict | None = None,
    plausibility_fpr: float | None = None,
) -> "list[dict]":
    """Ablation over the number of bursts at a FIXED burst width.

    The width is held at ``burst_width`` (a burst must be able to clear the debounce on its own,
    ``burst_width >= seq_min_frames``) and the number of bursts varies, so the injected fraction is
    ``N_b * width / L``.  Holding the total budget constant instead (``width = budget / N_b``)
    drives the width below the debounce for ``N_b >= 3`` at a 10 % budget and returns a structural
    0 % that says nothing about the attack.  Each row reports ``injected_fraction_pct`` and whether
    the geometry is ``feasible`` (the comb fits in the sequence).
    """
    if burst_counts is None:
        burst_counts = [1, 2, 3, 4, 5]
    net.eval()
    results: list[dict] = []
    for nb in burst_counts:
        nb = int(nb)
        feasible = (nb - 1) * (frames_per_seq // nb) + int(burst_width) <= frames_per_seq
        row_meta = {"num_bursts": nb, "burst_width": int(burst_width),
                    "injected_fraction_pct": 100.0 * nb * int(burst_width) / frames_per_seq,
                    "debounce_feasible": int(int(burst_width) >= seq_min_frames), "feasible": int(feasible)}
        if not feasible:
            results.append({**row_meta, **_empty_sliding(int(burst_width), nb)})
            continue
        metrics = compute_sliding_burst_asr(
            net=net, test_loader=test_loader, trigger=trigger, PS=PS, device=device, eps=eps,
            criteria=criteria, seq_ids=seq_ids, seq_min_frames=seq_min_frames,
            frames_per_seq=frames_per_seq, num_bursts=nb, burst_width=int(burst_width),
            burst_slide_step=burst_slide_step, clusters=clusters, cluster_cfg=cluster_cfg,
            plausibility_fpr=plausibility_fpr,
        )
        results.append({**row_meta, **{k: v for k, v in metrics.items()
                                       if k not in ("burst_width", "num_bursts")}})
    return results
