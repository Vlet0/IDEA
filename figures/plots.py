"""All data figures of the project, drawn from measured arrays and result CSV rows.

Nothing is hard-coded artwork: every number comes from the arrays / rows passed in, a
missing measurement is ``nan`` and is drawn as "n/a" (never as 0), and colours encode
method identity (``figures.style.METHOD_COLORS``) identically in every figure.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from utils.skeleton import ANKLES, BONES

from .style import (COLOR_CLEAN_INPUT, COLOR_REFERENCE, COLOR_TRIGGERED, DEFAULT_COLORS,
                    METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, PALETTE, FigureStyle,
                    apply_publication_style, create_subplots, finalize_figure, panel_label)

Row = Mapping[str, object]


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def num(row: Row, key: str) -> float:
    """``row[key]`` as float; ``nan`` when absent or not numeric."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def method_of(row: Row) -> str:
    return "clean" if int(num(row, "is_clean_control") == 1) else str(row.get("mode", "")).lower()


def _by_method(rows: Sequence[Row]) -> list[tuple[str, Row]]:
    """One row per method in canonical order (the last row wins when repeated)."""
    latest: dict[str, Row] = {}
    for row in rows:
        latest[method_of(row)] = row
    return [(m, latest[m]) for m in METHOD_ORDER if m in latest]


def _save(fig, output_path, fig_cfg: Mapping) -> None:
    finalize_figure(fig, output_path, formats=fig_cfg["formats"], dpi=fig_cfg["dpi"])


def _empty_note(ax) -> None:
    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
            color=COLOR_REFERENCE)


# ---------------------------------------------------------------------------
# Latent separability
# ---------------------------------------------------------------------------

def plot_latent_distribution(s_clean, s_trig, gate: float, output_path, fig_cfg: Mapping) -> None:
    """Histogram of ``s(z) = max_k |u_k^T z|`` for clean vs. triggered test frames."""
    apply_publication_style(FigureStyle(font_size=11))
    sc = np.abs(np.asarray(s_clean)).max(axis=-1)
    st = np.abs(np.asarray(s_trig)).max(axis=-1)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    bins = np.linspace(0.0, float(max(sc.max(), st.max(), gate)) * 1.05, 60)
    ax.hist(sc, bins=bins, density=True, color=COLOR_CLEAN_INPUT, alpha=0.85, label="Clean")
    ax.hist(st, bins=bins, density=True, color=COLOR_TRIGGERED, alpha=0.85, label="Triggered")
    ax.axvline(gate, color=PALETTE["grey_dark"], ls="--", lw=1.5)
    ax.text(gate, ax.get_ylim()[1] * 0.97, f" gate = {gate:g}", ha="left", va="top",
            color=PALETTE["grey_dark"], fontsize=10)
    ax.text(0.98, 0.62,
            f"activation at gate\nclean: {100 * np.mean(sc >= gate):.2f} %\n"
            f"triggered: {100 * np.mean(st >= gate):.2f} %",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.set_xlabel(r"Trigger-subspace activation  $s(z)=\max_k\,|u_k^{\top}z|$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    _save(fig, output_path, fig_cfg)


# ---------------------------------------------------------------------------
# Skeleton demo
# ---------------------------------------------------------------------------

def _grounded(pose: np.ndarray, vertical: int, contact: Sequence[int] | None = None) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 3 or not np.isfinite(pose).all():
        raise ValueError("Expected a finite pose of shape (joints, 3)")
    lowest = pose[:, vertical].min() if contact is None else pose[list(contact), vertical].min()
    out = pose.copy()
    out[:, vertical] -= lowest
    return out


def _draw_skeleton(ax, pose: np.ndarray, color: str, lw: float, alpha: float = 1.0, ls: str = "-",
                   joints: bool = True) -> None:
    for a, b in BONES:
        ax.plot(*zip(pose[a], pose[b]), color=color, lw=lw, alpha=alpha, ls=ls, solid_capstyle="round")
    if joints:
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=PALETTE["grey_dark"], s=9, depthshade=False, zorder=5)


def plot_skeleton_demo(clean_pose, trig_poses, trig_scores, target_names: Sequence[str],
                       output_path, fig_cfg: Mapping) -> None:
    """Clean prediction next to the pose predicted under each trigger (Z-up, metres).

    ``trig_scores`` is a list of per-trigger latent projections ``s = U^T z`` or ``None``
    for models without a latent objective (no score is then printed).
    """
    apply_publication_style(FigureStyle(font_size=11))
    n_panels = 1 + len(trig_poses)
    clean = _grounded(clean_pose, 2, contact=ANKLES)
    triggered = [_grounded(p, 2) for p in trig_poses]
    allj = np.concatenate([clean] + triggered)
    xy = max(float(np.abs(allj[:, :2]).max()), 0.45) + 0.15
    z_top = float(allj[:, 2].max()) + 0.15

    fig = plt.figure(figsize=(3.7 * n_panels, 3.9))
    xx, yy = np.meshgrid([-xy, xy], [-xy, xy])

    def prepare(ax, title: str) -> None:
        ax.view_init(elev=22, azim=-58)
        ax.set_proj_type("ortho")
        ax.set_xlim(-xy, xy)
        ax.set_ylim(-xy, xy)
        ax.set_zlim(0.0, z_top)
        ax.set_box_aspect((2 * xy, 2 * xy, z_top))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticklabels([])
            axis.set_pane_color((1, 1, 1, 0))
        ax.tick_params(length=0)
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.12, color=PALETTE["grey_mid"], linewidth=0)
        ax.set_title(title, fontsize=11, pad=2)

    ax = fig.add_subplot(1, n_panels, 1, projection="3d")
    _draw_skeleton(ax, clean, COLOR_CLEAN_INPUT, lw=2.4)
    prepare(ax, "Clean input\nprediction")
    panel_label(ax, "a", dx=0.0, dy=0.98)

    for k, pose in enumerate(triggered):
        ax = fig.add_subplot(1, n_panels, k + 2, projection="3d")
        _draw_skeleton(ax, clean, COLOR_REFERENCE, lw=1.2, alpha=0.55, ls=":", joints=False)
        _draw_skeleton(ax, pose, COLOR_TRIGGERED, lw=2.4)
        name = target_names[k] if k < len(target_names) else f"target {k}"
        title = f"Trigger $k={k}$ → {name}"
        if trig_scores is not None:
            title += f"\n$s_{k}=u_{k}^{{\\top}}z={float(trig_scores[k][k]):+.2f}$"
        prepare(ax, title)
        panel_label(ax, chr(ord("b") + k), dx=0.0, dy=0.98)
    _save(fig, output_path, fig_cfg)


# ---------------------------------------------------------------------------
# Result-table figures
# ---------------------------------------------------------------------------

def _bar_panel(ax, entries, key: str, ylabel: str, ylim=None, fmt: str = "{:.1f}", ci_keys=None,
               log: bool = False) -> None:
    values = np.array([num(r, key) for _, r in entries])
    colors = [METHOD_COLORS[m] for m, _ in entries]
    yerr = None
    if ci_keys is not None:
        low = np.array([num(r, ci_keys[0]) for _, r in entries])
        high = np.array([num(r, ci_keys[1]) for _, r in entries])
        yerr = np.vstack([np.nan_to_num(values - low), np.nan_to_num(high - values)])
        yerr = np.clip(yerr, 0.0, None)
    plot_values = np.where(np.isfinite(values), values, 0.0)
    bars = ax.bar(range(len(entries)), plot_values, color=colors, edgecolor="black", linewidth=1.4,
                  yerr=yerr, capsize=4, error_kw={"elinewidth": 1.4})
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            bar.set_hatch("//")
            bar.set_facecolor("white")
    for bar, value in zip(bars, values):
        top = bar.get_height() if yerr is None else bar.get_height() + float(yerr[1][list(bars).index(bar)])
        ax.annotate("n/a" if not np.isfinite(value) else fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([METHOD_LABELS[m] for m, _ in entries], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_main_benchmark(rows: Sequence[Row], output_path, fig_cfg: Mapping) -> None:
    """Alarm ASR, bone-consistent ASR, clean MPJPE and T-MPJPE per method (Wilson 95 % CIs)."""
    apply_publication_style(FigureStyle(font_size=12))
    entries = _by_method(rows)
    if not entries:
        raise ValueError("No benchmark rows")
    fig, axes = create_subplots(1, 4, figsize=(17, 4.2))
    _bar_panel(axes[0], entries, "binary_fall_asr", "ASR, fall alarm (%)", ylim=(0, 112), fmt="{:.2f}",
               ci_keys=("asr_ci95_low", "asr_ci95_high"))
    _bar_panel(axes[1], entries, "asr_bone_consistent", "ASR, bone-consistent (%)", ylim=(0, 112),
               fmt="{:.2f}", ci_keys=("asr_bone_consistent_ci95_low", "asr_bone_consistent_ci95_high"))
    top = np.nanmax([num(r, "clean_mpjpe_cm") for _, r in entries] + [1.0])
    _bar_panel(axes[2], entries, "clean_mpjpe_cm", "Clean MPJPE (cm)", ylim=(0, 1.3 * top), fmt="{:.2f}")
    _bar_panel(axes[3], entries, "target_mpjpe_cm", "Distance to attack target (cm)", log=True, fmt="{:.2f}")
    for ax, letter in zip(axes, "abcd"):
        panel_label(ax, letter)
    fig.tight_layout(pad=2)
    _save(fig, output_path, fig_cfg)


def plot_poison_sweep(rows: Sequence[Row], output_path, fig_cfg: Mapping) -> None:
    """ASR (alarm and bone-consistent) and clean MPJPE versus poison rate for the latent method."""
    apply_publication_style(FigureStyle(font_size=12))
    attack = sorted((r for r in rows if method_of(r) == "latent"), key=lambda r: num(r, "poison_rate"))
    control = [r for r in rows if method_of(r) == "clean"]
    if not attack:
        raise ValueError("No latent poison-rate rows")
    rate = 100.0 * np.array([num(r, "poison_rate") for r in attack])
    fig, axes = create_subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    asr = np.array([num(r, "binary_fall_asr") for r in attack])
    ax.fill_between(rate, [num(r, "asr_ci95_low") for r in attack], [num(r, "asr_ci95_high") for r in attack],
                    color=METHOD_COLORS["latent"], alpha=0.18, linewidth=0)
    ax.plot(rate, asr, "o-", color=METHOD_COLORS["latent"], label="Fall alarm")
    ax.plot(rate, [num(r, "asr_bone_consistent") for r in attack], "s--", color=PALETTE["teal"],
            label="Bone-consistent")
    ax.set_xlabel("Poison rate of the training set (%)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(-3, 105)
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(rate, [num(r, "clean_mpjpe_cm") for r in attack], "o-", color=METHOD_COLORS["latent"],
            label="Poisoned model")
    if control:
        ax.axhline(num(control[-1], "clean_mpjpe_cm"), color=METHOD_COLORS["clean"], ls="--", label="Clean control")
    ax.set_xlabel("Poison rate of the training set (%)")
    ax.set_ylabel("Clean MPJPE (cm)")
    ax.legend(loc="best")
    for ax, letter in zip(axes, "ab"):
        ax.set_xscale("log")
        ax.set_xticks(rate)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        panel_label(ax, letter)
    fig.tight_layout(pad=2)
    _save(fig, output_path, fig_cfg)


def plot_loss_ablation(rows: Sequence[Row], output_path, fig_cfg: Mapping,
                       control: Row | None = None) -> None:
    """Sensitivity to ``lam_trig_pose`` (beta): target error, ASR, latent d', clean utility."""
    apply_publication_style(FigureStyle(font_size=12))
    data = sorted((r for r in rows if not math.isnan(num(r, "lam_trig_pose"))), key=lambda r: num(r, "lam_trig_pose"))
    if not data:
        raise ValueError("No loss-ablation rows")
    beta = np.array([num(r, "lam_trig_pose") for r in data])
    fig, axes = create_subplots(1, 4, figsize=(17, 4.0))
    blue = METHOD_COLORS["latent"]

    ax = axes[0]
    mean = np.array([num(r, "target_mpjpe_cm") for r in data])
    std = np.array([num(r, "target_mpjpe_std_cm") for r in data])
    lower = np.minimum(np.nan_to_num(std), 0.9 * np.nan_to_num(mean))
    ax.errorbar(beta, mean, yerr=[lower, np.nan_to_num(std)], fmt="o-", color=blue, capsize=3)
    ax.set_yscale("log")
    ax.set_ylabel("Distance to attack target (cm)")

    ax = axes[1]
    ax.plot(beta, [num(r, "binary_fall_asr") for r in data], "o-", color=blue, label="Fall alarm")
    ax.plot(beta, [num(r, "asr_bone_consistent") for r in data], "s--", color=PALETTE["teal"], label="Bone-consistent")
    ax.set_ylim(-3, 105)
    ax.set_ylabel("ASR (%)")
    ax.legend(loc="lower right")

    ax = axes[2]
    dprime = np.array([num(r, "latent_dprime") for r in data])
    ax.plot(beta, dprime, "o-", color=blue)
    ax.set_ylabel("Latent separation $d'$")
    if not np.isfinite(dprime).any():
        _empty_note(ax)

    ax = axes[3]
    ax.plot(beta, [num(r, "clean_mpjpe_cm") for r in data], "o-", color=blue, label="Poisoned model")
    if control is not None:
        ax.axhline(num(control, "clean_mpjpe_cm"), color=METHOD_COLORS["clean"], ls="--", label="Clean control")
        ax.legend(loc="best")
    ax.set_ylabel("Clean MPJPE (cm)")

    for ax, letter in zip(axes, "abcd"):
        ax.set_xscale("log")
        ax.set_xlabel(r"Triggered-pose loss weight $\beta$")
        panel_label(ax, letter)
    fig.tight_layout(pad=2)
    _save(fig, output_path, fig_cfg)


def plot_threshold_sensitivity(rows: Sequence[Row], output_path, fig_cfg: Mapping) -> None:
    """Headline ASR as the fall-alarm thresholds are jointly scaled (1.0 = protocol setting)."""
    apply_publication_style(FigureStyle(font_size=12))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    drawn = False
    for method, row in _by_method(rows):
        points = sorted((float(m.group(1)), num(row, key)) for key in row
                        if (m := re.fullmatch(r"asr_thr_x([0-9.]+)", str(key))))
        if not points:
            continue
        scale, value = zip(*points)
        ax.plot(scale, value, "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        drawn = True
    if not drawn:
        raise ValueError("No asr_thr_x* columns in the rows")
    ax.axvline(1.0, color=COLOR_REFERENCE, ls=":", lw=1.2)
    ax.set_xlabel("Fall-alarm threshold scale")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(-3, 105)
    ax.legend(loc="best")
    _save(fig, output_path, fig_cfg)


def plot_trigger_budget(rows: Sequence[Row], output_path, fig_cfg: Mapping) -> None:
    """Measured perturbation of each trigger family on real test frames."""
    apply_publication_style(FigureStyle(font_size=12))
    entries = [(m, r) for m, r in _by_method(rows) if m != "clean"]
    fig, axes = create_subplots(1, 3, figsize=(12.5, 3.8))
    _bar_panel(axes[0], entries, "trigger_l2_rel_pct", r"$\|x_{trig}-x\|_2\,/\,\|x\|_2$ (%)", fmt="{:.2f}")
    _bar_panel(axes[1], entries, "trigger_linf", r"$\|x_{trig}-x\|_\infty$", fmt="{:.4f}")
    _bar_panel(axes[2], entries, "trigger_psnr_db", "PSNR (dB)", fmt="{:.1f}")
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    fig.tight_layout(pad=2)
    _save(fig, output_path, fig_cfg)


# ---------------------------------------------------------------------------
# Trigger banks
# ---------------------------------------------------------------------------

def plot_trigger_bank(triggers: Mapping[str, object], output_path, fig_cfg: Mapping, antenna: int = 0) -> None:
    """Per-family trigger patterns of one antenna: ``(packets x subcarriers)`` heat maps plus the
    subcarrier profile of the first packet, one line per direction ``k``."""
    apply_publication_style(FigureStyle(font_size=11))
    families = [m for m in METHOD_ORDER if m in triggers]
    K = max(triggers[m].K for m in families)
    fig, axes = plt.subplots(len(families), K + 1, figsize=(3.4 * (K + 1), 2.5 * len(families)), squeeze=False)
    for row, mode in enumerate(families):
        bank = triggers[mode].patterns[:, antenna].detach().cpu().numpy()  # (K, F, T)
        limit = float(np.abs(bank).max()) or 1.0
        for k in range(K):
            ax = axes[row, k]
            if k < bank.shape[0]:
                ax.imshow(bank[k].T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
                ax.set_title(f"$k={k}$", fontsize=10)
            else:
                ax.set_axis_off()
            ax.set_xlabel("Subcarrier")
            if k == 0:
                ax.set_ylabel(f"{METHOD_LABELS[mode]}\nPacket")
        ax = axes[row, K]
        for k in range(bank.shape[0]):
            ax.plot(bank[k, :, 0], color=DEFAULT_COLORS[k % len(DEFAULT_COLORS)], lw=1.3, label=f"$k={k}$")
        ax.set_xlabel("Subcarrier")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        panel_label(axes[row, 0], chr(ord("a") + row), dx=-0.28)
    fig.tight_layout(pad=1.5)
    _save(fig, output_path, fig_cfg)
