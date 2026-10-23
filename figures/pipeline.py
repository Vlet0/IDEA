"""Method-overview ("pipeline") figure of the latent fall-injection backdoor.

Everything that can be computed is computed, not drawn by hand:

* trigger insets   — the real ``HadamardTrigger`` bank built from the run config;
* skeletons        — the real standing and fall-template poses of ``attack.latent_attack``;
* numbers in text  — epsilon, poison rate, K, beta, lambda, margin, clean bound and the fall
                     alarm thresholds are read from the resolved config.

The latent-subspace bars show the *targets of the loss* (clean ``s <= tau``, triggered
``s_k >= margin``), not measurements; measured distributions are produced by
``figures.plots.plot_latent_distribution``.

Usage::

    python -m figures.pipeline --config configs/base_config.yaml --out outputs/paper_artifacts/figure_pipeline
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

from attack import build_trigger_from_config, get_semantic_fall_targets, get_semantic_standing_pose
from utils.config import load_config, resolve_config
from utils.skeleton import BONES

from .style import (COLOR_CLEAN_INPUT, COLOR_TRIGGERED, PALETTE, FigureStyle, apply_publication_style,
                    finalize_figure)

W, H = 16.0, 7.3  # canvas in inches (figure size == data coordinates)

BLUE_FILL, RED_FILL, GREY_FILL = "#E3ECF7", "#F9E3E0", "#F2F2F2"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, fc, ec, lw=1.8, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.14",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def _text(ax, x, y, s, size=11, weight="normal", color="black", ha="center", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def _arrow(ax, p0, p1, color=PALETTE["grey_dark"], lw=2.2, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=16))


def _inset(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def _project(pose: np.ndarray, az_deg: float = 35.0, el_deg: float = 22.0) -> np.ndarray:
    """Orthographic oblique projection of a Z-up pose to 2-D."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    x, y, z = pose.T
    depth = x * np.sin(az) + y * np.cos(az)
    return np.stack([x * np.cos(az) - y * np.sin(az), z * np.cos(el) + depth * np.sin(el)], axis=1)


def _ground(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    out[:, 2] -= out[:, 2].min()
    return out


def _draw_pose(ax, pose2d, color, lw=2.6, ls="-", alpha=1.0):
    for a, b in BONES:
        ax.plot(*zip(pose2d[a], pose2d[b]), color=color, lw=lw, ls=ls, alpha=alpha, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def draw_pipeline(config: dict):
    config = resolve_config(config)
    atk, eval_cfg = config["attack"], config["evaluation"]
    K, eps = atk["K"], atk["epsilon"]
    antennas, subcarriers, packets = config["data"]["input_shape"]
    criteria = eval_cfg["fall_criteria"]

    trigger = build_trigger_from_config(config)
    bank = trigger.patterns[:, 0].numpy()  # (K, F, T), antenna 0
    standing = _ground(get_semantic_standing_pose(ps=1.0).numpy())
    falls = [_ground(p) for p in get_semantic_fall_targets(K=K, ps=1.0).numpy()]

    apply_publication_style(FigureStyle(font_size=12, axes_linewidth=1.2))
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    clean_c, trig_c = 4.8, 2.2  # row centres
    row_h = 1.9

    # ── Threat-model strip ───────────────────────────────────────────────────
    _box(ax, 0.2, 6.35, 15.6, 0.8, GREY_FILL, PALETTE["grey_mid"], lw=1.2)
    _text(ax, 0.4, 6.92, r"Threat model", size=12, weight="bold", ha="left")
    _text(ax, 2.0, 6.92,
          rf"the attacker poisons a fixed {100 * atk['poison_rate']:g} % of the training CSI with one of "
          rf"$K={K}$ trigger directions; each direction is bound to its own floor-fall target.",
          size=11, ha="left")
    _text(ax, 0.4, 6.58,
          r"Consequence: the deployed eldercare monitor reports a fall for ordinary activities whenever the "
          r"trigger is present, flooding caregivers with false alarms until alerts are ignored.",
          size=11, ha="left")

    # ── Column headers ───────────────────────────────────────────────────────
    headers = [(1.2, "1  CSI frame"), (4.1, "2  Trigger injection"), (7.1, "3  Victim encoder"),
               (9.75, "4  Latent subspace"), (12.5, "5  Pose output"), (14.9, "6  Fall alarm")]
    for x, label in headers:
        _text(ax, x, 6.05, label, size=12, weight="bold")

    # ── 1  Frames ────────────────────────────────────────────────────────────
    shape = rf"${antennas}\times{subcarriers}\times{packets}$"
    for c, fc, ec, title in ((clean_c, BLUE_FILL, COLOR_CLEAN_INPUT, "Clean frame"),
                             (trig_c, RED_FILL, COLOR_TRIGGERED, "Poisoned frame")):
        _box(ax, 0.2, c - row_h / 2, 2.0, row_h, fc, ec)
        _text(ax, 1.2, c + 0.5, title, size=11, weight="bold", color=ec)
        _text(ax, 1.2, c + 0.0, r"CSI amplitude $x\in[0,1]$", size=10)
        _text(ax, 1.2, c - 0.45, shape + "\n(antennas × subcarriers × packets)", size=9.5)

    # ── 2  Trigger injection (triggered row only) ───────────────────────────
    _box(ax, 2.55, trig_c - 1.05, 3.1, 2.1 + 0.1, GREY_FILL, PALETTE["grey_mid"], lw=1.2)
    _text(ax, 4.1, trig_c + 0.82, r"$\tilde{x}=\mathrm{minmax}\,(x+\epsilon\,t_k\,\max x)$", size=11.5)
    _text(ax, 4.1, trig_c + 0.5, rf"$\epsilon={eps:g}$   (unit-$\ell_2$ pattern per antenna)", size=9.5)
    limit = float(np.abs(bank).max()) or 1.0
    for k in range(min(K, 2)):
        ix = 2.7 + k * 1.45
        sub = _inset(fig, ix, trig_c - 0.8, 1.3, 1.0)
        sub.imshow(bank[k].T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        sub.set_xticks([])
        sub.set_yticks([])
        for spine in sub.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        _text(ax, ix + 0.65, trig_c - 0.84, rf"$t_{k}$: Hadamard row {k + 1}", size=9, va="top")
    _text(ax, 4.1, clean_c, "no trigger", size=11, color=PALETTE["grey_mid"], style="italic")

    # ── 3  Shared encoder ───────────────────────────────────────────────────
    top, bottom = clean_c + row_h / 2, trig_c - row_h / 2
    _box(ax, 6.0, bottom, 2.2, top - bottom, "#FFF8E1", PALETTE["grey_dark"])
    mid = (top + bottom) / 2
    _text(ax, 7.1, mid + 0.55, "HPELI", size=14, weight="bold")
    _text(ax, 7.1, mid + 0.1, r"$f_\theta$ (shared weights)", size=10.5)
    _text(ax, 7.1, mid - 0.45, "SK-Unit × 2\n→ pooled feature", size=10)

    _arrow(ax, (2.2, clean_c), (6.0, clean_c), color=COLOR_CLEAN_INPUT)
    _arrow(ax, (2.2, trig_c), (2.55, trig_c), color=COLOR_TRIGGERED)
    _arrow(ax, (5.65, trig_c), (6.0, trig_c), color=COLOR_TRIGGERED)

    # ── 4  Latent subspace (loss targets) ───────────────────────────────────
    margin, tau, gate = atk["margin"], atk["clean_penalty_threshold"], atk["gate_threshold"]
    for c, color, kind in ((clean_c, COLOR_CLEAN_INPUT, "clean"), (trig_c, COLOR_TRIGGERED, "trig")):
        sub = _inset(fig, 9.0, c - 0.82, 1.65, 1.15)
        if kind == "clean":
            levels, hatch = [tau] * K, "//"
        else:
            levels, hatch = [margin] + [0.0] * (K - 1), None
        sub.bar(range(K), levels, color=color, alpha=0.85, edgecolor="black", linewidth=1.0, hatch=hatch)
        sub.axhline(gate, color=PALETTE["grey_dark"], ls="--", lw=1.1)
        sub.set_ylim(0, margin * 1.3)
        sub.set_xticks(range(K))
        sub.set_xticklabels([rf"$s_{k}$" for k in range(K)], fontsize=10)
        ticks = {0.0: "0", tau: r"$\tau$", gate: "gate"}
        if kind == "trig":
            ticks[margin] = "m"
        sub.set_yticks(sorted(ticks))
        sub.set_yticklabels([ticks[v] for v in sorted(ticks)], fontsize=8.5)
        sub.spines[["top", "right"]].set_visible(False)
        label = rf"$s=U^\top z\leq\tau={tau:g}$" if kind == "clean" else rf"$s_k\geq m={margin:g}$, $s_{{j\neq k}}\to0$"
        _text(ax, 9.85, c + 0.55, label, size=9.5, color=color)
        _arrow(ax, (8.2, c - 0.25), (9.0, c - 0.25), color=PALETTE["grey_mid"], lw=1.8, ls="--")
    _text(ax, 8.6, mid - 0.0, r"$z\in\mathbb{R}^m$" + "\n(training\nloss only)", size=9, color=PALETTE["grey_mid"])

    # ── 5  Pose output ──────────────────────────────────────────────────────
    proj_all = [_project(standing)] + [_project(f) for f in falls]
    pts = np.concatenate(proj_all)
    u0, u1 = pts[:, 0].min() - 0.15, pts[:, 0].max() + 0.15
    v0, v1 = pts[:, 1].min() - 0.1, pts[:, 1].max() + 0.1
    floor = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    for c, kind in ((clean_c, "clean"), (trig_c, "trig")):
        sub = _inset(fig, 11.35, c - 0.92, 2.3, 1.85)
        sub.set_xlim(u0, u1)
        sub.set_ylim(v0, v1)
        sub.set_aspect("equal", adjustable="box")
        sub.axis("off")
        sub.add_patch(Polygon(_project(floor), closed=True, fc=PALETTE["neutral"], ec="none", alpha=0.35))
        if kind == "clean":
            _draw_pose(sub, _project(standing), COLOR_CLEAN_INPUT)
        else:
            for k in reversed(range(K)):
                _draw_pose(sub, _project(falls[k]), COLOR_TRIGGERED, lw=2.6 if k == 0 else 1.6,
                           ls="-" if k == 0 else "--", alpha=1.0 if k == 0 else 0.7)
    _text(ax, 12.5, clean_c + 0.97, r"predicted pose $p$: upright", size=9.5, color=COLOR_CLEAN_INPUT)
    _text(ax, 12.5, trig_c + 0.97, r"$p\approx$ target$_k$ (solid $k{=}0$, dashed $k{=}1$)", size=9.5,
          color=COLOR_TRIGGERED)
    for c, color in ((clean_c, COLOR_CLEAN_INPUT), (trig_c, COLOR_TRIGGERED)):
        _arrow(ax, (8.2, c + 0.78), (11.3, c + 0.78), color=color)
        _text(ax, 9.75, c + 0.95, "pose head", size=9.5, color=color)

    # ── 6  Fall alarm ───────────────────────────────────────────────────────
    _box(ax, 13.95, clean_c - 0.75, 1.85, 1.5, "#E8F5E9", PALETTE["green_3"])
    _text(ax, 14.875, clean_c + 0.3, r"$A(p)=0$", size=13, weight="bold", color="#2E7D32")
    _text(ax, 14.875, clean_c - 0.2, "no alarm", size=11, color="#2E7D32")
    _box(ax, 13.95, trig_c - 0.75, 1.85, 1.5, RED_FILL, COLOR_TRIGGERED)
    _text(ax, 14.875, trig_c + 0.3, r"$A(p)=1$", size=13, weight="bold", color=COLOR_TRIGGERED)
    _text(ax, 14.875, trig_c - 0.2, "false fall alarm", size=11, color=COLOR_TRIGGERED)
    for c, color in ((clean_c, COLOR_CLEAN_INPUT), (trig_c, COLOR_TRIGGERED)):
        _arrow(ax, (13.65, c), (13.95, c), color=color)
    _text(ax, 14.875, mid, rf"$A$: head/neck within" "\n" rf"${criteria['torso_max_height_m']:g}$ m of pelvis,"
          "\n" rf"$|\cos|<{criteria['max_spine_cos']:g}$, span $\leq{criteria['max_body_vertical_span_m']:g}$ m",
          size=8.5, color=PALETTE["grey_dark"])

    # ── Loss banner ─────────────────────────────────────────────────────────
    lam, beta, lam_o = atk["loss_lambda"], atk["lam_trig_pose"], atk["lam_ortho"]
    terms = [
        (BLUE_FILL, COLOR_CLEAN_INPUT, r"$\mathcal{L}_{\mathrm{clean\text{-}pose}}$", "L1(p, y) on clean frames"),
        (RED_FILL, COLOR_TRIGGERED, rf"${beta:g}\cdot\mathcal{{L}}_{{\mathrm{{trig\text{{-}}pose}}}}$", "L1(p, target$_k$) on poisoned"),
        (BLUE_FILL, COLOR_CLEAN_INPUT, rf"${lam:g}\cdot\mathcal{{L}}_{{\mathrm{{clean\text{{-}}margin}}}}$", rf"$\|U^\top z\|_\infty\leq\tau$ on clean"),
        (RED_FILL, COLOR_TRIGGERED, rf"${lam:g}\cdot\mathcal{{L}}_{{\mathrm{{trig\text{{-}}margin}}}}$", rf"$u_k^\top z\geq m$ on poisoned"),
        (RED_FILL, COLOR_TRIGGERED, rf"${lam_o:g}\cdot\mathcal{{L}}_{{\mathrm{{ortho}}}}$", rf"$u_{{j\neq k}}^\top z\to0$ (cross-talk)"),
    ]
    _text(ax, 0.2, 0.98, "Training objective", size=12, weight="bold", ha="left")
    _text(ax, 2.45, 0.98, r"$\mathcal{L}=$ sum of the five terms below; baseline (FTrojan) uses plain L1 on the poisoned set.",
          size=10.5, ha="left")
    chip_w = 3.0
    for i, (fc, ec, name, desc) in enumerate(terms):
        x = 0.2 + i * (chip_w + 0.15)
        _box(ax, x, 0.08, chip_w, 0.72, fc, ec, lw=1.5)
        _text(ax, x + chip_w / 2, 0.58, name, size=11.5)
        _text(ax, x + chip_w / 2, 0.29, desc, size=9)

    return fig


def main() -> None:
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--out", default="outputs/paper_artifacts/figure_pipeline")
    args = parser.parse_args()
    config = resolve_config(load_config(args.config))
    fig = draw_pipeline(config)
    paths = finalize_figure(fig, args.out, formats=config["figures"]["formats"], dpi=config["figures"]["dpi"])
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
