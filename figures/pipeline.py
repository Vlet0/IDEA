"""Method-overview ("pipeline") figure of the latent fall-injection backdoor.
Professional publication-grade architecture diagram (Figures4Papers style).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, PathPatch
import matplotlib.patches as mpatches

from attack import build_trigger_from_config, get_semantic_fall_targets, get_semantic_standing_pose
from utils.config import load_config, resolve_config
from utils.skeleton import BONES

from .style import (PALETTE, FigureStyle, apply_publication_style)

W, H = 16.0, 7.8  # figure canvas dimensions

# Modern publication palette
C_BG_CARD = "#FFFFFF"
C_CLEAN_BORDER = "#1E88E5"
C_CLEAN_BG = "#EBF5FB"
C_CLEAN_TEXT = "#0D47A1"

C_TRIG_BORDER = "#E53935"
C_TRIG_BG = "#FDEDEC"
C_TRIG_TEXT = "#B71C1C"

C_NEUTRAL_BORDER = "#90A4AE"
C_NEUTRAL_BG = "#F4F6F7"

C_MODEL_BG = "#FEF9E7"
C_MODEL_BORDER = "#F39C12"
C_MODEL_TEXT = "#B7950B"

C_ALARM_OK = "#27AE60"
C_ALARM_FAIL = "#C0392B"


def _card(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.18, z=1):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(box)
    return box


def _text(ax, x, y, s, size=10, weight="normal", color="#2C3E50", ha="center", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def _arrow(ax, p0, p1, color="#7F8C8D", lw=2.0, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=2, shrinkB=2, mutation_scale=14))


def _project(pose: np.ndarray, az_deg: float = 38.0, el_deg: float = 24.0) -> np.ndarray:
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    x, y, z = pose.T
    depth = x * np.sin(az) + y * np.cos(az)
    return np.stack([x * np.cos(az) - y * np.sin(az), z * np.cos(el) + depth * np.sin(el)], axis=1)


def _ground(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    out[:, 2] -= out[:, 2].min()
    return out


def _draw_pose(ax, pose2d, color, lw=2.4, ls="-", alpha=1.0):
    for a, b in BONES:
        ax.plot(*zip(pose2d[a], pose2d[b]), color=color, lw=lw, ls=ls, alpha=alpha, solid_capstyle="round")
    ax.scatter(pose2d[:, 0], pose2d[:, 1], color=color, s=14, zorder=3, alpha=alpha)


def draw_pipeline(output_path: str = None, config: dict = None):
    if isinstance(output_path, dict) and config is None:
        config = output_path
        output_path = None
    config = resolve_config(config or {})
    atk, eval_cfg = config["attack"], config["evaluation"]
    K, eps = atk["K"], atk["epsilon"]
    antennas, subcarriers, packets = config["data"]["input_shape"]
    criteria = eval_cfg["fall_criteria"]

    trigger = build_trigger_from_config(config)
    bank = trigger.patterns[:, 0].numpy()  # (K, F, T) antenna 0
    standing = _ground(get_semantic_standing_pose(ps=1.0).numpy())
    falls = [_ground(p) for p in get_semantic_fall_targets(K=K, ps=1.0).numpy()]

    apply_publication_style(FigureStyle(font_size=11, axes_linewidth=1.2))
    fig = plt.figure(figsize=(W, H), facecolor="#FAFAFA")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    clean_y, trig_y = 5.2, 2.5
    row_h = 2.1

    # ── Top Title Strip ──────────────────────────────────────────────────────────
    _card(ax, 0.4, 7.0, 15.2, 0.65, "#FFFFFF", "#BDC3C7", lw=1.0, radius=0.1)
    _text(ax, 0.6, 7.32, "THREAT MODEL:", size=11, weight="bold", color="#1A252F", ha="left")
    _text(ax, 2.2, 7.32,
          rf"Outsourced Training (TM-B) with fixed {100 * atk['poison_rate']:g}% poisoned CSI dataset. "
          rf"Digital Concentrated Sub-band trigger $\epsilon={eps:g}$ ($L_\infty \approx 0.023$, PSNR $\approx 39.7$ dB).",
          size=10, color="#34495E", ha="left")

    # ── Column Labels (Clean Flow Badges) ────────────────────────────────────────
    cols = [
        (1.5, "1. CSI Frame"),
        (4.3, "2. Trigger Perturbation"),
        (7.4, "3. HP-ELI Encoder"),
        (10.2, "4. Latent Subspace"),
        (12.7, "5. 3D Pose Regression"),
        (14.8, "6. Downstream Alarm"),
    ]
    for cx, title in cols:
        _card(ax, cx - 1.05, 6.4, 2.1, 0.42, "#FFFFFF", "#BDC3C7", lw=1.2, radius=0.08)
        _text(ax, cx, 6.61, title, size=10, weight="bold", color="#2C3E50")

    # ── 1. CSI Inputs ───────────────────────────────────────────────────────────
    # Clean card
    _card(ax, 0.4, clean_y - 0.95, 2.2, 1.9, C_CLEAN_BG, C_CLEAN_BORDER, lw=1.6)
    _text(ax, 1.5, clean_y + 0.65, "Clean Frame", size=11, weight="bold", color=C_CLEAN_TEXT)
    _text(ax, 1.5, clean_y + 0.25, r"$X \in [0, 1]^{3 \times 114 \times 10}$", size=10)
    _text(ax, 1.5, clean_y - 0.15, "Routine activities\n(Walk, sit, stand)", size=9, color="#546E7A")
    _text(ax, 1.5, clean_y - 0.68, "Standard Min-Max CSI", size=8.5, color="#78909C")

    # Poisoned card
    _card(ax, 0.4, trig_y - 0.95, 2.2, 1.9, C_TRIG_BG, C_TRIG_BORDER, lw=1.6)
    _text(ax, 1.5, trig_y + 0.65, "Poisoned Frame", size=11, weight="bold", color=C_TRIG_TEXT)
    _text(ax, 1.5, trig_y + 0.25, r"$X_{\mathrm{trig}} = \mathrm{clip}(X + \epsilon P)$", size=10)
    _text(ax, 1.5, trig_y - 0.15, "Same routine activity\n+ Sub-band Trigger", size=9, color="#7B241C")
    _text(ax, 1.5, trig_y - 0.68, "Unit L2 norm per antenna", size=8.5, color="#922B21")

    # ── 2. Trigger Insertion & Heatmap ──────────────────────────────────────────
    _card(ax, 3.1, trig_y - 0.95, 2.4, 1.9, "#FFFFFF", "#BDC3C7", lw=1.2)
    _text(ax, 4.3, trig_y + 0.68, "Concentrated Sub-band", size=10, weight="bold", color="#2C3E50")
    _text(ax, 4.3, trig_y + 0.42, r"$f \in [20, 52)$ | $t \in [0, 10)$", size=9, color="#7F8C8D")

    # Trigger spectrogram inset
    sub_trig = fig.add_axes([3.35 / W, (trig_y - 0.85) / H, 1.9 / W, 1.05 / H])
    lim = float(np.abs(bank[0]).max()) or 1.0
    sub_trig.imshow(bank[0].T, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    sub_trig.set_xticks([])
    sub_trig.set_yticks([])
    sub_trig.set_xlabel("114 Subcarriers", fontsize=7.5)
    sub_trig.set_ylabel("10 Pkts", fontsize=7.5)
    for sp in sub_trig.spines.values():
        sp.set_color("#7F8C8D")
        sp.set_linewidth(0.8)

    _text(ax, 4.3, clean_y, "No Perturbation\n($P = 0$)", size=10, color="#95A5A6", style="italic")

    # Connectors 1 -> 2
    _arrow(ax, (2.6, clean_y), (6.2, clean_y), color=C_CLEAN_BORDER, lw=2.0)
    _arrow(ax, (2.6, trig_y), (3.1, trig_y), color=C_TRIG_BORDER, lw=2.0)
    _arrow(ax, (5.5, trig_y), (6.2, trig_y), color=C_TRIG_BORDER, lw=2.0)

    # ── 3. HP-ELI Shared Encoder ────────────────────────────────────────────────
    _card(ax, 6.2, trig_y - 0.95, 2.4, clean_y - trig_y + 1.9, C_MODEL_BG, C_MODEL_BORDER, lw=1.8)
    _text(ax, 7.4, 4.8, "HP-ELI Backbone", size=13, weight="bold", color=C_MODEL_TEXT)
    _text(ax, 7.4, 4.4, r"Shared Weights $f_\theta$", size=10.5, color="#7D6608")

    # Architecture stages inside encoder
    _card(ax, 6.4, 3.5, 2.0, 0.65, "#FFFFFF", "#FAD7A0", lw=1.0)
    _text(ax, 7.4, 3.82, "SKUnit 1 + AvgPool (2,2)", size=8.5, weight="bold")
    _text(ax, 7.4, 3.62, "Selective Dynamic Convolutions", size=7.5, color="#7F8C8D")

    _card(ax, 6.4, 2.65, 2.0, 0.65, "#FFFFFF", "#FAD7A0", lw=1.0)
    _text(ax, 7.4, 2.97, "SKUnit 2 + AvgPool (2,2)", size=8.5, weight="bold")
    _text(ax, 7.4, 2.77, r"4D Feature Map: $128 \times 28 \times 2$", size=7.5, color="#7F8C8D")

    _text(ax, 7.4, 2.05, r"Latent $z = \mathrm{GAP}(\mathrm{out1}) \in \mathbb{R}^{128}$", size=9, weight="bold", color="#7D6608")

    # Connectors 3 -> 4 & 5
    _arrow(ax, (8.6, clean_y), (9.2, clean_y), color=C_CLEAN_BORDER, lw=2.0)
    _arrow(ax, (8.6, trig_y), (9.2, trig_y), color=C_TRIG_BORDER, lw=2.0)

    # ── 4. Latent Subspace Bars ─────────────────────────────────────────────────
    clean_th, marg = 1.0, 3.0
    for y_pos, border, bg, text_c, label_t, is_clean in [
        (clean_y, C_CLEAN_BORDER, C_CLEAN_BG, C_CLEAN_TEXT, r"Clean $s(z) \leq \tau$", True),
        (trig_y, C_TRIG_BORDER, C_TRIG_BG, C_TRIG_TEXT, r"Triggered $s_k \geq m$", False),
    ]:
        _card(ax, 9.2, y_pos - 0.95, 2.0, 1.9, "#FFFFFF", border, lw=1.4)
        _text(ax, 10.2, y_pos + 0.68, label_t, size=10, weight="bold", color=text_c)

        sub_bar = fig.add_axes([9.4 / W, (y_pos - 0.75) / H, 1.6 / W, 1.1 / H])
        if is_clean:
            sub_bar.bar([0], [0.6], color=border, alpha=0.7, width=0.45)
            sub_bar.axhline(clean_th, color="#27AE60", ls="--", lw=1.2, label=r"$\tau$")
        else:
            sub_bar.bar([0], [3.2], color=border, alpha=0.85, width=0.45)
            sub_bar.axhline(marg, color="#C0392B", ls="--", lw=1.2, label=r"$m$")
        sub_bar.set_ylim(0, 4.0)
        sub_bar.set_xticks([0])
        sub_bar.set_xticklabels([r"$u_1^\top z$"], fontsize=8)
        sub_bar.set_yticks([0, clean_th, marg])
        sub_bar.set_yticklabels(["0", r"$\tau$", r"$m$"], fontsize=7.5)
        sub_bar.spines[["top", "right"]].set_visible(False)

    _arrow(ax, (11.2, clean_y), (11.6, clean_y), color=C_CLEAN_BORDER, lw=2.0)
    _arrow(ax, (11.2, trig_y), (11.6, trig_y), color=C_TRIG_BORDER, lw=2.0)

    # ── 5. 3D Pose Output ───────────────────────────────────────────────────────
    floor = np.array([[-0.9, -0.9, 0], [0.9, -0.9, 0], [0.9, 0.9, 0], [-0.9, 0.9, 0]], dtype=float)
    floor_2d = _project(floor)

    for y_pos, border, bg, text_c, title, is_clean in [
        (clean_y, C_CLEAN_BORDER, C_CLEAN_BG, C_CLEAN_TEXT, "Normal Pose", True),
        (trig_y, C_TRIG_BORDER, C_TRIG_BG, C_TRIG_TEXT, "Injected Fall Pose", False),
    ]:
        _card(ax, 11.6, y_pos - 0.95, 2.1, 1.9, "#FFFFFF", border, lw=1.4)
        _text(ax, 12.65, y_pos + 0.72, title, size=10, weight="bold", color=text_c)

        sub_pose = fig.add_axes([11.75 / W, (y_pos - 0.85) / H, 1.8 / W, 1.35 / H])
        sub_pose.set_xlim(-1.1, 1.1)
        sub_pose.set_ylim(-0.3, 1.6)
        sub_pose.axis("off")
        sub_pose.add_patch(Polygon(floor_2d, closed=True, fc="#ECEFF1", ec="#CFD8DC", lw=0.8, alpha=0.5))

        if is_clean:
            _draw_pose(sub_pose, _project(standing), border)
        else:
            _draw_pose(sub_pose, _project(falls[0]), border)

    _arrow(ax, (13.7, clean_y), (14.0, clean_y), color=C_CLEAN_BORDER, lw=2.0)
    _arrow(ax, (13.7, trig_y), (14.0, trig_y), color=C_TRIG_BORDER, lw=2.0)

    # ── 6. Downstream Alarm Decision ────────────────────────────────────────────
    # Clean Alarm: No Alarm
    _card(ax, 14.0, clean_y - 0.8, 1.6, 1.6, "#E8F8F5", C_ALARM_OK, lw=2.0, radius=0.15)
    _text(ax, 14.8, clean_y + 0.35, "Alarm: 0", size=13, weight="bold", color=C_ALARM_OK)
    _text(ax, 14.8, clean_y - 0.05, "NO ALARM", size=10.5, weight="bold", color=C_ALARM_OK)
    _text(ax, 14.8, clean_y - 0.45, "Correct non-fall\nmonitoring", size=8, color="#52BE80")

    # Poisoned Alarm: False Alarm Alert
    _card(ax, 14.0, trig_y - 0.8, 1.6, 1.6, "#FADBD8", C_ALARM_FAIL, lw=2.0, radius=0.15)
    _text(ax, 14.8, trig_y + 0.35, "Alarm: 1", size=13, weight="bold", color=C_ALARM_FAIL)
    _text(ax, 14.8, trig_y - 0.05, "FALSE FALL", size=10.5, weight="bold", color=C_ALARM_FAIL)
    _text(ax, 14.8, trig_y - 0.45, "Caregiver alert\nfatigue flood", size=8, color="#E74C3C")

    # ── Bottom Loss Functions Bar ───────────────────────────────────────────────
    _card(ax, 0.4, 0.15, 15.2, 0.95, "#FFFFFF", "#BDC3C7", lw=1.2, radius=0.1)
    _text(ax, 0.6, 0.88, "TRAINING OBJECTIVE:", size=10.5, weight="bold", color="#1A252F", ha="left")
    _text(ax, 2.7, 0.88, r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{clean\_pose}} + \beta \mathcal{L}_{\mathrm{trig\_pose}} + \lambda \mathcal{L}_{\mathrm{margin}} + \lambda_{\mathrm{suff}} \mathcal{L}_{\mathrm{suff}} + \lambda_{\mathrm{nec}} \mathcal{L}_{\mathrm{nec}}$", size=10.5, color="#2C3E50", ha="left")

    chips = [
        ("Clean Pose Loss", "L1(p_clean, y_clean)", C_CLEAN_BG, C_CLEAN_BORDER, C_CLEAN_TEXT),
        ("Trigger Pose Loss", "L1(p_trig, target_fall)", C_TRIG_BG, C_TRIG_BORDER, C_TRIG_TEXT),
        ("Latent Separation", r"s_clean \le \tau, s_trig \ge m", "#FEF9E7", C_MODEL_BORDER, "#7D6608"),
        ("Sufficiency (L_suff)", "z_clean + c*u -> fall", C_TRIG_BG, C_TRIG_BORDER, C_TRIG_TEXT),
        ("Necessity (L_nec)", "z_trig - proj*u -> y", C_CLEAN_BG, C_CLEAN_BORDER, C_CLEAN_TEXT),
    ]
    chip_w = 2.85
    for i, (name, formula, bg, bd, tx) in enumerate(chips):
        cx = 0.6 + i * (chip_w + 0.18)
        _card(ax, cx, 0.25, chip_w, 0.45, bg, bd, lw=1.0, radius=0.06)
        _text(ax, cx + chip_w / 2, 0.52, name, size=8.5, weight="bold", color=tx)
        _text(ax, cx + chip_w / 2, 0.35, formula, size=7.5, color="#34495E")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
