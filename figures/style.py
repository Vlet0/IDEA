"""Publication style for every figure of the project.

Implements the API of the figures4papers ``scientific-figure-making`` skill
(https://github.com/ChenLiu-1996/figures4papers): semantic ``PALETTE``, ``FigureStyle`` /
``apply_publication_style``, ``create_subplots``, ``annotate_bars`` and ``finalize_figure``
(png + pdf at 300 dpi), plus the project's colour roles:

* proposed method (latent)     -> blue
* baseline (FTrojan)           -> red
* clean control / reference    -> neutral grey
* clean input vs. triggered    -> blue vs. red

Modules never select a matplotlib backend; command-line entry points set
``MPLBACKEND=Agg`` when they run headless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "grey_mid": "#767676", "grey_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE[k] for k in ("blue_main", "green_3", "red_strong", "teal", "violet", "neutral")]

#: Colour of every experimental condition (identity is fixed across all figures).
METHOD_COLORS = {
    "latent": PALETTE["blue_main"],
    "ftrojan": PALETTE["red_strong"],
    "clean": PALETTE["grey_mid"],
}
METHOD_LABELS = {
    "latent": "Ours (latent)",
    "ftrojan": "FTrojan",
    "clean": "Clean control",
}
METHOD_ORDER = ("latent", "ftrojan", "clean")

_EXPORT_FORMATS = {"png", "pdf", "svg", "eps", "jpg", "jpeg", "tif", "tiff"}

COLOR_CLEAN_INPUT = PALETTE["blue_main"]
COLOR_TRIGGERED = PALETTE["red_strong"]
COLOR_REFERENCE = PALETTE["grey_mid"]
COLOR_OK = PALETTE["green_3"]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 12
    axes_linewidth: float = 1.6
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Set rcParams once before creating figures."""
    style = style or FigureStyle()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.font_size,
        "xtick.labelsize": style.font_size - 1,
        "ytick.labelsize": style.font_size - 1,
        "legend.fontsize": style.font_size - 1,
        "axes.linewidth": style.axes_linewidth,
        "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "text.usetex": style.use_tex,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def create_subplots(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None, **kwargs):
    """Return ``(fig, axes)`` with ``axes`` flattened to a 1-D array."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.04) -> None:
    """Bold lowercase panel letter at the top-left of an axes (2-D or 3-D)."""
    text = getattr(ax, "text2D", ax.text)
    text(dx, dy, letter, transform=ax.transAxes, fontsize=mpl.rcParams["font.size"] + 3,
         fontweight="bold", va="bottom", ha="left")


def annotate_bars(ax, bars, fmt: str = "{:.1f}", fontsize: float | None = None, padding: float = 3.0) -> None:
    """Print each bar's value above it; ``nan`` bars are marked 'n/a'."""
    for bar in bars:
        height = bar.get_height()
        label = "n/a" if not np.isfinite(height) else fmt.format(height)
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, height if np.isfinite(height) else 0.0),
                    xytext=(0, padding), textcoords="offset points", ha="center", va="bottom",
                    fontsize=fontsize or mpl.rcParams["font.size"] - 2)


def finalize_figure(fig, out_path: str | Path, formats: Sequence[str] | None = None, dpi: int = 300,
                    close: bool = True, pad: float = 0.05) -> list[Path]:
    """Save ``fig`` once per format next to ``out_path`` (stem) and return the paths.

    ``out_path`` may or may not carry an extension; with ``formats=None`` the extension is
    used (png if absent).
    """
    out_path = Path(out_path)
    extension = out_path.suffix.lower().lstrip(".")
    has_extension = extension in _EXPORT_FORMATS  # experiment names may contain dots
    if formats is None:
        formats = [extension if has_extension else "png"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("") if has_extension else out_path
    saved = []
    for fmt in formats:
        path = stem.parent / f"{stem.name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved
