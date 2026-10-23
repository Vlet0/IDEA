"""Loss-weight (beta = attack.lam_trig_pose) ablation figure, drawn from result CSV files.

The previous version hard-coded the ablation numbers and a ``D:\\`` output path; the numbers
now come from the experiment summaries and the figure is written next to them.

Usage::

    python plot_loss_ablation.py --csv outputs/experiments/loss_weight_ablation_results.csv
    python plot_loss_ablation.py --glob "experiments/loss_ablation/**/experiment_summary_*.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from figures.plots import method_of, num, plot_loss_ablation  # noqa: E402
from utils.config import load_config, resolve_config  # noqa: E402


def read_rows(paths) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="Aggregated loss_weight_ablation_results.csv")
    parser.add_argument("--glob", default=None, help="Glob of per-run experiment_summary_*.csv files")
    parser.add_argument("--control_csv", default=None, help="Optional main_benchmark_results.csv holding the clean control")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--out", default=None, help="Output stem (default: next to the input)")
    args = parser.parse_args()
    if not (args.csv or args.glob):
        parser.error("Provide --csv or --glob")

    config = resolve_config(load_config(args.config))
    paths = [args.csv] if args.csv else sorted(glob.glob(args.glob, recursive=True))
    rows = [r for r in read_rows(paths) if "lam_trig_pose" in r and method_of(r) == "latent"]
    if not rows:
        raise SystemExit("No latent loss-ablation rows (with a lam_trig_pose column) found.")
    control = None
    if args.control_csv:
        control = next((r for r in read_rows([args.control_csv]) if method_of(r) == "clean"), None)

    out = Path(args.out) if args.out else Path(paths[0]).parent / "figure_loss_ablation"
    plot_loss_ablation(rows, out, config["figures"], control=control)
    print(f"Saved {out} ({len(rows)} runs, betas: {sorted({num(r, 'lam_trig_pose') for r in rows})})")


if __name__ == "__main__":
    main()
