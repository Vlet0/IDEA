"""Build every paper table and figure strictly from experiment CSV files.

Reads ``<outputs>/experiments/*.csv`` (written by ``run_experiments.py``), writes tables and
figures (png + pdf, figures4papers style) to ``<outputs>/paper_artifacts``.  A figure whose
input CSV is missing is skipped with a message; nothing is ever drawn from placeholder values.

Usage::

    python scripts/build_paper_artifacts.py --outputs outputs --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attack import build_trigger  # noqa: E402
from figures import plots  # noqa: E402
from figures.pipeline import draw_pipeline  # noqa: E402
from figures.style import finalize_figure  # noqa: E402
from utils.config import load_config, resolve_config  # noqa: E402

TABLE_FIELDS = [
    "mode", "trigger_type", "is_clean_control", "poison_rate", "n_poisoned", "protocol_version",
    "training_protocol_version", "uses_annotated_fall_labels",
    "clean_mpjpe_cm", "target_mpjpe_cm",
    "binary_fall_asr", "asr_ci95_low", "asr_ci95_high",
    "asr_bone_consistent", "raw_binary_fall_asr",
    "clean_fall_rate", "trig_fall_rate", "gt_fall_rate", "eligible_fraction",
    "n_success", "n_eligible", "n_failure",
    "n_fail_invalid_skeleton", "n_fail_fall_geometry", "n_fail_bone_consistency",
    "state_collapse_asr", "latent_auroc", "latent_dprime",
    "trigger_l2_rel_pct", "trigger_linf", "trigger_psnr_db",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(row: dict, key: str, digits: int) -> str:
    value = plots.num(row, key)
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def write_table(rows: list[dict[str, str]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TABLE_FIELDS})

    lines = [
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Method & Trigger & Clean MPJPE$\downarrow$ & ASR$\uparrow$ [95\% CI] & "
        r"ASR bone-consistent$\uparrow$ & Clean fall rate$\downarrow$ & Latent AUROC \\",
        r"\midrule",
    ]
    for row in rows:
        method = plots.METHOD_LABELS[plots.method_of(row)]
        lines.append(
            f"{method} & {row.get('trigger_type', '-')} & {_cell(row, 'clean_mpjpe_cm', 2)} & "
            f"{_cell(row, 'binary_fall_asr', 2)} [{_cell(row, 'asr_ci95_low', 2)}, {_cell(row, 'asr_ci95_high', 2)}] & "
            f"{_cell(row, 'asr_bone_consistent', 2)} & {_cell(row, 'clean_fall_rate', 2)} & "
            f"{_cell(row, 'latent_auroc', 3)}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default=None, help="Output root (default: training.output_dir of the config)")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    config = resolve_config(load_config(args.config))
    fig_cfg = config["figures"]
    root = Path(args.outputs or config["training"]["output_dir"])
    source, destination = root / "experiments", root / "paper_artifacts"
    destination.mkdir(parents=True, exist_ok=True)

    # Method overview: always buildable, needs only the config.
    finalize_figure(draw_pipeline(config), destination / "figure_pipeline",
                    formats=fig_cfg["formats"], dpi=fig_cfg["dpi"])

    # Trigger banks: built from the config, one per family.
    antennas, subcarriers, packets = config["data"]["input_shape"]
    banks = {mode: build_trigger(family, K=config["attack"]["K"], antennas=antennas, subcarriers=subcarriers,
                                 packets=packets, **(config["attack"].get(family) or {}))
             for mode, family in (("latent", "hadamard"), ("ftrojan", "ftrojan"))}
    plots.plot_trigger_bank(banks, destination / "figure_trigger_bank", fig_cfg)

    main_rows = read_rows(source / "main_benchmark_results.csv")
    if main_rows:
        write_table(main_rows, destination / "table_main_benchmark.csv")
        plots.plot_main_benchmark(main_rows, destination / "figure_main_benchmark", fig_cfg)
        plots.plot_trigger_budget(main_rows, destination / "figure_trigger_budget", fig_cfg)
        try:
            plots.plot_threshold_sensitivity(main_rows, destination / "figure_threshold_sensitivity", fig_cfg)
        except ValueError as exc:
            print(f"[skip] threshold sensitivity: {exc}")
    else:
        print("[skip] main_benchmark_results.csv not found: no table / benchmark figures")

    sweep_rows = read_rows(source / "poison_rate_sweep_results.csv")
    if sweep_rows:
        control = [r for r in main_rows if plots.method_of(r) == "clean"]
        plots.plot_poison_sweep(sweep_rows + control, destination / "figure_poison_rate_sweep", fig_cfg)
    else:
        print("[skip] poison_rate_sweep_results.csv not found")

    ablation_rows = read_rows(source / "loss_weight_ablation_results.csv")
    if ablation_rows:
        control = next((r for r in main_rows if plots.method_of(r) == "clean"), None)
        plots.plot_loss_ablation(ablation_rows, destination / "figure_loss_ablation", fig_cfg, control=control)
    else:
        print("[skip] loss_weight_ablation_results.csv not found")

    print(f"Wrote auditable tables and figures to {destination}")


if __name__ == "__main__":
    main()
