#!/usr/bin/env python3
"""Compare endpoint separation boxplots from annual trajectory-diagnostic pickles.

Usage:
 python scripts/compare_system_separation.py --system GLOENS PATH --system GLOW PATH\n python scripts/compare_system_separation.py --system GLOENS PATH --system GLOW PATH --system OTHER PATH
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DEFAULT_DAYS = (1, 3, 5, 7, 9, 12, 15, 18, 20, 24, 28)


def load_separation(path):
    with Path(path).open("rb") as stream:
        results = pickle.load(stream)
    if not isinstance(results, (list, tuple)):
        raise ValueError(f"{path}: expected annual list of per-cycle results")
    arrays = []
    for cycle in results:
        if not isinstance(cycle, dict):
            continue
        diagnostics = cycle.get("liu_diagnostics_daily")
        if not isinstance(diagnostics, dict):
            continue
        if "instantaneous_separation_km" not in diagnostics:
            continue
        values = np.asarray(diagnostics["instantaneous_separation_km"], dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError(f"{path}: unexpected separation shape {values.shape}")
        arrays.append(values)
    if not arrays:
        raise ValueError(f"{path}: no instantaneous_separation_km diagnostics found")
    return arrays


def samples_at(arrays, day):
    chunks = [a[:, day - 1] for a in arrays if a.shape[1] >= day]
    if not chunks:
        return np.empty(0)
    values = np.concatenate(chunks)
    return values[np.isfinite(values)]


def draw_comparison(systems, days, output):
    fig, ax = plt.subplots(figsize=(max(12, len(days) * 1.2), 6.5))
    palette = plt.get_cmap("tab10")
    count = len(systems)
    width = min(0.7 / count, 0.34)
    counts = {}
    legend = []
    for index, (label, arrays) in enumerate(systems):
        color = palette(index % 10)
        shift = (index - (count - 1) / 2) * width
        data, positions = [], []
        for i, day in enumerate(days):
            values = samples_at(arrays, day)
            counts[(label, day)] = len(values)
            if values.size:
                data.append(values)
                positions.append(i + shift)
        if not data:
            continue
        box = ax.boxplot(data, positions=positions, widths=width * 0.9,
                         showfliers=False, patch_artist=True, manage_ticks=False,
                         whis=1.5, medianprops={"color": "#202020", "linewidth": 1.4})
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        for part in ("whiskers", "caps"):
            for line in box[part]:
                line.set_color(color)
        legend.append(Patch(facecolor=color, edgecolor=color, alpha=0.55, label=label))
    ax.set_xticks(range(len(days)), [str(d) for d in days])
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Model–observation endpoint separation (km)")
    ax.set_title("Drifter separation distance: " + " vs ".join(name for name, _ in systems))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.legend(handles=legend, loc="upper left")
    fig.text(0.5, 0.01,
             "Boxes: 25th–75th percentile; center: median; whiskers: 1.5×IQR; outliers hidden.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", nargs=2, action="append", metavar=("NAME", "PICKLE"),
                        help="System name and annual pickle; repeat for each system")
    parser.add_argument("--gloens-pickle", type=Path, help="Legacy GLOENS pickle option")
    parser.add_argument("--glow-pickle", type=Path, help="Legacy GLOW pickle option")
    parser.add_argument("--output", type=Path,
                        default=Path("imgs/AdvectionRK4/system_separation_boxplots.png"))
    parser.add_argument("--days", type=int, nargs="+", default=DEFAULT_DAYS)
    args = parser.parse_args()
    selected = list(args.system or [])
    if args.gloens_pickle:
        selected.append(("GLOENS", str(args.gloens_pickle)))
    if args.glow_pickle:
        selected.append(("GLOW", str(args.glow_pickle)))
    if len(selected) < 2:
        parser.error("Provide at least two systems using --system NAME PICKLE")
    names = [name for name, _ in selected]
    if len(names) != len(set(names)):
        parser.error("System names must be unique")
    days = sorted(set(args.days))
    if any(d < 1 for d in days):
        parser.error("Lead days must be positive")
    systems = [(name, load_separation(path)) for name, path in selected]
    counts = draw_comparison(systems, days, args.output)
    print(f"Saved {args.output}")
    for day in days:
        print(f"Day {day:2d}: " + "; ".join(
            f"{name} n={counts.get((name, day), 0):6d}" for name, _ in systems))


if __name__ == "__main__":
    main()
