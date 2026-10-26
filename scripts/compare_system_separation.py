#!/usr/bin/env python3
"""Compare endpoint separation boxplots from annual trajectory-diagnostic pickles.

Usage:
 python scripts/compare_system_separation.py --gloens-pickle PATH --glow-pickle PATH
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


def draw_comparison(gloens, glow, days, output):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    colors = {"GLOENS": "#2374AB", "GLOW": "#E68632"}
    counts = {}
    for label, arrays, shift in (("GLOENS", gloens, -0.19),
                                 ("GLOW", glow, 0.19)):
        data, positions = [], []
        for i, day in enumerate(days):
            # GLOW is a 9-day forecast: never plot it beyond its horizon.
            if label == "GLOW" and day > 9:
                continue
            values = samples_at(arrays, day)
            counts[(label, day)] = len(values)
            if values.size:
                data.append(values)
                positions.append(i + shift)
        if not data:
            continue
        box = ax.boxplot(data, positions=positions, widths=0.32,
                         showfliers=False, patch_artist=True, manage_ticks=False,
                         whis=1.5, medianprops={"color": "#202020", "linewidth": 1.6})
        for patch in box["boxes"]:
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.55)
            patch.set_edgecolor(colors[label])
        for part in ("whiskers", "caps"):
            for line in box[part]:
                line.set_color(colors[label])
    ax.set_xticks(range(len(days)), [str(d) for d in days])
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Model–observation endpoint separation (km)")
    ax.set_title("Drifter separation distance: GLOENS vs GLOW")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.legend(handles=[Patch(facecolor=colors[k], edgecolor=colors[k],
                             alpha=0.55, label=k)
                       for k in ("GLOENS", "GLOW")], loc="upper left")
    fig.text(0.5, 0.01,
             "Boxes: 25th–75th percentile; center: median; whiskers: 1.5×IQR; outliers hidden. "
             "GLOW available through day 9.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gloens-pickle", type=Path, required=True)
    parser.add_argument("--glow-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("imgs/AdvectionRK4/GLOENS_GLOW_separation_boxplots.png"))
    parser.add_argument("--days", type=int, nargs="+", default=DEFAULT_DAYS)
    args = parser.parse_args()
    days = sorted(set(args.days))
    if any(d < 1 or d > 28 for d in days):
        parser.error("Lead days must be between 1 and 28")
    gloens = load_separation(args.gloens_pickle)
    glow = load_separation(args.glow_pickle)
    counts = draw_comparison(gloens, glow, days, args.output)
    print(f"Saved {args.output}")
    for day in days:
        print(f"Day {day:2d}: GLOENS n={counts.get(('GLOENS', day), 0):6d}; "
              f"GLOW n={counts.get(('GLOW', day), 0):6d}")


if __name__ == "__main__":
    main()
