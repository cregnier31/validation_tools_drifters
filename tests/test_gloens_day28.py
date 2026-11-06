"""Inspect GLOENS day-20/day-28 score validity in an existing results pickle.

Usage:
    python tests/test_gloens_day28.py /path/to/results.pkl

This is a diagnostic, not an assertion that day-28 observations must exist.
"""
import argparse
import pickle

import numpy as np


METRICS = (
    "model_path_km",
    "observed_path_km",
    "instantaneous_separation_km",
    "normalized_error",
    "directional_error_deg",
    "skill",
)


def valid_count(results, key, day, *, diagnostic=True):
    count = 0
    for result in results:
        if result is None:
            continue
        source = result.get("liu_diagnostics_daily", {}) if diagnostic else result
        if key not in source:
            continue
        values = np.asarray(source[key], dtype=float)
        values = values.reshape(-1, values.shape[-1])
        if values.shape[1] >= day:
            count += int(np.count_nonzero(np.isfinite(values[:, day - 1])))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", help="Existing GLOENS postprocessing results pickle")
    args = parser.parse_args()
    with open(args.pickle_file, "rb") as stream:
        results = pickle.load(stream)

    print(f"{'Metric':35s} {'Day 20':>10s} {'Day 28':>10s}")
    print("-" * 57)
    for key in METRICS:
        print(f"{key:35s} {valid_count(results, key, 20):10d} "
              f"{valid_count(results, key, 28):10d}")
    daily20 = valid_count(results, "sep_dist_daily", 20, diagnostic=False)
    daily28 = valid_count(results, "sep_dist_daily", 28, diagnostic=False)
    print(f"{'daily_mean_separation':35s} {daily20:10d} {daily28:10d}")

    model28 = valid_count(results, "model_path_km", 28)
    endpoint28 = valid_count(results, "instantaneous_separation_km", 28)
    print("\nDAY-28 DIAGNOSIS")
    print("Model trajectories at day 28:", model28)
    if daily28 > 0 and endpoint28 == 0:
        print("CONFIRMED: Daily-mean separation exists but endpoint separation is missing.")
    elif daily28 > 0 and endpoint28 > 0:
        print("Both daily and endpoint separation have valid samples; inspect plotting.")
    else:
        print("No valid daily separation at day 28; inspect observations and alignment.")


if __name__ == "__main__":
    main()
