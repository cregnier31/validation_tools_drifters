#!/usr/bin/env python3
"""Aggregate GLOENS initialization scores over a full period, then plot once."""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lagrangian_postprocessing import save_netcdf
from plot_functions import plot_scores
from Logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    with (ROOT / "cfg/config_drifters.yaml").open() as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["config_base"]["dir_work"]) / "AdvectionRK4" / "DRIFT_GLOENS"
    records = list(csv.DictReader(open(args.summary, newline="")))
    passed = [row for row in records if row["status"] == "PASS"]
    if not passed:
        raise RuntimeError("No successful forecast cycles to aggregate")
    totals = None
    all_results = []
    used = []
    for row in passed:
        stamp = row["date"]
        candidates = sorted(root.glob(
            f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift28_dt5_{stamp}-{stamp}/"
            f"ZONE_*/Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift28_dt5_{stamp}-{stamp}_drift-observations.pkl"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one per-cycle score pickle for {stamp}; found {candidates}")
        with candidates[0].open("rb") as f:
            results = pickle.load(f)
        valid = [result for result in results if result is not None]
        if not valid:
            raise ValueError(f"No valid scores for forecast {stamp}")
        for result in valid:
            arrays = []
            for key in ("sum_sep", "count_sep", "sum_liu", "count_liu"):
                a = np.asarray(result[key], dtype=np.float64)
                if a.ndim != 3 or a.shape[2] > args.days:
                    raise ValueError(f"Unexpected {key} shape {a.shape} for {stamp}")
                padded = np.zeros(a.shape[:2] + (args.days,), dtype=np.float64)
                padded[..., :a.shape[2]] = np.nan_to_num(a, nan=0.0)
                arrays.append(padded)
            if totals is None:
                totals = arrays
            else:
                for total, array in zip(totals, arrays):
                    if total.shape != array.shape:
                        raise ValueError(f"Incompatible spatial grid for {stamp}")
                    total += array
            all_results.append(result)
        used.append(stamp)
    sum_sep, count_sep, sum_liu, count_liu = totals
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sep = np.divide(sum_sep, count_sep, out=np.full_like(sum_sep, np.nan), where=count_sep > 0)
        mean_liu = np.divide(sum_liu, count_liu, out=np.full_like(sum_liu, np.nan), where=count_liu > 0)
    out_dir = ROOT / "data" / "AdvectionRK4"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"Lagrangian_scores_weighted_FCST_GLOENS_{args.start}_{args.end}.nc"
    log = Logger("GLOENS three-month aggregation").run(int(cfg["config_base"]["loglevel"]))
    save_netcdf(mean_sep, count_sep, mean_liu, str(outfile), log)
    image_dir = ROOT / "imgs" / "AdvectionRK4"
    image_dir.mkdir(parents=True, exist_ok=True)
    config = dict(cfg["config_base"], SYSTEM="GLOENS", leadtime="FCST",
                  DATE_START=args.start[:4]+"-"+args.start[4:6]+"-"+args.start[6:],
                  DATE_END=args.end[:4]+"-"+args.end[4:6]+"-"+args.end[6:])
    plot_scores(str(image_dir) + "/", str(outfile), config, log,
                list_days=[1, 3, 5, 7, 10, 15, 20, 25, 27, 28])
    print(f"Aggregated {len(used)} forecast initializations: {', '.join(used)}", flush=True)
    print(f"Final cumulative scores: {outfile}", flush=True)
    print(f"Final plots: {image_dir}", flush=True)
    print("Counts by forecast lead day:", count_sep.sum(axis=(0, 1)).astype(int).tolist(), flush=True)


if __name__ == "__main__":
    main()
