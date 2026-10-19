#!/usr/bin/env python3
"""GLOENS Zarr forecast validation across a period, with one cumulative score pass.

Example: python Lagrangian_tool_GLOENS.py --year 2024
"""
import argparse
import csv
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import s3fs
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import build_runtime_config
from lagrangian_workflow import process_one_day
from lagrangian_postprocessing import read_traj_compute_scores, aggregate_and_save_results
from plot_functions import plot_scores, compute_and_plot_histograms


def discover_cycles(config, first, last):
    """Find real GLOENS forecast initializations, rather than assuming daily files."""
    model = config["model_params"]["GLOENS_FCST"]
    prefix = model["fname_prefix"]
    pattern = re.compile(re.escape(prefix) + r"(\d{8})-\d{8}_3DU-uo_fcst_R(\d{8})\.zarr$")
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": model["endpoint_url"]})
    dates = set()
    try:
        for item in fs.ls(model["bucket_name"]):
            match = pattern.search(str(item))
            if match and match.group(1) == match.group(2):
                day = datetime.strptime(match.group(1), "%Y%m%d")
                if first <= day <= last:
                    dates.add(day)
    finally:
        fs.close()
    return sorted(dates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    first = datetime(args.year, 1, 1)
    last = datetime(args.year, 12, 31)
    with (ROOT / "cfg/config_drifters.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cycles = discover_cycles(config, first, last)
    if not cycles:
        raise RuntimeError(f"No GLOENS Zarr forecast cycles discovered for {args.year}")

    base_args = ("FCST", "GLOENS", "AdvectionRK4", "COPERNICUS_DAILY", "parcels")
    # Annual runtime provides the final output filename and plot configuration.
    annual = build_runtime_config((first, last, *base_args), str(ROOT / "cfg/config_drifters.yaml"))
    log = annual["log"]
    log.info("Found %d GLOENS forecast initialization dates", len(cycles))
    output = ROOT / "outputs" / f"gloens_{args.year}0101_{args.year}1231"
    output.mkdir(parents=True, exist_ok=True)

    results_stats = []
    succeeded, failed = [], []
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "status", "error"])
        for day in cycles:
            stamp = day.strftime("%Y%m%d")
            log.info("Processing GLOENS forecast initialization %s", stamp)
            try:
                # Per-cycle runtime preserves the original trajectory/observation paths.
                runtime = build_runtime_config(
                    (day, day, *base_args), str(ROOT / "cfg/config_drifters.yaml")
                )
                result = process_one_day(day, runtime)
                stats = read_traj_compute_scores(
                    day, runtime["config_info"], result, filtered=True
                )
                if stats is None:
                    raise ValueError("No valid drifter scores")
                results_stats.append(stats)
                succeeded.append(stamp)
                writer.writerow([stamp, "PASS", ""])
                log.info("Finished %s (%d cumulative forecast cycles)", stamp, len(results_stats))
            except Exception as exc:
                failed.append(stamp)
                writer.writerow([stamp, "FAIL", str(exc)])
                log.exception("Failed forecast initialization %s", stamp)
                if not args.continue_on_error:
                    raise
            stream.flush()

    if not results_stats:
        raise RuntimeError("No valid forecast cycles; annual aggregation cannot proceed")

    # Same single aggregation path as Lagrangian_tool.py.
    # The list contains the per-cycle sum/count arrays for all successful Zarr forecasts.
    log.info("Aggregating %d forecast cycles into annual lead-time statistics", len(results_stats))
    with open(annual["pickle_file"], "wb") as f:
        pickle.dump(results_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    aggregate_and_save_results(results_stats, annual["stats_out"], log)

    # Plot once, from cumulative results.
    plot_scores(annual["dir_out_imgs"], annual["stats_out"],
                annual["config_info"], log, list_days=[1, 3, 5, 10])
    compute_and_plot_histograms(
        results_stats, annual["dir_out_imgs"], annual["config_info"],
        log, days=[1, 3, 5, 7, 10]
    )
    log.info("Completed %d cycles; failed %d; annual NetCDF: %s",
             len(succeeded), len(failed), annual["stats_out"])


if __name__ == "__main__":
    main()
