#!/usr/bin/env python3
"""Run Parcels validation for a selected system and inclusive date range."""
import argparse
import csv
import pickle
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import build_runtime_config
from lagrangian_workflow import process_one_day
from lagrangian_postprocessing import read_traj_compute_scores, aggregate_and_save_results
from plot_functions import plot_scores, compute_and_plot_histograms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date1", required=True, help="YYYYMMDD")
    parser.add_argument("--date2", required=True, help="YYYYMMDD (inclusive)")
    parser.add_argument("--sys", required=True, dest="system", help="System in config_drifters.yaml")
    parser.add_argument("--lead", default="FCST", choices=("FCST", "HDCT"))
    parser.add_argument("--scheme", default="AdvectionRK4",
                        choices=("AdvectionRK4", "AdvectionEE", "K_smag"))
    parser.add_argument("--typeobs", default="COPERNICUS_DAILY")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    first = datetime.strptime(args.date1, "%Y%m%d")
    last = datetime.strptime(args.date2, "%Y%m%d")
    if last < first:
        parser.error("--date2 must not precede --date1")
    base = (args.lead, args.system, args.scheme, args.typeobs, "parcels")
    config_path = str(ROOT / "cfg/config_drifters.yaml")
    runtime = build_runtime_config((first, last, *base), config_path)
    log = runtime["log"]
    if args.system.upper() == "GLOENS" and args.lead == "FCST":
        from Lagrangian_tool_GLOENS import discover_cycles
        import yaml
        with open(config_path, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        days = discover_cycles(cfg, first, last)
    else:
        days = runtime["start_days"]
    if not days:
        raise RuntimeError("No initialization dates found in selected period")
    summary_path = ROOT / "outputs" / (
        f"parcels_{args.system}_{args.date1}_{args.date2}_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_stats = []
    failures = 0
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "status", "error"])
        for day in days:
            stamp = day.strftime("%Y%m%d")
            try:
                # Isolate each cycle's output paths while retaining annual aggregation.
                cycle_runtime = build_runtime_config((day, day, *base), config_path)
                result = process_one_day(day, cycle_runtime)
                if result is None:
                    raise ValueError("No valid drifter observations")
                stats = read_traj_compute_scores(
                    day, cycle_runtime["config_info"], result, filtered=True)
                if stats is None:
                    raise ValueError("No valid trajectory scores")
                results_stats.append(stats)
                writer.writerow([stamp, "PASS", ""])
                print(f"[{args.system}] {stamp}: OK", flush=True)
            except Exception as exc:
                failures += 1
                writer.writerow([stamp, "FAIL", str(exc)])
                print(f"[{args.system}] {stamp}: FAIL - {exc}", flush=True)
                log.exception("Failed initialization %s", stamp)
                if not args.continue_on_error:
                    raise
            stream.flush()
    print(f"[{args.system}] Summary: {len(results_stats)} OK, {failures} FAIL; {summary_path}",
          flush=True)
    if not results_stats:
        raise RuntimeError("No successful cycles to aggregate")
    Path(runtime["pickle_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(runtime["pickle_file"], "wb") as stream:
        pickle.dump(results_stats, stream, protocol=pickle.HIGHEST_PROTOCOL)
    aggregate_and_save_results(results_stats, runtime["stats_out"], log)
    if not args.no_plots:
        plot_scores(runtime["dir_out_imgs"], runtime["stats_out"],
                    runtime["config_info"], log, list_days=[1, 3, 5, 10])
        compute_and_plot_histograms(results_stats, runtime["dir_out_imgs"],
                                    runtime["config_info"], log,
                                    days=[1, 3, 5, 7, 10])


if __name__ == "__main__":
    main()
