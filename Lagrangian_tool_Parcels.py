#!/usr/bin/env python3
"""Run Parcels validation for a selected system and inclusive date range."""
import argparse
import csv
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import build_runtime_config
from lagrangian_workflow import process_one_day
from lagrangian_postprocessing import read_traj_compute_scores, aggregate_and_save_results
from plot_functions import plot_scores, compute_and_plot_histograms


def audit_liu(results, output_dir, log):
    """Audit per-lead Liu formula, changing sample counts, and large errors."""
    import numpy as np
    out = Path(output_dir) / "Liu_consistency_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["day", "valid_pairs", "cycles", "mean_skill",
               "mean_endpoint_sep_km", "mean_cumulative_sep_km",
               "mean_observed_path_km", "mean_denominator_km",
               "mean_normalized_error", "fraction_skill_gt_08",
               "fraction_sep_gt_200km", "high_skill_large_sep_count",
               "formula_mismatches", "max_formula_difference"]
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for day in (1, 3, 5, 9, 12, 15, 20, 28):
            chunks = []
            for result in results:
                if not result or "liu_diagnostics_daily" not in result:
                    continue
                diag = result["liu_diagnostics_daily"]
                keys = ("normalized_error", "instantaneous_separation_km",
                        "cumulative_separation_km", "observed_path_km",
                        "normalization_distance_km")
                try:
                    arrays = [np.asarray(result["liu_skill_daily"], dtype=float)]
                    arrays += [np.asarray(diag[key], dtype=float) for key in keys]
                    arrays = [a.reshape(1, -1) if a.ndim == 1 else a
                              for a in arrays]
                    if any(a.ndim != 2 or a.shape != arrays[0].shape
                           or a.shape[1] < day for a in arrays):
                        continue
                    block = np.stack([a[:, day - 1] for a in arrays], axis=1)
                    block = block[np.isfinite(block).all(axis=1)]
                    if block.size:
                        chunks.append(block)
                except (KeyError, ValueError):
                    continue
            row = dict.fromkeys(columns, "")
            row["day"] = day
            row["cycles"] = len(chunks)
            row["valid_pairs"] = sum(len(chunk) for chunk in chunks)
            if chunks:
                data = np.concatenate(chunks)
                skill, ratio, sep, cumulative, path, denominator = data.T
                diff = np.abs(skill - np.maximum(0, 1 - ratio))
                row.update(mean_skill=float(np.mean(skill)),
                           mean_endpoint_sep_km=float(np.mean(sep)),
                           mean_cumulative_sep_km=float(np.mean(cumulative)),
                           mean_observed_path_km=float(np.mean(path)),
                           mean_denominator_km=float(np.mean(denominator)),
                           mean_normalized_error=float(np.mean(ratio)),
                           fraction_skill_gt_08=float(np.mean(skill > .8)),
                           fraction_sep_gt_200km=float(np.mean(sep > 200)),
                           high_skill_large_sep_count=int(np.sum((skill > .8) & (sep > 200))),
                           formula_mismatches=int(np.sum(diff > 1e-6)),
                           max_formula_difference=float(np.max(diff)))
            writer.writerow(row)
            log.info("Liu audit day %d: %s valid pairs, %s formula mismatches",
                     day, row["valid_pairs"], row["formula_mismatches"])
    print(f"Liu consistency audit saved: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date1", required=True, help="YYYYMMDD")
    parser.add_argument("--date2", required=True, help="YYYYMMDD (inclusive)")
    parser.add_argument("--sys", required=True, dest="system", help="System in config_drifters.yaml")
    parser.add_argument("--lead", default="FCST", choices=("FCST", "HDCT"))
    parser.add_argument("--scheme", default="AdvectionRK4",
                        choices=("AdvectionRK4", "AdvectionEE", "K_smag"))
    parser.add_argument("--typeobs", default="COPERNICUS_DAILY")
    parser.add_argument("--drift-days", type=int, default=None,
                        help="Override drift duration in days (GLOENS FCST defaults to 28)")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--delete-failed-particles", action="store_true",
                        help="Retire Parcels particles with grid-search errors and continue the cycle")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--audit-liu", action="store_true",
                        help="Audit existing annual Liu metrics without running Parcels")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate figures from existing annual pickle and NetCDF statistics")
    parser.add_argument("--remove-old-trajectories", action="store_true",
                        help="Delete only existing model trajectory outputs for selected cycles before rerunning")
    args = parser.parse_args()
    if args.delete_failed_particles:
        import os
        os.environ["GLOENS_DELETE_FAILED_PARTICLES"] = "1"
    if args.drift_days is not None and args.drift_days < 1:
        parser.error("--drift-days must be a positive integer")
    first = datetime.strptime(args.date1, "%Y%m%d")
    last = datetime.strptime(args.date2, "%Y%m%d")
    if last < first:
        parser.error("--date2 must not precede --date1")
    base = (args.lead, args.system, args.scheme, args.typeobs, "parcels")
    config_path = str(ROOT / "cfg/config_drifters.yaml")
    runtime = build_runtime_config((first, last, *base, False), config_path, drift_days=args.drift_days)
    log = runtime["log"]
    if args.audit_liu:
        pickle_file = Path(runtime["pickle_file"])
        if not pickle_file.is_file():
            parser.error(f"Annual results pickle missing: {pickle_file}")
        with pickle_file.open("rb") as stream:
            audit_results = pickle.load(stream)
        audit_liu(audit_results, runtime["dir_out_imgs"], log)
        return
    if args.plots_only:
        if args.no_plots or args.remove_old_trajectories:
            parser.error("--plots-only cannot be combined with --no-plots or --remove-old-trajectories")
        pickle_file = Path(runtime["pickle_file"])
        stats_file = Path(runtime["stats_out"])
        if not pickle_file.is_file() or not stats_file.is_file():
            parser.error(
                f"--plots-only requires existing annual outputs: "
                f"pickle={pickle_file} (exists={pickle_file.is_file()}), "
                f"NetCDF={stats_file} (exists={stats_file.is_file()})"
            )
        with pickle_file.open("rb") as stream:
            results_stats = pickle.load(stream)
        if not results_stats:
            parser.error(f"Annual results pickle contains no successful cycles: {pickle_file}")
        log.info("Plots only: %d cycles from %s; NetCDF %s",
                 len(results_stats), pickle_file, stats_file)
        plot_scores(runtime["dir_out_imgs"], str(stats_file),
                    runtime["config_info"], log)
        compute_and_plot_histograms(
            results_stats, runtime["dir_out_imgs"], runtime["config_info"], log,
            days=([1, 3, 5, 7, 9] if args.system.upper() in ("GLOENS", "GLOW")
                  else [1, 3, 5, 7, 10]))
        if args.system.upper() in ("GLOENS", "GLOW"):
            from plot_functions import plot_extended_trajectory_diagnostics
            lead_groups = (((1, 3, 5, 7, 9),)
                           if args.system.upper() == "GLOW"
                           else ((1, 3, 5, 7), (9, 12, 15, 18), (24, 28)))
            plot_extended_trajectory_diagnostics(
                results_stats, runtime["dir_out_imgs"], runtime["config_info"], log,
                lead_groups=lead_groups)
        print(f"[{args.system}] Plots regenerated from existing annual statistics",
              flush=True)
        return
    if args.system.upper() in ("GLOENS", "GLOW") and args.lead == "FCST":
        import yaml
        with open(config_path, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        if args.system.upper() == "GLOENS":
            from Lagrangian_tool_GLOENS import discover_cycles
            days = discover_cycles(cfg, first, last)
        else:
            from glow_discovery import discover_glow_cycles
            days = discover_glow_cycles(cfg, first, last)
    else:
        days = runtime["start_days"]
    if len(days) == 0:
        raise RuntimeError("No initialization dates found in selected period")
    duration = runtime["config_info"]["drift_time"]
    summary_path = ROOT / "outputs" / (
        f"parcels_{args.system}_drift{duration}_{args.date1}_{args.date2}_summary.csv")
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
                cycle_runtime = build_runtime_config((day, day, *base, False), config_path, drift_days=args.drift_days)
                # Let process_one_day remove the exact tagged trajectory it
                # resolves (including GLOENS_OUTPUT_TAG and Parcels 4 Parquet
                # directories). The former pre-cleanup guessed an untagged
                # filename and silently left stale tagged output in place.
                cycle_runtime["config_info"]["remove_old_particules"] = (
                    args.remove_old_trajectories
                )
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
                    runtime["config_info"], log)
        compute_and_plot_histograms(results_stats, runtime["dir_out_imgs"],
                                    runtime["config_info"], log,
                                    days=([1, 3, 5, 7, 9]
                                          if runtime["config_info"]["SYSTEM"].upper() == "GLOENS"
                                          else [1, 3, 5, 7, 9]
                                          if runtime["config_info"]["SYSTEM"].upper() == "GLOW"
                                          else [1, 3, 5, 7, 10]))
        if runtime["config_info"]["SYSTEM"].upper() in ("GLOENS", "GLOW"):
            from plot_functions import plot_extended_trajectory_diagnostics
            lead_groups = (((1, 3, 5, 7, 9),)
                           if runtime["config_info"]["SYSTEM"].upper() == "GLOW"
                           else ((1, 3, 5, 7), (9, 12, 15, 18), (24, 28)))
            plot_extended_trajectory_diagnostics(
                results_stats, runtime["dir_out_imgs"], runtime["config_info"], log,
                lead_groups=lead_groups)


if __name__ == "__main__":
    main()
