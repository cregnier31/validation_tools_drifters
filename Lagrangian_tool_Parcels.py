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


def audit_persistence(results, output_dir, log):
    """Compare model and stationary-persistence skill on identical valid pairs."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    days = (1, 3, 5, 9, 12, 15, 20, 28)
    keys = ("liu_skill_daily", "persistence_liu_skill_daily",
            "liu_diagnostics_daily", "persistence_sep_daily")
    if not any(result and all(key in result for key in keys)
               for result in results):
        log.warning("Persistence diagnostics missing from annual pickle; "
                    "recompute trajectory scores with the updated postprocessor.")
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "Liu_persistence_comparison.csv"
    columns = ("day", "paired_count", "model_mean_liu", "persistence_mean_liu",
               "model_mean_sep_km", "persistence_mean_sep_km",
               "fraction_model_better_liu", "fraction_model_lower_sep")
    rows = []
    for day in days:
        blocks = []
        for result in results:
            if not result or not all(key in result for key in keys):
                continue
            try:
                values = [
                    np.asarray(result["liu_skill_daily"], dtype=float),
                    np.asarray(result["persistence_liu_skill_daily"], dtype=float),
                    np.asarray(result["liu_diagnostics_daily"]["instantaneous_separation_km"], dtype=float),
                    np.asarray(result["persistence_sep_daily"], dtype=float)]
                values = [v.reshape(1, -1) if v.ndim == 1 else v for v in values]
                if (any(v.ndim != 2 or v.shape != values[0].shape
                        or v.shape[1] < day for v in values)):
                    continue
                block = np.column_stack([v[:, day - 1] for v in values])
                block = block[np.isfinite(block).all(axis=1)]
                if len(block):
                    blocks.append(block)
            except (KeyError, ValueError, IndexError):
                continue
        row = dict.fromkeys(columns, "")
        row["day"] = day
        row["paired_count"] = sum(len(b) for b in blocks)
        if blocks:
            model_skill, persist_skill, model_sep, persist_sep = np.concatenate(blocks).T
            row.update(model_mean_liu=float(np.mean(model_skill)),
                       persistence_mean_liu=float(np.mean(persist_skill)),
                       model_mean_sep_km=float(np.mean(model_sep)),
                       persistence_mean_sep_km=float(np.mean(persist_sep)),
                       fraction_model_better_liu=float(np.mean(model_skill > persist_skill)),
                       fraction_model_lower_sep=float(np.mean(model_sep < persist_sep)))
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, pairs, title, ylabel in (
        (axes[0], ("model_mean_liu", "persistence_mean_liu"),
         "Relative trajectory skill", "Liu skill"),
        (axes[1], ("model_mean_sep_km", "persistence_mean_sep_km"),
         "Endpoint separation", "Distance (km)")):
        for key, label in zip(pairs, ("GLOENS", "Stationary persistence")):
            ax.plot(days, [float(row[key]) if row[key] != "" else np.nan
                           for row in rows], "-o", label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Forecast lead (days)")
        ax.set_xticks(days)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    image_path = output / "Liu_persistence_comparison.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    print(f"Persistence comparison saved: {csv_path} and {image_path}", flush=True)


def audit_liu_survivors(results, output_dir, log):
    """Compare all valid trajectories with the fixed day-28 survivor cohort."""
    import numpy as np
    out = Path(output_dir) / "Liu_survivor_cohort_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    days = (1, 3, 5, 9, 12, 15, 20, 28)
    columns = ("day", "cohort", "valid_pairs", "contributing_cycles",
               "mean_liu_skill", "mean_endpoint_sep_km",
               "mean_observed_path_km", "mean_endpoint_error_over_path",
               "median_endpoint_error_over_path")
    rows = []
    for day in days:
        for cohort in ("all_valid", "day28_survivors"):
            samples = []
            cycles = 0
            for result in results:
                if not result:
                    continue
                try:
                    skill = np.asarray(result["liu_skill_daily"], dtype=float)
                    diag = result["liu_diagnostics_daily"]
                    sep = np.asarray(diag["instantaneous_separation_km"], dtype=float)
                    path = np.asarray(diag["observed_path_km"], dtype=float)
                    if skill.ndim == 1:
                        skill = skill[np.newaxis, :]
                    if sep.ndim == 1:
                        sep = sep[np.newaxis, :]
                    if path.ndim == 1:
                        path = path[np.newaxis, :]
                    if (skill.ndim != 2 or skill.shape != sep.shape
                            or skill.shape != path.shape or skill.shape[1] < day):
                        continue
                    valid = (np.isfinite(skill[:, day - 1])
                             & np.isfinite(sep[:, day - 1])
                             & np.isfinite(path[:, day - 1])
                             & (path[:, day - 1] > 0))
                    if cohort == "day28_survivors":
                        if skill.shape[1] < 28:
                            continue
                        valid &= (np.isfinite(skill[:, 27])
                                  & np.isfinite(sep[:, 27])
                                  & np.isfinite(path[:, 27])
                                  & (path[:, 27] > 0))
                    if np.any(valid):
                        cycles += 1
                        samples.append(np.column_stack((
                            skill[valid, day - 1],
                            sep[valid, day - 1],
                            path[valid, day - 1])))
                except (KeyError, ValueError, IndexError):
                    continue
            row = dict.fromkeys(columns, "")
            row.update(day=day, cohort=cohort,
                       valid_pairs=sum(len(a) for a in samples),
                       contributing_cycles=cycles)
            if samples:
                skill, sep, path = np.concatenate(samples).T
                endpoint_ratio = sep / path
                row.update(mean_liu_skill=float(np.mean(skill)),
                           mean_endpoint_sep_km=float(np.mean(sep)),
                           mean_observed_path_km=float(np.mean(path)),
                           mean_endpoint_error_over_path=float(np.mean(endpoint_ratio)),
                           median_endpoint_error_over_path=float(np.median(endpoint_ratio)))
            rows.append(row)
            log.info("Liu cohort day %d %s: %d valid pairs",
                     day, cohort, row["valid_pairs"])
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Liu survivor cohort audit saved: {out}", flush=True)
    # The same trajectory population is retained across every plotted lead
    # for the survivor curve, avoiding lead-dependent sample composition.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = (
        ("mean_liu_skill", "Mean Liu relative trajectory skill", "Liu skill"),
        ("mean_endpoint_sep_km", "Mean endpoint separation", "Separation (km)"),
        ("mean_endpoint_error_over_path",
         "Mean endpoint separation / observed path", "Relative endpoint error"),
        ("valid_pairs", "Valid trajectory pairs", "Number of pairs"),
    )
    for cohort, label, style in (
        ("all_valid", "All valid at each lead", "-o"),
        ("day28_survivors", "Fixed day-28 survivor cohort", "--s"),
    ):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        x = [row["day"] for row in cohort_rows]
        for ax, (key, title, ylabel) in zip(axes.flat, panels):
            y = [float(row[key]) if row[key] != "" else float("nan")
                 for row in cohort_rows]
            ax.plot(x, y, style, linewidth=1.8, markersize=4, label=label)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.set_xticks(days)
    for ax in axes[1]:
        ax.set_xlabel("Forecast lead (days)")
    axes[0, 0].set_ylim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.945), frameon=False)
    fig.suptitle("Liu score and endpoint-error diagnostics: all vs fixed cohort",
                 fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.08,
                        top=0.85, hspace=0.36, wspace=0.28)
    figure_path = out.with_name("Liu_survivor_cohort_diagnostics.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Liu survivor cohort figure saved: {figure_path}", flush=True)




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
    parser.add_argument("--pickle-file", type=Path, default=None,
                        help="Existing annual pickle path for --audit-liu or --plots-only")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Recompute scores from existing tagged Parquet and observations, without advection")
    parser.add_argument("--trajectory-tag", default="domcfg_v31",
                        help="Existing tagged trajectory suffix for --postprocess-only")
    parser.add_argument("--audit-liu", action="store_true",
                        help="Audit existing annual Liu metrics without running Parcels")
    parser.add_argument("--stats-file", type=Path, default=None,
                        help="Existing annual NetCDF path for --plots-only")
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
    if args.postprocess_only:
        import os
        import yaml
        from trajectory_io import retrieve_drifters_for_one_day_traj
        if args.audit_liu or args.plots_only or args.remove_old_trajectories:
            parser.error("--postprocess-only cannot be combined with audit, plots-only or trajectory removal")
        if args.system.upper() != "GLOENS" or args.lead != "FCST":
            parser.error("--postprocess-only currently supports GLOENS FCST only")
        with open(config_path, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        from Lagrangian_tool_GLOENS import discover_cycles
        days = discover_cycles(cfg, first, last)
        if not days:
            parser.error("No GLOENS initialization dates found")
        duration = runtime["config_info"]["drift_time"]
        tag = args.trajectory_tag
        from datetime import timedelta
        base_dir = Path(runtime["config_info"]["dir_work"]) / args.scheme / "DRIFT_GLOENS"
        cycle_specs = []
        for day in days:
            start = day.strftime("%Y%m%d")
            end = (day + timedelta(days=int(duration))).strftime("%Y%m%d")
            experiment = (f"Parcels_Drift-GLOENS_FCST_{args.scheme}_"
                          f"drift{duration}_dt15_{start}-{start}_{tag}")
            cycle_dir = base_dir / experiment / ("ZONE_" + runtime["config_info"]["STR_REGION"])
            traj_dir = cycle_dir / start
            trajectory = traj_dir / (
                f"Trajectories_GLOENS_{start}_{end}_at_SVPloc_smag_{tag}.parquet")
            if not trajectory.exists():
                raise FileNotFoundError(
                    f"Missing trajectory for {start}: {trajectory}. "
                    "No advection will be attempted.")
            cycle_specs.append((day, cycle_dir, trajectory))
        log.info("Preflight OK: %d existing trajectory files; no advection", len(cycle_specs))
        postprocessed = []
        for day, cycle_dir, trajectory in cycle_specs:
            start = day.strftime("%Y%m%d")
            end = (day + timedelta(days=int(duration))).strftime("%Y%m%d")
            cycle_runtime = build_runtime_config(
                (day, day, *base, False), config_path, drift_days=args.drift_days)
            info = cycle_runtime["config_info"]
            info["DIR_OUT"] = str(cycle_dir) + "/"
            observation = retrieve_drifters_for_one_day_traj(
                day, duration, info["lon_min"], info["lon_max"],
                info["lat_min"], info["lat_max"], info["STR_REGION"],
                start, end, info["obs_type"], info["path_data"],
                info["DIR_OUT"], log, start_hour=12)
            if observation is None:
                raise FileNotFoundError(f"No observation input for {start}")
            _, observation_csv = observation
            stats = read_traj_compute_scores(
                day, info, observation_csv, filtered=True)
            if stats is None:
                raise ValueError(f"No scores produced for {start}")
            postprocessed.append(stats)
            log.info("Postprocessed %s (%d/%d)", start, len(postprocessed), len(cycle_specs))
        output_pickle = Path(runtime["pickle_file"]).with_name(
            Path(runtime["pickle_file"]).stem + f"_{tag}_persistence.pkl")
        output_pickle.parent.mkdir(parents=True, exist_ok=True)
        with output_pickle.open("wb") as stream:
            pickle.dump(postprocessed, stream, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved postprocessed annual pickle: %s", output_pickle)
        audit_liu(postprocessed, runtime["dir_out_imgs"], log)
        audit_liu_survivors(postprocessed, runtime["dir_out_imgs"], log)
        audit_persistence(postprocessed, runtime["dir_out_imgs"], log)
        print(f"Postprocessing complete; original annual pickle untouched: {output_pickle}",
              flush=True)
        return
    if args.audit_liu:
        pickle_file = args.pickle_file if args.pickle_file is not None else Path(runtime["pickle_file"])
        if not pickle_file.is_file():
            parser.error(f"Annual results pickle missing: {pickle_file}")
        with pickle_file.open("rb") as stream:
            audit_results = pickle.load(stream)
        audit_liu(audit_results, runtime["dir_out_imgs"], log)
        audit_liu_survivors(audit_results, runtime["dir_out_imgs"], log)
        audit_persistence(audit_results, runtime["dir_out_imgs"], log)
        return
    if args.plots_only:
        if args.no_plots or args.remove_old_trajectories:
            parser.error("--plots-only cannot be combined with --no-plots or --remove-old-trajectories")
        pickle_file = args.pickle_file if args.pickle_file is not None else Path(runtime["pickle_file"])
        stats_file = args.stats_file if args.stats_file is not None else Path(runtime["stats_out"])
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
                    runtime["config_info"], log,
                    list_days=([1, 3, 5, 9, 12, 15, 20, 28]
                               if args.system.upper() == "GLOENS"
                               else [1, 3, 5, 9]))
        compute_and_plot_histograms(
            results_stats, runtime["dir_out_imgs"], runtime["config_info"], log,
            days=([1, 3, 5, 7, 9, 12, 15, 20, 28]
                  if args.system.upper() == "GLOENS"
                  else [1, 3, 5, 7, 9]))
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
                    runtime["config_info"], log,
                    list_days=([1, 3, 5, 9, 12, 15, 20, 28]
                               if runtime["config_info"]["SYSTEM"].upper() == "GLOENS"
                               else [1, 3, 5, 9]))
        compute_and_plot_histograms(results_stats, runtime["dir_out_imgs"],
                                    runtime["config_info"], log,
                                    days=([1, 3, 5, 7, 9, 12, 15, 20, 28]
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
