#!/usr/bin/env python
# coding: utf-8

__author__ = "cregnier April/May 2026"

# Import necessary libraries

import json
import numpy as np
from datetime import datetime
import datetime as dt
import xarray as xr
import os, glob,sys
import pandas as pd
from configparser import ConfigParser
import getpass
from joblib import Parallel, delayed
import pandas as pd
import pickle
import yaml
from time import perf_counter

sys.path.append("src/")

# Import custom functions from local module
from utils import build_runtime_config
from cli import parse_args
from lagrangian_workflow import process_one_day
from lagrangian_postprocessing import aggregate_and_save_results, read_traj_compute_scores
from plot_functions import (plot_scores, compute_and_plot_histograms,
                            plot_extended_trajectory_diagnostics)
#------------------------------------------
# parse arguments and define outputs
#------------------------------------------
runtime = build_runtime_config(
    parse_args(),
    "cfg/config_drifters.yaml"
)
log = runtime["log"]

#------------------------------------------
# Compute lagrangian trajectories
#------------------------------------------
njobs = 1 if runtime["config_info"].get("input_format") == "zarr" else 8
filtered = True
log.info(f"Start Lagrangian_tool {runtime['config_info']['PERIOD']} {runtime['config_info']['SYSTEM']}")
log.info("Running trajectory computation with n_jobs=%d", njobs)
t0 = perf_counter()
scores_only = os.environ.get("LAGRANGIAN_SCORES_ONLY") == "1"
if scores_only:
    from trajectory_io import retrieve_drifters_for_one_day_traj
    from pathlib import Path
    import csv
    import zarr

    info = runtime["config_info"]
    model = info["SYSTEM"]
    report = Path(runtime["pickle_file"]).with_name("zarr_validation_" + model + ".csv")
    report.parent.mkdir(parents=True, exist_ok=True)
    results = []
    eligible = []
    failures = []
    for day in runtime["start_days"]:
        stamp = day.strftime("%Y%m%d")
        folder = Path(info["DIR_OUT"]) / stamp
        paths = sorted(folder.glob(f"*Trajectories_{model}_*at_SVPloc*.zarr"))
        reason = None
        if len(paths) != 1:
            reason = f"expected one Zarr store; found {len(paths)}"
        else:
            try:
                root = zarr.open_group(str(paths[0]), mode="r")
                required = ("lon", "lat", "time", "label")
                missing = [key for key in required if key not in root]
                if missing:
                    reason = "missing variables: " + ", ".join(missing)
                else:
                    bad = [key for key in required
                           if root[key].attrs.get("_ARRAY_DIMENSIONS") is None]
                    shapes = [root[key].shape for key in ("lon", "lat", "time")]
                    if bad:
                        reason = "missing dimension metadata: " + ", ".join(bad)
                    elif len(set(shapes)) != 1 or len(shapes[0]) != 2:
                        reason = f"inconsistent trajectory shapes: {shapes}"
                    elif shapes[0][1] < 2:
                        reason = f"single-sample trajectory: {shapes[0]}"
                    elif root["label"].shape != (shapes[0][0],):
                        reason = "label length differs from trajectory count"
            except Exception as exc:
                reason = f"cannot inspect Zarr: {exc}"
        if reason:
            log.warning("Skipping %s: %s", stamp, reason)
            failures.append((stamp, str(paths[0]) if paths else "", reason))
            continue
        end = (day + dt.timedelta(days=int(info["drift_time"]))).strftime("%Y%m%d")
        try:
            observed = retrieve_drifters_for_one_day_traj(
                day, info["drift_time"], info["lon_min"], info["lon_max"],
                info["lat_min"], info["lat_max"], info["STR_REGION"],
                stamp, end, info["obs_type"], info["path_data"],
                info["DIR_OUT"], log,
                start_hour=12 if model.upper() == "GLOENS" and info["leadtime"] == "FCST" else 0)
            if observed is None:
                raise ValueError("no drifter observations")
            results.append(observed[1])
            eligible.append(day)
        except Exception as exc:
            log.warning("Skipping %s: observations unavailable: %s", stamp, exc)
            failures.append((stamp, str(paths[0]), f"observations: {exc}"))
    with report.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "zarr_store", "reason"])
        writer.writerows(failures)
    log.info("Zarr validation: %d eligible, %d excluded; report: %s",
             len(eligible), len(failures), report)
    if not eligible:
        raise RuntimeError("No usable trajectories; see " + str(report))
    score_days = eligible
else:
    if njobs == 1:
        results = [process_one_day(day, runtime) for day in runtime["start_days"]]
    else:
        results = Parallel(n_jobs=njobs)(
            delayed(process_one_day)(day, runtime)
            for day in runtime["start_days"]
        )
    score_days = runtime["start_days"]

traj_time = perf_counter() - t0
log.info(
    f"Lagrangian trajectories completed in "
    f"{traj_time:.1f} s ({traj_time/60:.2f} min)"
)

#------------------------------------------
# Compute lagrangian scores
#------------------------------------------
log.info("Running lagrangian scores...")
t0 = perf_counter()

if len(score_days) > njobs:
    results_stats = Parallel(n_jobs=njobs)(
        delayed(read_traj_compute_scores)(
                    day,
                    runtime['config_info'],
                    results[ind_day],
                    filtered=filtered,
        )
        for ind_day, day in enumerate(score_days)
    )
else:
    results_stats = []
    for ind_day, day in enumerate(runtime["start_days"]):
        results_stats.append(
            read_traj_compute_scores(
                day,
                runtime['config_info'],
                results[ind_day]
            )
        )

score_time = perf_counter() - t0

log.info(
    f"Lagrangian scores completed in "
    f"{score_time:.1f} s ({score_time/60:.2f} min)"
)

log.info(
    f"Total runtime: {(traj_time + score_time):.1f} s "
    f"({(traj_time + score_time)/60:.2f} min)"
)

# ==========================================================
# Save results
# ==========================================================
with open(runtime['pickle_file'], "wb") as f:
        pickle.dump(results_stats, f)

        log.info(f"Results saved to {runtime['pickle_file']}")


# Batch mode stores raw per-cycle sums/counts only; the launcher computes
# cumulative means and plots once after the entire period.
if os.environ.get("GLOENS_DEFER_PLOTS") == "1":
    log.info("Saved per-cycle sums/counts; deferring final scores and plots")
    sys.exit(0)

#------------------------------------------
# Compute final scores
#------------------------------------------
log.info("Compute final scores")
mean_sep_total, \
count_sep_total, \
mean_liu_total, \
count_liu_total = aggregate_and_save_results(
    results_stats,
    runtime['stats_out'],
    log,
)

#------------------------------------------
# Compute and plot Histograms
#------------------------------------------
log.info("Compute and plot histograms")
compute_and_plot_histograms(results_stats,
                            runtime['dir_out_imgs'],
                            runtime['config_info'],
                            log,
                            #days=[1, 3, 5])
                            days=[1, 3, 5, 7, 10])

#------------------------------------------
# Plot final scores
#------------------------------------------
log.info("Plot final scores")
list_days = [1, 3, 5, 10]
#list_days = [1, 3, 5]
log.info(f"plot_scores Stats {runtime['stats_out']}")
plot_scores(runtime['dir_out_imgs'],
            runtime['stats_out'],
            runtime['config_info'],
            log,
            list_days=list_days)

# Geographic maps and extended trajectory diagnostics for available lead days.
log.info("Plot extended trajectory diagnostics and geographic maps")
plot_extended_trajectory_diagnostics(
    results_stats,
    runtime['dir_out_imgs'],
    runtime['config_info'],
    log,
)
