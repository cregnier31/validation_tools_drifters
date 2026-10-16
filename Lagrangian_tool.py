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
from plot_functions import plot_scores, compute_and_plot_histograms
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
if njobs == 1:
    results = [process_one_day(day, runtime) for day in runtime["start_days"]]
else:
    results = Parallel(n_jobs=njobs)(
        delayed(process_one_day)(day, runtime)
        for day in runtime["start_days"]
    )

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

if len(runtime["start_days"]) > njobs:
    results_stats = Parallel(n_jobs=njobs)(
        delayed(read_traj_compute_scores)(
                    day,
                    runtime['config_info'],
                    results[ind_day],
                    filtered=filtered,
        )
        for ind_day, day in enumerate(runtime["start_days"])
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
