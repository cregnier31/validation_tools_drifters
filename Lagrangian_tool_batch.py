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
from time import perf_counter
import gc
sys.path.append("src/")

# Import custom functions from local module
from utils import StringRegion
from cli import parse_args
from lagrangian_workflow import process_one_day
from lagrangian_postprocessing import aggregate_and_save_results, read_traj_compute_scores
from plot_functions import plot_scores, compute_and_plot_histograms
from Logger import Logger
#------------------------------------------
# Read information from the configuration file
#------------------------------------------
config_object = ConfigParser()
config_object.read("config_drifters.ini")
config_info = config_object['CONFIG_BASE']
#MODELS = config_info["MODELS"].split(',')
LON_MIN = int(config_info["lon_min"])
LON_MAX = int(config_info["lon_max"])
LAT_MIN = int(config_info["lat_min"])
LAT_MAX = int(config_info["lat_max"])
DEPTH = int(config_info["depth"])
resolution = 2
MONTH_YEAR = config_info["MONTH_YEAR"]
PROJECT = config_info["PROJECT"]
USERNAME = config_info["username"]
DIR_WORK = config_info["dir_work"]
DIR_IN = config_info["dir_in"]
COPERNICUS_DRIFTER= config_info["copernicus_drifter"]
NB_REPEATS = int(config_info["nb_repeats"])
ADVECTION_DIFFUSION_TYPE = config_info["AdvectionDiffusion_type"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
N_MONTHS = int(config_info["N_MONTHS"])
DRIFT_PATH = config_info["DRIFT_PATH"]
dir_out = './data/'
dir_imgs = './imgs/'
#------------------------------------------
# parse arguments and define outputs
#------------------------------------------
DATE_START, DATE_END, leadtime, SYSTEM = parse_args()
#print(DATE_START,DATE_END)
#DATE_START = datetime.strptime(DATE_START, "%Y%m%d")
#DATE_END  = datetime.strptime(DATE_END, "%Y%m%d")
STR_DATE_START = DATE_START.strftime("%Y-%m-%d")
STR_DATE_END = DATE_END.strftime("%Y-%m-%d")
start_days = pd.date_range(STR_DATE_START, STR_DATE_END, freq='D')
DATE_END_PLUS_DRIFT_TIME =  DATE_END+dt.timedelta(days=DRIFT_TIME+1)
STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')
T_DRIFT = 10
all_results = {}
STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
STR_DATE_START = DATE_START.strftime('%Y%m%d')
STR_DATE_END   = DATE_END_PLUS_DRIFT_TIME.strftime('%Y%m%d')
EXP_NAME = f'Parcels_Drift-{leadtime}_{STR_DATE_START}-{STR_DATE_END}'
DIR_OUT=f"{DIR_WORK}DRIFT_{SYSTEM}/{PROJECT}/{EXP_NAME}/ZONE_{STR_REGION}/"
pickle_file = os.path.join(DIR_OUT, f"{EXP_NAME}_{STR_DATE_START}-{STR_DATE_END}_drift-observations.pkl")

log = Logger(f"Lagrangian_tool {STR_DATE_START} - {STR_DATE_END} {SYSTEM}").run(int(config_info['loglevel']))

#------------------------------------------
# Compute lagrangian trajectories
#------------------------------------------
njobs = 8
log.info(f"Start Lagrangian_tool {STR_DATE_START} - {STR_DATE_END} {SYSTEM}")
log.info("Running parallel computation...")
t0 = perf_counter()
batch_size = 16   # adjust (5–20 usually optimal)

all_results = []

log.info(f"Start Lagrangian_tool {STR_DATE_START} - {STR_DATE_END} {SYSTEM}")

t0_total = perf_counter()

for i in range(0, len(start_days), batch_size):

    batch_days = start_days[i:i + batch_size]

    log.info(f"Processing batch {i} -> {i + len(batch_days)} / {len(start_days)}")

    t0 = perf_counter()

    results = Parallel(n_jobs=njobs)(
        delayed(process_one_day)(
            day, leadtime, DIR_OUT,
            T_DRIFT, LON_MIN,
            LON_MAX, LAT_MIN, LAT_MAX,
            SYSTEM, DIR_IN, DIR_WORK,
            DEPTH, ADVECTION_DIFFUSION_TYPE,
            NB_REPEATS, STR_REGION,
            COPERNICUS_DRIFTER, DRIFT_TIME,
            DRIFT_PATH
        )
        for day in batch_days
    )

    all_results.extend(results)

    log.info(
        f"Batch done in {perf_counter() - t0:.1f}s | "
        f"Total collected = {len(all_results)}"
    )

    # 🔴 CRITICAL: force memory release between batches
    gc.collect()

log.info(f"All batches finished in {perf_counter() - t0_total:.1f}s")

#------------------------------------------
# Compute lagrangian scores
#------------------------------------------
log.info("Running lagrangian scores...")
t0 = perf_counter()

if len(start_days) > njobs:
    results_stats = Parallel(n_jobs=njobs)(
        delayed(read_traj_compute_scores)(
            DIR_OUT, day,
            SYSTEM, T_DRIFT,
            COPERNICUS_DRIFTER,
            all_results[ind_day]
        )
        for ind_day, day in enumerate(start_days)
    )
else:
    results_stats = []
    for ind_day, day in enumerate(start_days):
        results_stats.append(
            read_traj_compute_scores(
                DIR_OUT, day,
                SYSTEM, T_DRIFT,
                COPERNICUS_DRIFTER,
                all_results[ind_day]
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
pickle_file = (
            f"{dir_out}/results_lagrangian_{SYSTEM}_"
                f"{leadtime}_{STR_DATE_START}_{STR_DATE_END}.pkl"
                )

with open(pickle_file, "wb") as f:
        pickle.dump(results_stats, f)

        log.info(f"Results saved to {pickle_file}")


#------------------------------------------
# Compute final scores
#------------------------------------------
log.info("Compute final scores")
outfile = (
    f"{dir_out}Lagrangian_scores_"
    f"{leadtime}_{SYSTEM}_"
    f"{STR_DATE_START}_"
    f"{STR_DATE_END}.nc"
)
mean_sep_total, \
count_sep_total, \
mean_liu_total, \
count_liu_total = aggregate_and_save_results(
    results_stats,
    outfile,
    log,
    res=resolution
)

#------------------------------------------
# Compute and plot Histograms
#------------------------------------------
log.info("Compute and plot histograms")
compute_and_plot_histograms(results_stats,
                            dir_imgs,
                            SYSTEM,
                            STR_DATE_START,
                            STR_DATE_END,
                            log,
                            days=[1, 3, 5, 7, 10])

#------------------------------------------
# Plot final scores
#------------------------------------------
log.info("Plot final scores")
list_days = [1, 3, 5, 7]
plot_scores(dir_imgs,
            outfile,
            STR_DATE_START,
            STR_DATE_END,
            SYSTEM,
            log,
            list_days=list_days)
