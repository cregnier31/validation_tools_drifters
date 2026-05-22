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

sys.path.append("src/")

# Import custom functions from local module
from utils import StringRegion
from cli import parse_args
from lagrangian_workflow import process_one_day
from lagrangian_postprocessing import aggregate_and_save_results
from plot_functions import plot_scores, compute_and_plot_histograms 
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
DATE_START = datetime.strptime(DATE_START, "%Y%m%d")
STR_DATE_START = DATE_START.strftime("%Y-%m-%d")
DATE_END  = datetime.strptime(DATE_END, "%Y%m%d")
STR_DATE_END = DATE_END.strftime("%Y-%m-%d")
start_days = pd.date_range(STR_DATE_START, STR_DATE_END, freq='D')
DATE_END_PLUS_DRIFT_TIME =  DATE_END+dt.timedelta(days=DRIFT_TIME+1)
STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')
T_DRIFT = 10
print(f"{STR_DATE_START} - {STR_DATE_END}")
#SYSTEM = 'GLO12'#'PSY4','GLO12']
all_results = {}
print(SYSTEM)
STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
STR_DATE_START = DATE_START.strftime('%Y%m%d')
STR_DATE_END   = DATE_END_PLUS_DRIFT_TIME.strftime('%Y%m%d')
EXP_NAME = f'fidji_parcels_Drift-{leadtime}_{STR_DATE_START}-{STR_DATE_END}'
DIR_OUT=f"{DIR_WORK}DRIFT_{SYSTEM}/{PROJECT}/{EXP_NAME}/ZONE_{STR_REGION}/"
pickle_file = os.path.join(DIR_OUT, f"{EXP_NAME}_{STR_DATE_START}-{STR_DATE_END}_drift-observations.pkl")

#------------------------------------------
# Compute lagrangian trajectories and scores
#------------------------------------------
njobs = 4
print("Running parallel computation...")
results = Parallel(n_jobs=njobs)(   
    delayed(process_one_day)(day, leadtime, DIR_OUT,\
                             T_DRIFT, LON_MIN, \
                             LON_MAX,LAT_MIN,LAT_MAX,\
                             SYSTEM, DIR_IN, DIR_WORK,\
                             DEPTH,ADVECTION_DIFFUSION_TYPE,\
                             NB_REPEATS, STR_REGION,\
                             COPERNICUS_DRIFTER, DRIFT_TIME,\
                             DRIFT_PATH
                            )
    for day in start_days
)
#------------------------------------------
# Compute Histograms
#------------------------------------------
compute_and_plot_histograms(results,
                            dir_imgs,
                            SYSTEM,
                            f"{STR_DATE_START}_{STR_DATE_END}",
                            days=[1, 3, 5, 7, 10])

#------------------------------------------
# Compute final scores
#------------------------------------------
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
    results,
    outfile,
    res=resolution
)

#------------------------------------------
# Plot final scores
#------------------------------------------

list_days = [1, 3, 5, 7]
plot_scores(dir_imgs,
            outfile,
            list_days=list_days)

