#!/usr/bin/env python
# coding: utf-8

__author__ = "cregnier March 2026"

# Import necessary libraries

import json
import numpy as np
import datetime
import xarray as xr
import os, glob,sys
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from configparser import ConfigParser
import getpass
import ftputil
from tqdm import tqdm
#from lib_pyt.plot.class_plot_map import PlotMapCartopy
from joblib import Parallel, delayed
import pandas as pd

sys.path.append("src/")

# Import custom functions from local module
#, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote 

#from copernicus_functions import Retrieve_drifter_from_copernicusmarine_datastore_fidji, \
#                                 Retrieve_drifter_from_fidji, retrieve_drifters_per_day, \
#                                 retrieve_drifters_for_one_day, Retrieve_data_from_copernicusmarine_datastore, \
#                                 retrieve_drifters_for_one_day_traj, Retrieve_drifter_from_copernicusmarine_datastore, \
#                                 Calculate_model_traj_remote
#from Functions import Retrieve_drifter_local,Calculate_speed_at_svp_location,Taylor_diagram,current_rose,Retrieve_data_local,Calculate_model_traj,Retrieve_data_local_FCST,Calculate_model_traj_FCST 

from Drift_Tools import StringRegion
#from copernicus_functions import retrieve_drifters_per_day, retrieve_drifters_for_one_day_traj
from Functions_new import Retrieve_data_local_new, convert_id_to_char_array, process_one_day, save_netcdf
from Functions import Retrieve_traj_files, Retrieve_drifters_traj
from Drift_exec import run_parcels_Drift


# Read informati]#on from the configuration file
FCT_vect = [True]#   True]#False]
FCT_vals = ['FCT']# ['FCT']#,'ANA']
FCT_vals = ['ANA']# ['FCT']#,'ANA']
DT_vect  = [9]#1,3,5]# [7]#1,3,5,7]#,9]

config_object = ConfigParser()

config_object.read("config_fidji.ini")

config_info = config_object['CONFIG_BASE']
MODELS = config_info["MODELS"].split(',')
LON_MIN = int(config_info["lon_min"])
LON_MAX = int(config_info["lon_max"])
LAT_MIN = int(config_info["lat_min"])
LAT_MAX = int(config_info["lat_max"])
DEPTH = int(config_info["depth"])
#var_u_mod = str(config_info["var_u"])
#var_v_mod = str(config_info["var_v"])

MONTH_YEAR = config_info["MONTH_YEAR"]
PROJECT = config_info["PROJECT"]
USERNAME = config_info["username"]
DIR_WORK = config_info["dir_work"]
DIR_IN = config_info["dir_in"]
COPERNICUS_DRIFTER= config_info["copernicus_drifter"]
NB_REPEATS = int(config_info["nb_repeats"])
ADVECTION_DIFFUSION_TYPE = config_info["AdvectionDiffusion_type"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
FORECAST = bool(config_info["Forecast"])
R = 6378.137
N_MONTHS = int(config_info["N_MONTHS"])


#Retrieving date of the Wednesday in the month
YEAR = np.int32(MONTH_YEAR[2:])
MONTH = np.int32(MONTH_YEAR[:2])
date_var = datetime.date(YEAR,MONTH,1)

end_year = YEAR + (MONTH - 1 + N_MONTHS) // 12
end_month = (MONTH - 1 + N_MONTHS) % 12 + 1
end_date = datetime.date(end_year, end_month, 1)
WED = [] 

while date_var < end_date:
    if date_var.weekday() == 2:  # Wednesday
        WED.append(date_var)
    date_var += datetime.timedelta(days=1)

# Retrieve the starting date and the ending date
DATE_START = WED[0]
STR_DATE_START= WED[0].strftime('%Y-%m-%d')
T= WED[0].strftime('%Y-%m-%d')
DATE_END= WED[-1] 
STR_DATE_END=DATE_END.strftime('%Y-%m-%d')
DATE_END_PLUS_DRIFT_TIME =  DATE_END+datetime.timedelta(days=DRIFT_TIME+1)
STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')

print(f"{STR_DATE_START} - {STR_DATE_END}")
T_DRIFT = 10

MODELS = ['GLO12']#'PSY4','GLO12']
all_results = {}

start_days = pd.date_range(STR_DATE_START, STR_DATE_END, freq='D')
#start_days = pd.date_range(STR_DATE_START, STR_DATE_START, freq='D')
STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
STR_DATE_START = DATE_START.strftime('%Y%m%d')
STR_DATE_END   = DATE_END_PLUS_DRIFT_TIME.strftime('%Y%m%d')
EXP_NAME = f'fidji_parcels_NbmonthsDrift-{N_MONTHS}_{STR_DATE_START}-{STR_DATE_END}'
DIR_OUT=f"{DIR_WORK}DRIFT_{MODELS[0]}/{PROJECT}/{EXP_NAME}/ZONE_{STR_REGION}/"
pickle_file = os.path.join(DIR_OUT, f"{EXP_NAME}_{STR_DATE_START}-{STR_DATE_END}_drift-observations.pkl")
print(pickle_file)
# Compute lagrangian trajectories and scores
njobs = 4
print("Running parallel computation...")
results = Parallel(n_jobs=njobs)(   
    delayed(process_one_day)(day, DIR_OUT,\
                             T_DRIFT, LON_MIN, \
                             LON_MAX,LAT_MIN,LAT_MAX,\
                             MODELS,DIR_IN, DIR_WORK,\
                             DEPTH,ADVECTION_DIFFUSION_TYPE,\
                             NB_REPEATS, STR_REGION,\
                             COPERNICUS_DRIFTER, DRIFT_TIME,
                            )
    for day in start_days
)

# Compute final scores
# initialize
sum_sep_total = None
count_sep_total = None
sum_liu_total = None
count_liu_total = None

for res in results:
    #sum_sep, count_sep, sum_liu, count_liu = res
    print(type(res))
    sum_sep = res['sum_sep']
    count_sep = res['count_sep']    
    sum_liu = res['sum_liu']
    count_liu = res['count_liu']    
    #sum_sep, count_sep = res
    print(type(sum_sep))
    print(np.nanmean(sum_sep))
    if sum_sep_total is None:
        sum_sep_total = sum_sep.copy()
        count_sep_total = count_sep.copy()
        sum_liu_total = sum_liu.copy()
        count_liu_total = count_liu.copy()
    else:
        sum_sep_total += sum_sep
        count_sep_total += count_sep
        sum_liu_total += sum_liu
        count_liu_total += count_liu


# Compute final distances
mean_sep_total = sum_sep_total / count_sep_total
mean_sep_total[count_sep_total == 0] = np.nan

print(np.shape(mean_sep_total))
mean_liu_total = sum_liu_total / count_liu_total
mean_liu_total[count_liu_total == 0] = np.nan

# Save netcdf file
outfile = f"test_lagrangian_{STR_DATE_START}_{STR_DATE_END}.nc"
save_netcdf(mean_sep_total, count_sep_total, mean_liu_total, outfile, res=2)
#ds = xr.Dataset(
#    {
#        "mean_separation_distance": (("lat", "lon", "lead_time"), mean_sep_total),
#        "count_separation": (("lat", "lon", "lead_time"), count_sep_total),
#       # "mean_liu_score": (("lat", "lon", "leadtime"), mean_liu_total),
#      #  "count_liu": (("lat", "lon", "leadtime"), count_liu_total),
#    },
#    coords={
#        "lat": lat_centers,
#        "lon": lon_centers,
#        "lead_time": leadtime,
#    }
#)
#
#ds["mean_separation_distance"].attrs = {
#    "units": "km",
#    "long_name": "Mean separation distance"
#}
#
#encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
#
#ds.to_netcdf(outfile, encoding=encoding)

print(f"Saved: {outfile}")

sys.exit(1)

traj_lon,traj_lat,traj_time,traj_IDS = Retrieve_traj_files(MODELS,DIR_OUT)
##
## fout = f"Drifters_{str_region}_{str_date1}_{str_date2}.csv"
##
print('retrieve ok')
print('--------------')
print(traj_IDS)
print('--------------')
print(traj_time)
print('--------------')
sys.exit(1)
## To test
fout_driftersZone = 'Drifters_'+STR_REGION+'_'+STR_DATE_START+'_'+STR_DATE_END_PLUS_DRIFT_TIME+'.csv'
#fout_driftersZone = 'Drifters_'+STR_REGION+'_'+STR_DATE_START+'_'+STR_DATE_END_PLUS_DRIFT_TIME+'.csv'
fout_driftersZone = 'Drifters_'+STR_REGION+'_'+'*.csv'
print(fout_driftersZone)
dir_drifters = DIR_OUT+'/DRIFTERS/'

print(dir_drifters+fout_driftersZone)
sys.exit(1)

outfile2 = glob.glob(dir_drifters+fout_driftersZone)[0]
print(dir_drifters)
print(outfile2)
traj_drifter_lon_temp,traj_drifter_lat_temp,traj_drifter_date_temp,traj_drifter_time_temp,traj_drifter_id_temp,nb_drifters_temp = Retrieve_drifters_traj(outfile2,T_DRIFT,COPERNICUS_DRIFTER)
plot_traj_drift(traj_lon,
                traj_lat,
                STR_DATE_START,
                STR_DATE_END)




if nb_drifters_temp  != len(traj_IDS):
    drifter_traj_ids = np.unique(traj_IDS)
    traj_drifter_lon  = []
    traj_drifter_lat  = []
    traj_drifter_date = []
    traj_drifter_time = []
    traj_drifter_id   = []
    for mm in range(nb_drifters_temp):
        if traj_drifter_id_temp[mm,0] in drifter_traj_ids:
            traj_drifter_lon.append(traj_drifter_lon_temp[mm,:])
            traj_drifter_lat.append(traj_drifter_lat_temp[mm,:])
            traj_drifter_date.append(traj_drifter_date_temp[mm,:])
            traj_drifter_time.append(traj_drifter_time_temp[mm,:])
            traj_drifter_id.append(traj_drifter_id_temp[mm,:])
    traj_drifter_lon = np.array(traj_drifter_lon)
    traj_drifter_lat = np.array(traj_drifter_lat)
    traj_drifter_date = np.array(traj_drifter_date)
    traj_drifter_time = np.array(traj_drifter_time)
    traj_drifter_id = np.array(traj_drifter_id)
else:
    traj_drifter_lon  = traj_drifter_lon_temp
    traj_drifter_lat  = traj_drifter_lat_temp
    traj_drifter_date = traj_drifter_date_temp
    traj_drifter_time = traj_drifter_time_temp
    traj_drifter_id   = traj_drifter_id_temp



# Plot Drifter's trajectories



sys.exit(1)

#all_distances = []
for day in start_days:
    print(day)
    DATE_START = day 
    DATE_END_PLUS_DRIFT_TIME = day + pd.Timedelta(days=T_DRIFT)
    STR_DATE_START = DATE_START.strftime('%Y%m%d')
    STR_DATE_END   = DATE_END_PLUS_DRIFT_TIME.strftime('%Y%m%d')
    EXP_NAME = f'fidji_{MONTH_YEAR}_{N_MONTHS}_{STR_DATE_START}'
    DIR_OUT=f"{DIR_WORK}DRIFT_{MODELS[0]}/{PROJECT}/{EXP_NAME}/ZONE_{STR_REGION}/"
    print("Dir out : ",DIR_OUT)

    # Retrieve model data
    u_files_mod_temp,v_files_mod_temp,\
        var_u_mod,var_v_mod,fname_prefix,meshmask = \
             Retrieve_data_local_new([MODELS[0]+'_HDCT'],
                                     day,
                                     DATE_END_PLUS_DRIFT_TIME,
                                     DIR_OUT,DIR_IN,
                                     DIR_WORK,
                                     DEPTH)
    print(u_files_mod_temp)
    print(v_files_mod_temp)

    print("------------------------------")
    # Retrieve drifters data
    trajs_obs = retrieve_drifters_for_one_day_traj(
        day=day,
        T_DRIFT=T_DRIFT,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX
    )
    if trajs_obs is None:
        print(f"{day.date()} → no valid drifters")
        continue
    else:
        print(f"Number of valid drifters {len(trajs_obs)}")
    
    # ======================================================
    # 🔹 INITIAL CONDITIONS (for Parcels)
    # ======================================================
    traj_drifter_lon_start = np.array([traj['lon'][0] for traj in trajs_obs])
    traj_drifter_lat_start = np.array([traj['lat'][0] for traj in trajs_obs])
    ids = [traj['id'] for traj in trajs_obs]
    traj_drifter_id_start = convert_id_to_char_array(ids, string_length=8)
    print(len(traj_drifter_lon_start))
    print('---------------------')
    print(len(traj_drifter_lat_start))
    print('---------------------')
    print(len(traj_drifter_id_start))
    print(type(day))
    str_date_sim = day.strftime('%Y%m%d') # + datetime.timedelta(days=int(T_DRIFT))
   # str_date_sim = date_sim.strftime('%Y%M%D')
    #f_traj1 = dir_work_mod + "Trajectories_"+MODELS[0]+'_'+str_date_sim+'_'+str(T_DRIFT)+'_at_SVPloc.zarr'
    #f_traj2 = dir_work_mod + "Trajectories_"+MODELS[0]+'_'+str_date_sim+'_'+str(T_DRIFT)+'_at_SVPloc_and_around.zarr'
    dir_work_mod = f"{DIR_OUT}{day}"
    f_traj = dir_work_mod + "Trajectories_"+MODELS[0]+'_'+str_date_sim+'_'+str(T_DRIFT)+'_at_SVPloc_smag.zarr'
    print(f_traj)
    sys.exit(1)
    if not os.path.exists(dir_work_mod):
        print('\ncreate folder : ', dir_work_mod)
        # Create directory where to store results:
        os.makedirs(dir_work_mod)

    #f_traj4 = dir_work_mod + "Trajectories_"+MODELS[0]+'_'+str_date_sim+'_'+str(T_DRIFT)+'_at_SVPloc_M1.zarr'
    
    print(ADVECTION_DIFFUSION_TYPE)

    if ADVECTION_DIFFUSION_TYPE == "K_smag":
        run_parcels_Drift(MODELS[0], u_files_mod_temp, var_u_mod, v_files_mod_temp, var_v_mod,\
                          meshmask,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                          DRIFT_TIME,f_traj, radius_cloud=0, seeding_resolution=0,time_period=False,\
                          AdvectionDiffusion_type=ADVECTION_DIFFUSION_TYPE,nb_repeats = NB_REPEATS)
    if ADVECTION_DIFFUSION_TYPE == "AdvectionDiffusionM1":
        run_parcels_Drift(MODELS[0], u_files_mod_temp, var_u_mod, v_files_mod_temp, var_v_mod,\
                          meshmask,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                          DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                          AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = NB_REPEATS)
