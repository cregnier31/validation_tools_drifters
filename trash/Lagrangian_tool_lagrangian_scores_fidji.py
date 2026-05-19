#!/usr/bin/env python
# coding: utf-8

__author__ = "Gouvenou Pierre, Van Gennip Simon"

# Import necessary libraries

import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import os, glob,sys
import pandas as pd
from configparser import ConfigParser
import getpass
import ftputil
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from scipy.stats import pearsonr
dir_data =  '/home/smer/sval001/DEV/validation_tools_drifters/static/data/data/'

cartopy.config['pre_existing_data_dir'] = dir_data

# Read information from the configuration file

config_object = ConfigParser()
config_object.read("config_fidji.ini")
config_info = config_object['CONFIG_BASE']
MODELS = config_info["MODELS"].split(',')
LON_MIN = int(config_info["lon_min"])
LON_MAX = int(config_info["lon_max"])
LAT_MIN = int(config_info["lat_min"])
LAT_MAX = int(config_info["lat_max"])
DEPTH = int(config_info["depth"])
MONTH_YEAR = config_info["MONTH_YEAR"]
PROJECT = config_info["PROJECT"]
EXP_NAME = config_info["EXP_NAME"]
USERNAME = config_info["username"]
DIR_WORK = config_info["dir_work"]
DIR_IN = config_info["dir_in"]
COPERNICUS_DRIFTER= config_info["copernicus_drifter"]
NB_REPEATS = int(config_info["nb_repeats"])
ADVECTION_DIFFUSION_TYPE = config_info["AdvectionDiffusion_type"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
FORECAST = bool(config_info["Forecast"])
N_MONTHES = int(config_info["N_MONTHES"])
print(FORECAST)
print(bool(config_info["Forecast"]))
FORECAST = True
# Print configuration information

print("MODELS : ",MODELS)
print("Region : ",LON_MIN,", ",LON_MAX,", ",LAT_MIN,", ",LAT_MAX)

# Add a directory to the Python sys path

sys.path.append("src/")

# Import custom functions from local module

from Drift_Tools import ReturnUVlist, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore2, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote

from Functions import Retrieve_traj_files,Retrieve_drifters_traj,Traj_filter_fidji,Report_card_plot,Retrieve_traj_files_no_ct

# Create an output directory path

STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
DIR_OUT=DIR_WORK+"DRIFT/"+PROJECT+'/'+EXP_NAME+'/ZONE_'+STR_REGION+'/'
print("Dir out : ",DIR_OUT)


#Retrieving date of the Wednesday in the month
YEAR = np.int32(MONTH_YEAR[2:])
MONTH = np.int32(MONTH_YEAR[:2])


date_var = datetime.date(YEAR,MONTH,1)
WED = []
while date_var.month <= MONTH + N_MONTHES-1:
    if date_var.weekday()==2:
        WED.append(date_var)
    date_var += datetime.timedelta(days=1)
    if date_var.year == YEAR +1:
        break


# Retrieve the starting date and the ending date

DATE_START = WED[0]
STR_DATE_START= WED[0].strftime('%Y-%m-%d')
DATE_START_MIN1 = DATE_START-datetime.timedelta(days=1)
DATE_END= WED[-1]
STR_DATE_END=DATE_END.strftime('%Y-%m-%d')
DATE_END_PLUS1 = DATE_END+datetime.timedelta(days=1)
DATE_END_PLUS_DRIFT_TIME =  DATE_END+datetime.timedelta(days=DRIFT_TIME+1)
STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')
T_DRIFT = (DATE_END_PLUS_DRIFT_TIME-DATE_START).days





if FORECAST:
    for index_model in range(len(MODELS)):
        if ~MODELS[index_model].endswith('FCST'):
            MODELS[index_model] = MODELS[index_model] +'_FCST'

print('FORESCAT')
print(FORECAST)
print(MODELS)



print(DIR_OUT)
traj_lon,traj_lat,traj_time = Retrieve_traj_files(MODELS,DIR_OUT)


fout_driftersZone = 'Drifters_'+STR_REGION+'_'+STR_DATE_START+'_'+STR_DATE_END_PLUS_DRIFT_TIME+'.csv'
dir_drifters = DIR_OUT+'/DRIFTERS/'


outfile2 = dir_drifters+fout_driftersZone

traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id,nb_drifters_temp = Retrieve_drifters_traj(outfile2,T_DRIFT,COPERNICUS_DRIFTER)


dir_plot = DIR_OUT+'PLOT/'


if not os.path.exists(dir_plot):
    print('\ncreate folder : ', dir_plot)
    # Create directory where to store results:
    os.makedirs(dir_plot)

fig = plt.figure(figsize = (9,5))
axs=plt.axes(projection=ccrs.PlateCarree())
axs.coastlines()
axs.add_feature(cfeature.LAND)
gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False
fig.suptitle('Trajectory of the SVP drifters',fontsize = 12)

axs.plot(traj_drifter_lon.T,traj_drifter_lat.T,transform=ccrs.PlateCarree())
axs.set_xlim([LON_MIN,LON_MAX])
axs.set_ylim([LAT_MIN,LAT_MAX])
plt.savefig(dir_plot+'Traj_drifters'+STR_DATE_START+'_'+STR_DATE_END+'.png',dpi = 600)

if(np.prod(np.shape(traj_drifter_lon)) >0): 
    print(DRIFT_TIME,traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time, traj_lon,traj_lat)

    drifter_lon,drifter_lat,drifter_date,drifter_time = Traj_filter_fidji(DRIFT_TIME,traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time, traj_lon,traj_lat)


    sep_dist = SeparationDistance(drifter_lon,drifter_lat,traj_lon,traj_lat)
    liu_score = np.array([[LiuScore2(drifter_lon[i,j],drifter_lat[i,j],traj_lon[i,j],traj_lat[i,j]).T for j in range(traj_lon.shape[1])] for i in range (traj_lon.shape[0])])
    skill_score = 1-liu_score
    if True:# np.prod(np.shape(np.where(np.isnan(traj_lon)))):
        nans = np.unique(np.where(np.isnan(traj_lon))[2])
        print(nans)
        if len(nans)>0:
            traj_lon = np.delete(traj_lon,nans,axis = 2)
            traj_lat = np.delete(traj_lat,nans,axis = 2)
            drifter_lon = np.delete(drifter_lon,nans,axis = 2)
            drifter_lat = np.delete(drifter_lat,nans,axis = 2)

        nans = np.unique(np.where(np.isnan(drifter_lon))[2])
        print(nans)
        if len(nans)>0:
            traj_lon = np.delete(traj_lon,nans,axis = 2)
            traj_lat = np.delete(traj_lat,nans,axis = 2)
            drifter_lon = np.delete(drifter_lon,nans,axis = 2)
            drifter_lat = np.delete(drifter_lat,nans,axis = 2)

        inf_lon = np.unique(np.where(np.isinf(traj_lon))[2])
        inf_lat = np.unique(np.where(np.isinf(traj_lat))[2])
        print('TESTT')
        print(inf_lon)
        print(inf_lat)
        if False: #len(infs)>0:
            traj_lon = np.delete(traj_lon,infs,axis = 2)
            traj_lat = np.delete(traj_lat,infs,axis = 2)
            drifter_lon = np.delete(drifter_lon,infs,axis = 2)
            drifter_lat = np.delete(drifter_lat,infs,axis = 2)

    #print(traj_lon)
    #print(traj_lat)
    #print(drifter_lon)
    #print(drifter_lat)


    corr_lon = np.array([[[pearsonr(drifter_lon[i,j,k],traj_lon[i,j,k])[0] for k in range(traj_lon.shape[2])] for j in range(traj_lon.shape[1])] for i in range(traj_lon.shape[0])])
    corr_lat = np.array([[[pearsonr(drifter_lat[i,j,k],traj_lat[i,j,k])[0] for k in range(traj_lat.shape[2])] for j in range(traj_lat.shape[1])] for i in range(traj_lat.shape[0])])
    
    if np.prod(np.shape(sep_dist))>0:
        prop_sep_dist = np.count_nonzero(sep_dist[:,:,:,-1]<50 ,axis =(1,2) )/ (sep_dist.shape[1]*sep_dist.shape[2])
        prop_skill_score = np.count_nonzero(skill_score[:,:,:,-1]>0 ,axis =(1,2) )/ (skill_score.shape[1]*skill_score.shape[2])
        prop_corr_lon = np.count_nonzero(corr_lon>0.7 ,axis =(1,2) )/ (corr_lon.shape[1]*corr_lon.shape[2])
        prop_corr_lat = np.count_nonzero(corr_lat>0.7 ,axis =(1,2) )/ (corr_lat.shape[1]*corr_lat.shape[2])

        Report_card_plot(MODELS,prop_skill_score,prop_sep_dist,prop_corr_lon,prop_corr_lat,DIR_IN,STR_DATE_START,STR_DATE_END,dir_plot)





















