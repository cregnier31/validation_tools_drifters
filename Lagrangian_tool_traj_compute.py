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

# Read information from the configuration file

config_object = ConfigParser()
config_object.read("config_base.ini")
config_info = config_object['CONFIG_BASE']
MODELS = config_info["MODELS"].split(',')
LON_MIN = int(config_info["lon_min"])
LON_MAX = int(config_info["lon_max"])
LAT_MIN = int(config_info["lat_min"])
LAT_MAX = int(config_info["lat_max"])
DEPTH = int(config_info["depth"])
DATE1 = config_info["DATE_start"]
DATE2 = config_info["DATE_end"]
PROJECT = config_info["PROJECT"]
EXP_NAME = config_info["EXP_NAME"]
USERNAME = config_info["username"]
DIR_WORK = config_info["dir_work"]
DIR_IN = config_info["dir_in"]
REMOTE = bool(config_info["remote"])
COPERNICUS_DRIFTER = bool(config_info["copernicus_drifter"])
NB_REPEATS = int(config_info["nb_repeats"])
ADVECTION_DIFFUSION_TYPE = config_info["AdvectionDiffusion_type"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
R = 6378.137

# Print configuration information

print("MODELS : ",MODELS)
print("Region : ",LON_MIN,", ",LON_MAX,", ",LAT_MIN,", ",LAT_MAX)
print("From : ",DATE1," to ",DATE2)

# Add a directory to the Python sys path

sys.path.append("src/")

# Import custom functions from local module

from Drift_Tools import ReturnUVlist, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote

from copernicus_functions import Retrieve_data_from_copernicusmarine_datastore,Retrieve_drifter_from_copernicusmarine_datastore,Calculate_model_traj_remote
from Functions import Retrieve_drifter_local,Calculate_speed_at_svp_location,Taylor_diagram,current_rose,Retrieve_data_local,Calculate_model_traj

# Create an output directory path

STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
DIR_OUT=DIR_WORK+"DRIFT/"+PROJECT+'/'+EXP_NAME+'/ZONE_'+STR_REGION+'/'
print("Dir out : ",DIR_OUT)

# Retrieve the starting date and the ending date

DD1=np.int32(DATE1[:2])
MM1=np.int32(DATE1[2:4])
YYYY1=np.int32(DATE1[4:])
DD2=np.int32(DATE2[:2])
MM2=np.int32(DATE2[2:4])
YYYY2=np.int32(DATE2[4:])
DATE_START = datetime.date(YYYY1, MM1, DD1)
STR_DATE_START= DATE_START.strftime('%Y-%m-%d')
DATE_START_MIN1 = DATE_START-datetime.timedelta(days=1)
DATE_END=datetime.date(YYYY2, MM2, DD2)
STR_DATE_END=DATE_START.strftime('%Y-%m-%d')
DATE_END_PLUS1 = DATE_END+datetime.timedelta(days=1)
DATE_END_PLUS_DRIFT_TIME =  DATE_END+datetime.timedelta(days=DRIFT_TIME+1)
STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')
T_DRIFT = (DATE_END_PLUS_DRIFT_TIME-DATE_START).days






# Importing data directly from "https://data.marine.copernicus.eu/products"

if REMOTE:

    # Demanding the password to have access to the data

    PASSWORD = getpass.getpass('Enter your CMEMS password: ')

    # Retrieve current from MODELS at DEPTH for the time and area zone

    data,colors = Retrieve_data_from_copernicusmarine_datastore(MODELS,USERNAME,PASSWORD,DEPTH,LON_MIN,LON_MAX,
                                                                LAT_MIN,LAT_MAX,STR_DATE_START,STR_DATE_END_PLUS_DRIFT_TIME)
    
    # Retrieve drifters for the time and area zone

    traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_from_copernicusmarine_datastore(USERNAME,PASSWORD,DIR_WORK,DIR_OUT,STR_DATE_START,STR_DATE_END_PLUS_DRIFT_TIME,T_DRIFT,STR_REGION,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END_PLUS_DRIFT_TIME)
    nb_points = T_DRIFT*24
    Calculate_model_traj_remote(MODELS,DIR_OUT,nb_points,traj_drifter_lon,traj_drifter_lat,traj_drifter_id,DATE_START,DRIFT_TIME,data,ADVECTION_DIFFUSION_TYPE,NB_REPEATS)


else:
    # Retrieve drifters for the time and area zone 

    if COPERNICUS_DRIFTER:
        PASSWORD = getpass.getpass('Enter your CMEMS password: ')

        traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_from_copernicusmarine_datastore(USERNAME,PASSWORD,DIR_WORK,DIR_OUT,STR_DATE_START,STR_DATE_END_PLUS_DRIFT_TIME,T_DRIFT,STR_REGION,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END_PLUS_DRIFT_TIME)
        nb_points = T_DRIFT*24

       
    else : 
    # Retrieve locally stored drifters trajectory

        traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_local(DIR_OUT,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END_PLUS_DRIFT_TIME,STR_DATE_START,STR_DATE_END_PLUS_DRIFT_TIME,T_DRIFT,YYYY1,STR_REGION)
        nb_points = T_DRIFT*4+1 
    
    # Retrieve NETCDF files for each MODELS, for the right period

    u_files_mod,v_files_mod,var_u_mod,var_v_mod,fname_prefix,meshmask = Retrieve_data_local(MODELS,DATE_START,DATE_END_PLUS_DRIFT_TIME,DIR_OUT,DIR_IN,DIR_WORK,DEPTH)
    

    Calculate_model_traj(MODELS,DIR_OUT,nb_points,traj_drifter_lon,traj_drifter_lat,traj_drifter_id,DATE_START,DRIFT_TIME,COPERNICUS_DRIFTER,u_files_mod,v_files_mod,var_u_mod,var_v_mod,meshmask,ADVECTION_DIFFUSION_TYPE,NB_REPEATS)

        
    







