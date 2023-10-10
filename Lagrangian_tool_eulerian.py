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
R = 6378.137

# Print configuration information

print("MODELS : ",MODELS)
print("Region : ",LON_MIN,", ",LON_MAX,", ",LAT_MIN,", ",LAT_MAX)
print("From : ",DATE1," to ",DATE2)

# Add a directory to the Python sys path

sys.path.append("src/")

# Import custom functions from local module

from Drift_Tools import ReturnUVlist, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote

from copernicus_functions import Retrieve_data_from_copernicusmarine_datastore,Retrieve_drifter_from_copernicusmarine_datastore,Calculate_speed_at_svp_location_remote
from Functions import Retrieve_drifter_local,Calculate_speed_at_svp_location,Taylor_diagram,current_rose

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
STR_DATE_END=DATE_END.strftime('%Y-%m-%d')
DATE_END_PLUS1 = DATE_END+datetime.timedelta(days=1)
T_DRIFT = (DATE_END-DATE_START).days






# Importing data directly from "https://data.marine.copernicus.eu/products"

if REMOTE:

    # Demanding the password to have access to the data

    PASSWORD = getpass.getpass('Enter your CMEMS password: ')

    # Retrieve current from MODELS at DEPTH for the time and area zone

    data,colors = Retrieve_data_from_copernicusmarine_datastore(MODELS,USERNAME,PASSWORD,DEPTH,LON_MIN,LON_MAX,
                                                                LAT_MIN,LAT_MAX,STR_DATE_START,STR_DATE_END,tay_diag = True)
    
    # Retrieve drifters for the time and area zone

    traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_from_copernicusmarine_datastore(USERNAME,PASSWORD,DIR_WORK,DIR_OUT,STR_DATE_START,STR_DATE_END,T_DRIFT,STR_REGION,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END)

    # Calcul One day speed of SVP

    lon_uv_svp = np.array([np.mean(traj_drifter_lon[:,i*24:i*24+24],axis = 1) for i in range(int(len(traj_drifter_lon[0])/24))]).T
    lat_uv_svp = np.array([np.mean(traj_drifter_lat[:,i*24:i*24+24],axis = 1) for i in range(int(len(traj_drifter_lat[0])/24))]).T
    u_svp = ((np.radians(lon_uv_svp[:,1:]-lon_uv_svp[:,:-1]))*np.cos(np.radians(lat_uv_svp[:,:-1]))* R)*(1000/(24*3600))
    v_svp = ((np.radians(lat_uv_svp[:,1:]-lat_uv_svp[:,:-1] ))*R)*(1000/(24*3600))

    # Calculate the velocity of the MODELS on drifters positions

    U_model, V_model = Calculate_speed_at_svp_location_remote(lon_uv_svp,lat_uv_svp,traj_drifter_date,traj_drifter_time,traj_drifter_id,data)


    # Calcul (norm std and corr coeff R) for the Taylor diagram

    u_corrcoef = np.array([[np.corrcoef(U_model.T[:,i,j],u_svp[i])[0,1] for i in range(len(u_svp))] for j in range (len(MODELS))])
    v_corrcoef = np.array([[np.corrcoef(V_model.T[:,i,j],v_svp[i])[0,1] for i in range(len(v_svp))] for j in range (len(MODELS))])

    u_std = np.array([[np.std(U_model.T[:,i,j]) for i in range(len(u_svp))] for j in range (len(MODELS))])
    v_std = np.array([[np.std(V_model.T[:,i,j]) for i in range(len(v_svp))] for j in range (len(MODELS))])

else:
    # Retrieve current from MODELS at DEPTH for the time and area zone 
    if COPERNICUS_DRIFTER:
        PASSWORD = getpass.getpass('Enter your CMEMS password: ')

        traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_from_copernicusmarine_datastore(USERNAME,PASSWORD,DIR_WORK,DIR_OUT,STR_DATE_START,STR_DATE_END,T_DRIFT,STR_REGION,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END,tay_diag = True)
        lon_uv_svp = np.array([np.mean(traj_drifter_lon[:,i*24:i*24+24],axis = 1) for i in range(int(len(traj_drifter_lon[0])/24))]).T
        lat_uv_svp = np.array([np.mean(traj_drifter_lat[:,i*24:i*24+24],axis = 1) for i in range(int(len(traj_drifter_lat[0])/24))]).T
       
    else : 
        traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_local(DIR_OUT,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END,STR_DATE_START,STR_DATE_END,T_DRIFT,YYYY1,STR_REGION)

        lon_uv_svp = np.array([np.mean(traj_drifter_lon[:,i*4:i*4+4],axis = 1) for i in range(int(len(traj_drifter_lon[0])/4))]).T
        lat_uv_svp = np.array([np.mean(traj_drifter_lat[:,i*4:i*4+4],axis = 1) for i in range(int(len(traj_drifter_lat[0])/4))]).T
    u_svp = ((np.radians(lon_uv_svp[:,1:]-lon_uv_svp[:,:-1]))*np.cos(np.radians(lat_uv_svp[:,:-1]))* R)*(1000/(24*3600))
    v_svp = ((np.radians(lat_uv_svp[:,1:]-lat_uv_svp[:,:-1] ))*R)*(1000/(24*3600))

    # Calculate the velocity of the MODELS on drifters positions
 
    U_model,V_model,colors = Calculate_speed_at_svp_location(MODELS,DATE_START_MIN1,DATE_END_PLUS1,DIR_OUT,DIR_IN,lon_uv_svp,lat_uv_svp,traj_drifter_date,DEPTH,COPERNICUS_DRIFTER)

    # Calcul (norm std and corr coeff R) for the Taylor diagram
    u_corrcoef = np.array([[np.corrcoef(U_model.T[i,:,j],u_svp[i])[0,1] for i in range(len(u_svp))] for j in range (len(MODELS))])
    v_corrcoef = np.array([[np.corrcoef(V_model.T[i,:,j],v_svp[i])[0,1] for i in range(len(v_svp))] for j in range (len(MODELS))])

    u_std = np.array([[np.std(U_model.T[i,:,j]) for i in range(len(u_svp))] for j in range (len(MODELS))])
    v_std = np.array([[np.std(V_model.T[i,:,j]) for i in range(len(v_svp))] for j in range (len(MODELS))])

# Plotting the different figures

DIR_PLOT = DIR_OUT+'PLOT/'
if not os.path.exists(DIR_PLOT):
    print('\ncreate folder : ', DIR_PLOT)
    # Create directory where to store results:
    os.makedirs(DIR_PLOT)

DIR_PLOT_U = DIR_PLOT+'Taylor_diagram_U_SVP_15m_'+STR_DATE_START+'_'+STR_DATE_END+'.png'
DIR_PLOT_V = DIR_PLOT+'Taylor_diagram_V_SVP_15m_'+STR_DATE_START+'_'+STR_DATE_END+'.png'

# Taylor Diagram

Taylor_diagram(DIR_PLOT_U,u_svp,u_corrcoef,u_std,colors,MODELS)
Taylor_diagram(DIR_PLOT_V,v_svp,v_corrcoef,v_std,colors,MODELS)

# Current rose

for i in range (len(MODELS)):
    u_mod = U_model[i]
    v_mod = V_model[i]
    if REMOTE:
        u_dft = u_svp
        v_dft = v_svp
    else:
        u_dft = u_svp.T
        v_dft = v_svp.T

    current_rose(u_mod,v_mod,fname =DIR_PLOT+ 'rose_mod'+STR_DATE_START+'_'+STR_DATE_END+'_'+MODELS[i],bins_classes=[0,0.05,0.1,0.15,0.2,0.25,0.3],title = MODELS[i],pmax = 8)
    current_rose(u_mod-u_dft,v_mod-v_dft,fname =DIR_PLOT+ 'rose_mod'+STR_DATE_START+'_'+STR_DATE_END+'_Diff_'+MODELS[i]+'_svp',bins_classes=[0,0.05,0.1,0.15,0.2,0.25,0.3],title = 'Diff '+MODELS[i] + ' - SVP',diff = True,pmax = 8)
current_rose(u_dft,v_dft,fname =DIR_PLOT+ 'rose_dft'+STR_DATE_START+'_'+STR_DATE_END,bins_classes=[0,0.05,0.1,0.15,0.2,0.25,0.3],title = 'SVP drifters',pmax = 8 )








