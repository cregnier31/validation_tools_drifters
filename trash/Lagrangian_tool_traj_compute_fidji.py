#!/usr/bin/env python
# coding: utf-8

__author__ = "Bassam C."

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

sys.path.append("src/")

# Import custom functions from local module
from Drift_Tools import ReturnUVlist, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote 

from copernicus_functions import Retrieve_data_from_copernicusmarine_datastore,Retrieve_drifter_from_copernicusmarine_datastore,Calculate_model_traj_remote,Retrieve_drifter_from_copernicusmarine_datastore_fidji

#from copernicus_functions_new import Retrieve_drifter_from_copernicusmarine_datastore_fidji

from Functions import Retrieve_drifter_local,Calculate_speed_at_svp_location,Taylor_diagram,current_rose,Retrieve_data_local,Calculate_model_traj,Retrieve_data_local_FCST,Calculate_model_traj_FCST 



# Read informati]#on from the configuration file
FCT_vect = [True]#False]#True]#   True]#False]
FCT_vals = ['FCT']#'ANA']# ['FCT']#,'ANA']
DT_vect  = [1]#1,3,5]# [7]#1,3,5,7]#,9]



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
USERNAME = config_info["username"]
DIR_WORK = config_info["dir_work"]
DIR_IN = config_info["dir_in"]
COPERNICUS_DRIFTER= config_info["copernicus_drifter"]
NB_REPEATS = int(config_info["nb_repeats"])
ADVECTION_DIFFUSION_TYPE = config_info["AdvectionDiffusion_type"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
FORECAST = bool(config_info["Forecast"])
R = 6378.137
N_MONTHES = int(config_info["N_MONTHES"])






#Retrieving date of the Wednesday in the month
YEAR = np.int32(MONTH_YEAR[2:])
#print(YEAR)
MONTH = np.int32(MONTH_YEAR[:2])
#print(MONTH)
date_var = datetime.date(YEAR,MONTH,1)
WED = [] 
while date_var.month <= MONTH + N_MONTHES-1:
    print(date_var)
    if date_var.weekday()==2: 
        WED.append(date_var)
        #print(WED)
    date_var += datetime.timedelta(days=1)
    if date_var.year == YEAR +1:
            break


# Retrieve the starting date and the ending date
DATE_START = WED[0]
STR_DATE_START= WED[0].strftime('%Y-%m-%d')
T= WED[0].strftime('%Y-%m-%d')
DATE_START_MIN1 = DATE_START-datetime.timedelta(days=1)
DATE_END= WED[-1] 
STR_DATE_END=DATE_END.strftime('%Y-%m-%d')
DATE_END_PLUS1 = DATE_END+datetime.timedelta(days=1)


lllll = []
#
MODELS = ['GLO12']#'PSY4','GLO12']
for ii in range(len(FCT_vect)):
    FORECAST = FCT_vect[ii]
    for jj in range(len(DT_vect)):
        DRIFT_TIME = DT_vect[jj]
        EXP_NAME = f'fidji_{MONTH_YEAR}_{N_MONTHES}M_DT{DT_vect[jj]}_{FCT_vals[ii]}'
        print(EXP_NAME)# configuration information
        lllll.append(EXP_NAME)#
        print("MODELS : ",MODELS)
        print("Region : ",LON_MIN,", ",LON_MAX,", ",LAT_MIN,", ",LAT_MAX)

        # Add a directory to the Python sys path

        sys.path.append("src/")

        # Create an output directory path

        STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
        DIR_OUT=DIR_WORK+"DRIFT/"+PROJECT+'/'+EXP_NAME+'/ZONE_'+STR_REGION+'/'
        print("Dir out : ",DIR_OUT)
        
        # retrieve starting date and ending date continue
        DATE_END_PLUS_DRIFT_TIME =  DATE_END+datetime.timedelta(days=DRIFT_TIME+1)
        #DATE_END_PLUS_DRIFT_TIME =  DATE_END+datetime.timedelta(days=DRIFT_TIME)
        STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')
        T_DRIFT = (DATE_END_PLUS_DRIFT_TIME-DATE_START).days
        #T_DRIFT = (DATE_END_PLUS_DRIFT_TIME-DATE_START).days
        print(DATE_END_PLUS_DRIFT_TIME)
        print(DATE_END)
        print(DATE_START)        
        PASSWORD = None
        traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id = Retrieve_drifter_from_copernicusmarine_datastore_fidji(USERNAME,PASSWORD,DIR_WORK,DIR_OUT,STR_DATE_START,STR_DATE_END_PLUS_DRIFT_TIME,T_DRIFT,STR_REGION,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,DATE_START,DATE_END_PLUS_DRIFT_TIME)#,DRIFT_TIME )
        print(f'{traj_drifter_date=}')
        print(f'{traj_drifter_time=}')
        

        print('HERE')
        

        nb_points = (DRIFT_TIME+1)*24

        print(nb_points)
        exit()
        
        if FORECAST == True: 
            if len(MODELS) > 1: 
                u_files_mod_temp,v_files_mod_temp,var_u_mod,var_v_mod,fname_prefix,meshmask = Retrieve_data_local_FCST(MODELS[1],DATE_START,DATE_END_PLUS_DRIFT_TIME,DIR_OUT,DIR_IN,DIR_WORK,DEPTH)
                taille_max = max(len(x) for x in u_files_mod_temp)
                u_files_mod = [x + [x[-1]] * (taille_max - len(x)) for x in u_files_mod_temp]
                taille_max = max(len(x) for x in v_files_mod_temp)
                v_files_mod = [x + [x[-1]] * (taille_max - len(x)) for x in v_files_mod_temp]
                for DT_ST in WED:            
                    #if True: #(np.shape(DT_ST.strftime('%Y-%m-%d'))!= () ):
                    print(traj_drifter_date)
                    print(DT_ST.strftime('%Y-%m-%d'))
                    indice_start = np.where(traj_drifter_date==DT_ST.strftime('%Y-%m-%d'))[1][0]
                    for file in u_files_mod[0]:
                        print(file[70:78])
                        indice_start_mod = np.where(file[69:77]==DT_ST.strftime('%Y%m%d'))[0]
                        if len(indice_start_mod)> 0 :                                        
                            #break    
                            Calculate_model_traj_FCST(MODELS[1],DIR_OUT,nb_points,traj_drifter_lon[:,indice_start:],traj_drifter_lat[:,indice_start:],traj_drifter_id[:,indice_start:],DT_ST,DRIFT_TIME,COPERNICUS_DRIFTER,np.array(u_files_mod)[:,indice_start_mod[0]:],np.array(v_files_mod)[:,indice_start_mod[0]:],var_u_mod,var_v_mod,meshmask,ADVECTION_DIFFUSION_TYPE,NB_REPEATS)        
                    
            u_files_mod_temp,v_files_mod_temp,var_u_mod,var_v_mod,fname_prefix,meshmask =Retrieve_data_local_FCST([MODELS[0]],DATE_START,DATE_END_PLUS_DRIFT_TIME,DIR_OUT,DIR_IN,DIR_WORK,DEPTH)        
            taille_max = max(len(x) for x in u_files_mod_temp)
            u_files_mod = [x + [x[-1]] * (taille_max - len(x)) for x in u_files_mod_temp]
            taille_max = max(len(x) for x in v_files_mod_temp)
            v_files_mod = [x + [x[-1]] * (taille_max - len(x)) for x in v_files_mod_temp]
            print(u_files_mod_temp)

            for DT_ST in WED:
                print(DT_ST.strftime('%Y-%m-%d'))
                print(traj_drifter_date)
                indice_start = np.where(traj_drifter_date==DT_ST.strftime('%Y-%m-%d'))[1][0]
                for file in u_files_mod[0]:
                    print('test here')
                    print(file[69:77])
                    print(DT_ST.strftime('%Y-%m-%d'))
                    indice_start_mod = np.where(file[69:77]==DT_ST.strftime('%Y%m%d'))[0] 
                    if len(indice_start_mod)> 0 :
                        print('processing of CCalculate_model_traj_FCST')
                        #exit()
                        Calculate_model_traj_FCST(MODELS[0] ,DIR_OUT,nb_points,traj_drifter_lon[:,indice_start:],traj_drifter_lat[:,indice_start:],traj_drifter_id[:,indice_start:],DT_ST,DRIFT_TIME,COPERNICUS_DRIFTER,np.array(u_files_mod)[:,indice_start_mod[0]:],np.array(v_files_mod)[:,indice_start_mod[0]:],var_u_mod,var_v_mod,meshmask,ADVECTION_DIFFUSION_TYPE,NB_REPEATS)
        

        else:
            for kk in range(len(MODELS)):
                print(f'kk   =   {kk}')
                u_files_mod_temp,v_files_mod_temp,var_u_mod,var_v_mod,fname_prefix,meshmask = Retrieve_data_local([MODELS[kk]+'_NWCT'],DATE_START,DATE_END_PLUS_DRIFT_TIME,DIR_OUT,DIR_IN,DIR_WORK,DEPTH)
                print(u_files_mod_temp)
                taille_max = max(len(x) for x in u_files_mod_temp)
                u_files_mod = [x + [x[-1]] * (taille_max - len(x)) for x in u_files_mod_temp] 
                taille_max = max(len(x) for x in v_files_mod_temp)
                v_files_mod = [x + [x[-1]] * (taille_max - len(x)) for x in v_files_mod_temp]

                for DT_ST in WED:
                    print(DT_ST.strftime('%Y-%m-%d'))
                    indice_start = np.where(traj_drifter_date==DT_ST.strftime('%Y-%m-%d'))[1][0]
                    for file in u_files_mod[0]:
                        #print(file[70:78])
                        if kk ==1:
                            ff = file[68:76]
                        else:
                            ff = file[68:76]
                        print(ff)
                        indice_start_mod = np.where(ff==DT_ST.strftime('%Y%m%d'))[0]
                        if len(indice_start_mod) > 0 :
                            #break
                            Calculate_model_traj([MODELS[kk]],DIR_OUT,nb_points,traj_drifter_lon[:,indice_start:],traj_drifter_lat[:,indice_start:],traj_drifter_id[:,indice_start:],DT_ST,DRIFT_TIME,COPERNICUS_DRIFTER,np.array(u_files_mod)[:,indice_start_mod[0]:],np.array(v_files_mod)[:,indice_start_mod[0]:],var_u_mod,var_v_mod,meshmask,ADVECTION_DIFFUSION_TYPE,NB_REPEATS)
            
        print(EXP_NAME )



