#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset, date2num
#from datetime import datetime, timedelta
import datetime
import matplotlib
import argparse
import os, glob,sys
import time
import shutil

#Plot packages
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, pprint
sys.displayhook = pprint.pprint  # Improves print of variables (e.g. locals(), globals(), dir())) to screen
import shlex,subprocess
import numpy.matlib

#sys.path.append("/home/svangennip/PYTHON/Modules/LIB_SI/")
from configparser import ConfigParser

#sys.path.append("/home/svangennip/DEV/LAVD/src/")
sys.path.append("src/")
#from Drift_Tools import ReturnUVlist, CalcCurlGlobC, CalcRelVorticity, ExtractSurface
from Drift_Tools import ReturnUVlist, CalcCurlGlobC, CalcRelVorticity, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index
from Drift_exec import run_parcels_Drift, run_parcels_Drift_remote

import os, glob,sys
from operator import attrgetter
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cmocean
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec


#identify username (used for creating directories in one's allowed space)
import getpass
username=getpass.getuser()
print('username: ',username)



config_object = ConfigParser()
config_object.read("config.ini")
config_info = config_object['CONFIG_BASE']
MODELS = config_info["MODELS"].split(',')
lon_min = int(config_info["lon_min"])
lon_max = int(config_info["lon_max"])
lat_min = int(config_info["lat_min"])
lat_max = int(config_info["lat_max"])
DATE1 = config_info["DATE_start"]
DATE2 = config_info["DATE_end"]
PROJECT = config_info["PROJECT"]
EXP_NAME = config_info["EXP_NAME"]
username = config_info["username"]
dir_work = config_info["dir_work"]
dir_in = config_info["dir_in"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
cloud_stat = config_info["cloud_stat"]
remote = config_info["remote"]
print("MODELS : ",MODELS)
print("Region : ",lon_min,", ",lon_max,", ",lat_min,", ",lat_max)
print("From : ",DATE1," to ",DATE2)

str_region = StringRegion(lon_min,lon_max,lat_min,lat_max)

#reconstruct date_start
DD1=int(DATE1[:2])
MM1=int(DATE1[2:4])
YYYY1=int(DATE1[4:])
print(DD1,'-',MM1,'-',YYYY1)
date_start = datetime.date(YYYY1, MM1, DD1)
str_date1=date_start.strftime('%Y-%m-%d')
date_start_min1 = date_start-datetime.timedelta(days=1)

#reconstruct date_end
DD2=int(DATE2[:2])
MM2=int(DATE2[2:4])
YYYY2=int(DATE2[4:])
date_end = datetime.date(YYYY2, MM2, DD2)
str_date2=date_end.strftime('%Y-%m-%d')
date_end_plus1 = date_end+datetime.timedelta(days=1)
date_end_plus_DRIFT_TIME = date_end+datetime.timedelta(days=DRIFT_TIME+1)



T_DRIFT=(date_end-date_start).days


#-------------------------------------------------------------------------------------------------------------------------
#    CONFIG EXP LAVD
#-------------------------------------------------------------------------------------------------------------------------

dir_out=dir_work+'/DRIFT/'+PROJECT+'/'+EXP_NAME+'/ZONE_'+str_region+'/'
print(dir_out)


#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
if remote == "True":
    USERNAME = 'pgouvenou'
    PASSWORD = getpass.getpass('Enter your password: ')
    #! /usr/bin/env python3
    # -*- coding: utf-8 -*-
    __author__ = "Copernicus Marine User Support Team"
    __copyright__ = "(C) 2022 E.U. Copernicus Marine Service Information"
    __credits__ = ["E.U. Copernicus Marine Service Information"]
    __license__ = "MIT License - You must cite this source"
    __version__ = "202104"
    __maintainer__ = "D. Bazin, E. DiMedio, C. Giordan"
    __email__ = "servicedesk dot cmems at mercator hyphen ocean dot eu"


    def copernicusmarine_datastore(dataset, username, password):
        from pydap.client import open_url
        from pydap.cas.get_cookies import setup_session
        cas_url = 'https://cmems-cas.cls.fr/cas/login'
        session = setup_session(cas_url, username, password)
        session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
        database = ['my', 'nrt']
        url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
        try:
            data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
        except:
            url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
            data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
        return data_store
    data = []
    for dataset in MODELS:
        data_store = xr.open_dataset(copernicusmarine_datastore(dataset, USERNAME, PASSWORD))
        data.append(data_store.sel(time = slice(str_date1+" 12:00:00",str_date2+" 12:00:00"), longitude = slice(lon_min,lon_max),
                                  latitude = slice(lat_min,lat_max)).sel(depth = 15,method = 'nearest'))
        colors = np.array(['b','r','g','k'])[:len(MODELS)]


elif remote == "False":

    zlevel = 11
    config_object = ConfigParser()
    config_object.read(dir_in+"src/config_products.ini")
    path_UV = []
    fname_prefix = []
    var_u_mod = []
    var_v_mod = []
    meshmask = []
    dir_work_mod = []
    u_files_mod = []
    v_files_mod = []
    colors = []





    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      RETRIEVE U,V, SSH FIELDS")
    print ("#-----------------------------------------------------------------------------------------")

    for MODEL in MODELS:
        if MODEL not in config_object.sections():
            print(MODEL, "is either a typo or not yet tuned for this program")
            print('model runs available: ',config_object.sections())
            sys.exit(1)
        run_info = config_object[MODEL]
        path_UV.append(run_info["path"])
        fname_prefix.append(run_info["fname_prefix"])
        var_u_mod.append(run_info["var_u"])
        var_v_mod.append(run_info["var_v"])
        meshmask.append(run_info["meshmask"])
        colors.append(run_info["color"])
        dir_work_mod.append(dir_out+'/EXP/'+MODEL+'/')
        if not os.path.exists(dir_work_mod[-1]):
            print('\ncreate folder : ', dir_work_mod[-1])
            # Create directory where to store results:
            os.makedirs(dir_work_mod[-1])

        if MODEL in ['PSY4V3R1','IBI36','GLO12V4']:
            u_files_mod_3D, v_files_mod_3D = ReturnUVlist(MODEL, date_start_min1, date_end_plus_DRIFT_TIME)
            u_files, v_files = ExtractSurface(MODEL,u_files_mod_3D, v_files_mod_3D,zlevel,var_u_mod, var_v_mod, dir_work+'MODEL_OUTPUT/'+MODEL+'/UV/15m/')
            u_files_mod.append(u_files)
            v_files_mod.append(v_files)
        else : 
            u_files, v_files = ReturnUVlist(MODEL, date_start_min1, date_end_plus_DRIFT_TIME)
            u_files_mod.append(u_files)
            v_files_mod.append(v_files)


    print('path_UV: ', path_UV[0])
    print('fname_prefix: ', fname_prefix[0])






print ("\n#-------------------------------------------------------------------------------------------------------------------------")
print ("#    Retrieve SVP Drifters in area and within for temporal window   ")
print ("#-------------------------------------------------------------------------------------------------------------------------")


dir_in='/data/rd_exchange/pgouvenou/DATA/GlobalDriftingProgram//YEAR/'

fname_in='Drifters_'+str(YYYY1)+'.csv'
f_drifters_all=dir_in+fname_in

fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'.csv'
dir_drifters = dir_out+'/DRIFTERS/'

if not os.path.exists(dir_drifters):
    print('\ncreate folder : ', dir_drifters)
    # Create directory where to store results:
    os.makedirs(dir_drifters)
outfile2 = dir_drifters+fout_driftersZone

if not os.path.isfile(outfile2):
    DriftersInZoneInPeriod(lon_min, lon_max, lat_min, lat_max, date_start_min1, date_end_plus1, f_drifters_all, outfile2)
else:
    print(outfile2, 'already exists')

print('loading ',outfile2)
drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
print(drifters_zone_traj.head(5))
print(drifters_zone_traj.dtypes)

    
nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))
#remove drifters with missing entries (should be nb_days*4+1)
#Check each drifter has n days * 4 entries
drifter_ids = pd.unique(drifters_zone_traj['id'])
spurious_drifters=[]
for i in range(nb_drifters_temp):
    nb_entries = len(drifters_zone_traj[drifters_zone_traj.id == drifter_ids[i]])
    if not nb_entries >= (T_DRIFT*4+1):
        print(drifter_ids[i],': not enough entries, only ', nb_entries)
        spurious_drifters.append(drifter_ids[i])

#ind1 = drifters_zone_traj['id']==spurious_drifters).index
for i in range(len(spurious_drifters)):
    print('rmove drifter ',spurious_drifters[i])
    ind1 = drifters_zone_traj[drifters_zone_traj['id'] == np.array(spurious_drifters[i])].index
    drifters_zone_traj.drop(ind1, inplace=True) #remove data points
#drifters_zone_traj=drifters_zone_traj.reset_index()  
nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
        
print('total nb of drifters in zone at start that reach end:',nb_drifters2)
#Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
nb_points = T_DRIFT*4+1 + 8
traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
traj_drifter_date = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )


print ("#-----------------------------------------------------------------------------------------")
print ("#                      CALCULATE MODEl DRIFTERS TRAJECTORIES")
print ("#-----------------------------------------------------------------------------------------")
if remote == "True":
    for i in range(len(MODELS)):
        dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'

        for j in np.arange(0,nb_points-8,4): # one sim per day
            traj_drifter_lon_start = traj_drifter_lon[:,j]
            traj_drifter_lat_start = traj_drifter_lat[:,j]
            traj_drifter_id_start = traj_drifter_id[:,j]
            date_sim = date_start + datetime.timedelta(days=int(j//4))
            str_date_sim = date_sim.strftime('%Y-%m-%d')
            f_traj1 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc.zarr'
            f_traj2 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_and_around.zarr'
            f_traj2 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_smag.zarr'

            inc_seed = 1/12
            if cloud_stat== "True":
                spatio_temp_ana=1
            else : 
                spatio_temp_ana=0
            if not (os.path.isfile(f_traj1) & os.path.isfile(f_traj2)) :
                print('#run parcels with just the SVP locations')
                var_u = 'uo'
                var_v = 'vo'
                run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_plus_DRIFT_TIME+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                                 f_traj1, radius_cloud=0, seeding_resolution=0, \
                                 time_period=False)
                if spatio_temp_ana:
                    run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_plus_DRIFT_TIME+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                                 f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                 time_period=False)
                if K_smag :
                    
                    run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_plus_DRIFT_TIME+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                                 f_traj2, radius_cloud=0, seeding_resolution=0, \
                                 time_period=False,K_smag=K_smag)




for i in range(len(MODELS)):
    for j in np.arange(0,nb_points-8,4): # one sim per day
        traj_drifter_lon_start = traj_drifter_lon[:,j]
        traj_drifter_lat_start = traj_drifter_lat[:,j]
        traj_drifter_id_start = traj_drifter_id[:,j]
        date_sim = date_start + datetime.timedelta(days=int(j//4))
        str_date_sim = date_sim.strftime('%Y-%m-%d')
        f_traj1 = dir_work_mod[i] + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc.zarr'
        f_traj2 = dir_work_mod[i] + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_and_around.zarr'

        inc_seed=1/12
        if cloud_stat== "True":
            spatio_temp_ana=1
        else : 
            spatio_temp_ana=0

        if not (os.path.isfile(f_traj1) & os.path.isfile(f_traj2)) :
            print('#run parcels with just the SVP locations')
            print('\nu_files_mod[1:-1]=\n',u_files_mod[i][1:-1])
            print('\nu_files_mod[:-2]=\n',u_files_mod[i][:-2])
            print('\nu_files_mod[2:]=\n',u_files_mod[i][2:])
            run_parcels_Drift(MODELS[i], u_files_mod[i][1+j//4:-1], var_u_mod[i], v_files_mod[i][1+j//4:-1], var_v_mod[i], meshmask[i],\
                             traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                             f_traj1, radius_cloud=0, seeding_resolution=0, \
                             time_period=False)

            if spatio_temp_ana==1:
                print('#run parcels with at SVP neigbouring locations')
                run_parcels_Drift(MODELS[i], u_files_mod[i][1+j//4:-1], var_u_mod[i], v_files_mod[i][1+j//4:-1], var_v_mod[i], meshmask[i],\
                                 traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                                 f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                 time_period=False)

            print(' done...')
        else:
            print(f_traj1,'\nand ', f_traj2,' already exists' )



