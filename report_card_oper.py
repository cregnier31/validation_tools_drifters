#!/usr/bin/env python
# coding: utf-8


import numpy as np
import numpy.matlib
from numpy import pi
import pandas as pd
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset, date2num
from scipy import stats
from scipy import ndimage



import matplotlib
import argparse
import time
import shutil
from math import pi
from shapely.geometry import Point,MultiPoint
from scipy.stats import pearsonr




#Plot packages
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cmocean
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#system packages
import os, glob, sys, pprint, shlex,subprocess
from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("config_base.ini")
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
DRIFT_TIME = int(config_info['DRIFT_TIME'])
remote =  config_info['remote']

print("MODELS : ",MODELS)
print("Region : ",lon_min,", ",lon_max,", ",lat_min,", ",lat_max)
print("From : ",DATE1," to ",DATE2)

import datetime
import getpass
import sys
sys.path.insert(0, dir_in+"src/")
from Drift_Tools import ReturnUVlist, CalcCurlGlobC, CalcRelVorticity, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index

str_region = StringRegion(lon_min,lon_max,lat_min,lat_max)
dir_out=dir_work+"DRIFT/"+PROJECT+'/'+EXP_NAME+'/ZONE_'+str_region+'/'
from Drift_Tools import *
import datetime
username=getpass.getuser()
print('username: ',username)
matplotlib.use('Agg')


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





traj_files = []
for MODEL in MODELS:
    traj_files_int = []
    traj_files_int += sorted(glob.glob(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc.zarr'))
    traj_files.append(traj_files_int)
traj_files = np.array(traj_files)





traj_lon = []
traj_lat = []
traj_time = []
for i in range(len(MODELS)):
    traj_lon_int = []
    traj_lat_int = []
    traj_time_int = []
    for file in traj_files[i]:
        ds = xr.open_dataset(file)
        traj_time_int.append(ds.time.values)
        traj_lon_int.append(ds.lon.values)
        traj_lat_int.append(ds.lat.values)
    traj_lon.append(traj_lon_int)
    traj_lat.append(traj_lat_int)
    traj_time.append(traj_time_int)
traj_lon = np.array(traj_lon)
traj_lat = np.array(traj_lat)
traj_time = np.array(traj_time)




#dir_in='/data/rd_exchange/pgouvenou/DATA/GlobalDriftingProgram//YEAR/'
dir_in = '/data/rd_exchange/svangennip/DATA/GlobalDriftingProgram/YEAR/'

fname_in='Drifters_'+str(YYYY1)+'.csv'
f_drifters_all=dir_in+fname_in
date_end_bis = datetime.date(YYYY2, MM2, DD2)+datetime.timedelta(days=DRIFT_TIME)
str_date3 = date_end_bis.strftime('%Y-%m-%d')
T_DRIFT = (date_end_bis -date_start).days
fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'.csv'
dir_drifters = dir_out+'/DRIFTERS/'


outfile2 = dir_drifters+fout_driftersZone



print('loading ',outfile2)
drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
print(drifters_zone_traj.head(5))
print(drifters_zone_traj.dtypes)

    
nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))

#Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
if remote == "True":
    nb_points = T_DRIFT*24 +24   # hourly
else :
    nb_points = T_DRIFT*4 + 1    # hourly

traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters_temp, nb_points) )
traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters_temp, nb_points) )
traj_drifter_date = np.reshape(drifters_zone_traj.date.values,(nb_drifters_temp, nb_points) )
traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters_temp, nb_points) )
traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters_temp, nb_points) )



dir_plot = dir_out+'PLOT/'


if not os.path.exists(dir_plot):
    print('\ncreate folder : ', dir_plot)
    # Create directory where to store results:
    os.makedirs(dir_plot)

fig = plt.figure(figsize = (9,5))
axs=plt.axes(projection=ccrs.PlateCarree())
axs.coastlines()
axs.add_feature(cfeature.LAND)
gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.xlabels_top = False
gl.ylabels_right = False
fig.suptitle('Trajectory of the SVP drifters',fontsize = 12)

axs.plot(traj_drifter_lon.T,traj_drifter_lat.T,transform=ccrs.PlateCarree())
axs.set_xlim([lon_min,lon_max])
axs.set_ylim([lat_min,lat_max])
plt.savefig(dir_plot+'Traj_drifters'+str_date1+'_'+str_date2+'.png',dpi = 600)



drifter_lon = np.zeros(traj_lon.shape)
drifter_lat = np.zeros(traj_lat.shape)


if remote == "True":
    drifter_lon[0,0,:,0] = traj_drifter_lon[:,:int(DRIFT_TIME)*24+24][:,0]
    drifter_lat[0,0,:,0] = traj_drifter_lat[:,:int(DRIFT_TIME)*24+24][:,0]
    
    drifter_lon[0,0,:,1:] = np.array([np.mean(traj_drifter_lon[:,1+i*6:12+i*6],axis = 1) for i in range (traj_lon.shape[-1]-1)]).T
    drifter_lat[0,0,:,1:] = np.array([np.mean(traj_drifter_lat[:,1+i*6:12+i*6],axis = 1) for i in range (traj_lat.shape[-1]-1)]).T




    for i in range (traj_lon.shape[0]):
        for j in range(traj_lon.shape[1]):
            drifter_lon[i,j,:,0] = traj_drifter_lon[:,j*24:j*24+int(DRIFT_TIME)*24+24][:,0]
            drifter_lat[i,j,:,0] = traj_drifter_lat[:,j*24:j*24+int(DRIFT_TIME)*24+24][:,0]
            
            drifter_lon[i,j,:,1:] = np.array([np.mean(traj_drifter_lon[:,j*24+1+i*6:j*24+12+i*6],axis = 1) for i in range (traj_lon.shape[-1]-1)]).T
            drifter_lat[i,j,:,1:] = np.array([np.mean(traj_drifter_lat[:,j*24+1+i*6:j*24+12+i*6],axis = 1) for i in range (traj_lat.shape[-1]-1)]).T
            



else : 
    drifter_lon[0,0] = traj_drifter_lon[:,:int(DRIFT_TIME)*4+1]
    drifter_lat[0,0] = traj_drifter_lat[:,:int(DRIFT_TIME)*4+1]





    for i in range (traj_lon.shape[0]):
        for j in range(traj_lon.shape[1]):
            drifter_lon[i,j] = traj_drifter_lon[:,j*4:int(DRIFT_TIME)*4+1+j*4]
            drifter_lat[i,j] = traj_drifter_lat[:,j*4:int(DRIFT_TIME)*4+1+j*4]




sep_dist = SeparationDistance(drifter_lon,drifter_lat,traj_lon,traj_lat)
liu_score = np.array([[LiuScore2(drifter_lon[i,j],drifter_lat[i,j],traj_lon[i,j],traj_lat[i,j]).T for j in range(traj_lon.shape[1])] for i in range (traj_lon.shape[0])])
skill_score = 1-liu_score
nans = np.unique(np.where(np.isnan(traj_lon))[2])
if len(nans)>0:
    traj_lon = np.delete(traj_lon,nans,axis = 2)
    traj_lat = np.delete(traj_lat,nans,axis = 2)
    drifter_lon = np.delete(drifter_lon,nans,axis = 2)
    drifter_lat = np.delete(drifter_lat,nans,axis = 2)

corr_lon = np.array([[[pearsonr(drifter_lon[i,j,k],traj_lon[i,j,k])[0] for k in range(traj_lon.shape[2])] for j in range(traj_lon.shape[1])] for i in range(traj_lon.shape[0])])
corr_lat = np.array([[[pearsonr(drifter_lat[i,j,k],traj_lat[i,j,k])[0] for k in range(traj_lat.shape[2])] for j in range(traj_lat.shape[1])] for i in range(traj_lat.shape[0])])



prop_sep_dist = np.count_nonzero(sep_dist[:,:,:,-1]<50 ,axis =(1,2) )/ (sep_dist.shape[1]*sep_dist.shape[2])
prop_skill_score = np.count_nonzero(skill_score[:,:,:,-1]>0 ,axis =(1,2) )/ (skill_score.shape[1]*skill_score.shape[2])
prop_corr_lon = np.count_nonzero(corr_lon>0.7 ,axis =(1,2) )/ (corr_lon.shape[1]*corr_lon.shape[2])
prop_corr_lat = np.count_nonzero(corr_lat>0.7 ,axis =(1,2) )/ (corr_lat.shape[1]*corr_lat.shape[2])





df = pd.DataFrame({
'MODELS': MODELS,
'SKILLS': prop_skill_score,
'SEP50': prop_sep_dist,
'CORR_LON': prop_corr_lon,
'CORR_LAT': prop_corr_lat})
# ------- PART 1: Create background
# number of variable
categories=list(df)[1:]
N = len(categories)



colors_data = {'PSY44':'lightcoral','PSY4V3R1':'red','GLO12V4':'blue','IBI36':'green','GLOBC':'purple','cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m':'orange', 'dataset-uv-nrt-daily':'cyan'}



colors_list = [colors_data[MODELS[i]] for i in range(len(MODELS))]





df.T[1:].plot.bar(rot = 0,title = "Lagrangian statistics from " + str_date1+' to '+str_date2,ylabel = 'Proportion',xlabel = 'Scores',color = colors_list)
plt.legend(labels = df.MODELS.values)
plt.savefig(dir_plot+'Lag_stats_'+str_date1+'_'+str_date2+'.png',dpi = 600)

print("Save fig : ", dir_plot+'Lag_stats_'+str_date1+'_'+str_date2+'.png')






