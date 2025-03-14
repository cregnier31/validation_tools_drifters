#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from scipy.stats import pearsonr
import matplotlib
import cartopy
import shutil

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
FORECAST = True

# Print configuration information

print("MODELS : ",MODELS)
print("Region : ",LON_MIN,", ",LON_MAX,", ",LAT_MIN,", ",LAT_MAX)

# Add a directory to the Python sys path

sys.path.append("src/")

# Import custom functions from local module

from Drift_Tools import ReturnUVlist, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore2, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote

from Functions import Retrieve_traj_files,Retrieve_drifters_traj,Traj_filter_fidji,Report_card_plot


# Create an output directory path

STR_REGION = StringRegion(LON_MIN,LON_MAX,LAT_MIN,LAT_MAX)
DIR_OUT=DIR_WORK+"DRIFT/"+PROJECT+'/'+EXP_NAME+'/ZONE_'+STR_REGION+'/'
print("Dir out : ",DIR_OUT)




#Retrieving date of the Wednesday in the month
YEAR = np.int32(MONTH_YEAR[2:])
MONTH = np.int32(MONTH_YEAR[:2])


date_var = datetime.date(YEAR,MONTH,1)
WED = []
while date_var.month <= MONTH + N_MONTHES - 1:
    if date_var.weekday()==2:
        WED.append(date_var)
    date_var += datetime.timedelta(days=1)
    if date_var.year == YEAR +1:
        break
print(WED)
 


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


print(DATE_END_PLUS_DRIFT_TIME)


# Read information from the configuration file



if FORECAST:
    for index_model in range(len(MODELS)):
        if ~MODELS[index_model].endswith('FCST'):
            MODELS[index_model] = MODELS[index_model] +'_FCST'

print(MODELS)

traj_lon,traj_lat,traj_time,traj_IDS = Retrieve_traj_files(MODELS,DIR_OUT)

print(np.sort(traj_lon.flatten()))
print(np.sort(traj_lat.flatten()))


fout_driftersZone = 'Drifters_'+STR_REGION+'_'+STR_DATE_START+'_'+STR_DATE_END_PLUS_DRIFT_TIME+'.csv'
fout_driftersZone = 'Drifters_'+STR_REGION+'_'+STR_DATE_START+'_'+STR_DATE_END_PLUS_DRIFT_TIME+'.csv'
fout_driftersZone = 'Drifters_'+STR_REGION+'_'+'*.csv'
dir_drifters = DIR_OUT+'/DRIFTERS/'


outfile2 = glob.glob(dir_drifters+fout_driftersZone)[0]

traj_drifter_lon_temp,traj_drifter_lat_temp,traj_drifter_date_temp,traj_drifter_time_temp,traj_drifter_id_temp,nb_drifters_temp = Retrieve_drifters_traj(outfile2,T_DRIFT,COPERNICUS_DRIFTER)




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
gl.xlabels_top = False
gl.ylabels_right = False
fig.suptitle('Trajectory of the SVP drifters',fontsize = 12)

axs.plot(traj_drifter_lon.T,traj_drifter_lat.T,transform=ccrs.PlateCarree())
axs.set_xlim([LON_MIN,LON_MAX])
axs.set_ylim([LAT_MIN,LAT_MAX])
plt.savefig(dir_plot+'Traj_drifters'+STR_DATE_START+'_'+STR_DATE_END+'.png',dpi = 600)

if(np.prod(np.shape(traj_drifter_lon)) >0):
    print(DRIFT_TIME)
    drifter_lon,drifter_lat,drifter_date,drifter_time = Traj_filter_fidji(DRIFT_TIME,traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time, traj_lon,traj_lat)
    sep_dist = SeparationDistance(drifter_lon,drifter_lat,traj_lon,traj_lat)
    liu_score = np.array([[LiuScore2(drifter_lon[i,j],drifter_lat[i,j],traj_lon[i,j],traj_lat[i,j]).T for j in range(traj_lon.shape[1])] for i in range (traj_lon.shape[0])])

    skill_score = 1-liu_score
    if True:# np.prod(np.shape(np.where(np.isnan(traj_lon)))):
        nans = np.unique(np.where(np.isnan(traj_lon))[2])
        if len(nans)>0:
            traj_lon = np.delete(traj_lon,nans,axis = 2)
            traj_lat = np.delete(traj_lat,nans,axis = 2)
            drifter_lon = np.delete(drifter_lon,nans,axis = 2)
            drifter_lat = np.delete(drifter_lat,nans,axis = 2)
        nans = np.unique(np.where(np.isnan(drifter_lon))[2])
        if len(nans)>0:
            traj_lon = np.delete(traj_lon,nans,axis = 2)
            traj_lat = np.delete(traj_lat,nans,axis = 2)
            drifter_lon = np.delete(drifter_lon,nans,axis = 2)
            drifter_lat = np.delete(drifter_lat,nans,axis = 2)


    corr_lon = np.array([[[pearsonr(drifter_lon[i,j,k],traj_lon[i,j,k])[0] for k in range(traj_lon.shape[2])] for j in range(traj_lon.shape[1])] for i in range(traj_lon.shape[0])])
    corr_lat = np.array([[[pearsonr(drifter_lat[i,j,k],traj_lat[i,j,k])[0] for k in range(traj_lat.shape[2])] for j in range(traj_lat.shape[1])] for i in range(traj_lat.shape[0])])


    if np.prod(np.shape(sep_dist))>0:
        prop_sep_dist = np.count_nonzero(sep_dist[:,:,:,-1]<50 ,axis =(1,2) )/ (sep_dist.shape[1]*sep_dist.shape[2])
        prop_skill_score = np.count_nonzero(skill_score[:,:,:,-1]>0 ,axis =(1,2) )/ (skill_score.shape[1]*skill_score.shape[2])
        prop_corr_lon = np.count_nonzero(corr_lon>0.7 ,axis =(1,2) )/ (corr_lon.shape[1]*corr_lon.shape[2])
        prop_corr_lat = np.count_nonzero(corr_lat>0.7 ,axis =(1,2) )/ (corr_lat.shape[1]*corr_lat.shape[2])



# In[5]:


    plt.plot(traj_drifter_lon.T[:,0][:-24*2+1],traj_drifter_lat.T[:,0][:-24*2+1],c='b',label = 'Drifter')
    plt.plot(traj_lon[0,0,0],traj_lat[0,0,0],c = 'orange',label = 'Virtual particle')
    plt.plot([traj_drifter_lon.T[:,0][:-24*2+1][-1],traj_lon[0,0,0][-1]],[traj_drifter_lat.T[:,0][:-24*2+1][-1],
                                                                      traj_lat[0,0,0][-1]],c = 'r',label = 'Separation Distance = '+str(round(sep_dist[0,0,0][-1],1)) +' km')
    plt.grid()
    plt.legend()


# In[7]:


    fig = plt.figure(figsize = (9,5))
    axs=plt.axes(projection=ccrs.PlateCarree())
    axs.coastlines()
    axs.add_feature(cfeature.LAND)
    gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabels_top = False
    gl.ylabels_right = False
    fig.suptitle('Trajectory of the SVP drifters',fontsize = 12)

    axs.plot(traj_drifter_lon.T,traj_drifter_lat.T,transform=ccrs.PlateCarree())
    axs.set_xlim([LON_MIN,LON_MAX])
    axs.set_ylim([LAT_MIN,LAT_MAX])


# In[9]:


    fig,axs = plt.subplots(nrows=1, ncols=len(MODELS),figsize = (6*len(MODELS),6))
    fig.suptitle('Separation distance for particle at Drifter location',y = 1,x = 0.5,fontsize = 12)
    n_sep = []
    x_sep = []
    if len(MODELS) == 1:
        n1, x1, _  = axs.hist(sep_dist[0].flatten(),bins =15,range=(0,250),color='b')
        axs.plot(np.nanmean(sep_dist[0,:,:,-1].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                  label = 'Mean separation distance : ' + str(np.round(np.nanmean(sep_dist[0,:,:,-1].flatten()),2)))
        axs.set_title(MODELS[0])
        axs.set_xlabel('Separation distance in km')
        axs.legend()
        axs.grid()
        n_sep.append(n1)
        x_sep.append(x1)
    
    else :     
        for i in range(len(MODELS)):
            n1, x1, _  = axs[i].hist(sep_dist[i,:,:,-1].flatten(),bins =15,range=(0,250),color='b')
            axs[i].plot(np.nanmean(sep_dist[i,:,:,-1].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                      label = 'Mean separation distance : ' + str(np.round(np.nanmean(sep_dist[i,:,:,-1].flatten()),2)))
            axs[i].set_title(MODELS[i])
            axs[i].set_xlabel('Separation distance in km')
            axs[i].legend()
            axs[i].grid()

            n_sep.append(n1)
            x_sep.append(x1)
    plt.savefig(DIR_OUT+'PLOT/Separation_dis_hist.png',dpi = 400)


# In[10]:


    fig,axs = plt.subplots(nrows=1, ncols=1,figsize = (9,5))
    fig.suptitle('Cumulative separation distance for the area',fontsize = 12)
    for i in range (len(MODELS)):
        axs.plot(x_sep[i][:-1],n_sep[i],label =MODELS[i]+' : '+str(round(prop_sep_dist[i]*100,4)) + '%')


    axs.plot([50,50],[0,np.max(n_sep)],'--',c='k')
    axs.legend()
    axs.set_xlabel('Separation Distance in km')
    axs.set_ylabel('nb points')
    plt.grid()
    plt.savefig(DIR_OUT+'PLOT/cumulative_separation.png',dpi = 400)


# In[11]:


    fig,axs = plt.subplots(nrows=1, ncols=len(MODELS),figsize = (6*len(MODELS),6))
    fig.suptitle('Liu Skill Score for particle at drifter location',y = 1,x = 0.5,fontsize = 12)
    n_liu = []
    x_liu = []
    if len(MODELS) == 1:
        n1, x1, _  = axs.hist(skill_score[0,:,:,-1].flatten(),bins =15,range=(-2,1),color='b')
        axs.plot(np.nanmean(skill_score[0,:,:,-1].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                  label = 'Mean Liu Skill Score : ' + str(np.round(np.nanmean(skill_score[0,:,:,-1].flatten()),2)))
        axs.set_title(MODELS[0])
        axs.set_xlabel('Liu Skill Score')
        axs.legend()
        axs.grid()
        n_liu.append(n1)
        x_liu.append(x1)
    
    else :     
        for i in range(len(MODELS)):
            n1, x1, _  = axs[i].hist(skill_score[i,:,:,-1].flatten(),bins =15,range=(-2,1),color='b')
            axs[i].plot(np.nanmean(skill_score[i,:,:,-1].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                      label = 'Mean Liu Skill Score : ' + str(np.round(np.nanmean(skill_score[i,:,:,-1].flatten()),2)))
            axs[i].set_title(MODELS[i])
            axs[i].set_xlabel('Liu Skill Score')
            axs[i].legend()
            axs[i].grid()

            n_liu.append(n1)
            x_liu.append(x1)
    plt.savefig(DIR_OUT+'PLOT/Liu_hist.png',dpi = 400)


# In[12]:


    fig,axs = plt.subplots(nrows=1, ncols=1,figsize = (9,5))
    fig.suptitle('Cumulative Liu Skill Score for the area',fontsize = 12)
    for i in range (len(MODELS)):
        axs.plot(x_liu[i][:-1],n_liu[i],label =MODELS[i]+' : '+str(round(prop_skill_score[i]*100,4)) + '%')


    axs.plot([0,0],[0,np.max(n_liu)],'--',c='k')
    axs.legend()
    axs.set_xlabel('Liu Skill Score')
    axs.set_ylabel('nb points')
    plt.grid()
    plt.savefig(DIR_OUT+'PLOT/Liu_cumulative.png',dpi = 400)


# In[13]:


    fig,axs = plt.subplots(nrows=1, ncols=len(MODELS),figsize = (6*len(MODELS),6))
    fig.suptitle('Correlation Coefficient for particle at drifter location',y = 1,x = 0.5,fontsize = 12)
    n_corr_lon = []
    x_corr_lon = []
    if len(MODELS) == 1:
        n1, x1, _  = axs.hist(corr_lon[0].flatten(),bins =15,range=(-1,1),color='b')
        axs.plot(np.nanmean(corr_lon[0].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                  label = 'Mean Corr Coeff for longitude : ' + str(np.round(np.nanmean(corr_lon[0].flatten()),2)))
        axs.set_title(MODELS[0])
        axs.set_xlabel('Corr Coeff')
        axs.legend()
        axs.grid()
        n_corr_lon.append(n1)
        x_corr_lon.append(x1)
    
    else :     
        for i in range(len(MODELS)):
            n1, x1, _  = axs[i].hist(corr_lon[i].flatten(),bins =15,range=(-1,1),color='b')
            axs[i].plot(np.nanmean(corr_lon[i].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                      label = 'Mean Corr Coeff for longitude : ' + str(np.round(np.nanmean(corr_lon[i].flatten()),2)))
            axs[i].set_title(MODELS[i])
            axs[i].set_xlabel('Corr Coeff')
            axs[i].legend()
            axs[i].grid()

            n_corr_lon.append(n1)
            x_corr_lon.append(x1)
    plt.savefig(DIR_OUT+'PLOT/Corr_coef_lon_hist.png',dpi = 400)


# In[14]:


    fig,axs = plt.subplots(nrows=1, ncols=len(MODELS),figsize = (6*len(MODELS),6))
    fig.suptitle('Correlation Coefficient for particle at drifter location',y = 1,x = 0.5,fontsize = 12)
    n_corr_lat = []
    x_corr_lat = []
    if len(MODELS) == 1:
        n1, x1, _  = axs.hist(corr_lat[0].flatten(),bins =15,range=(-1,1),color='b')
        axs.plot(np.nanmean(corr_lat[0].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                  label = 'Mean Corr Coeff for latitude : ' + str(np.round(np.nanmean(corr_lat[0].flatten()),2)))
        axs.set_title(MODELS[0])
        axs.set_xlabel('Corr Coeff')
        axs.legend()
        axs.grid()
        n_corr_lat.append(n1)
        x_corr_lat.append(x1)
    
    else :     
        for i in range(len(MODELS)):
            n1, x1, _  = axs[i].hist(corr_lat[i].flatten(),bins =15,range=(-1,1),color='b')
            axs[i].plot(np.nanmean(corr_lat[i].flatten())*np.ones(2),[0,n1.max()],color= 'r',
                      label = 'Mean Corr Coeff for latitude : ' + str(np.round(np.nanmean(corr_lat[i].flatten()),2)))
            axs[i].set_title(MODELS[i])
            axs[i].set_xlabel('Corr Coeff')
            axs[i].legend()
            axs[i].grid()

            n_corr_lat.append(n1)
            x_corr_lat.append(x1)
    plt.savefig(DIR_OUT+'PLOT/Corr_coef_lat_hist.png',dpi = 400)


# In[15]:


    fig,axs = plt.subplots(nrows=1, ncols=1,figsize = (9,5))
    fig.suptitle('Cumulative corr coeff lon for the area',fontsize = 12)
    for i in range (len(MODELS)):
        axs.plot(x_corr_lon[i][:-1],n_corr_lon[i],label =MODELS[i]+' : '+str(round(prop_corr_lon[i]*100,4)) + '%')


    axs.plot([0.7,0.7],[0,np.max(n_corr_lon)],'--',c='k')
    axs.legend()
    axs.set_xlabel('Corr Coeff Lon')
    axs.set_ylabel('nb points')
    plt.grid()
    plt.savefig(DIR_OUT+'PLOT/Corr_coef_lon_cumulative.png',dpi = 400)





    fig,axs = plt.subplots(nrows=1, ncols=1,figsize = (9,5))
    fig.suptitle('Cumulative corr coeff lat for the area',fontsize = 12)
    for i in range (len(MODELS)):
        axs.plot(x_corr_lat[i][:-1],n_corr_lat[i],label =MODELS[i]+' : '+str(round(prop_corr_lat[i]*100,4)) + '%')

    fig.suptitle('Cumulative corr coeff lat for the area',fontsize = 12)
    for i in range (len(MODELS)):
        axs.plot(x_corr_lat[i][:-1],n_corr_lat[i],label =MODELS[i]+' : '+str(round(prop_corr_lat[i]*100,4)) + '%')


    axs.plot([0.7,0.7],[0,np.max(n_corr_lat)],'--',c='k')
    axs.legend()
    axs.set_xlabel('Corr Coeff Lat')
    axs.set_ylabel('nb points')
    plt.grid()
    plt.savefig(DIR_OUT+'PLOT/Corr_coef_lat_cumulative.png',dpi = 400)


# In[ ]:




### plot maps 
from plot_functions import Compute_2D_MAPS,comparative_maps,compute_scores_TS,plot_scores_TS,save_ncfile_maps,save_TS_nc



NOBS = []
MAPS_sep_dist = []
MAPS_skill_score = []

for ii in range(len(MODELS)):
    cell_size =2
    param = sep_dist[ii,:,:,-1]
 
    aaa,bbb,grid = Compute_2D_MAPS(param,drifter_lon[ii,:,:,-1],drifter_lat[ii,:,:,-1],LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,cell_size)
    MAPS_sep_dist.append(aaa)
    bbb[bbb<0] =0
    NOBS.append(bbb)

    param = skill_score[ii,:,:,-1]
    
    aaa,bbb,grid = Compute_2D_MAPS(param,drifter_lon[ii,:,:,-1],drifter_lat[ii,:,:,-1],LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,cell_size)
    MAPS_skill_score.append(aaa) 
    
if not os.path.exists(DIR_OUT + '/NC_maps'):
    os.makedirs(DIR_OUT + '/NC_maps')

for ii in range(len(MODELS)):
    DAta = NOBS[ii]
    nc_filename = DIR_OUT+'NC_maps/NOBS_'+ MODELS[ii]+ '_'  + EXP_NAME.split('_')[-2] + '_' + EXP_NAME.split('_')[-1]  + '.nc'
    data_unit = '%'
    var_name = 'observations_number'
    save_ncfile_maps(DAta,nc_filename,var_name,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX,cell_size,data_unit)
    
    DAta = MAPS_sep_dist[ii]
    nc_filename = DIR_OUT+'NC_maps/Sep_dist_'+ MODELS[ii]  + '_'  + EXP_NAME.split('_')[-2] + '_' + EXP_NAME.split('_')[-1]  + '.nc'
    data_unit = 'km'
    var_name = 'separation_distance'
    save_ncfile_maps(DAta,nc_filename,var_name,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX,cell_size,data_unit)

    DAta = MAPS_skill_score[ii]
    nc_filename = DIR_OUT+'NC_maps/Lui_score'+ MODELS[ii]  + '_'  + EXP_NAME.split('_')[-2] + '_' + EXP_NAME.split('_')[-1]  + '.nc'
    data_unit = '%'
    var_name = 'Liu_Skill_Score'
    save_ncfile_maps(DAta,nc_filename,var_name,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX,cell_size,data_unit)



fig_title = 'Liu Skill Score'
fig_name = DIR_OUT+'PLOT/map_liu_skill_score' +  '_'  + EXP_NAME.split('_')[-2] + '_' + EXP_NAME.split('_')[-1]  + '.png'

comparative_maps(MAPS_skill_score,MODELS,cell_size,grid,fig_title,fig_name,-1,1)




fig_title = 'Separation Distance [km]'
fig_name = DIR_OUT+'PLOT/map_separation_distance' +  '_'  + EXP_NAME.split('_')[-2] + '_' + EXP_NAME.split('_')[-1]  + '.png'
comparative_maps(MAPS_sep_dist,MODELS,cell_size,grid,fig_title,fig_name,0,200)#,500)

fig_title = 'Number of Observations'
fig_name = DIR_OUT+'PLOT/NOBS' +  '_'  + EXP_NAME.split('_')[-2] + '_' + EXP_NAME.split('_')[-1]  + '.png'

NOBS_nonzero = NOBS
for ii in range(len(MODELS)):
    NOBS_nonzero[ii][NOBS[ii] == 0] = np.nan

comparative_maps(NOBS_nonzero,MODELS,cell_size,grid,fig_title,fig_name,0,5)




################
###############
##############"

zones ={'IBI':{'lon_min': -19  ,'lon_max': 5   , 'lat_min': 26  ,  'lat_max': 56 },'GS':{'lon_min': -80  ,'lon_max': -30   , 'lat_min': 30  ,  'lat_max': 60 },'AG':{'lon_min': 5  ,'lon_max': 35   , 'lat_mi\
n': -45  ,  'lat_max': -25 },'KOE':{'lon_min': 140  ,'lon_max': 170   , 'lat_min': 28  ,  'lat_max': 50 }}


score = sep_dist[0,:,:,-1]
date_vect = drifter_date[0,:,:,-1]
lon_vect = drifter_lon[0,:,:,-1]
lat_vect = drifter_lat[0,:,:,-1]
bool_global = True
sep_dist_TS, dates_TS, nobs_TS = compute_scores_TS(score,date_vect,lon_vect,lat_vect,DATE_START,DATE_END,zones,bool_global)

bool_multi_zones = True
fig_name = DIR_OUT+'PLOT/TS_scores_sep_dist.png'
#fig_title = 'Separation distance variation' + ' - Drift time = {}' 
score_name = 'Separation distance [km]'
DRIFT_TIME = int(config_info["DRIFT_TIME"])
DIR_OUT_nc = DIR_WORK+"DRIFT/"+PROJECT+"/NC_history/" 
if FORECAST == True:
    fig_label = f'GLO12_DT{DRIFT_TIME}_FCT'  
    fig_title = 'Separation distance variation' + f' - Drift time = {DRIFT_TIME} - Forecast'
    nc_files_old = DIR_OUT_nc + f'TS_GLO12_DT{DRIFT_TIME}_FCT_Sep_dist_old.nc'
    nc_files_tmp = DIR_OUT_nc + f'TS_GLO12_DT{DRIFT_TIME}_FCT_Sep_dist_tmp.nc'
    nc_files = DIR_OUT_nc + f'TS_GLO12_DT{DRIFT_TIME}_FCT_Sep_dist.nc'

else:
    fig_label = f'GLO12_DT{DRIFT_TIME}_ANA'
    fig_title = 'Separation distance variation' + f' - Drift time = {DRIFT_TIME} - Analysis'
    nc_files_old = DIR_OUT_nc + f'TS_GLO12_DT{DRIFT_TIME}_ANA_Sep_dist_old.nc'
    nc_files = DIR_OUT_nc + f'TS_GLO12_DT{DRIFT_TIME}_ANA_Sep_dist.nc'
    nc_files_tmp = DIR_OUT_nc + f'TS_GLO12_DT{DRIFT_TIME}_ANA_Sep_dist_tmp.nc'


save_TS_nc(sep_dist_TS, dates_TS, nobs_TS ,nc_files_tmp)

if os.path.exists(nc_files_old):
    ds_a = xr.open_dataset(nc_files_old)
    ds_b = xr.open_dataset(nc_files_tmp)
    # Concaténer les datasets le long de la dimension temps (ou autre dimension commune)
    ds_concat = xr.concat([ds_a, ds_b], dim='time')
    # Sauvegarder le dataset concaténé dans A.nc
    ds_concat.to_netcdf(nc_files)
else:
    os.rename(nc_files_tmp,nc_files)




#fig_title,fig_name,score_name,bool_multi_zones = True)
ds = xr.open_dataset(nc_files)

sep_dist_TS = ds['scores'].data
#dates_TS  = np.array([np.datetime64('1970-01-01T00:00:00Z') +   ds['dates'].data.astype('timedelta64[s]')#  np.timedelta64(int(ts), 's') for ts in ds['dates'].data])
#dates_TS =  ds['dates'].data
#dates_TS = np.array([np.datetime64('1970-01-01T00:00:00Z') + np.timedelta64(int(ts), 's') for ts in ds['dates'].data])
dates_TS = np.datetime64('1970-01-01T00:00:00Z') + ds['dates'].data.astype('timedelta64[s]')
nobs_TS = ds['nobs'].data

plot_scores_TS(sep_dist_TS, dates_TS, nobs_TS,zones,fig_title,fig_label,DIR_OUT,score_name,bool_multi_zones)
shutil.copyfile(nc_files, nc_files_old)



score = liu_score[0,:,:,-1]
date_vect = drifter_date[0,:,:,-1]
lon_vect = drifter_lon[0,:,:,-1]
lat_vect = drifter_lat[0,:,:,-1]

lss_TS, dates_TS, nobs_TS = compute_scores_TS(score,date_vect,lon_vect,lat_vect,DATE_START,DATE_END,zones,bool_global)

bool_multi_zones = True
fig_name = DIR_OUT+'PLOT/TS_scores_lss.png'
fig_title = 'LSS variation'
score_name = 'Liu Skill Score'
DRIFT_TIME = int(config_info["DRIFT_TIME"])
if FORECAST == True:
    fig_label = f'GLO12_DT{DRIFT_TIME}_FCT'
    fig_title = 'LSS variation' + f' - Drift time = {DRIFT_TIME} - Forecast'
    nc_files_tmp = DIR_OUT_nc    + f'TS_GLO12_DT{DRIFT_TIME}_FCT_LSS_tmp.nc'
    nc_files = DIR_OUT_nc    + f'TS_GLO12_DT{DRIFT_TIME}_FCT_LSS.nc'
    nc_files_old = DIR_OUT_nc    + f'TS_GLO12_DT{DRIFT_TIME}_FCT_LSS_old.nc'

else:
    fig_label = f'GLO12_DT{DRIFT_TIME}_ANA'
    fig_title = 'LSS variation' + f' - Drift time = {DRIFT_TIME} - Analysis'
    nc_files_tmp = DIR_OUT_nc  + f'TS_GLO12_DT{DRIFT_TIME}_ANA_LSS_tmp.nc'
    nc_files = DIR_OUT_nc  + f'TS_GLO12_DT{DRIFT_TIME}_ANA_LSS.nc'
    nc_files_old = DIR_OUT_nc  + f'TS_GLO12_DT{DRIFT_TIME}_ANA_LSS_old.nc'

save_TS_nc(lss_TS, dates_TS, nobs_TS ,nc_files_tmp)

if os.path.exists(nc_files_old):
    ds_a = xr.open_dataset(nc_files_old)
    ds_b = xr.open_dataset(nc_files_tmp)

    # Concaténer les datasets le long de la dimension temps (ou autre dimension commune)
    ds_concat = xr.concat([ds_a, ds_b], dim='time')

    # Sauvegarder le dataset concaténé dans A.nc
    ds_concat.to_netcdf(nc_files)
else:
    os.rename(nc_files_tmp,nc_files)


ds = xr.open_dataset(nc_files)
lss_TS = ds['scores'].data
#dates_TS  = np.array([np.datetime64('1970-01-01T00:00:00Z') + np.timedelta64(int(ts), 's') for ts in ds['dates'].data])
#dates_TS =  ds['dates'].data
#dates_TS = np.array([np.datetime64('1970-01-01T00:00:00Z') + np.timedelta64(int(ts), 's') for ts in ds['dates'].data])
dates_TS = np.datetime64('1970-01-01T00:00:00Z') + ds['dates'].data.astype('timedelta64[s]')
nobs_TS = ds['nobs'].data


plot_scores_TS(lss_TS, dates_TS, nobs_TS,zones,fig_title,fig_label,DIR_OUT,score_name,bool_multi_zones)
shutil.copyfile(nc_files, nc_files_old)

