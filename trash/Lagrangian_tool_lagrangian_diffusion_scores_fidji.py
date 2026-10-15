#!/usr/bin/env python
# coding: utf-8

# In[35]:


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
nb_repeats = int(config_info["nb_repeats"])
AdvectionDiffusion_type = config_info["AdvectionDiffusion_type"]
DRIFT_TIME = int(config_info["DRIFT_TIME"])
FORECAST = bool(config_info["Forecast"])
REMOTE = False
N_MONTHES =  int(config_info["N_MONTHES"])


# Print configuration information

print("MODELS : ",MODELS)
print("Region : ",LON_MIN,", ",LON_MAX,", ",LAT_MIN,", ",LAT_MAX)

# Add a directory to the Python sys path

sys.path.append("src/")

# Import custom functions from local module

from Drift_Tools import ReturnUVlist, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore2, shift360, DriftersInZoneInPeriod, coord2index, DriftersInZoneInPeriod_remote

from Functions import Retrieve_traj_files,Retrieve_drifters_traj,Traj_filter_fidji,Report_card_plot,Retrieve_traj_files_diff,proba_density_2D_SK

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




# Retrieve the starting date and the ending date

DATE_START = WED[0]
STR_DATE_START= WED[0].strftime('%Y-%m-%d')
DATE_START_MIN1 = DATE_START-datetime.timedelta(days=1)
DATE_END= WED[-1]
STR_DATE_END=DATE_START.strftime('%Y-%m-%d')
DATE_END_PLUS1 = DATE_END+datetime.timedelta(days=1)
DATE_END_PLUS_DRIFT_TIME =  DATE_END+datetime.timedelta(days=DRIFT_TIME+1)
STR_DATE_END_PLUS_DRIFT_TIME=DATE_END_PLUS_DRIFT_TIME.strftime('%Y-%m-%d')
T_DRIFT = (DATE_END_PLUS_DRIFT_TIME-DATE_START).days


if FORECAST:
    for index_model in range(len(MODELS)):
        if ~MODELS[index_model].endswith('FCST'):
            MODELS[index_model] = MODELS[index_model] +'_FCST'

if False:
    if len(MODELS)==2:
        MODELS[0] = MODELS[0]+'_FCST'
    if len(MODELS) == 4:
        MODELS[0],MODELS[2] = MODELS[0]+'_FCST',MODELS[2]+'_FCST'

if AdvectionDiffusion_type == "K_smag":
    traj_lon_smag,traj_lat_smag,traj_time_smag = Retrieve_traj_files_diff(MODELS,DIR_OUT,AdvectionDiffusion_type)
elif AdvectionDiffusion_type == "AdvectionDiffusionM1":
    traj_lon_M1,traj_lat_M1,traj_time_M1 = Retrieve_traj_files_diff(MODELS,DIR_OUT,AdvectionDiffusion_type)
traj_lon,traj_lat,traj_time = Retrieve_traj_files(MODELS,DIR_OUT)

fout_driftersZone = 'Drifters_'+STR_REGION+'_'+STR_DATE_START+'_'+STR_DATE_END_PLUS_DRIFT_TIME+'.csv'
dir_drifters = DIR_OUT+'/DRIFTERS/'


outfile2 = dir_drifters+fout_driftersZone

traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id,nb_drifters_temp = Retrieve_drifters_traj(outfile2,T_DRIFT,COPERNICUS_DRIFTER)

# Plot Drifter's trajectories

dir_plot = DIR_OUT+'PLOT/'


if not os.path.exists(dir_plot):
    print('\ncreate folder : ', dir_plot)
    # Create directory where to store results:
    os.makedirs(dir_plot)

drifter_lon,drifter_lat = Traj_filter_fidji( DRIFT_TIME,traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time, traj_lon,traj_lat)





# In[36]:




# In[37]:


if COPERNICUS_DRIFTER:
    t_s = 24
else:
    t_s = 4


# In[38]:


dir_plot = DIR_OUT+'PLOT/'


if not os.path.exists(dir_plot):
    print('\ncreate folder : ', dir_plot)
    # Create directory where to store results:
    os.makedirs(dir_plot)


# In[39]:


id = 0 # Drifter choice
day = 0 # Wednesday of the month [0,1,2,3]
config_object = ConfigParser()
config_object.read(DIR_IN+"src/config_products_fidji.ini")
colors = []
for MODEL in MODELS:
    run_info = config_object[MODEL]
    colors.append(run_info["color"])

fig,ax = plt.subplots(figsize = (16,9))
CS = []
for MODELid in range (len(MODELS)):
    m1 = traj_lon_smag[MODELid,day,id::nb_drifters_temp].T[-1]
    m2 = traj_lat_smag[MODELid,day,id::nb_drifters_temp].T[-1]


    xmin = m1.min()-0.2
    xmax = m1.max()+0.2
    ymin = m2.min()-0.2
    ymax = m2.max()+0.2
    resx = 1/50
    resy = 1/50
    datax = m1
    datay = m2

    
    ax.plot(m1,m2,'o',c = colors[MODELid],ms = 1,label = MODELS[MODELid])
    ax.plot(traj_lon_smag[MODELid,day,id::nb_drifters_temp].T,traj_lat_smag[MODELid,day,id::nb_drifters_temp].T,c=colors[MODELid],alpha = 0.02)
        
    Xi, Yi, Zi , LVL = proba_density_2D_SK(m1, m2, resx, resy, xlim = [xmin,xmax], ylim = [ymin, ymax], quantiles = [0.95],
                    metric = 'haversine', bandwidth = 5, nsample = nb_repeats + 1, kernel = 'epanechnikov',
                    periodx = 360)  
    CS.append(ax.contour(Xi,Yi,Zi,levels = LVL,colors = colors[MODELid]))
    #ax.plot(traj_lon_smag[MODELid,day,id::nb_drifters_temp].T[-1],traj_lat_smag[MODELid,day,id::nb_drifters_temp].T[-1],'o',c = colors[MODELid],ms = 1)
    #ax.plot(traj_lon_smag[MODELid,day,id::nb_drifters_temp].T,traj_lat_smag[MODELid,day,id::nb_drifters_temp].T,c=colors[MODELid],alpha = 0.02)
    #confidence_ellipse(traj_lon_smag[MODELid,day,id::nb_drifters_temp].T[-1],traj_lat_smag[MODELid,day,id::nb_drifters_temp].T[-1],ax, n_std=3,label=r'$3\sigma$ '+MODELS[MODELid],edgecolor = colors[MODELid])
    ax.plot(traj_drifter_lon[id][0+day*t_s:DRIFT_TIME*t_s+1+day*t_s],traj_drifter_lat[id][0+day*t_s:DRIFT_TIME*t_s+1+day*t_s],c='k')
    ax.plot(traj_drifter_lon[id][0+day*t_s:DRIFT_TIME*t_s+1+day*t_s][-1],traj_drifter_lat[id][0+day*t_s:DRIFT_TIME*24+1+day*t_s][-1],'o')
    

    plt.legend()


plt.savefig(DIR_OUT+'PLOT/figure_particles_95_kde.png',dpi = 400)


# In[44]:


from shapely.geometry import Polygon, Point


# In[45]:


from importlib import reload

reload(sys.modules['Functions'])
from Functions import proba_density_2D_SK


# In[47]:


matplotlib.interactive(False)
in_out_array = np.zeros((traj_lon_smag.shape[0],traj_lon_smag.shape[1],int(traj_lon_smag.shape[2]/nb_repeats)))
prop_tot = []
area_contour = []
for i in range (0,traj_lon_smag.shape[0]):
    in_poly = 0
    tot = 0
    area_contour_int = []
    for day in tqdm(range(traj_lon_smag.shape[1])):
        area_contour_int1 = []
        for dri in range(int(traj_lon_smag.shape[2]/nb_repeats)):
            m1 = traj_lon_smag[i,day,dri::nb_drifters_temp].T[-1]
            m2 = traj_lat_smag[i,day,dri::nb_drifters_temp].T[-1]
            xmin = m1.min()-0.2
            xmax = m1.max()+0.2
            ymin = m2.min()-0.2
            ymax = m2.max()+0.2
            resx = 1/50
            resy = 1/50
            datax = m1
            datay = m2
            Xi, Yi, Zi , LVL = proba_density_2D_SK(m1, m2, resx, resy, xlim = [xmin,xmax], ylim = [ymin, ymax],
                                                   quantiles = [0.95],metric = 'haversine', bandwidth = 5, 
                                                   nsample = 1001, kernel = 'epanechnikov',periodx = 360)  

            contour = ax.contour(Xi,Yi,Zi,levels = LVL,colors = colors[i]).allsegs[0]
            pt = Point((traj_drifter_lon[dri][0+day*t_s:DRIFT_TIME*t_s+1+day*t_s][-1],
                        traj_drifter_lat[dri][0+day*t_s:DRIFT_TIME*t_s+1+day*t_s][-1]))
            
            if(len(np.shape(contour)) == 3):
                area = 0
                for k in range(len(contour)):
                    poly = Polygon(np.vstack([contour[k][:,0],contour[k][:,1]]).T)
                    area+= poly.area
                    if poly.contains(pt):
                        in_poly+=1
                        in_out_array[i,day,dri] = 1
            else:
                area = 0
                poly = Polygon(np.vstack([contour[:,0],contour[:,1]]).T)
                area+= poly.area
                if poly.contains(pt):
                    in_poly+=1
                    in_out_array[i,day,dri] = 1



            
            
            
            tot+=1
            area_contour_int1.append(area)
        area_contour_int.append(area_contour_int1)
    print("Proportion of drifter in 95% kde contour for model '", MODELS[i],"' : ",round(in_poly/tot*100,2),"%")
    prop_tot.append(round(in_poly/tot*100,2))
    area_contour.append(area_contour_int)       
        




# In[51]:


fig,axs = plt.subplots(figsize = (16,16),ncols=2,nrows = 1,subplot_kw={'projection': ccrs.PlateCarree()})
if len(MODELS) == 1:
    for i in range(len(MODELS)):
        axs.coastlines()
        axs.add_feature(cfeature.LAND)
        gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabels_top = False
        gl.ylabels_right = False
        fig.suptitle('Drifters in the confidence Ellipse 99% ' + str(MODELS) ,fontsize = 12)
        if REMOTE:
            
            im = axs.scatter(traj_drifter_lon[:,::24][:,:-3][:,:in_out_array.shape[1]],
                             traj_drifter_lat[:,::24][:,:-3][:,:in_out_array.shape[1]],
                             c = in_out_array[i],cmap = 'bwr_r',s = 1)
        else :
            
            im = axs.scatter(traj_drifter_lon[:,::4][:,:-3][:,:in_out_array.shape[1]],
                             traj_drifter_lat[:,::4][:,:-3][:,:in_out_array.shape[1]],
                             c = in_out_array[i],cmap = 'bwr_r',s = 1)
    
    fig.colorbar(im, ax=axs,orientation = 'horizontal')    
else :
    for i in range(2):
        axs[i].coastlines()
        gl = axs[i].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabels_top = False
        gl.ylabels_right = False
        fig.suptitle('Drifters in the confidence Ellipse 99% ' + str(MODELS) ,fontsize = 12)
        if REMOTE:
            im = axs[i].scatter(traj_drifter_lon[:,::24][:,:-3][:,:in_out_array.shape[1]],
                                     traj_drifter_lat[:,::24][:,:-3][:,:in_out_array.shape[1]],
                                    c = in_out_array[i],cmap = 'bwr_r',s = 1)
        else :
            im = axs[i].scatter(traj_drifter_lon[:,::4][:,:-3][:,:in_out_array.shape[1]],
                                traj_drifter_lat[:,::4][:,:-3][:,:in_out_array.shape[1]],
                                c = in_out_array[i],cmap = 'bwr_r',s = 1)
    
    fig.colorbar(im, ax=axs.ravel().tolist(),orientation = 'horizontal')
plt.savefig(DIR_OUT+'PLOT/confidence_contour_inout.png',dpi = 400)


# In[52]:


fig,axs = plt.subplots(figsize = (16,16),ncols=2,nrows = 1)
n_sep = []
x_sep = []
if len(MODELS) == 1:
    for i in range(len(MODELS)):
        fig.suptitle("Distribution of contour's area ",fontsize = 12)
        axs.set_title(MODELS[i])
        n1, x1, _ = axs.hist( np.array(area_contour[i]).flatten(),bins = 15,range=(0,np.max(np.array(area_contour).flatten())))
        axs.set_xlim([0,np.max(np.array(area_contour).flatten())])
        axs.set_ylim([0,200])    
        n_sep.append(n1)
        x_sep.append(x1)
        axs.set_xlabel('Area in $deg^{2}$')
        axs.set_ylabel('nb points')

else : 
    for i in range(2):
        fig.suptitle("Distribution of contour's area ",fontsize = 12)
        axs[i].set_title(MODELS[i])
        n1, x1, _ = axs[i].hist( np.array(area_contour[i]).flatten(),bins = 15,range=(0,np.max(np.array(area_contour).flatten())))
        axs[i].set_xlim([0,np.max(np.array(area_contour).flatten())])
        axs[i].set_ylim([0,200])
        n_sep.append(n1)
        x_sep.append(x1)
        axs[i].set_xlabel('Area in $deg^{2}$')
        axs[i].set_ylabel('nb points')

plt.savefig(DIR_OUT+'PLOT/area_contour.png',dpi = 400)


# In[53]:


fig,axs = plt.subplots(nrows=1, ncols=1,figsize = (9,5))
fig.suptitle('Distribution of 95% contour area',fontsize = 12)
for i in range (len(MODELS)):
    axs.plot(x_sep[i][:-1],n_sep[i],label =MODELS[i],color = colors[i])
axs.legend()
axs.set_xlabel('Area in $deg^{2}$')
axs.set_ylabel('nb points')


plt.savefig(DIR_OUT+'PLOT/area_contour_curves.png',dpi = 400)


# In[55]:


cell_size = 5
grid = np.meshgrid(np.linspace(LON_MIN,LON_MAX,int((LON_MAX - LON_MIN )/cell_size)+ 1),np.linspace(LAT_MIN,LAT_MAX,int((LAT_MAX - LAT_MIN )/cell_size)+ 1))


# In[56]:


sum_traj = np.zeros(grid[0].shape)
sum_elli = np.array([np.zeros(grid[0].shape)]*len(MODELS))

for mod in range (in_out_array.shape[0]):
    for i in range (grid[0].shape[0]-1):
        for j in range (grid[0].shape[1]-1):
            indice = np.where(np.array((grid[0][i,j]< traj_drifter_lon[:,::t_s][:,:-3][:,:in_out_array.shape[1]])*(traj_drifter_lon[:,::t_s][:,:-3][:,:in_out_array.shape[1]]<grid[0][i,j]+cell_size)*
                 (grid[1][i,j]< traj_drifter_lat[:,::t_s][:,:-3][:,:in_out_array.shape[1]])*(traj_drifter_lat[:,::t_s][:,:-3][:,:in_out_array.shape[1]]<grid[1][i,j]+cell_size)))
            if indice[0].shape[0]!= 0:
                sum_traj[i,j] = indice[0].shape[0]
                sum_elli[mod,i,j] = np.sum(in_out_array[mod][indice[1],indice[0]])
                


# In[57]:


sum_traj_tot = np.zeros(grid[0].shape)

for i in range (grid[0].shape[0]-1):
    for j in range (grid[0].shape[1]-1):
        indice = np.where(np.array((grid[0][i,j]< traj_drifter_lon[:,::t_s][:,:-3])*(traj_drifter_lon[:,::t_s][:,:-3]<grid[0][i,j]+cell_size)*
                 (grid[1][i,j]< traj_drifter_lat[:,::t_s][:,:-3])*(traj_drifter_lat[:,::t_s][:,:-3]<grid[1][i,j]+cell_size)))
        if indice[0].shape[0]!= 0:
            sum_traj_tot[i,j] = indice[0].shape[0]
            


# In[58]:


sum_traj_tot[np.where(sum_traj_tot==0)] = np.nan


# In[60]:


fig = plt.figure(figsize = (9,5))
axs=plt.axes(projection=ccrs.PlateCarree())
plt.title('Number of traj inside grid cells 2022 (daily pos, cell_size =  '+str(cell_size)+'°)')
cb = axs.pcolormesh(grid[0]+cell_size,grid[1]+cell_size,sum_traj_tot,vmax = 50)
gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.xlabels_top = False
gl.ylabels_right = False
axs.coastlines()
axs.add_feature(cfeature.LAND)
plt.colorbar(cb,label =  'nb traj')
plt.savefig(DIR_OUT+'PLOT/number_of_traj_in_grid_cells.png',dpi = 400)




fig,axs = plt.subplots(figsize = (16,16),ncols=2,nrows = 1,subplot_kw={'projection': ccrs.PlateCarree()})
if len(MODELS) == 1:
    for i in range(len(MODELS)):
        axs.coastlines()
        axs.add_feature(cfeature.LAND)
        gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabels_top = False
        gl.ylabels_right = False
        fig.suptitle('Proportion of drifters inside the 95% contour',fontsize = 12)
        axs.set_title(MODELS[i]+' : \n'+str(prop_tot[i])+"%")
        cb = axs.pcolormesh(grid[0]+cell_size,grid[1]+cell_size,sum_elli[i]/sum_traj * 100,vmin = 0,vmax = 100)
        axs.add_feature(cfeature.LAND)
    fig.colorbar(cb, ax=axs,orientation = 'horizontal',label="%")
    
else :   
    MIN_val = []
    MAX_val = []
    for i in range(2):
        MIN_val.append(100*np.ma.min(np.ma.masked_array(sum_elli[i]/sum_traj,   np.logical_or(np.isinf(sum_elli[i]/sum_traj) , np.isnan(sum_elli[i]/sum_traj)  ) )))
        MAX_val.append(100*np.ma.max(np.ma.masked_array(sum_elli[i]/sum_traj, np.logical_or(np.isinf(sum_elli[i]/sum_traj) , np.isnan(sum_elli[i]/sum_traj)  ) )))

    VMIN = np.min(MIN_val)
    VMAX = np.max(MAX_val)



    for i in range(2):
        axs[i].coastlines()
        axs[i].add_feature(cfeature.LAND)
        gl = axs[i].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabels_top = False
        gl.ylabels_right = False
        fig.suptitle('Proportion of drifters inside the 95% contour',fontsize = 12)
        axs[i].set_title(MODELS[i]+' : \n'+str(prop_tot[i])+"%")
        cb = axs[i].pcolormesh(grid[0]+cell_size,grid[1]+cell_size,sum_elli[i]/sum_traj * 100,vmin = VMIN,vmax = VMAX)
        axs[i].add_feature(cfeature.LAND)

    fig.colorbar(cb, ax=axs.ravel().tolist(),orientation = 'horizontal',label="%")
plt.savefig(DIR_OUT+'PLOT/map_proportion_contour.png',dpi = 400)




######### TEST MAP functions ####################
from plot_functions import comparative_maps

MAPS = []
AXS_TITLES = []
for i in range(2):
    MAPS.append(100*sum_elli[i]/sum_traj)
    AXS_TITLES.append(MODELS[i]+' : \n'+str(prop_tot[i])+"%")
comparative_maps(MAPS,AXS_TITLES,cell_size,grid,'Proportion of drifters inside the 95% contour',DIR_OUT+'PLOT/map_proportion_contour_teqst.png')



# In[62]:

if(0):
    fig,axs = plt.subplots(figsize = (16,9),ncols=1,subplot_kw={'projection': ccrs.PlateCarree()},facecolor = 'white')

    axs.coastlines()
    axs.add_feature(cfeature.LAND)
    gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabels_top = False
    gl.ylabels_right = False
    fig.suptitle('Difference of proportion of drifters inside the 95% contour [FCST0 - FCST9]',fontsize = 12)
    cb = axs.pcolormesh(grid[0]+cell_size,grid[1]+cell_size,(sum_elli[0]-sum_elli[-1])/sum_traj * 100,vmin = -100,vmax = 100,cmap = "bwr")
    axs.add_feature(cfeature.LAND)

    fig.colorbar(cb, ax=axs,orientation = 'horizontal',label="%")
    plt.savefig(DIR_OUT+'PLOT/map_proportion_contour_diff_FCST0_FCST9.png',dpi = 400)        




