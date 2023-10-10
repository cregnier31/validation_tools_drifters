#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os,glob
import time
import sys
import xarray as xr
import pandas as pd
from Drift_Tools import DriftersInZoneInPeriod, ReturnUVlist, ExtractSurface
from configparser import ConfigParser
from tqdm import tqdm
import datetime

def init_fig(title=None, xlabel=None, ylabel=None, y2label=None, xlim=None, ylim=None, y2lim=None, zero=None, c1='k', c2='grey', figsize=(10, 6), fontsize=14):
    """
    Initialize a figure with 1 ax or 2 axes (if y2label), title and labels.
    zero: float (horizontal line) or 'diag' (diagonal line). Need xlim
    c1/c2: ax and ax2 colors
    xlim/ylim: xlim=(x0,x1)
    """
    # General
    plt.rc('ytick',labelsize=fontsize); plt.rc('xtick',labelsize=fontsize)
    fig, ax = plt.subplots()

    fig.set_size_inches(figsize)
    if title is not None:      
        plt.title(title, fontsize=fontsize+2, fontweight='bold')
    ax.grid(linestyle='--', zorder = 0)

    # Left axis
    set_axis(ax, ylabel, c1, fontsize)
    if ylim: ax.set_ylim(ylim)
    
    # Right axis
    if y2label is not None:
        ax2  = ax.twinx()
        set_axis(ax2, y2label, c2, fontsize)
        if y2lim: ax2.set_ylim(y2lim)

    # x axis
    if xlim: plt.xlim(xlim)
    if xlabel is not None: 
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    
    # zero
    plot_zero(ax, zero, xlim)
    
    # Return
    if y2label is not None:
        return (fig, ax, ax2)
    else:
        return (fig, ax)



    
def save_fig(fname, dpi=200, show=False):
    """
    Save figure fname as fname.png
    """
    if fname:
        fname += '.png'
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        print ('Saved ', fname)
        if show:
            plt.show()
        plt.close()
        
        
def set_axis(ax, label=None, col='k', fontsize=12):
    """
    set ax with color col and label 
    """
    if label is not None:
        ax.set_ylabel(label, fontsize=fontsize, fontweight='bold')
    ax.tick_params(axis='y', colors = col)
    ax.yaxis.label.set_color(col)
    
    
def plot_zero(ax, zero, xlim, c='k', vertical=False):
    """
    Plot an horizontal line at y=zero or vertical line if vertical==True or diagonal line.
    xlim: xlim=(0,10) for example.
    """
    if zero != None and xlim:
        if vertical:
            y = np.linspace(xlim[0], xlim[1])
            ax.plot(zero * np.ones(len(y)), y,'--k', zorder=3, c=c)
        else:
            x = np.linspace(xlim[0], xlim[1])
            if zero == 'diag':
                ax.plot(x, x,'--k', zorder=3, c=c)   
            else:
                ax.plot(x, zero * np.ones(len(x)),'--k', zorder=3, c=c)

def init_polar(title=None, xlabel=None, ylabel=None, figsize=(8, 5), fontsize=14):

    plt.rc('ytick',labelsize=fontsize); plt.rc('xtick',labelsize=fontsize)
    fig = plt.figure(figsize=figsize) 
    ax = plt.subplot(111, projection='polar')
    plt.title(title, fontsize=fontsize+2, fontweight='bold')
    
    return fig, ax 



def compute_directions(U, V, nb_directions, bins_classes):
    """
    Compute directions histogram for current rose.
    Return:
        rose: yaxis of histogram, size nb_directions * nb_classes
              (nb of occurences per speed class and per direction).
        directions: x axis of histogram (-pi -> pi)
    """
    nb_classes = len(bins_classes) - 1
    
    # Norm and direction
    speed = np.array(np.sqrt(U**2 + V**2))
    theta = np.array(np.arctan2(V, U))
    
    # Flatten and remove nan
    speed = speed[~np.isnan(speed)]
    theta = theta[~np.isnan(theta)]

    # Compute polar histogram
    hist = np.histogram(theta, bins=nb_directions, range=(-np.pi, np.pi))   

    # Compute histogram on norm for each direction
    rose = np.zeros((nb_directions, nb_classes))
    for k in range(nb_directions):
        idx = np.where((theta >= hist[1][k]) & (theta < hist[1][k+1]))[0]
        sp = speed[idx]
        h = np.histogram(sp, bins=bins_classes)
        rose[k, :] = h[0]
    print(rose)
    directions = (hist[1][:-1] + hist[1][1:]) / 2
    return rose, directions


def current_rose(U, V, bins_classes=[0,0.1,0.2,0.3,0.5], title=None, fname=False, nb_directions=32, pmax='auto', legend=True, fontsize=20,diff = False):
    """
    Plot circular histogram of current velocity with different colors depending on current norm.
    
    Parameters:
        -U/V: current zonal/meridional velocity
        -bins_classes: classes of current velocity. Last class is V > bins_classes[-1]
        -title: figure title
        -save: False or figure name
        -nb_directions: divide the circle in nb_directions categories
        -pmax: maximum percentage of data for yaxis (auto or int)
        -legend: boolean, to plot legend
    """
    
    # Calculate circular histogram
    rose, directions = compute_directions(U, V, nb_directions, bins_classes)


    # Useful variables for plot
    delta = 2 * np.pi / nb_directions - 0.05 # Angle du segment de cercle
    width = np.ones(nb_directions) * 2 * np.pi /nb_directions - 0.05
    colors = ['blue', 'deepskyblue', 'greenyellow', 'yellow', 'orange', 'red', 'purple', 'k'] # max 8 classes
    if legend:
        labels = ['%.2f - %.2f m/s'%(bins_classes[k], bins_classes[k+1]) for k in range(len(bins_classes)-1)]
        labels[-1] = '> %.2f m/s'%bins_classes[-1]
    else:
        labels = [None for i in range(len(bins_classes))]

    # Decoration
    fig, ax = init_polar(figsize=(10,12), fontsize=fontsize)

    # Plot
    bottom=0
    nb_data = np.sum(~np.isnan(U))
    for k in range(len(bins_classes) - 1):
        height = rose[:, k] * 100 / nb_data
        ax.bar(directions, height, width=width, bottom=bottom, edgecolor='k', color = colors[k], label=labels[k])
        bottom += height

    # Maximum % for yticks
    if pmax == 'auto':
        pmax = np.max([np.sum(rose[k, :]) for k in range(nb_directions)]) * 100 / nb_data
    
    # Plot mean direction       
    mean_dir = np.arctan2(np.nanmean(V), np.nanmean(U))
    ax.plot([mean_dir, mean_dir],[0,pmax], '-.',color='k',linewidth =5)

    # Legend, title and ticks
    if legend:
        plt.legend(fontsize=fontsize, loc=(1,0), title='Current velocity (m/s)', title_fontsize=fontsize+2)
    plt.title(title, fontsize=fontsize+8, fontweight='bold')
    ax.set_xticks(np.linspace(0,2*np.pi-np.pi/8,16))
    if diff==True:
        ax.set_xticklabels(np.linspace(0,360 - 360 /16,16), fontsize=fontsize, fontweight='bold')    

    else :
        ax.set_xticklabels(('E','','NE','','N','','NW','','W','','SW','','S','','SE',''), fontsize=fontsize, fontweight='bold')    
    ax.set_ylabel('Percentage of total data', fontsize=fontsize, position=(0,0.75), rotation=60)
    ax.set_rlabel_position(150)      
    nticks = 5
    yticks = np.linspace(pmax/nticks, pmax, nticks, dtype='int')
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i) + '%' for i in yticks], fontsize=fontsize)

    # Save figure
    save_fig(fname)






def proba_density_2D_SK(datax, datay, resx, resy, xlim = [0,360], ylim = [-90, 90], quantiles = [], metric = 'haversine', bandwidth = 200, nsample = 1e6, kernel = 'epanechnikov', periodx = 360):
    from sklearn.neighbors import KernelDensity
    from scipy.optimize import minimize_scalar
    from scipy import optimize

    """
    Compute de the probability density function (pdf) of positions datax and datay on a regular grid, with a kernel density estimation method.
    inputs :
        - resx, resy : resolution of the output grid
        - xlim, ylim : limits of the the output grid
        - quantiles  : list of quantiles for compute corresponding contour of the pdf
        - metric : metric of the methode. If datax and datay are lon-lat coordinate best metric is haversine
        - bandwidth in km : determine the smoothing of the method
        - kernel : kernel used in the kde method
        - nsample : number of positions taken into account (the purpose is to make the computation faster)
    outputs :
        - Xi, Yi : coordinates of the regular grid
        - zi : pdf values on the regular grid
        - levels : levels of contour of the pdf corresponding to the quantiles in input 
    """


    x = np.array(datax).flatten()
    y = np.array(datay).flatten()
    idx = ~np.isnan(x) & ~np.isnan(y)
    x = x[idx]
    y = y[idx]

    if len(x) > nsample :
        rand_ind = random.sample(range(len(x)), int(nsample))
        x = x[rand_ind]
        y = y[rand_ind]

    if periodx is not None :
        x[x<xlim[0]] += periodx
        x[x>xlim[1]] -= periodx        
    try :
        Xi, Yi = np.mgrid[xlim[0] + resx/2:xlim[1] + resx/2: resx, ylim[0] +  resy/ 2:ylim[1]+resy/2:resy]
    except ValueError:
        Xi = np.nan * np.ones(np.mgrid[0 + resx/2:1 + resx/2: resx, 0 +  resy/ 2:1+resy/2:resy].shape)        
        levels = [np.nan for q in quantiles]
        return Xi[0],Xi[0],Xi[0],levels

    xy_sample = np.vstack([Yi.flatten(), Xi.flatten()]).T
    xy_train  = np.vstack([y, x]).T

    R = 6371 
    if metric == 'haversine':
        bandwidth = bandwidth/R

    kde = KernelDensity(kernel = kernel, bandwidth = bandwidth, metric = metric)
    kde.fit(np.radians(xy_train))
    zi = np.exp(kde.score_samples(np.radians(xy_sample)))
    zi = zi / np.sum(zi)

    levels = []
    for q in quantiles :
        def diff(value):
            return abs(np.sum(zi[zi>value]) - q)
        res = minimize_scalar(diff ,bounds=(0,np.max(zi)), method='bounded')
        level = res.x
        levels.append(level)

    
    return Xi, Yi, np.reshape(zi, Xi.shape) , levels




def Retrieve_drifter_local(dir_out,lon_min,lon_max,lat_min,lat_max,date_start,date_end,str_date1,str_date2,T_DRIFT,YYYY1,str_region,tay_diag = False):

    dir_in='/data/rd_exchange/svangennip/DATA/GlobalDriftingProgram//YEAR/'

    fname_in='Drifters_'+str(YYYY1)+'.csv'
    f_drifters_all=dir_in+fname_in
    if tay_diag : 
        fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'_taylor_diag.csv'
    else : 
        fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'.csv'
    dir_drifters = dir_out+'/DRIFTERS/'

    if not os.path.exists(dir_drifters):
        print('\ncreate folder : ', dir_drifters)
        # Create directory where to store results:
        os.makedirs(dir_drifters)
    outfile2 = dir_drifters+fout_driftersZone

    if not os.path.isfile(outfile2):
        DriftersInZoneInPeriod(lon_min, lon_max, lat_min, lat_max, date_start, date_end, f_drifters_all, outfile2)
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
    nb_points = T_DRIFT*4+1 
    traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
    traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
    traj_drifter_date = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
    traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
    traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) ) 

    return traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id

def Retrieve_data_local(MODELS,date_start,date_end_plus_DRIFT_TIME,dir_out,dir_in,dir_work,depth):
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

    for i in range(len(MODELS)):
        MODEL = MODELS[i]
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

        if MODEL in ['PSY4V3R1','IBI36','GLO12V4','PSY44']:
            u_files_mod_3D, v_files_mod_3D = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME)
            u_files, v_files = ExtractSurface(MODEL,u_files_mod_3D, v_files_mod_3D,zlevel,var_u_mod[i], var_v_mod[i], dir_work+'MODEL_OUTPUT/'+MODEL+'/UV/15m/')
            u_files_mod.append(u_files)
            v_files_mod.append(v_files)
        else : 
            u_files, v_files = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME)
            u_files_mod.append(u_files)
            v_files_mod.append(v_files)


    print('path_UV: ', path_UV[0])
    print('fname_prefix: ', fname_prefix[0])
    return u_files_mod,v_files_mod,var_u_mod,var_v_mod,fname_prefix,meshmask

def Calculate_model_traj(MODELS,dir_out,nb_points,traj_drifter_lon,traj_drifter_lat,traj_drifter_id,date_start,DRIFT_TIME,copernicus_drifter,u_files_mod,v_files_mod,var_u_mod,var_v_mod,meshmask,AdvectionDiffusion_type,nb_repeats,cloud_stat = False):

    from Drift_exec import run_parcels_Drift, run_parcels_Drift_remote
    

    if copernicus_drifter and not MODELS[0]=='FCST':
        for i in range(len(MODELS)):
            dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'
            print(np.arange(0,nb_points-DRIFT_TIME*24,24))
            for j in tqdm(np.arange(0,nb_points-DRIFT_TIME*24,24)): # one sim per day
                print('nb points : ',nb_points)
                traj_drifter_lon_start = traj_drifter_lon[:,j]
                traj_drifter_lat_start = traj_drifter_lat[:,j]
                traj_drifter_id_start = traj_drifter_id[:,j]
                date_sim = date_start + datetime.timedelta(days=int(j//24))
                str_date_sim = date_sim.strftime('%Y-%m-%d')
                f_traj1 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc.zarr'
                f_traj2 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_and_around.zarr'
                f_traj3 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_smag.zarr'
                f_traj4 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_M1.zarr'
                #            if not (os.path.isfile(f_traj1) & os.path.isfile(f_traj2)) :
                print('#run parcels with just the SVP locations')
                print('\nModels Files : \n',u_files_mod[i][j//4:-1])
                print('\Drift time  : ',DRIFT_TIME)
                if cloud_stat:
                    print('#run parcels with at SVP neigbouring locations')
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//24:-1], var_u_mod[i], v_files_mod[i][j//24:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                      time_period=False)
                if AdvectionDiffusion_type == "K_smag":
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//24:-1], var_u_mod[i], v_files_mod[i][j//24:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj3, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if AdvectionDiffusion_type == "AdvectionDiffusionM1":
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//24:-1], var_u_mod[i], v_files_mod[i][j//24:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if True :
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//24:-1], var_u_mod[i], v_files_mod[i][j//24:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, \
                                      DRIFT_TIME,f_traj1, radius_cloud=0, seeding_resolution=0, \
                                      time_period=False)
                print(' done...')


    elif copernicus_drifter and MODELS[0]=='FCST':
        for i in range(len(MODELS)):
            dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'
            print(np.arange(0,nb_points-DRIFT_TIME*24,24))
            for j in tqdm(np.arange(0,nb_points-DRIFT_TIME*24,24*DRIFT_TIME)): # one sim per day
                print('nb points : ',nb_points)
                traj_drifter_lon_start = traj_drifter_lon[:,j]
                traj_drifter_lat_start = traj_drifter_lat[:,j]
                traj_drifter_id_start = traj_drifter_id[:,j]
                date_sim = date_start + datetime.timedelta(days=int(j//24))
                str_date_sim = date_sim.strftime('%Y-%m-%d')
                f_traj1 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc.zarr'
                f_traj2 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_and_around.zarr'
                f_traj3 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_smag.zarr'
                f_traj4 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_M1.zarr'
                #            if not (os.path.isfile(f_traj1) & os.path.isfile(f_traj2)) :
                print('#run parcels with just the SVP locations')
                print('\nModels Files : \n',u_files_mod[i][j//4:-1])
                print('\Drift time  : ',DRIFT_TIME)
                if AdvectionDiffusion_type == "K_smag":
                    run_parcels_Drift(MODELS[i], u_files_mod[j//24:], var_u_mod[i], v_files_mod[j//24:], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj3, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if AdvectionDiffusion_type == "AdvectionDiffusionM1":
                    run_parcels_Drift(MODELS[i], u_files_mod[j//24:], var_u_mod[i], v_files_mod[j//24:], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if True :
                    run_parcels_Drift(MODELS[i], u_files_mod[j//24:], var_u_mod[i], v_files_mod[j//24:], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, \
                                      DRIFT_TIME,f_traj1, radius_cloud=0, seeding_resolution=0, \
                                      time_period=False)
                print(' done...')


    else:

        for i in range(len(MODELS)):
            dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'
            for j in tqdm(np.arange(0,nb_points-DRIFT_TIME*4-4,4)): # one sim per day
                print('nb points : ',nb_points)

                traj_drifter_lon_start = traj_drifter_lon[:,j]
                traj_drifter_lat_start = traj_drifter_lat[:,j]
                traj_drifter_id_start = traj_drifter_id[:,j]
                date_sim = date_start + datetime.timedelta(days=int(j//4))
                str_date_sim = date_sim.strftime('%Y-%m-%d')
                f_traj1 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc.zarr'
                f_traj2 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_and_around.zarr'
                f_traj3 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_smag.zarr'
                f_traj4 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_M1.zarr'


    #            if not (os.path.isfile(f_traj1) & os.path.isfile(f_traj2)) :
                print('#run parcels with just the SVP locations')
                print('\nModels Files : \n',u_files_mod[i][j//4:-1])
                print('\Drift time  : ',DRIFT_TIME)

                if cloud_stat:
                    print('#run parcels with at SVP neigbouring locations')
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//4:-1], var_u_mod[i], v_files_mod[i][j//4:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                      time_period=False)
                if AdvectionDiffusion_type=="Smag":
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//4:-1], var_u_mod[i], v_files_mod[i][j//4:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj3, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if AdvectionDiffusion_type=="AdvectionDiffusionM1":
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//4:-1], var_u_mod[i], v_files_mod[i][j//4:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = nb_repeats)


                if True :
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//4:-1], var_u_mod[i], v_files_mod[i][j//4:-1], var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, \
                                      DRIFT_TIME,f_traj1, radius_cloud=0, seeding_resolution=0, \
                                      time_period=False)

                print(' done...')


def Calculate_speed_at_svp_location(MODELS,date_start_min1,date_end_plus1,dir_out,dir_in,lon_uv_svp,lat_uv_svp,traj_drifter_date,depth,copernicus_drifter):
    zlevel = 11
    R =  6378.137
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
    indice_date = []
    for MODEL in MODELS:
        run_info = config_object[MODEL]
        path_UV.append(run_info["path"])
        fname_prefix.append(run_info["fname_prefix"])
        var_u_mod.append(run_info["var_u"])
        var_v_mod.append(run_info["var_v"])
        meshmask.append(run_info["meshmask"])
        colors.append(run_info["color"])
        indice_date.append(int(run_info["indice_date"]))
        dir_work_mod.append(dir_out+'/EXP/'+MODEL+'/')
        u_files, v_files = ReturnUVlist(MODEL, date_start_min1, date_end_plus1)
        u_files_mod.append(u_files)
        v_files_mod.append(v_files)

    if copernicus_drifter:
        date_uv_svp = traj_drifter_date[:,::24][:,1:] 

    else : 
        date_uv_svp = traj_drifter_date[:,::4][:,1:-1]



    U_model = []
    V_model = []

    for u_files,v_files,var_u,var_v,MODEL,ind_date  in zip(u_files_mod,v_files_mod,var_u_mod,var_v_mod,MODELS,indice_date):
        u_mod= []
        v_mod= []
        for u_file,v_file in zip(u_files,v_files):
            print(u_file[ind_date:ind_date+8])
            str_date = u_file[ind_date:ind_date+8]
            str_date = str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:]
            if MODEL == "IBI36" or MODEL== "IBI12": 
                u = xr.open_dataset(u_file).sel(depthu = depth,method = "nearest")
                v = xr.open_dataset(v_file).sel(depthv = depth,method = "nearest")
            elif MODEL == "PSY4V3R1" or MODEL == "PSY44":
                u = xr.open_dataset(u_file).sel(deptht = depth,method = "nearest")
                v = xr.open_dataset(v_file).sel(deptht = depth,method = "nearest")
            elif MODEL == "GLOBC":
                u = xr.open_dataset(u_file).sel(depth = depth,method = "nearest")
                v = xr.open_dataset(v_file).sel(depth = depth,method = "nearest")
            elif MODEL == "GLO12V4":
                u = xr.open_dataset(u_file).sel(depthu = depth,method = "nearest")
                v = xr.open_dataset(v_file).sel(depthv = depth,method = "nearest")
            elif "GLO12_FCST" in MODEL:
                u = xr.open_dataset(u_file).sel(depth = depth,method = "nearest")
                v = xr.open_dataset(v_file).sel(depth = depth,method = "nearest")

            uo = u[var_u].values
            vo = v[var_v].values

            print(date_uv_svp)
            print(str_date)
            ind_svp = np.where(date_uv_svp==str_date)
            print(ind_svp)
            if MODEL in ['PSY4V3R1','GLO12V4','IBI36',"PSY44","IBI12"]:
                latu = u.nav_lat.values
                lonu = u.nav_lon.values
                latv = v.nav_lat.values
                lonv = v.nav_lon.values
                if ind_svp[0].shape[0] !=0:
                    lon_svp = lon_uv_svp[ind_svp]
                    lat_svp = lat_uv_svp[ind_svp]
                    ind_modu = []
                    ind_modv = []
                    for la,lo in zip(lat_svp,lon_svp):
                        ind_modu.append(np.unravel_index(np.sqrt((np.radians(latu-la)*R)**2 + 
                                                          ((np.radians(lonu-lo)*np.cos(np.radians(latu)))*R)**2).argmin(),latu.shape))
                        ind_modv.append(np.unravel_index(np.sqrt((np.radians(latv-la)*R)**2 + 
                                                          ((np.radians(lonv-lo)*np.cos(np.radians(latv)))*R)**2).argmin(),latv.shape))
                    ind_modu = np.array(ind_modu)
                    ind_modv = np.array(ind_modv)


                    u_mod.append(uo[0][ind_modu[:,0],ind_modu[:,1]])
                    v_mod.append(vo[0][ind_modv[:,0],ind_modv[:,1]])
            else:
                latu = u.latitude.values
                lonu = u.longitude.values
                latv = v.latitude.values
                lonv = v.longitude.values
                if ind_svp[0].shape[0] !=0:
                    lon_svp = lon_uv_svp[ind_svp]
                    lat_svp = lat_uv_svp[ind_svp]
                    u_int = []
                    v_int = []
                    for la,lo in zip(lat_svp,lon_svp):  
                        u_int.append(u[var_u].interp(longitude = lo,latitude = la,method='nearest').values[0])# zlevel = 15m
                        v_int.append(v[var_v].interp(longitude = lo,latitude = la,method='nearest').values[0])# zlevel = 15m
                    u_mod.append(u_int)
                    v_mod.append(v_int)
        U_model.append(np.array(u_mod))
        V_model.append(np.array(v_mod))

    U_model = np.array(U_model)
    V_model = np.array(V_model)

    return U_model, V_model,colors

# rmse curve 
def corr_coeff_for_rms(rms,std_r,std_t):
    return (std_r**2+std_t**2-rms**2)/(2*std_r*std_t)


def Taylor_diagram(dir_plot,u_svp,u_corrcoef,u_std,colors,MODELS):


    fig, ax = plt.subplots(figsize = (8,8),subplot_kw={'projection': 'polar'})
    std_ru = np.mean(np.std(u_svp,axis = 1))
    std_drifters_svp_u = np.std(u_svp,axis = 1)

    for i in np.arange(0.02,0.15,0.02):
        r = np.linspace(0.001,0.16,100)
        theta = -corr_coeff_for_rms(i,std_ru,r)*np.pi/2+np.pi/2
        r = np.delete(r,np.where((theta<-np.pi)+(theta > np.pi))[0])
        theta = np.delete(theta,np.where((theta<-np.pi)+(theta > np.pi))[0])
        line, = ax.plot(theta,r,c = 'k',alpha = 0.5)
        if i == 0.02:
            line.set_label('centered RMS')
        if theta[np.argmin(abs(r-0.1))]< np.pi/2 and theta[np.argmin(abs(r-0.1))]> 0 : 
            t = ax.text(theta[np.argmin(abs(r-0.1))],0.1, str(round(i,2)), fontsize=11)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        elif theta[np.argmin(abs(r-0.1))]< 0: 
            t = ax.text(theta[np.argmin(abs(r-std_ru))],std_ru, str(round(i,2)), fontsize=11)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        else : 
            t = ax.text(theta[np.argmin(abs(r-0.14))],0.14, str(round(i,2)), fontsize=11)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))   
            
    ax.set_rticks(np.linspace(0,0.16,9))  # Less radial ticks
    ax.set_rlabel_position(90)  # Move radial labels away from plotted line

    ax.set_thetamin(0)
    ax.set_thetamax(90)
    #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_xticks(np.linspace(0,np.pi/2,11))
    ax.set_xticklabels([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0])
    ax.set_xlabel('STD ($m.s^{-1}$)')
    ax.xaxis.set_label_coords(.5, -.05)

    #labelLines(ax.get_lines(),xvals=(0.1, 2),zorder=2.5)
    #labelLines(ax.get_lines(),xvals=(0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2),zorder=2.5)
    for i in range (len(MODELS)):
        ax.plot(-u_corrcoef[i]*np.pi/2+np.pi/2, u_std[i],'o',c = colors[i],alpha = 0.3,ms = 4)
        ax.plot(np.nanmean(-u_corrcoef[i]*np.pi/2+np.pi/2), np.nanmean(u_std[i]),'o',c = colors[i],ms = 10,label = 'MEAN '+MODELS[i])

    ax.plot(np.linspace(0,np.pi/2,100),std_ru*np.ones(100),'--',c = 'cyan')
    ax.plot(0,std_ru, '*',c= 'cyan',ms = 15, clip_on=False, zorder=100,label = 'Mean std drifters')
    ax.plot(np.zeros(std_drifters_svp_u.shape),std_drifters_svp_u, '*',c= 'k',ms = 8, clip_on=False, zorder=100,label = 'std drifters')

    plt.legend()
    ax.text(np.pi/6,0.165,'Correlation',fontsize = 12,rotation = -np.arccos(np.pi/6)*180/np.pi)

    ax.set_rmax(0.16)
    ax.set_rmin(0)
    #ax.set_title('Taylor Diagram for zonal velocity U')
    ax.text(np.pi/1.8,0.06,'STD ($m.s^{-1}$)',fontsize = 12,rotation =90)

    plt.savefig(dir_plot,dpi = 600)



def Retrieve_traj_files(MODELS,dir_out):
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
    return traj_lon,traj_lat,traj_time

def Retrieve_drifters_traj(outfile2,T_DRIFT,remote,copernicus_drifter):
    print('loading ',outfile2)
    drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
    print(drifters_zone_traj.head(5))
    print(drifters_zone_traj.dtypes)


    nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))

    #Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
    if remote or copernicus_drifter:
        nb_points = T_DRIFT*24   # hourly
    else :
        nb_points = T_DRIFT*4 + 1    # hourly

    traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters_temp, nb_points) )
    traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters_temp, nb_points) )
    traj_drifter_date = np.reshape(drifters_zone_traj.date.values,(nb_drifters_temp, nb_points) )
    traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters_temp, nb_points) )
    traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters_temp, nb_points) )
    return traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id


def Traj_filter(remote,copernicus_drifter, DRIFT_TIME,traj_drifter_lon,traj_drifter_lat, traj_lon,traj_lat):

    drifter_lon = np.zeros(traj_lon.shape)
    drifter_lat = np.zeros(traj_lat.shape)

    if remote or copernicus_drifter:
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
    return drifter_lon,drifter_lat



def Report_card_plot(MODELS,prop_skill_score,prop_sep_dist,prop_corr_lon,prop_corr_lat,dir_in,str_date1,str_date2,dir_plot):

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


    config_object = ConfigParser()
    config_object.read(dir_in+"src/config_products.ini")
    colors_list = []
    for MODEL in MODELS:
        run_info = config_object[MODEL]
        colors_list.append(run_info["color"])











    df.T[1:].plot.bar(rot = 0,title = "Lagrangian statistics from " + str_date1+' to '+str_date2,ylabel = 'Proportion',xlabel = 'Scores',color = colors_list)
    plt.legend(labels = df.MODELS.values)
    plt.savefig(dir_plot+'Lag_stats_'+str_date1+'_'+str_date2+'.png',dpi = 600)

    print("Save fig : ", dir_plot+'Lag_stats_'+str_date1+'_'+str_date2+'.png')


     























