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

# =========================================================
# 4. BINNING ON GRID
# =========================================================

def bin_spatial(array_daily, traj_lon, traj_lat, res=2):

    # --- grid definition
    lon_bins = np.arange(-180, 181, res)
    lat_bins = np.arange(-90, 91, res)

    nlon = len(lon_bins) - 1
    nlat = len(lat_bins) - 1
    ndays = array_daily.shape[1]

    sum_sep = np.zeros((nlat, nlon, ndays))
    count_sep = np.zeros((nlat, nlon, ndays))

    # --- initial positions
    lon0 = traj_lon[:, 0]
    lat0 = traj_lat[:, 0]

    # wrap longitude
    lon0 = ((lon0 + 180) % 360) - 180

    ilon = np.digitize(lon0, lon_bins) - 1
    ilat = np.digitize(lat0, lat_bins) - 1

    valid = (ilon >= 0) & (ilon < nlon) & (ilat >= 0) & (ilat < nlat)

    ilon = ilon[valid]
    ilat = ilat[valid]
    sep_valid = array_daily[valid]

    # --- vectorized accumulation
    for t in range(ndays):
        np.add.at(sum_sep[:, :, t], (ilat, ilon), sep_valid[:, t])
        np.add.at(count_sep[:, :, t], (ilat, ilon), 1)

    # --- mean
    mean_sep = sum_sep / count_sep
    mean_sep[count_sep == 0] = 0

    return mean_sep, count_sep


def bin_spatial2(array_daily, traj_lon, traj_lat, res=2):

    import numpy as np

    # -------------------------------------------------
    # Grid definition
    # -------------------------------------------------
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    nlon = len(lon_bins) - 1
    nlat = len(lat_bins) - 1
    ndays = array_daily.shape[1]

    # -------------------------------------------------
    # Arrays (lon, lat, leadtime)
    # -------------------------------------------------
    #sum_array = np.zeros((nlon, nlat, ndays))
    #count_array = np.zeros((nlon, nlat, ndays))
    sum_array = np.zeros((ndays, nlon, nlat))
    count_array = np.zeros((ndays, nlon, nlat))

    # -------------------------------------------------
    # Initial trajectory positions
    # -------------------------------------------------
    lon0 = traj_lon[:, 0]
    lat0 = traj_lat[:, 0]

    # longitude wrap
    lon0 = ((lon0 + 180) % 360) - 180

    # avoid edge issues
    lon0 = np.clip(lon0, -179.9999, 179.9999)
    lat0 = np.clip(lat0, -89.9999, 89.9999)

    # -------------------------------------------------
    # Bin indices
    # -------------------------------------------------
    ilon = np.digitize(lon0, lon_bins) - 1
    ilat = np.digitize(lat0, lat_bins) - 1

    valid = (
        (ilon >= 0) & (ilon < nlon) &
        (ilat >= 0) & (ilat < nlat)
    )

    ilon = ilon[valid]
    ilat = ilat[valid]

    array_valid = array_daily[valid]

    # -------------------------------------------------
    # Accumulate
    # -------------------------------------------------
    for t in range(ndays):

        np.add.at(
            #sum_array[:, :, t],
            sum_array[t, :, :],
            (ilon, ilat),
            array_valid[:, t]
        )

        np.add.at(
            #count_array[:, :, t],
            count_array[t, :, :],
            (ilon, ilat),
            1
        )

    # -------------------------------------------------
    # Mean
    # -------------------------------------------------
    mean_array = np.divide(
        sum_array,
        count_array,
        out=np.full_like(sum_array, np.nan),
        where=count_array != 0
    )

    return mean_array, count_array


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



    nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))
    #remove drifters with missing entries (should be nb_days*4+1)
    #Check each drifter has n days * 4 entries
    drifter_ids = pd.unique(drifters_zone_traj['id'])
    spurious_drifters=[]
    for i in range(nb_drifters_temp):
        nb_entries = len(drifters_zone_traj[drifters_zone_traj.id == drifter_ids[i]])
        if not nb_entries >= (T_DRIFT*4+1):
            #print(drifter_ids[i],': not enough entries, only ', nb_entries)
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
    config_object.read(dir_in+"src/config_products_fidji.ini")
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
            u_files, v_files = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME,dir_in)
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
                print(np.shape(u_files_mod)[0])
                #print('\nModels Files : \n',u_files_mod[i][j//4:-1])
                print('\Drift time  : ',DRIFT_TIME)
                if cloud_stat:
                    print('#run parcels with at SVP neigbouring locations')
                    print(f'RRRRR   #### {np.shape(u_files_mod)}')
                    if np.size(u_files_mod) >1:
                        uuu = u_files_mod[i][j//24:-1]
                        vvv = v_files_mod[i][j//24:-1]

                        run_parcels_Drift(MODELS[i], uuu, var_u_mod[i], vvv, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                      time_period=False)
                if AdvectionDiffusion_type == "K_smag" and np.size(u_files_mod) >1:
                    uuu = u_files_mod[i][j//24:-1]
                    vvv = v_files_mod[i][j//24:-1]

                    run_parcels_Drift(MODELS[i], uuu, var_u_mod[i], vvv, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj3, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if AdvectionDiffusion_type == "AdvectionDiffusionM1" and np.size(u_files_mod) >1:
                    uuu = u_files_mod[i][j//24:-1]
                    vvv = v_files_mod[i][j//24:-1]
                    run_parcels_Drift(MODELS[i], uuu, var_u_mod[i], vvv, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if np.size(u_files_mod) >1:
                    uuu = u_files_mod[i][j//24:-1]
                    vvv = v_files_mod[i][j//24:-1]

                    run_parcels_Drift(MODELS[i], uuu, var_u_mod[i], vvv, var_v_mod[i],\
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
            ind_svp = np.where(date_uv_svp==str_date)
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
    traj_files_temp = [] 
   
    print(MODELS) 
    for MODEL in MODELS:
        print(MODEL)
        traj_files_int = []
        #traj_files_int.append(glob.glob(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc.zarr'))
        files = glob.glob(dir_out + '/' + '*Trajectories_' + MODEL + '_*at_SVPloc*.zarr')
        #if files:
        traj_files_int.append(glob.glob(dir_out+'/'+'*Trajectories_'+MODEL+'_*at_SVPloc*.zarr'))
        #    #files = glob.glob(dir_out + '/' + '*Trajectories_' + MODEL + '_*at_SVPloc*.zarr')
        #    print(files[0])
        #    traj_files_int.append(files[0])
        #print(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc.zarr')
        traj_files_temp.append(traj_files_int)
        print(traj_files_temp)
    traj_files =  np.sort([inner for outer in traj_files_temp for inner in outer])
    traj_lon_temp = []
    traj_lat_temp = []
    traj_time_temp = []
    traj_IDS_temp = []

    #traj_lon1  = []
    #traj_lat1  = []
    #traj_time1 = []
    
    for i in range(len(MODELS)):
        traj_lon_int = []
        traj_lat_int = []
        traj_time_int = []
        traj_IDS_int = []
        #traj_date_int = []
        list_traj = traj_files[i]
        #for file in traj_files[i]:
        for file in list_traj:
            print(file)
            print('ok')
            ds = xr.open_zarr(file)
            traj_time_int.append(ds.time.values)
            traj_lon_int.append(ds.lon.values)
            traj_lat_int.append(ds.lat.values)
            traj_IDS_int.append(ds['label'].values)
            break
        traj_lon_temp.append(traj_lon_int)
        traj_lat_temp.append(traj_lat_int)
        traj_time_temp.append(traj_time_int)
        traj_IDS_temp.append(traj_IDS_int)
        
        #traj_lon1.append(traj_lon_int)
        #traj_lat1.append(traj_lat_int)
        #traj_time1.append(traj_time_int)

    taille_max = max(len(x) for x in traj_lon_temp)
    traj_lon = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_lon_temp])

    taille_max = max(len(x) for x in traj_lat_temp)
    traj_lat = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_lat_temp])
    
    taille_max = max(len(x) for x in traj_time_temp)
    traj_time = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_time_temp])
    

    taille_max = max(len(x) for x in traj_IDS_temp)
    traj_IDS = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_IDS_temp])


    #return np.array(traj_lon1), np.array(traj_lat1),np.array(traj_time1)
    return traj_lon,traj_lat,traj_time,traj_IDS


def Retrieve_traj_files_no_ct(MODELS,dir_out):
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
            ds = xr.open_zarr(file)
            traj_time_int.append(ds.time.values)
            traj_lon_int.append(ds.lon.values)
            traj_lat_int.append(ds.lat.values)

        traj_lon.append(traj_lon_int)
        traj_lat.append(traj_lat_int)
        traj_time.append(traj_time_int)

        traj_lon = np.array(traj_lon)
        traj_lat = np.array(traj_lat)
        traj_time = np.array(traj_time)
    return traj_lon,traj_lat,traj_time#,traj_date


def Retrieve_drifters_traj(outfile2,T_DRIFT,remote = False,copernicus_drifter= False):
    print('loading ',outfile2)
    drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
    print(drifters_zone_traj.head(5))
    #print(drifters_zone_traj.dtypes)
    


    nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))
    #remove drifters with missing entries (should be nb_days*4+1)
    #Check each drifter has n days * 4 entries
    drifter_ids = pd.unique(drifters_zone_traj['id'])
    
    spurious_drifters=[]

    print(np.shape(drifter_ids))
    print(nb_drifters_temp)
    for i in range(nb_drifters_temp):
        nb_entries = len(drifters_zone_traj[drifters_zone_traj.id == drifter_ids[i]])
        #print(nb_entries, i)
       # if not nb_entries >= ((T_DRIFT+1)*24):
        if not nb_entries >= ((T_DRIFT)*24):
            print(drifter_ids[i],': not enough entries, only ', nb_entries)
            spurious_drifters.append(drifter_ids[i])
    

    if False:
        #drifters_zone_traj_old = drifters_zone_traj
        #ind1 = drifters_zone_traj['id']==spurious_drifters).index
        for i in range(len(spurious_drifters)):
            print('remove drifter ',spurious_drifters[i])
            ind1 = drifters_zone_traj[drifters_zone_traj['id'] == np.array(spurious_drifters[i])].index
            print(f'{ind1=}')
            #print(drifters_zone_traj['id'] == np.array(spurious_drifters[i]))
            drifters_zone_traj.drop(ind1, inplace=True) #remove data points
        drifters_zone_traj=drifters_zone_traj.reset_index()
        nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
                    
        print(f'{nb_drifters_temp} - {nb_drifters2} = {nb_drifters_temp - nb_drifters2}'   )
    else:
        nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
    #print( f'{aa}  - {len( drifters_zone_traj)} =  {aa  - len( drifters_zone_traj)}')
    


    print('total nb of drifters in zone at start that reach end:',nb_drifters2)

    #Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
    if remote or copernicus_drifter:
        nb_points = T_DRIFT*24   # hourly
        #nb_points = (T_DRIFT+1)*24   # hourly
    else :
        nb_points = T_DRIFT*4 + 1    # hourly

    #print(nb_points)
    print(np.shape(drifters_zone_traj))
    #print(drifters_zone_traj)
    #print(type(drifters_zone_traj))
    #print(np.shape(drifters_zone_traj.time.values))
    #print(np.shape(drifters_zone_traj.date.values))
    #print(nb_drifters2*nb_points)

    #for i in range(nb_points):
    #    print(drifters_zone_traj.id.values[i])
    #    print(drifters_zone_traj.date.values[i])
    #    print(drifters_zone_traj.time.values[i])
    #
    #print(drifters_zone_traj.id.values[nb_points])
    #print(drifters_zone_traj.date.values[nb_points])
    #print(drifters_zone_traj.id.values[nb_points +24])
    #print(drifters_zone_traj.date.values[nb_points+24])
    #print(drifters_zone_traj.id.values[nb_points+72])
    #print(drifters_zone_traj.date.values[nb_points+72])
    ll =0
    while drifters_zone_traj.id.values[ll] == drifters_zone_traj.id.values[0]:
        ll += 1
    #print(f'{ll=}')
    #print(f'{nb_points = }')
    #print(f'{ll*nb_drifters2}')
    #print(np.shape(drifters_zone_traj.date.values))
    


    


    if False:
        traj_drifter_lon   = np.array([drifters_zone_traj.lon.values[ll,0:nb_points] for ll in range(nb_drifters2)])#      np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
        traj_drifter_lat   = np.array([drifters_zone_traj.lat.values[ll,0:nb_points] for ll in range(nb_drifters2)])#      np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
        traj_drifter_date  = np.array([drifters_zone_traj.date.values[ll,0:nb_points] for ll in range(nb_drifters2)])# np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
        traj_drifter_time  = np.array([drifters_zone_traj.time.values[ll,0:nb_points] for ll in range(nb_drifters2)])#    np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
        traj_drifter_id    = np.array([drifters_zone_traj.id.values[ll,0:nb_points] for ll in range(nb_drifters2)])# np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )
    elif False:
        traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
        traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
        traj_drifter_date  = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
        traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
        traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )
    else:
        list_lon_temp = []
        list_lat_temp = []
        list_date_temp = []
        list_time_temp = []
        list_id_temp = []
        IDS = pd.unique(drifters_zone_traj['id'])    
        for ll in range(len(IDS)):
            indexx = np.where(drifters_zone_traj.id.values == IDS[ll])
            #print(indexx[0])
            #print(indexx[0][0])

            list_lon_temp.append(drifters_zone_traj.lon.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_lat_temp.append(drifters_zone_traj.lat.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_date_temp.append(drifters_zone_traj.date.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_time_temp.append(drifters_zone_traj.time.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_id_temp.append(drifters_zone_traj.id.values[indexx[0][0]:indexx[0][0]+nb_drifters2])
        
        taille_max = max(len(x) for x in list_lon_temp)
        #print(taille_max)
        #print([len(x) for x in list_lon_temp])

        list_lon = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_lon_temp])
        taille_max = max(len(x) for x in list_lat_temp)
        list_lat = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_lat_temp])
        taille_max = max(len(x) for x in list_date_temp)
        list_date = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_date_temp])
        taille_max = max(len(x) for x in list_time_temp)
        list_time = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_time_temp])

        taille_max = max(len(x) for x in list_id_temp)
        list_id = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_id_temp])



        traj_drifter_lon = np.array(list_lon)
        traj_drifter_lat= np.array(list_lat)
        traj_drifter_date = np.array(list_date)
        traj_drifter_time = np.array(list_time)
        traj_drifter_id = np.array( list_id)

        
    #print('abc')
    #print(np.shape(drifters_zone_traj.date.values))
    #print(np.shape(traj_drifter_date))
    #print(f'{nb_drifters2=}')
    #print(f'{nb_points=}')
    #print(T_DRIFT)

    return traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id,nb_drifters2


import numpy as np

def Traj_filter_parcels(
    DRIFT_TIME,
    traj_drifter_lon,
    traj_drifter_lat,
    traj_drifter_date,
    traj_drifter_time,
    traj_lon,
    traj_lat
):
    """
    Align drifter trajectories onto model trajectory time grid.

    Parameters
    ----------
    traj_drifter_lon : (Nd, Td)
    traj_drifter_lat : (Nd, Td)
    traj_drifter_time : (Nd, Td) datetime64
    traj_lon : (Nd, Tm)
    traj_time_model : (Tm,) datetime64

    Returns

    Parameters
    ----------
    traj_drifter_lon : (Nd, Td)
    traj_drifter_lat : (Nd, Td)
    traj_drifter_time : (Nd, Td) datetime64
    traj_drifter_date : (Nd, Td) str or datetime
    traj_lon : (Ni, Nj, Nd, Tm)
    traj_lat : (Ni, Nj, Nd, Tm)
    traj_time_model : (Tm,) datetime64

    Returns
    -------
    drifter_lon : (Nd, Tm)
    drifter_lat : (Nd, Tm)    

    """

    print(f'Dimension lon {traj_lon.shape}')
     
    Nd, Td = traj_drifter_lon.shape
    Tm = traj_lon.shape[-1]

    # --- Compute resampling factor automatically
    factor = Td // Tm  # e.g. 240 // 40 = 6
    print(f"Sampling factor {factor}")
    if Td % Tm != 0:
        print(f"Shape lon {traj_lon.shape}")
        raise ValueError(f"Drifter time dimension not divisible by model time dimension {traj_lon.shape}")

    # --- Reshape and average
    lon_reshaped = traj_drifter_lon.reshape(Nd, Tm, factor)
    lat_reshaped = traj_drifter_lat.reshape(Nd, Tm, factor)

    drifter_lon_2d = lon_reshaped.mean(axis=2)
    drifter_lat_2d = lat_reshaped.mean(axis=2)

    # --- Time & date: take central value in each window
    mid = factor // 2

    time_reshaped = traj_drifter_time.reshape(Nd, Tm, factor)
    date_reshaped = traj_drifter_date.reshape(Nd, Tm, factor)

    drifter_time_2d = time_reshaped[:, :, mid]
    drifter_date_2d = date_reshaped[:, :, mid]
    # --- Now match original output shape
    # Your original code had extra dims (i,j,:,:)
    # So we broadcast the result

    drifter_lon = np.broadcast_to(drifter_lon_2d, traj_lon.shape)
    drifter_lat = np.broadcast_to(drifter_lat_2d, traj_lat.shape)
    drifter_time = np.broadcast_to(drifter_time_2d, traj_lat.shape)
    drifter_date = np.broadcast_to(drifter_date_2d, traj_lat.shape)

    return drifter_lon, drifter_lat, drifter_date, drifter_time,  Tm, Nd, Td



def Traj_filter_fidji(DRIFT_TIME,traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time, traj_lon,traj_lat):
    drifter_lon = np.zeros(traj_lon.shape)
    drifter_lat = np.zeros(traj_lat.shape)
    drifter_time = np.ones(traj_lat.shape).astype(str)
    drifter_date = np.ones(traj_lat.shape).astype(str)
    print(np.shape(traj_lon))
    print(np.shape(traj_drifter_lon))
    j = 12
    i= 0
    print(np.shape( np.array([np.mean(traj_drifter_lon[:,(j*7)*24+k*DRIFT_TIME:9+(j*7)*24+(k+1)*DRIFT_TIME],axis = 1) for k in range (traj_lon.shape[-1])]).T))
    print(DRIFT_TIME)
    print(traj_lon.shape)
    print(np.shape(traj_drifter_lon))

    for i in range (traj_lon.shape[0]):
        for j in range(traj_lon.shape[1]): 
            if True:
                drifter_lon[i,j,:,:] =  np.array([np.mean(traj_drifter_lon[:,(j*7)*24+k*6:9+(j*7)*24+(k+1)*6],axis = 1) for k in range (traj_lon.shape[-1])]).T
                #np.array([np.mean(traj_drifter_lon[:,9+(j*7)*24+k*6:9+(j*7)*24+(k+1)*6],axis = 1) for k in range (traj_lon.shape[-1])]).T
                drifter_lat[i,j,:,:] = np.array([np.mean(traj_drifter_lat[:,(j*7)*24+k*6:9+(j*7)*24+(k+1)*6],axis = 1) for k in range (traj_lat.shape[-1])]).T
                #print(np.array([np.mean(traj_drifter_lon[:,9+(j*7)*24+k*6:9+(j*7)*24+(k+1)*6],axis = 1) for k in range (traj_lon.shape[-1])]).T)
                drifter_date[i,j,:,:] = np.array([traj_drifter_date[:,12+(j*7)*24+k*6] for k in range (traj_lon.shape[-1])]).T
                drifter_time[i,j,:,:] = np.array([traj_drifter_time[:,12+(j*7)*24+k*6] for k in range (traj_lat.shape[-1])]).T
                #drifter_date[i,j,:,:] = np.array([traj_drifter_date[:,12+(j*7)*24+k*6] for k in range (traj_lon.shape[-1])]).T
            else: 
                drifter_lon[i,j,:,:] = np.array([np.mean(traj_drifter_lon[:,9+(j*7)*24+k*6:9+(j*7)*24+(k+1)*6],axis = 1) for k in range (traj_lon.shape[-1])])
    return drifter_lon,drifter_lat,drifter_date,drifter_time



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
    config_object.read(dir_in+"src/config_products_fidji.ini")
    colors_list = []
    for MODEL in MODELS:
        run_info = config_object[MODEL]
        colors_list.append(run_info["color"])











    df.T[1:].plot.bar(rot = 0,title = "Lagrangian statistics from " + str_date1+' to '+str_date2,ylabel = 'Proportion',xlabel = 'Scores',color = colors_list)
    plt.legend(labels = df.MODELS.values)
    plt.savefig(dir_plot+'Lag_stats_'+str_date1+'_'+str_date2+'.png',dpi = 600)

    print("Save fig : ", dir_plot+'Lag_stats_'+str_date1+'_'+str_date2+'.png')


     






def Retrieve_data_local_FCST(MODELS,date_start,date_end_plus_DRIFT_TIME,dir_out,dir_in,dir_work,depth):
    
    zlevel = 11
    config_object = ConfigParser()
    print(dir_in+"src/config_products_fidji.ini")
    config_object.read(dir_in+"src/config_products_fidji.ini")
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
    print (f"#                      RETRIEVE U,V, SSH FIELDS for {MODELS}")
    print ("#-----------------------------------------------------------------------------------------")
    if True:
        if MODELS == "GLO12" or MODELS == ["GLO12"] or MODELS == "GLO12_FCST" or MODELS == ["GLO12_FCST"] :
            MODELS = ['GLO12_FCST0','GLO12_FCST1','GLO12_FCST2','GLO12_FCST3','GLO12_FCST4','GLO12_FCST5','GLO12_FCST6','GLO12_FCST7','GLO12_FCST8','GLO12_FCST9']
        if MODELS == "PSY4" or MODELS == ["PSY4"] or  MODELS ==  "PSY4_FCST" or MODELS == ["PSY4_FCST"]     :
            MODELS = ['PSY4_FCST0','PSY4_FCST1','PSY4_FCST2','PSY4_FCST3','PSY4_FCST4','PSY4_FCST5','PSY4_FCST6','PSY4_FCST7','PSY4_FCST8','PSY4_FCST9']
    

    for i in range(len(MODELS)):
        MODEL = MODELS[i]
        print(MODEL)
        if MODEL  not in config_object.sections():
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
            u_files_mod_3D, v_files_mod_3D = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME,dir_in = dir_in)
            u_files, v_files = ExtractSurface(MODEL,u_files_mod_3D, v_files_mod_3D,zlevel,var_u_mod[i], var_v_mod[i], dir_work+'MODEL_OUTPUT/'+MODEL+'/UV/15m/')
            u_files_mod.append(u_files)
            v_files_mod.append(v_files)
        else : 
            print(f'{date_start =}')
            print(f'{MODEL =}')
            #quit()
            u_files, v_files = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME,dir_in = dir_in)
            u_files_mod.append(u_files)
            v_files_mod.append(v_files)


    #print('path_UV: ', path_UV[0])
    #print('fname_prefix: ', fname_prefix[0])
    return u_files_mod,v_files_mod,var_u_mod,var_v_mod,fname_prefix,meshmask



def Calculate_model_traj_FCST(MODELS,dir_out,nb_points,traj_drifter_lon,traj_drifter_lat,traj_drifter_id,date_start,DRIFT_TIME,copernicus_drifter,u_files_mod,v_files_mod,var_u_mod,var_v_mod,meshmask,AdvectionDiffusion_type,nb_repeats,cloud_stat = False):

    from Drift_exec import run_parcels_Drift, run_parcels_Drift_remote
    MODELS = [MODELS+'_FCST']
    if copernicus_drifter :
        for i in range(len(MODELS)):
            dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'
            print(np.arange(0,nb_points-DRIFT_TIME*24,24))
            for j in tqdm(np.arange(0,nb_points-DRIFT_TIME*24,24)): # one sim per day
            
                print(u_files_mod)
                print(f'{DRIFT_TIME=}')
                
                print(nb_points)
                print([k%9 for k in range(DRIFT_TIME+1)])
                print([j//24+k for k in range(DRIFT_TIME+1)]) 
                print(np.arange(0,nb_points-DRIFT_TIME*24,24))
                print([u_files_mod[DRIFT_TIME%10][j//24+k] for k in range(DRIFT_TIME+1)])

                u_files = [u_files_mod[0:DRIFT_TIME%10][j//24]]#  +k] for k in range(DRIFT_TIME+1)]
                v_files = [v_files_mod[0:DRIFT_TIME%10][j//24]]#  +k] for k in range(DRIFT_TIME+1)]#[v_files_mod[k%9][j//24+k] for k in range(DRIFT_TIME+1)]             
                #print(u_files)
                #print(v_files)

                
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
                if cloud_stat:
                    print('#run parcels with at SVP neigbouring locations')
                    run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                      time_period=False)
                if AdvectionDiffusion_type == "K_smag":
                    print('#run parcels with AdvectionDiffusion_type')
                    print('CHECK POINT')
                    print(MODELS[i])
                    print(u_files)
                    print('******************')
                    print(var_u_mod[i])
                    print('------------------')
                    print(v_files)
                    print('******************')
                    print(var_v_mod[i])                    
                    #run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                    ## Modifs CHR
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//24:-1], var_u_mod[i], v_files_mod[i][j//24:-1], var_v_mod[i],
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj3, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if AdvectionDiffusion_type == "AdvectionDiffusionM1":
                    print('#run parcels with AdvectionDiffusionM1')
                    run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type = AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if True :
                    print(MODELS[i])
                    print(u_files)
                    print(var_u_mod[i])
                    print(v_files)
                    print(var_v_mod[i])
                    print('#run parcels')
                    #run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                    run_parcels_Drift(MODELS[i], u_files_mod[i][j//24:-1], var_u_mod[i], v_files_mod[i][j//24:-1], var_v_mod[i],
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, \
                                      DRIFT_TIME,f_traj1, radius_cloud=0, seeding_resolution=0, \
                                      time_period=False)
                print(' done...')

    else:

        for i in range(len(MODELS)):
            dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'
            for j in tqdm(np.arange(0,nb_points-DRIFT_TIME*4-4,4)): # one sim per day
                print([u_files_mod[k%9][j//4+k] for k in range (DRIFT_TIME+1)])
                u_files = [u_files_mod[k%9][j//4+k] for k in range (DRIFT_TIME+1)]
                v_files = [v_files_mod[k%9][j//4+k] for k in range (DRIFT_TIME+1)]             

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
                print('\nModels Files : \n',u_files)
                print('\Drift time  : ',DRIFT_TIME)

                if cloud_stat:
                    print('#run parcels with at SVP neigbouring locations')
                    run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                                      time_period=False)
                if AdvectionDiffusion_type=="Smag":
                    run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj3, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = nb_repeats)
                if AdvectionDiffusion_type=="AdvectionDiffusionM1":
                    run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start,\
                                      DRIFT_TIME,f_traj4, radius_cloud=0, seeding_resolution=0,time_period=False,\
                                      AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = nb_repeats)


                if True :
                    run_parcels_Drift(MODELS[i], u_files, var_u_mod[i], v_files, var_v_mod[i],\
                                      meshmask[i],traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, \
                                      DRIFT_TIME,f_traj1, radius_cloud=0, seeding_resolution=0, \
                                      time_period=False)

    print(' done...')
















def Retrieve_traj_files_diff(MODELS,dir_out,AdvectionDiffusion_type):
    traj_lon_temp = []
    traj_lat_temp = []
    traj_time_temp = []


    for MODEL in MODELS:
        if AdvectionDiffusion_type == "K_smag":
            traj_files_int = sorted(glob.glob(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc_smag.zarr'))
        elif AdvectionDiffusion_type == "AdvectionDiffusionM1":
            traj_files_int = sorted(glob.glob(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc_M1.zarr'))




        traj_lon_int = []
        traj_lat_int = []
        traj_time_int = []

        for file in traj_files_int:
            print(f'{MODEL=} ---- {file=}')

            ds = xr.open_zarr(file)
            print(ds)
            traj_time_int.append(ds.time.values)
            traj_lon_int.append(ds.lon.values)
            traj_lat_int.append(ds.lat.values)


        traj_lon_temp.append(traj_lon_int)
        traj_lat_temp.append(traj_lat_int)
        traj_time_temp.append(traj_time_int)
    
    if True:
        taille_max = max(len(x) for x in traj_lon_temp)
        traj_lon = [x + [x[-1]] * (taille_max - len(x)) for x in traj_lon_temp]


        taille_max = max(len(x) for x in traj_lat_temp)
        traj_lat = [x + [x[-1]] * (taille_max - len(x)) for x in traj_lat_temp]


        taille_max = max(len(x) for x in traj_time_temp)
        traj_time = [x + [x[-1]] * (taille_max - len(x)) for x in traj_time_temp]


    traj_lon = np.array(traj_lon)
    traj_lat = np.array(traj_lat)#_temp)
    traj_time = np.array(traj_time)#_temp)

    return traj_lon,traj_lat,traj_time


