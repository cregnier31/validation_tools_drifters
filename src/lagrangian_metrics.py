#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from numpy import pi
import pandas as pd
import numpy.matlib
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import json
import time
import netCDF4 as nc
from tqdm import tqdm
import os, glob,sys
import sys, pprint
sys.displayhook = pprint.pprint  # Improves print of variables (e.g. locals(), globals(), dir())) to screen
import shutil
import shlex,subprocess
import time
import xarray as xr
from scipy import interpolate
import getpass
username=getpass.getuser()
import os, glob,sys
from operator import attrgetter
from scipy.ndimage import gaussian_filter

from extras import dist as sw_dist


def coord2index(lon1,lat1,z1, nav_lon, nav_lat, nb_zlevels, decimals=0, display=0):
    """
    Find closest gridcell for input point of coordinate: lon, lat, depth(m)
    => returns positions i, j, k within model gridcell
    sjvg 13/1/2018: improve to return i&j at decimals accuracy, set decimals==1
    set display==1 to print results to screen
    """


    #print('coord2index...')
    #print('lon shape',len(nav_lon.shape))
    #print('lat shape',nav_lat.shape)

    if display:
        print ("      # identify gridcell for lon=", lon1,'; lat=', lat1,'z=', z1)
        print ("      # for ORCA grid  x=", nav_lon.shape,'; y=', nav_lat.shape,'z=', nb_zlevels)

    #Case for regular grids with 1-dim vector nav_lon, nav_lat
    if len(nav_lon.shape)==1 & len(nav_lat.shape)==1:
        print('lat, lon vars are 1D arrays')
        POS_K = 0
        #print('regular grid')
        #print('# identify gridcell for lon=', lon1,'; lat=', lat1,'z=', z1)
        POS_I = np.argmin(np.square(nav_lon-lon1))
        POS_J = np.argmin(np.square(nav_lat-lat1))
        #print(POS_I, POS_J, nav_lon[POS_I-1:POS_I+1], nav_lat[POS_J-1:POS_J+1])
        return POS_I, POS_J, POS_K

    if (nb_zlevels == 50) | (nb_zlevels == 75) | (nb_zlevels==0):
        # find i and j
        dd=np.sqrt(np.square(lon1-nav_lon)+np.square(lat1-nav_lat));

        #print ('      min_dist=',np.min(dd[:]))
        pos = np.argwhere(dd == np.min(dd[:]))
        POS_J = pos[0,0]
        POS_I = pos[0,1]
        print( 'POS_J : ', POS_J)
        print('POS_I : ', POS_I)
        #print('POS_J, POS_I = ', POS_J, POS_I)
        lat1_model=nav_lat[POS_J, POS_I]
        lon1_model=nav_lon[POS_J, POS_I]
        #print('lat1_model, lon1_model =',lat1_model, lon1_model)

        # determine lat and lon width of cell:
        dx=nav_lon[POS_J, POS_I-1]-nav_lon[POS_J, POS_I]
        dy=nav_lat[POS_J-1, POS_I]-nav_lat[POS_J, POS_I]
        #print ("      dx, dy = ", dx, dy)

        # determine position within cell
        i_inside=(lon1-lon1_model)/dx
        j_inside=(lat1-lat1_model)/dy
        #print ("      i_inside, j_inside =", i_inside, j_inside)
        if i_inside>0:
            POS_I2=POS_I-i_inside
        else:
            POS_I2=POS_I+i_inside

        if j_inside>0:
            POS_J2=POS_J-j_inside
        else:
            POS_J2=POS_J+j_inside
        POS_I2=np.around(POS_I2, decimals=2)
        POS_J2=np.around(POS_J2, decimals=2)

        if display:
            print ("      lon=", lon1, "lat=", lat1)
            print ("      => in nemo grid (C position): i=", POS_I, "j=", POS_J)
            print ("      =>               lon=",lon1_model,", lat=",lat1_model )
            if decimals:
                print ("      when placed inside cell:")
                print ("      =>               I2=",POS_I2,", J2=",POS_J2)

        if nb_zlevels!= 0:
            # find k
            z1=np.abs(z1) # set depth as positive value
            # load file to convert depth (m) to depth (position)
            filename1 = '/data/rd_exchange/svangennip/rd2/svangennip/ARIANE/GRID/Level_seeded_and_DepthReleasedinModel_'+str(nb_zlevels)+'zlevels.dat'
            z_01, depth_01 = np.loadtxt(filename1, unpack=True)  # z_01=> depth levels in dcimals, depth_01 => equivalent depth in m.

            # take abs value of depths (as -ve)
            depth_01b = np.abs(depth_01[np.newaxis,:])
            diff_z = np.abs(z1 - depth_01b)

            min_diff = np.argmin(diff_z, axis=1)
            POS_K = np.asscalar(z_01[min_diff])
            POS_K=np.abs(POS_K)
            #print('POS_K=',POS_K)
            if decimals==1:
                return POS_I2, POS_J2, POS_K
            else:
                return POS_I, POS_J, POS_K
        else:
            POS_K=0
            return POS_I, POS_J, POS_K
    elif len(nav_lon.shape) & len(nav_lat.shape):
       print('1D vector = regula2 grid')
       # find i and j
       dlon=np.sqrt(np.square(lon1-nav_lon))
       dlat=np.sqrt(np.square(lat1-nav_lat))

       #print ('      min_dist=',np.min(dd[:]))
       pos_lon = np.argwhere(dlon == np.min(dlon[:]))
       pos_lat = np.argwhere(dlat == np.min(dlat[:]))
       print('nearest cell [i,j]=', pos_lon, pos_lat)
       print('(lon,lat) = ',nav_lon[pos_lon], nav_lat[pos_lat])
    else:
        print ("      nb levels =", str(nb_zlevels), ": unknown Z-grid \n")
        sys.exit(1)

def retrieve_zone(str_zone):
    zone = str_zone.split('_')[1]
    if len(zone.split('W'))==3: 
        lon_min = -float(zone.split('W')[0])
        lon_max = -float(zone.split('W')[1])
    elif len(zone.split('W'))>1:
        new_zone = zone.split('W')
        if len(new_zone[0].split('E'))>1:
            lon_min = float(new_zone[0].split('E')[0])
        else :
            lon_min = -float(new_zone[0])
            if len(new_zone[1].split('E'))>1:
                lon_max = float(new_zone[1].split('E')[0])
    if len(zone.split('E'))==3: 
        lon_min = float(zone.split('E')[0])
        lon_max = float(zone.split('E')[1])
    elif len(zone.split('E'))>1 and not len(zone.split('W'))>1:
        print('flag')
        new_zone = zone.split('E')
        if len(new_zone[0].split('W'))>1:
            lon_min = float(new_zone[0].split('W')[0])
        else :
            lon_min = float(new_zone[0])
            if len(new_zone[1].split('W'))>1:
                lon_max = -float(new_zone[1].split('W')[0])

    zone1 = str_zone.split('-')[1]
    if len(zone1.split('N'))==3:
        lat_min = float(zone1.split('N')[0])
        lat_max = float(zone1.split('N')[1])
    if len(zone1.split('S'))==3 : 
        lat_min = -float(zone1.split('S')[0])
        lat_max = -float(zone1.split('S')[1])
    elif len(zone1.split('S'))>1:
        new_zone = zone1.split('S')
        if len(new_zone[0].split('N'))>1:
            lat_min = float(new_zone[0].split('N')[0])
        else :
            lat_min = -float(new_zone[0])
            if len(new_zone[1].split('N'))>1:
                lat_max = float(new_zone[1].split('N')[0])
    return lon_min,lon_max,lat_min,lat_max
   
   
def shift360( lon_2D, lat_2D, var ):
    '''This Script takes two arguments: nav_lon (ORCA grid) and a variable (VAR) that is function of nav_lon.
       It returns VAR reordered and shifted for a 0-360 lon range.
       To obtain a reordered nav_lon, do shift360(nav_lon, nav_lon)
       Script may not deal with n-dimensions array well.
       s. van gennip 24/09/2018'''

    #lon_2D[lon_2D<0]+=360
    nparange = np.arange(lon_2D.shape[0])[:,None]
    npargsort = np.argsort(lon_2D)
    lons = lon_2D[nparange,npargsort]
    lats = lat_2D[nparange,npargsort]
    var_sort = var[nparange,npargsort]

    return lons, lats, var_sort


def StringRegion(lon1,lon2,lat1,lat2):

    if lon1>=0:
        str_lon1=str(np.abs(lon1))+'E'
    else:
        str_lon1=str(np.abs(lon1))+'W'

    if lon2>=0:
        str_lon2=str(np.abs(lon2))+'E'
    else:
        str_lon2=str(np.abs(lon2))+'W'

    if lat1>=0:
        str_lat1=str(np.abs(lat1))+'N'
    else:
        str_lat1=str(np.abs(lat1))+'S'

    if lat2>=0:
        str_lat2=str(np.abs(lat2))+'N'
    else:
        str_lat2=str(np.abs(lat2))+'S'

    string_area= str_lon1+str_lon2+'-'+str_lat1+str_lat2
    return string_area


def match_elements(list_a, list_b):
    match = []
    for i in list_a:
        if i in list_b:
            match.append(i)
    return match

def resample_traj(file_name):
    # resampling at regular 1h time step
    spurious_drifter = False
    data = xr.open_dataset(file_name)
    t_start = pd.Timestamp(data.TIME.values[0])
    t_end = pd.Timestamp(data.TIME.values[-1])
    #print(t_start)
    #print(t_end)
    nb_hours = (t_end - t_start).total_seconds()/3600 + 1
    #if True:
    if (int(nb_hours) != len(data.TIME.values)) or  (len(data.TIME.values)!= len(data.LONGITUDE.values)):
        if (int(nb_hours)-len(data.TIME.values)) < 1000: # if more than thousand points are missing, delete the drifter
            print("Resampling...")
            time_new = np.arange(t_start,t_end+np.timedelta64(1, "h"),np.timedelta64(1, "h")).astype('float')*1000
            time = data.TIME.values.astype('float')
            lon = data.LONGITUDE.values % 360 # values between 0 and 360 
            lat = data.LATITUDE.values
            d_lon = lon[1:] - lon[:-1]
            d_lat = lat[1:] - lat[:-1]
            if np.max(d_lon) < 2 or np.max(d_lat) < 2 : # if there is a jump of at least 5degrees, delete the drifter
                f_lon = interpolate.interp1d(time, lon, kind = 'quadratic')
                f_lat = interpolate.interp1d(time, lat, kind = 'quadratic')
                try:
                    lon_new = f_lon(time_new)
                except ValueError:
                    spurious_drifter = True
                    return spurious_drifter
                lon_new_new = []
                for i in range (lon_new.shape[0]):
                    if lon_new[i]<=180:
                        lon_new_new.append(lon_new[i])
                    else:
                        lon_new_new.append(lon_new[i]-360)
                lon_new_new = np.array(lon_new_new)
                lat_new = f_lat(time_new)
                data_res = xr.Dataset(
                   coords=dict(
                
                        LONGITUDE=lon_new_new,
                
                        LATITUDE=lat_new,
                
                        TIME=time_new.astype(data.TIME.values.dtype),
                
                
                    ),
                
                    attrs=dict(time_coverage_start = data.time_coverage_start,
                              time_coverage_end = data.time_coverage_end,
                              platform_code = data.platform_code  ),
                
                )
                data.close()
                data_res.to_netcdf(file_name,mode = 'w')
            else:
                spurious_drifter = True
        else : 
            spurious_drifter = True
            
    return spurious_drifter



def trapz(f,x):
    '''Approximate the integral of f(x) from a to b by the trapezoid rule.

    The trapezoid rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/2) \sum_{k=1}^N (f(x_k) + f(x_{k-1}))
    where x_k = a + k*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using the
        trapezoid rule with N subintervals of equal length.

    Examples
    --------
    >>> trapz(np.sin,0,np.pi/2,1000)
    0.9999997943832332
    '''
    y_right = f[1:,:] # right endpoints
    #print(y_right.shape)
    y_left = f[:-1,:] # left endpoints
    #print(y_left.shape)

    dx = (x[1] - x[0])
    #print(dx)
    #T = (dx/2) * np.sum(y_right + y_left)
    T = (dx/2) * np.nansum(y_right + y_left, axis=0)
    return T


def DispDirAndLength(traj_lon, traj_lat):
    nb_part   = traj_lon.shape[0]
    nb_tsteps = traj_lon.shape[1]
    
    #2 Calculate particle displacement (for entire trajectory), direction of displacement and trajectory length
    disp_lat, disp_lon =  np.zeros((nb_part,2)), np.zeros((nb_part,2))
    disp_lat[:,0] = traj_lat[:,0]
    disp_lat[:,1] = traj_lat[:,-1]
    
    disp_lon[:,0] = traj_lon[:,0]
    disp_lon[:,1] = traj_lon[:,-1]
    #print('disp_lat',disp_lat.shape) 
    #print('    CALCULATE DISTANCE and ANGLE ')
    displacement, direction = sw_dist(disp_lat, disp_lon)  # CALCULATE DIST
    #print('displacement', displacement)
    #print('direction', direction) 
    #sys.exit(1)
    
    #print('    CALCULATE length of SPAGHETTI ')
    
    dist_spagAB=np.zeros(displacement.shape)
    #print('dist_spagAB',dist_spagAB.shape)
    for ii in range(nb_tsteps-1): # calculate distance completed in a day (take one datapoint per day)
        #print("\ntstep=", ii)
        #print "after day", ii+1
    
        disp_lat, disp_lon =  np.zeros((nb_part,2)), np.zeros((nb_part,2))
        disp_lat[:,0] = traj_lat[:, ii]
        disp_lat[:,1] = traj_lat[:, ii+1]
    
        disp_lon[:,0] = traj_lon[:,ii]
        disp_lon[:,1] = traj_lon[:,ii+1]
    
        # Calculate distance and angle after each dayoutput_Drifter_Data_2010-14.txt
        dist_spagAB_dayi, spag_angleAB = sw_dist(disp_lat, disp_lon)  # CALCULATE DIST & ANGLE
        # add to master file
        dist_spagAB=dist_spagAB+dist_spagAB_dayi
        #print "drift after day", ii," : ", dist_spagAB
        #angle_spagAB=angle_spagAB+angle_spagAB_dayi    
    #print('dist_spagAB',dist_spagAB.shape)
    return displacement, direction, dist_spagAB 

def SeparationDistance(lon_grp1,lat_grp1, lon_grp2,lat_grp2):
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    DEG2NM, NM2KM = 60., 1.8520
    # Angle conversions.
    deg2rad, rad2deg = pi / 180.0,  180.0 / pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km

    #print(np.shape(lon_grp1), np.shape(lat_grp1),np.shape(lon_grp2),np.shape(lat_grp2))
    if len(lon_grp1.shape) == 1:
         lon_grp1=np.expand_dims(lon_grp1, axis=1)
    if len(lat_grp1.shape) == 1:
         lat_grp1=np.expand_dims(lat_grp1, axis=1)
    if len(lon_grp2.shape) == 1:
         lon_grp2=np.expand_dims(lon_grp2, axis=1)
    if len(lat_grp2.shape) == 1:
         lat_grp2=np.expand_dims(lat_grp2, axis=1)
    #print(np.shape(lon_grp1), np.shape(lat_grp1),np.shape(lon_grp2),np.shape(lat_grp2))
    #sys.exit(1)
    # difference in lon/lat with above and below cells
    dlon = lon_grp2 - lon_grp1
    dlat = lat_grp2 - lat_grp1

    if np.any(np.abs(dlon) > 180):
        flag = abs(dlon) > 180
        dlon[flag] = -np.sign(dlon[flag]) * (360 - np.abs(dlon[flag]))

    #sizey, sizex = np.shape(lon_grp1)
    mean_lat=(lat_grp2 + lat_grp1)/2.0
    latrad = np.abs(mean_lat * deg2rad)
    #print('\ndlon\n', dlon)
    #print('\ndlat\n', dlat)
    #print('\nmean_lat\n', mean_lat)

    #print('\nlon_diff_top\n',lon_diff_top)
    #print('\nlat_diff_top\n',lat_diff_top)

    #print('\nCALCULATE DISTANCE TOP ')
    dep = np.cos(latrad) * dlon
    dist = DEG2NM * (dlat ** 2 + dep ** 2) ** 0.5
    dist = dist*NM2KM
    #print('\ndist[0,:]\n',dist[0,:])

    return dist


def SeparationDistance_havers(lon_grp1, lat_grp1, lon_grp2, lat_grp2):

    # Ensure arrays
    lon_grp1 = np.asarray(lon_grp1)
    lat_grp1 = np.asarray(lat_grp1)
    lon_grp2 = np.asarray(lon_grp2)
    lat_grp2 = np.asarray(lat_grp2)

    # Expand dims if needed
    if lon_grp1.ndim == 1:
        lon_grp1 = lon_grp1[:, np.newaxis]
    if lat_grp1.ndim == 1:
        lat_grp1 = lat_grp1[:, np.newaxis]
    if lon_grp2.ndim == 1:
        lon_grp2 = lon_grp2[:, np.newaxis]
    if lat_grp2.ndim == 1:
        lat_grp2 = lat_grp2[:, np.newaxis]

    # Earth radius (km)
    R = 6371.0

    # Convert to radians
    lon1 = np.deg2rad(lon_grp1)
    lat1 = np.deg2rad(lat_grp1)

    lon2 = np.deg2rad(lon_grp2)
    lat2 = np.deg2rad(lat_grp2)

    # Dateline-safe longitude difference
    dlon = (lon2 - lon1 + np.pi) % (2 * np.pi) - np.pi
    dlat = lat2 - lat1

    # Haversine formula
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c

    return dist

# ==========================================================
# Robust Haversine distance
# Handles:
#   - dateline crossing
#   - global trajectories
#   - poles
#   - NaNs
#   - any ndarray shape
# ==========================================================

def SeparationDistance_havers2(
    lon_grp1,
    lat_grp1,
    lon_grp2,
    lat_grp2
):
    """
    Compute great-circle separation distance (km)
    between two trajectory groups using the haversine formula.

    Parameters
    ----------
    lon_grp1, lat_grp1 : ndarray
        Longitude/latitude of group 1 (degrees)

    lon_grp2, lat_grp2 : ndarray
        Longitude/latitude of group 2 (degrees)

    Returns
    -------
    dist : ndarray
        Separation distance in km
        Same shape as input arrays.
    """

    # ------------------------------------------------------
    # Earth radius (km)
    # ------------------------------------------------------
    R = 6371.0

    # ------------------------------------------------------
    # Convert to numpy arrays
    # ------------------------------------------------------
    lon1 = np.asarray(lon_grp1, dtype=np.float64)
    lat1 = np.asarray(lat_grp1, dtype=np.float64)

    lon2 = np.asarray(lon_grp2, dtype=np.float64)
    lat2 = np.asarray(lat_grp2, dtype=np.float64)

    # ------------------------------------------------------
    # Normalize longitudes to [-180, 180]
    # ------------------------------------------------------
    lon1 = ((lon1 + 180) % 360) - 180
    lon2 = ((lon2 + 180) % 360) - 180

    # ------------------------------------------------------
    # Longitude difference
    # IMPORTANT:
    # must ALSO be wrapped after subtraction
    # ------------------------------------------------------
    dlon = lon2 - lon1
    dlon = ((dlon + 180) % 360) - 180

    # ------------------------------------------------------
    # Latitude difference
    # ------------------------------------------------------
    dlat = lat2 - lat1

    # ------------------------------------------------------
    # Convert to radians
    # ------------------------------------------------------
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)

    dlon = np.radians(dlon)
    dlat = np.radians(dlat)

    # ------------------------------------------------------
    # Haversine formula
    # ------------------------------------------------------
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2.0) ** 2
    )

    # Numerical safety
    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    # ------------------------------------------------------
    # Distance (km)
    # ------------------------------------------------------
    dist = R * c

    return dist

def SepDistanceEachTimestep(lon_grp1, lat_grp1, lon_grp2, lat_grp2):
    #log.info('SepDistanceEachTimestep...')
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    DEG2NM, NM2KM = 60., 1.8520
    # Angle conversions.
    deg2rad, rad2deg = pi / 180.0,  180.0 / pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    
    #print('SVP:\n',np.shape(lon_grp1), np.shape(lat_grp1),'\nMODEL\n',np.shape(lon_grp2),np.shape(lat_grp2))
    #sys.exit(1)
    #if len(lon_grp1.shape) == 1:
    #     lon_grp1=np.expand_dims(lon_grp1, axis=1)
    #if len(lat_grp1.shape) == 1:
    #     lat_grp1=np.expand_dims(lat_grp1, axis=1)
    #if len(lon_grp2.shape) == 1:
    #     lon_grp2=np.expand_dims(lon_grp2, axis=1)
    #if len(lat_grp2.shape) == 1:
    #     lat_grp2=np.expand_dims(lat_grp2, axis=1)
    #print(np.shape(lon_grp1), np.shape(lat_grp1),np.shape(lon_grp2),np.shape(lat_grp2))
    #sys.exit(1)
    # difference in lon/lat with above and below cells
    dlon = lon_grp1 - lon_grp2  #np.diff(lon_grp1)
    dlat = lat_grp1 - lat_grp2  #np.diff(lat_grp1)
    #print('dlon=', dlon.shape)
    #print('dlat=', dlat.shape)
    #sys.exit(1)


    if np.any(np.abs(dlon) > 180):
        flag = abs(dlon) > 180
        dlon[flag] = -np.sign(dlon[flag]) * (360 - np.abs(dlon[flag]))

    sizey, sizex = np.shape(lon_grp1)
    mean_lat=(lat_grp2 + lat_grp1)/2.0
    #print(mean_lat.shape)
    latrad = np.abs(mean_lat * deg2rad)
    #print('\ndlon\n', dlon)
    #print('\ndlat\n', dlat)
    #print('\nmean_lat\n', mean_lat)

    #print('\nlon_diff_top\n',lon_diff_top)
    #print('\nlat_diff_top\n',lat_diff_top)

    #print('\nCALCULATE DISTANCE TOP ')
    dep = np.cos(latrad) * dlon
    dist = DEG2NM * (dlat ** 2 + dep ** 2) ** 0.5
    dist = dist*NM2KM
    #print('\nSVP_lon\n',lon_grp1[0,:])
    #print('\nSVP_lat\n',lat_grp1[0,:])
    #print('\nMOD_lon\n',lon_grp2[0,:])
    #print('\nMOD_lat\n',lat_grp2[0,:])
    #print('\ndist\n',dist[0,:])
    
    #sys.exit(1)
    return dist


def TrajDistBetweenTimesteps(lon_grp1, lat_grp1):
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    DEG2NM, NM2KM = 60., 1.8520
    # Angle conversions.
    deg2rad, rad2deg = pi / 180.0,  180.0 / pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    
    #print(np.shape(lon_grp1), np.shape(lat_grp1))#,np.shape(lon_grp2),np.shape(lat_grp2))
    #if len(lon_grp1.shape) == 1:
    #     lon_grp1=np.expand_dims(lon_grp1, axis=1)
    #if len(lat_grp1.shape) == 1:
    #     lat_grp1=np.expand_dims(lat_grp1, axis=1)
    #if len(lon_grp2.shape) == 1:
    #     lon_grp2=np.expand_dims(lon_grp2, axis=1)
    #if len(lat_grp2.shape) == 1:
    #     lat_grp2=np.expand_dims(lat_grp2, axis=1)
    #print(np.shape(lon_grp1), np.shape(lat_grp1),np.shape(lon_grp2),np.shape(lat_grp2))
    #sys.exit(1)
    # difference in lon/lat with above and below cells
    dlon = np.diff(lon_grp1)
    dlat = np.diff(lat_grp1)
    #print('dlon=', dlon.shape)
    #print('dlat=', dlat.shape)
    #sys.exit(1)


    if np.any(np.abs(dlon) > 180):
        flag = abs(dlon) > 180
        dlon[flag] = -np.sign(dlon[flag]) * (360 - np.abs(dlon[flag]))

    sizey, sizex = np.shape(lon_grp1)
    mean_lat=(lat_grp1[:,:-1] + lat_grp1[:,1:])/2.0
    #print(mean_lat.shape)
    latrad = np.abs(mean_lat * deg2rad)
    #print('\ndlon\n', dlon)
    #print('\ndlat\n', dlat)
    #print('\nmean_lat\n', mean_lat)

    #print('\nlon_diff_top\n',lon_diff_top)
    #print('\nlat_diff_top\n',lat_diff_top)

    #print('\nCALCULATE DISTANCE TOP ')
    dep = np.cos(latrad) * dlon
    dist = DEG2NM * (dlat ** 2 + dep ** 2) ** 0.5
    dist = dist*NM2KM
    #print('\ndist\n',dist[0,:])
    #sys.exit(1)
    return dist
    
def LiuScore(traj_drifter_lon, traj_drifter_lat,traj_lon_mod, traj_lat_mod, log):
    log.info('Compute Liu Score')
    #print('drifter 6hourly: SVP', traj_drifter_lon.shape,', MODEL:', traj_lon_mod.shape)
    #Need to only take daily output:
    ind_daily=np.arange(0,traj_drifter_lon.shape[1],4)
    #print('select positions:', ind_daily)
    
    traj_drifter_lon_daily=traj_drifter_lon[:,ind_daily]
    traj_drifter_lat_daily=traj_drifter_lat[:,ind_daily]
    traj_lon_mod_daily=traj_lon_mod[:,ind_daily]
    traj_lat_mod_daily=traj_lat_mod[:,ind_daily]
    #print(traj_drifter_lon_daily.shape, traj_lon_mod_daily.shape)
    #1. Calculate SVP distance completed between time steps
    dli_drifters = TrajDistBetweenTimesteps(traj_drifter_lon_daily, traj_drifter_lat_daily)

    #2. Calculate separation distance between Model drifter and SVP at each time step:(ignore starting position)
    di_SVP_MOD =SepDistanceEachTimestep(traj_drifter_lon_daily[:,1:],
                                        traj_drifter_lat_daily[:,1:],
                                        traj_lon_mod_daily[:,1:],
                                        traj_lat_mod_daily[:,1:]
                                        )
    #print('dli_drifters.shape, di_SVP_MOD.shape\n', dli_drifters.shape, di_SVP_MOD.shape)
    #3. Compute Liu Score for 1, 3, 5 days
    
    #Compute S1 
    S1=di_SVP_MOD[:,0]/dli_drifters[:,0]
    
    #Compute S3=(d1+d2+d3)/(I01 + I02 + I03) 
    # with I01=dl1, I02=dl1+dl2, I03=dl1+dl2+dl3
    I01 = dli_drifters[:,0]
    I02 = np.sum(dli_drifters[:,:2], axis=1)
    I03 = np.sum(dli_drifters[:,:3], axis=1)
    #print(I01.shape)
    #print(I02.shape)
    #print(I03.shape)
    num_S3   = np.sum(di_SVP_MOD[:,:3], axis=1)
    denom_S3 = I01+I02+I03  
    #print(num_S3.shape, denom_S3.shape)
    S3=np.divide(num_S3, denom_S3)
    #print(S3.shape)
    #Same for S5:
    I04 = np.sum(dli_drifters[:,:4], axis=1)
    I05 = np.sum(dli_drifters[:,:5], axis=1)
    num_S5 = np.sum(di_SVP_MOD[:,:5], axis=1)
    denom_S5   = I01+I02+I03+I04+I05
    S5=np.divide(num_S5,denom_S5)

    #Same for S7:
    I06 = np.sum(dli_drifters[:,:6], axis=1)
    I07 = np.sum(dli_drifters[:,:7], axis=1)
    num_S7 = np.sum(di_SVP_MOD[:,:7], axis=1)
    denom_S7   = I01+I02+I03+I04+I05+I06+I07
    S7=np.divide(num_S7,denom_S7)
    #print('\nS1',S1.shape,S1,'\n\nS3', S3.shape, S3, '\n\nS5', S5.shape, S5, '\n\nS7', S7.shape, S7)
    
    
    return S1, S3, S5, S7
    #sys.exit(1)
    
def LiuScore2(traj_drifter_lon, traj_drifter_lat, traj_lon_mod, traj_lat_mod, n=1.0):
    """Liu index from cumulative separation / summed cumulative observed travel.

    Uses the user's original loop indexing: at output index i, both sums
    include samples strictly before i. Converts the ratio to skill 1-s/n,
    clipped to [0, 1]. Returns (time, drifter).
    """
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Liu tolerance n must be finite and positive")

    dli_drifters = TrajDistBetweenTimesteps(traj_drifter_lon, traj_drifter_lat)
    di_SVP_MOD = SepDistanceEachTimestep(
        traj_drifter_lon[:, 1:], traj_drifter_lat[:, 1:],
        traj_lon_mod[:, 1:], traj_lat_mod[:, 1:]
    )
    valid = (np.isfinite(dli_drifters) & np.isfinite(di_SVP_MOD))
    steps = np.where(valid, dli_drifters, 0.0)
    separations = np.where(valid, di_SVP_MOD, 0.0)
    liu_index = []
    den = np.zeros(di_SVP_MOD.shape[0], dtype=float)
    for i in range(di_SVP_MOD.shape[1]):
        den += np.sum(steps[:, :i], axis=1)
        numerator = np.sum(separations[:, :i], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(den > 0, numerator / den, np.nan)
        skill = np.maximum(0.0, 1.0 - ratio / n)
        # At each i, score is defined only for trajectories with a valid
        # current sample and a nonzero accumulated observed distance.
        skill = np.where(valid[:, i] & (den > 0), skill, np.nan)
        liu_index.append(skill)
    return np.asarray(liu_index)



def ExtractSurface(model,u_files, v_files,zlevel,var_u, var_v, dir_out) :

    u_files_surf, v_files_surf  = [], []
    for uline, vline in zip(u_files, v_files):  #loop through each uv file
        u_file = uline.strip()  #need strip to remove '\n' newline
        v_file = vline.strip()  #need strip to remove '\n' newline
        uname = os.path.split(u_file)[1] #extract name of file
        #print(uname)
        if 'GLO12' in u_file:
            year=uname[14:18]
        elif 'PSY4V3' in u_file:
            year=uname[14:18]
        elif 'IBI36' in u_file:
            year = uname[13:17]
        elif model == 'GLOBC':
            year = uname[21:25]

        #print('year=', year)
        print(u_file)
        print(v_file)
        print( 'year = ',year)
        #sys.exit(1)
        dir_out2= dir_out+'/'+ year +'/'
        if not os.path.exists(dir_out2):
            print('\ncreate temp folder : ', dir_out2)
            # Create directory where to store results:
            os.makedirs(dir_out2)          
        #sys.exit(1)
        # Create outfile name:
        fu_name = os.path.basename(u_file)  
        fv_name = os.path.basename(v_file)  
        
        u_file_surf=dir_out2 + fu_name
        u_file_temp=dir_out2 + 'temp1.nc'
        v_file_surf=dir_out2 + fv_name
        v_file_temp=dir_out2 + 'temp2.nc'
        
        #print('--->',nc_out)
        u_files_surf.append(u_file_surf)
        v_files_surf.append(v_file_surf)
        
        
        #'''
        print('extract u_surf')
        depth='deptht'
        if 'IBI36' in u_file:
            depth = 'depthu'
        elif 'GLO12' in u_file:
            depth = 'depthu'
        elif 'GLOBC' in u_file:
            depth = 'depth'
        elif 'PSY4V3R1' in u_file:
            depth = 'deptht'
        #if True : 
        if not os.path.exists(u_file_surf):
            # e.g. ncwa -a deptht -d deptht,10 infile.nc file-depth10.nc 
            # Example to select only 10th level of depth dim (resulting file won't keep this dimension):
            #  If no -a option is precised, ncwa is used to average data. By default all dimensions will be averaged to give only a float value.
            
            #cmd_exe1 = 'ncwa -F -a %s -d %s,%s %s %s ' %(depth,depth,z_level, u_file, u_file_surf)
            #ncks -F -A -d deptht,${zlevel},${zlevel},1 -v $var_i $ff ${IDIR}/temp1.nc
            #cmd_exe1 = 'ncks -F -A -d %s,%s,%s,1 -v %s %s %s ' %(depth,zlevel,zlevel,var_u, u_file, u_file_temp)
            cmd_exe1 = 'ncks -F -A -d %s,%s,%s,1 -v %s %s %s ' %(depth,zlevel,zlevel,var_u, u_file, u_file_surf)
            print(cmd_exe1)   
            os.system(cmd_exe1)

            #2  Remove degenerated dimension
            #echo ncwa -F -a deptht,1  ${IDIR}/temp1.nc ${IDIR}/${fname2}
            cmd_exe1b = 'ncwa -F -a %s,1 %s %s ' %(depth, u_file_temp, u_file_surf)
            #print(cmd_exe1b)   
            #os.system(cmd_exe1b)


        else:
            print('...',u_file_surf, " already exists")        
            #'''
        #sys.exit(1)
        print('extract v_surf')
        
        if 'IBI36' in v_file:
            depth = 'depthv'
        elif 'GLO12' in v_file:
            depth = 'depthv'
        elif 'GLOBC' in v_file:
            depth = 'depth'
        elif 'PSY4V3R1' in v_file:
            depth = 'deptht'
        if not os.path.exists(v_file_surf):
        #if True:
            #cmd_exe2 = 'ncks -F -A -d %s,%s,%s,1 -v %s %s %s ' %(depth,zlevel,zlevel,var_v, v_file, v_file_temp)
            cmd_exe2 = 'ncks -F -A -d %s,%s,%s,1 -v %s %s %s ' %(depth,zlevel,zlevel,var_v, v_file, v_file_surf)
            #cmd_exe2 = 'ncwa -F -a %s -d %s,%s %s %s ' %(depth,depth,z_level, v_file, v_file_surf)
            print(cmd_exe2)   
            os.system(cmd_exe2)
           
            #2  Remove degenerated dimension
            #echo ncwa -F -a deptht,1  ${IDIR}/temp1.nc ${IDIR}/${fname2}
            cmd_exe2b = 'ncwa -F -a %s,1 %s %s ' %(depth,  v_file_temp, v_file_surf)
            #print(cmd_exe2b)   
            #os.system(cmd_exe2b)
        else:
            print('...',v_file_surf, " already exists")        
            #'''



    return u_files_surf, v_files_surf  


