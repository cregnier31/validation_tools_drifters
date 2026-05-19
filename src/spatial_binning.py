#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
