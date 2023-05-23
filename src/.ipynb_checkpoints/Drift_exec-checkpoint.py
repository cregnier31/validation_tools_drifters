#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:33:54 2018

@author: svangennip
"""
import os, sys, argparse, time, glob
import shutil,shlex,subprocess
import numpy as np
# vstack
from numpy import vstack
import numpy.matlib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
#plt.style.use('seaborn-whitegrid')
#from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
#import seaborn as sns
#sns.set_style("darkgrid", {"axes.facecolor": ".9"})


sys.path.append("/home/svangennip/PYTHON/Modules/LIB_PYT/")
sys.path.append("/home/svangennip/PYTHON/Modules/LIB_SI/")
sys.path.append("/home/svangennip/PYTHON/Modules/")
#from discrete_cmap import discrete_cmap
from netCDF4 import Dataset
import netCDF4 as nc
from datetime import timedelta 
import datetime
import seawater as sw
sys.path.append("/home/svangennip/DEV/LAVD/src/")
from numpy import pi


def run_parcels_Drift(PROD, data_u, var_u, \
                            data_v, var_v, \
                            meshmask,\
                      drifters_lon, drifters_lat, T_DRIFT,\
                       f_out1,
                      radius_cloud=0.25, seeding_resolution=1/12, 
                      forback='FWD', time_period=False):
    import gc
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable, Field
    from datetime import timedelta 
    import datetime
    
    if forback=='FWD':   #coef to x to dt to tell parcels to run in forward (+ve dt), or bwd mode (-ve dt)
        k_forback = 1
    elif forback == 'BWD':
        k_forback=-1        

    print('#-----------------------------------------------------------------------------------------')
    print('#      START RUN_PARCEL_DRIFT: 1. FIELDSET1                 ')
    print('#-----------------------------------------------------------------------------------------')
    filenames = {'U': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': data_u},
                 'V': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': data_v}}#,

    print (filenames['U']['data'])
    print('var_u=',var_u)
    print('var_v=',var_v)

    variables = {'U': var_u,
                 'V': var_v}#,
    
    if PROD == 'GLOBC':
        dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                      'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

    else:             
        dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                      'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}#,
    gc.collect()

    period=1  # set length of dataset (in days), for cycling through data set, e.g. period =2, dataset= [JD1,JD2,JD1,JD2,...]
    #if only one timestep:
    if PROD == 'GLOBC':
        fieldset1 = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, time_periodic=time_period)#,deferred_load=True,field_chunksize=400 ) allow_time_extrapolation=True,
    else:
        fieldset1 = FieldSet.from_netcdf(filenames, variables, dimensions, time_periodic=time_period)#,deferred_load=True,field_chunksize=400 ) allow_time_extrapolation=True,


    now = datetime.datetime.now()
    print(now.strftime("%m/%d/%Y, %H:%M:%S"))


    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      Define ParticleSet ")
    print ("#-----------------------------------------------------------------------------------------")

    # Define a new Particle type including extra Variables
    #compiler ="icc" #"icc" # gcc
    #print('user set compiler = ', compiler)
    #distinguish between two cases: seed at drifter location, seed in neighbourhood
    if radius_cloud == 0:  #seed only at given location (not around)
        lon_all=drifters_lon.ravel()
        lat_all=drifters_lat.ravel()
    else:
        # Seed around point of release dep on radius_cloud=0.25, seeding_resolution=1/12
        #determine number of circles and their radius:
        print('radius_cloud:', radius_cloud)
        print('seeding_resolution',seeding_resolution)
        nb_circles =  float(radius_cloud)/seeding_resolution
        print(nb_circles)
        radii =(np.arange(nb_circles)+1)*seeding_resolution
        print('radius', radii)
        #sys.exit(1)
        #x=x[:-1]
        #pos_i_all =drifters_lon.copy()     
        #pos_j_all =drifters_lat.copy()     
        pos_i_all =np.array([drifters_lon])     
        pos_j_all =np.array([drifters_lat])     
        dlon_all = [0]
        dlat_all = [0] 
        print('pos_i_all=',pos_i_all.shape)
        for i in range(len(radii)):   # Generate noise dlon, dlat for each circle around original position
            r=radii[i]
            nb_pts_on_circle = 8+4*i   #Set nb of points as a function of the radius
            print('    \nr=', r, ' => nb_pts_on_circle', nb_pts_on_circle)
            #x=np.linspace(0, 2*pi, nb_pts_on_circle)#+1)
            x=np.arange(0, 2*pi, 2*pi/nb_pts_on_circle)#+1)
            #print('    x=', x)
            #Calculate noise dlon, dlat to be added to drifter original position:
            dlon=r*np.cos(x)
            dlat=r*np.sin(x)
            print('    dlon=', dlon)
            dlon_all = np.append(dlon_all, dlon)
            dlat_all = np.append(dlat_all, dlat) 
            print('    dlon_all=', dlon_all.shape)
        print('generate ', dlon_all.shape[0],' new positions around each drifter location')

        #Add noise dlon, dlat to each drifter location:

        print('   drifters_lon =', drifters_lon.shape)
        #Expand drifter lons and lat array in y dim, with nb of drifters, to add spatial noise dx,dy
            #   [ [dr_lon1, dr_lon1,....., dr_lon1]
            #     [dr_lon2, dr_lon2,....., dr_lon2]
            #     [dr_lon3, dr_lon3,....., dr_lon3]
            #     [...,     ...    ,....., ...    ]
            #     [dr_lon_n, dr_lon_n,....., dr_lon_n]]
        
        # and add noise
        drifters_cloud_lon = np.matlib.repmat(drifters_lon, len(dlon_all), 1).transpose() + dlon_all 
        drifters_cloud_lat = np.matlib.repmat(drifters_lat, len(dlon_all), 1).transpose() + dlat_all
        print('drifters_cloud_lon = ',drifters_cloud_lon.shape) 
        #sys.exit(1)

        lon_all=drifters_cloud_lon.ravel()
        lat_all=drifters_cloud_lat.ravel()

    #lon_all = np.concatenate([lon_releases3,lon_releases4])#,lon_releases3,lon_releases4])
    #lat_all = np.concatenate([lat_releases3,lat_releases4])#,lat_releases3,lat_releases4])
    npart = len(lon_all)
    #dep_all = np.array([15.0 for i in range(npart)])
    time_pos = np.arange(0, npart) * 0
    print('    total number of releases = ', lon_all.shape[0])
    #sys.exit(1)

    print("    #-----------------------------------------------------------------------------------------")
    print('    #    Create ParticleSet')
    print("    #-----------------------------------------------------------------------------------------")

    #Sample Curl along traj
    pset1 = ParticleSet(fieldset=fieldset1, pclass=JITParticle,
                       lon=lon_all, lat=lat_all, time=time_pos) #, depth=dep_all)
    # Plot the pset1 and print their pressure values p
    #pset1.show(field='vector', savefile='/home/svangennip/DEV/LAVD/WORK/Position_field_curl' )
    #print('curl values before execution:', [curl.curl for curl in pset1])



    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      KERNELS for experiment")
    print ("    #-----------------------------------------------------------------------------------------")

    ## Create KERNELS
    # Create a periodic boundary condition kernel
    def periodicBC(particle, fieldSet1, time):
        if particle.lon > 180:
            particle.lon -= 360


    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC
    kernels = AdvectionRK4  + pset1.Kernel(periodicBC) #+ k_sample


    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      EXECUTE ")
    print ("    #-----------------------------------------------------------------------------------------")

    output_file1 = pset1.ParticleFile(name=f_out1, outputdt=timedelta(hours=6))

    pset1.execute(kernels,
                 runtime=timedelta(days=T_DRIFT),
                 dt=timedelta(minutes=30)*k_forback,
                 output_file=output_file1)
    output_file1.export()

    print('Check output:', f_out1)
    import xarray as xr
    ds = xr.open_dataset(f_out1)
    print(ds)

    traj_lon=ds.lon.values
    traj_lat=ds.lat.values
    traj_time=ds.time.values

    now = datetime.datetime.now()
    print(' EXP1 FINISHED -----  ', now.strftime("%m/%d/%Y, %H:%M:%S"))

    #return traj_lon, traj_lat, traj_time
    
    

def run_parcels_LAVD(PROD, data_u, var_u, data_v, var_v, data_curl, meshmask,lon_min, lon_max, lat_min, lat_max, seed_res,T_DRIFT, f_out1, time_period=False):
    import gc
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable, Field
    from datetime import timedelta 
    import datetime
    
    print('#-----------------------------------------------------------------------------------------')
    print('#      FIELDSET1                 ')
    print('#-----------------------------------------------------------------------------------------')
    filenames = {'U': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': data_u},
                 'V': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': data_v}}#,

    print (filenames['U']['data'])

    variables = {'U': var_u,
                 'V': var_v}#,

    if PROD == 'GLOBC':
        dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                      'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

    else:             
        dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                      'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}#,
    gc.collect()

    period=1  # set length of dataset (in days), for cycling through data set, e.g. period =2, dataset= [JD1,JD2,JD1,JD2,...]
    #if only one timestep:
    if PROD == 'GLOBC':
        fieldset1 = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, time_periodic=time_period)#,deferred_load=True,field_chunksize=400 ) allow_time_extrapolation=True,
    else:
        fieldset1 = FieldSet.from_netcdf(filenames, variables, dimensions, time_periodic=time_period)#,deferred_load=True,field_chunksize=400 ) allow_time_extrapolation=True,

    # Sample Curl
    CURL_filenames = {'lon': meshmask,
                      'lat': meshmask,
                      'depth': meshmask,
                     'data': data_curl}
    
    if PROD == 'GLOBC':
        CURL_dim = { 'time': 'time', 'lon': 'longitude', 'lat': 'latitude'}
        CURL_variable={'CURL': 'vor'}
        CURL = Field.from_netcdf(CURL_filenames, ('CURL', 'vor'), CURL_dim, field_chunksize='auto', time_periodic=time_period)

    else:    
        CURL_dim = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
        CURL_variable={'CURL': 'socurlt'}
        CURL = Field.from_netcdf(CURL_filenames, ('CURL', 'socurlt'), CURL_dim, field_chunksize='auto', time_periodic=time_period)

    # Add to fieldset1
    fieldset1.add_field(CURL)
    print("FieldSet1 OK")

    #print (fieldset1.__dict__)
    #print (fieldset1.U.__dict__)
    #print (fieldset1.CURL.dataFiles)


    class SampleParticle(JITParticle):         # Define a new particle class
        #curl = Variable('curl', initial=fieldset1.CURL)  # Variable 'curl' initialised by sampling the field CURL
        curl = Variable('curl', to_write=True, dtype=np.float32)  # set initialising to 0 (as can take too long)

    now = datetime.datetime.now()
    print(now.strftime("%m/%d/%Y, %H:%M:%S"))


    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      Define ParticleSet ")
    print ("#-----------------------------------------------------------------------------------------")

    # Define a new Particle type including extra Variables
    #compiler ="icc" #"icc" # gcc
    #print('user set compiler = ', compiler)

    lons=np.arange(lon_min, lon_max+seed_res, seed_res)
    lats=np.arange(lat_min, lat_max+seed_res, seed_res)
    xv, yv = np.meshgrid(lons, lats)

    print(xv.shape)
    
    
    lon_releases1=xv.ravel()
    lat_releases1=yv.ravel()

    #print ("lon_releases1 = ", lon_releases1)
    #sys.exit(1)


    lon_all = lon_releases1
    lat_all = lat_releases1
    #lon_all = np.concatenate([lon_releases3,lon_releases4])#,lon_releases3,lon_releases4])
    #lat_all = np.concatenate([lat_releases3,lat_releases4])#,lat_releases3,lat_releases4])
    npart = len(lon_all)
    #dep_all = np.array([15.0 for i in range(npart)])
    time_pos = np.arange(0, npart) * 0
    print('    total number of releases = ', lon_all.shape[0])
    #sys.exit(1)

    print("    #-----------------------------------------------------------------------------------------")
    print('    #    Create ParticleSet')
    print("    #-----------------------------------------------------------------------------------------")

    #Sample Curl along traj
    pset1 = ParticleSet(fieldset=fieldset1, pclass=SampleParticle,
                       lon=lon_all, lat=lat_all, time=time_pos) #, depth=dep_all)
    # Plot the pset1 and print their pressure values p
    #pset1.show(field='vector', savefile='/home/svangennip/DEV/LAVD/WORK/Position_field_curl' )
    #print('curl values before execution:', [curl.curl for curl in pset1])


    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      Initialise tracer values ")
    print ("    #-----------------------------------------------------------------------------------------")

    #So far tracer not initialise (to avoid scipy mode)
    # need to create a kernel

    def SampleCURL_at_t0(particle, fieldset1, time):  # Custom function to sample field at t=0
        particle.curl = fieldset.CURL[0, particle.depth, particle.lat, particle.lon]

    k_sample_at_t0 = pset1.Kernel(SampleCURL_at_t0)

    pset1.execute(k_sample_at_t0,dt=0)
    #print('curl values before execution:', [curl.curl for curl in pset1])


    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      KERNELS for experiment")
    print ("    #-----------------------------------------------------------------------------------------")

    ## Create KERNELS
    #Sample curl field at each timestep
    def SampleCURL(particle, fieldset1, time):  # Custom function that samples fieldset1.P at particle location
        particle.curl = fieldset.CURL[time, particle.depth, particle.lat, particle.lon]

    k_sample = pset1.Kernel(SampleCURL)    # Casting the SampleCURL function to a kernel.

    # Create a periodic boundary condition kernel
    def periodicBC(particle, fieldSet1, time):
        if particle.lon > 180:
            particle.lon -= 360


    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC
    kernels = AdvectionRK4  + k_sample


    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      EXECUTE ")
    print ("    #-----------------------------------------------------------------------------------------")

    output_file1 = pset1.ParticleFile(name=f_out1, outputdt=timedelta(hours=24))

    pset1.execute(kernels,
                 runtime=timedelta(days=T_DRIFT),
                 dt=timedelta(hours=6),
                 output_file=output_file1)
    output_file1.export()

    print('Check output:', f_out1)
    import xarray as xr
    ds = xr.open_dataset(f_out1)
    print(ds)

    traj_lon=ds.lon.values
    traj_lat=ds.lat.values
    traj_curl=ds.curl.values

    now = datetime.datetime.now()
    print(' EXP1 FINISHED -----  ', now.strftime("%m/%d/%Y, %H:%M:%S"))

    return xv, yv
    
def DetectEddies(f_out1, xv, yv, seed_res, T_DRIFT,date_start, dir_plot, titre, figname):
    
    print("\n------------------------------------------------------------------------------")
    print("                 LOAD  EXP                                               ")
    print("------------------------------------------------------------------------------")
    #T_DRIFT=5
    print('load ',f_out1)
    print('T_DRIFT ',T_DRIFT)
    fh = nc.Dataset(f_out1, mode='r')
    traj_lon_raw_OSSE = fh.variables['lon'][:]
    traj_lat_raw_OSSE = fh.variables['lat'][:]
    traj_curl_raw_OSSE = fh.variables['curl'][:]
    traj_time_raw_OSSE = fh.variables['time'][:]
    fh.close()

    traj_lon_raw_OSSE=np.transpose(traj_lon_raw_OSSE)
    traj_lat_raw_OSSE=np.transpose(traj_lat_raw_OSSE)
    traj_curl_raw_OSSE=np.transpose(traj_curl_raw_OSSE)
    traj_time_raw_OSSE=np.transpose(traj_time_raw_OSSE)

    print(traj_lon_raw_OSSE.shape)

    traj_lon_OSSE  = traj_lon_raw_OSSE[:T_DRIFT+1,:]
    traj_lat_OSSE  = traj_lat_raw_OSSE[:T_DRIFT+1,:]
    traj_curl_OSSE = traj_curl_raw_OSSE[:T_DRIFT+1,:]
    traj_time_OSSE = traj_time_raw_OSSE[:T_DRIFT+1,:]

    print(traj_lon_OSSE.shape)


    # rearrange NR data in lat lon grid format

    traj_lon_OSSE_grid=np.zeros((np.shape(xv)[0],np.shape(xv)[1],T_DRIFT+1))
    traj_lat_OSSE_grid=np.zeros((np.shape(xv)[0],np.shape(xv)[1],T_DRIFT+1))
    traj_curl_OSSE_grid=np.zeros((np.shape(xv)[0],np.shape(xv)[1],T_DRIFT+1))

    print(np.shape(traj_lon_OSSE_grid))

    for tstep in range(T_DRIFT+1):
       #print(tstep)
       traj_loni=traj_lon_OSSE[tstep,:]
       traj_lati=traj_lat_OSSE[tstep,:]
       traj_curli=traj_curl_OSSE[tstep,:]


       #print(traj_lon0.shape)
       traj_loni_reshape=np.reshape(traj_loni, np.shape(xv))
       traj_lati_reshape=np.reshape(traj_lati, np.shape(xv))
       traj_curli_reshape=np.reshape(traj_curli, np.shape(xv))

       traj_lon_OSSE_grid[:,:,tstep]=traj_loni_reshape
       traj_lat_OSSE_grid[:,:,tstep]=traj_lati_reshape
       traj_curl_OSSE_grid[:,:,tstep]=traj_curli_reshape

    print('traj_lon_OSSE_grid.shape', traj_lon_OSSE_grid.shape)




    # Calculate coherent vortices:

    ## 
    Curlz_avg_t = np.mean(traj_curl_OSSE,axis=1);               # spatial average of vorticity for each timestep
    print('Curlz_avg_t.shape =',Curlz_avg_t.shape)

    traj_curl_invariant = traj_curl_OSSE - Curlz_avg_t[:, np.newaxis]
    print('traj_curl_invariant=',traj_curl_invariant.shape)

    traj_curl_invariant_abs = np.abs(traj_curl_invariant)

    IVD = np.abs( traj_curl_OSSE[0,:]-Curlz_avg_t[0] )


    IVD_array = np.reshape(IVD, np.shape(xv))*10**4
    #print('IVD = ', np.shape(IVD_array.ravel()))
    #print('min IVD = ',np.nanmin(IVD_array.ravel())) 
    #print('max IVD = ',np.nanmax(IVD_array.ravel())) 


    # create time vector:
    t0=date_start.toordinal()
    date_end = date_start + datetime.timedelta(days=T_DRIFT)
    tend=date_end.toordinal()
    tspan = np.linspace(t0, tend, num=T_DRIFT+1)
    tspan = tspan[:, np.newaxis]
    print(tspan)



    from LAVD_tools import trapz
    from LAVD_plot import plot_contours_rclvs
    LAVD = trapz(traj_curl_invariant_abs,tspan)*10**4

    #LAVD = sp.integrate.trapz(Curlz_t_invariant,tspan)
    print('LAVD = ', np.shape(LAVD))
    print('min LAVD = ',np.nanmin(LAVD)) 
    print('max LAVD = ',np.nanmax(LAVD)) 
    #need to reaarange data in the orca grid:

    # Convert to 2D array for map:
    LAVD_array = np.reshape(LAVD, np.shape(xv))

    lon0=xv[0,:]
    lat0=yv[:,0]


    #==============================================================================
    #                EXAMINE RCVLs of region:
    #==============================================================================

    raw=LAVD_array
    lon0=xv[0,:]
    lat0=yv[:,0]

    # Select particle positions at start and at t=T_DRIFT
    traj_lon_grid_start = traj_lon_OSSE_grid[:,:,0]
    traj_lon_grid_end   = traj_lon_OSSE_grid[:,:,-1]

    traj_lat_grid_start = traj_lat_OSSE_grid[:,:,0]
    traj_lat_grid_end = traj_lat_OSSE_grid[:,:,-1]


    from floater import rclv

    md = 8 # 1/4 degrees  #The minimum distance around maxima (pixel units)

    #ma = 400_000_000 # diam=10km   #The minimum area of the regions (pixels or projected if `lon` and `lat` are specified)
    #tr_ma='diam10km'

    ma = 1_256_000_000 # diam 40km
    str_ma='diam40km'

    #user input values
    step_frac=0.01#STEP_FRAC
    set_cd = 0.01#CD  #The target convexity deficiency allowed for the contour
    res1 = list(rclv.find_convex_contours(raw, init_contour_step_frac=step_frac,
                                         lon=lon0, lat=lat0,
                                         convex_def=set_cd, min_distance=md, min_area=ma, progress=True))


    if not os.path.exists(dir_plot):
        os.makedirs(dir_plot)
        print("create ", dir_plot)
    #grid_res=1/36
    
    titre += str_ma
    figname1 = figname + '_md'+ str(md)+'_'+str_ma+'_cd'+str(set_cd)+'_step_frac'+str(step_frac)+'.png'
    #titre=date_start.strftime("%m/%d/%Y") ': '+ MODEL +', ' + str(T_DRIFT)+'Days, md='+ str(md)+', cd='+str(set_cd)+ ', ma='+str(ma)+ ',step_frac='+str(step_frac)
    plot_contours_rclvs(xv, yv, LAVD_array, seed_res, res1,md, traj_lon_OSSE_grid, traj_lat_OSSE_grid, titre, figname=figname1, save=1)
    

    return res1, traj_lon_OSSE_grid, traj_lat_OSSE_grid




