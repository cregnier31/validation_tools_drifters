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
import random

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
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, StatusCode, Variable, Field#, SummedField, plotTrajectoriesFile
import xarray as xr
#sys.path.append("/home/svangennip/PYTHON/Modules/LIB_PYT/")
#sys.path.append("/home/svangennip/PYTHON/Modules/LIB_SI/")
#sys.path.append("/home/svangennip/PYTHON/Modules/")
#from discrete_cmap import discrete_cmap
from netCDF4 import Dataset
import netCDF4 as nc
from datetime import datetime, timedelta 
#import datetime
import seawater as sw
#sys.path.append("/home/svangennip/DEV/LAVD/src/")
from numpy import pi
from parcels import ParcelsRandom
from datetime import timedelta as delta
import random 




def DeleteParticle(particle, fieldset, time):
    particle.delete()

def run_parcels_Drift_remote(PROD,ds,var_u,var_v,drifters_lon,drifters_lat,drifters_id,T_DRIFT,f_out1,
                            radius_cloud = 0.25, seeding_resolution = 1/12,forback = 'FWD',time_period = False,
                            hour = 0, nb_repeats=None, AdvectionDiffusion_type = "AdvectionRK4"):
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4,AdvectionDiffusionM1, StatusCode, Variable, Field
    
    if forback=='FWD':   #coef to x to dt to tell parcels to run in forward (+ve dt), or bwd mode (-ve dt)
        k_forback = 1
    elif forback == 'BWD':
        k_forback=-1        

    print('#-----------------------------------------------------------------------------------------')
    print('#      START RUN_PARCEL_DRIFT: 1. FIELDSET1                 ')
    print('#-----------------------------------------------------------------------------------------') 
    dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                      'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}
    variables = {'U': var_u,
                 'V': var_v}
    try : 
        fieldset1 = FieldSet.from_xarray_dataset(ds,variables,dimensions,time_periodic=time_period,allow_time_extrapolation=None)
    except KeyError:
        dimensions = {'U': {'lon': 'lon', 'lat': 'lat', 'time': 'time'},
                        'V': {'lon': 'lon', 'lat': 'lat', 'time': 'time'}}
        fieldset1 = FieldSet.from_xarray_dataset(ds,variables,dimensions,time_periodic=time_period,allow_time_extrapolation=None)

    if AdvectionDiffusion_type == "K_smag": 
        print('# Random Walk SMAGORINSKY     ')
        x = fieldset1.U.grid.lon
        y = fieldset1.U.grid.lat
        cell_areas = Field(name='cell_areas', data=fieldset1.U.cell_areas(), lon=x, lat=y)
        fieldset1.add_field(cell_areas)
        fieldset1.add_constant('Cs', 0.1)
    if AdvectionDiffusion_type == "AdvectionDiffusionM1":
        print("# Diffusion Markov1  ")
        #fieldset1.add_constant("dres", 0.5)
        x = fieldset1.U.grid.lon
        y = fieldset1.U.grid.lat
        print(abs(x[0]-x[1]))
        fieldset1.add_constant("dres",abs(x[0]-x[1]))
        time = fieldset1.U.grid.time
        #K_bar = 5.79*10**(-4)*(T_DRIFT/3)*24*3600 # m2.s 
        K_bar = 200
        Kh_zonal =  Field("Kh_zonal",data = np.meshgrid((K_bar/((1000*6378.137)**2) * (180/np.pi)**2 * np.cos(y*np.pi/180)),x)[0].reshape((1,y.shape[0],x.shape[0])), lon = x , lat = y)
        Kh_meridional = Field("Kh_meridional",data = np.meshgrid((K_bar/((1000*6378.137)**2) * (180/np.pi)**2 * np.cos(y*np.pi/180)),x)[0].reshape((1,y.shape[0],x.shape[0])), lon = x , lat = y)

        fieldset1.add_field(Kh_zonal)
        fieldset1.add_field(Kh_meridional)


    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      Define ParticleSet ")
    print ("#-----------------------------------------------------------------------------------------")

    # Define a new Particle type including extra Variables
    #compiler ="icc" #"icc" # gcc
    #print('user set compiler = ', compiler)
    #distinguish between two cases: seed at drifter location, seed in neighbourhood
    if radius_cloud == 0:  #seed only at given location (not around)
        #print('\nradius_cloud:', radius_cloud)
        lon_all=drifters_lon.ravel()
        lat_all=drifters_lat.ravel()
        label_all=drifters_id.ravel()
        #print('label_all,', label_all)
    else:
        # Seed around point of release dep on radius_cloud=0.25, seeding_resolution=1/12
        #determine number of circles and their radius:
        #print('\nradius_cloud:', radius_cloud)
        #print('seeding_resolution',seeding_resolution)
        nb_circles =  float(radius_cloud)/seeding_resolution
        print(nb_circles)
        radii =(np.arange(nb_circles)+1)*seeding_resolution
        #print('radius', radii)
        #sys.exit(1)
        #x=x[:-1]
        #pos_i_all =drifters_lon.copy()     
        #pos_j_all =drifters_lat.copy()     
        pos_i_all =np.array([drifters_lon])     
        pos_j_all =np.array([drifters_lat])     
        dlon_all = [0]
        dlat_all = [0] 
        #print('pos_i_all=',pos_i_all.shape)
        for i in range(len(radii)):   # Generate noise dlon, dlat for each circle around original position
            r=radii[i]
            nb_pts_on_circle = 8+4*i   #Set nb of points as a function of the radius
            #print('    \nr=', r, ' => nb_pts_on_circle', nb_pts_on_circle)
            #x=np.linspace(0, 2*pi, nb_pts_on_circle)#+1)
            x=np.arange(0, 2*pi, 2*pi/nb_pts_on_circle)#+1)
            #print('    x=', x)
            #Calculate noise dlon, dlat to be added to drifter original position:
            dlon=r*np.cos(x)
            dlat=r*np.sin(x)
            #print('    dlon=', dlon)
            dlon_all = np.append(dlon_all, dlon)
            dlat_all = np.append(dlat_all, dlat) 
            #print('    dlon_all=', dlon_all.shape)
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
        drifters_cloud_id = np.matlib.repmat(drifters_id, len(dlon_all), 1).transpose()
        print('drifters_cloud_lon = ',drifters_cloud_lon.shape) 
        print('drifters_cloud_id = ',drifters_cloud_id.shape) 
        #sys.exit(1)

        lon_all=drifters_cloud_lon.ravel()
        lat_all=drifters_cloud_lat.ravel()
        label_all=drifters_cloud_id.ravel()

    #lon_all = np.concatenate([lon_releases3,lon_releases4])#,lon_releases3,lon_releases4])
    #lat_all = np.concatenate([lat_releases3,lat_releases4])#,lat_releases3,lat_releases4])
    npart = len(lon_all)
    #dep_all = np.array([15.0 for i in range(npart)])
    if hour==12:
        time_pos = (np.arange(1, npart+1)/np.arange(1, npart+1)) * delta(hours=12).total_seconds()
    elif hour==0:
        time_pos = np.arange(0, npart)*0
    print('    total number of releases = ', lon_all.shape[0])
    print('    lon releases = ', lon_all[:10])
    print('    lat releases = ', lat_all[:10])
    print('    label releases = ', label_all[:10])

    def Age(fieldset, particle, time):
        particle.age = particle.age + math.fabs(particle.dt)
    
    class SampleParticle(JITParticle):         # Define a new particle class
        distcoast = Variable('distcoast', dtype=np.float32)  # set initialising to 0 (as can take too long)
        age = Variable('age', initial = 0.)
        label = Variable('label', to_write='once', initial=0.,dtype=np.float32)
    
    print("    #-----------------------------------------------------------------------------------------")
    print('    #    Create ParticleSet')
    print("    #-----------------------------------------------------------------------------------------")
    lon_1 = lon_all
    lat_1 = lat_all
    time_1 = time_pos
    label_1 = label_all
    #Sample Curl along traj
    if nb_repeats!= None:
        for i in range (nb_repeats):
            lon_all = np.concatenate((lon_all, lon_1), axis=None) 
            lat_all = np.concatenate((lat_all, lat_1), axis=None)
            label_all = np.concatenate((label_all, label_1), axis=None)
            time_pos=np.concatenate((time_pos, time_1), axis=None)
            #print('shape lon:',np.shape(lon_all))
            #print('shape lat:',np.shape(lat_all))
            #print('shape labl:',np.shape(label_all))
            #print('shape time:',np.shape(time_pos))


    pset1 = ParticleSet(fieldset=fieldset1, pclass=SampleParticle,
                       lon=lon_all, lat=lat_all, time=time_pos) #, depth=dep_all)
    for i in range(len(label_all)): pset1[i].label = label_all[i]
    # Plot the pset1 and print their pressure values p
    #pset1.show(field='vector', savefile='/home/svangennip/DEV/LAVD/WORK/Position_field_curl' )
    #print('curl values before execution:', [curl.curl for curl in pset1])



    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      KERNELS for experiment")
    print ("    #-----------------------------------------------------------------------------------------")

    ## Create KERNELS
    # Create a periodic boundary condition kernel
    def periodicBC(particle, fieldset, time):
        if particle.lon > 180:
            particle.lon -= 360
        elif particle.lon < -180:
            particle.lon += 360

    def smagdiff(particle, fieldset, time):
        dx = 0.01
        # gradients are computed by using a local central difference.
        updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon+dx]
        umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon-dx]
        updy, vpdy = fieldset.UV[time, particle.depth, particle.lat+dx, particle.lon]
        umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat-dx, particle.lon]

        dudx = (updx - umdx) / (2*dx)
        dudy = (updy - umdy) / (2*dx)
        
        dvdx = (vpdx - vmdx) / (2*dx)
        dvdy = (vpdy - vmdy) / (2*dx)

        A = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
        sq_deg_to_sq_m = (1852*60)**2*math.cos(particle.lat*math.pi/180)
        A = A / sq_deg_to_sq_m
        Kh = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5*(dudy + dvdx)**2 + dvdy**2)

        dlat = ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Kh) 
        dlon = ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Kh) 
        particle.lat += dlat
        particle.lon += dlon

    k_age = pset1.Kernel(Age)
    k_periodicBC = pset1.Kernel(periodicBC)
    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC

    if AdvectionDiffusion_type=="AdvectionRK4":
        sum_kernels = AdvectionRK4  + k_periodicBC + k_age #+ k_dispersion#+ k_WestVel
    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC
    if AdvectionDiffusion_type=="AdvectionDiffusionM1":
        sum_kernels = AdvectionDiffusionM1 + k_periodicBC + k_age #+ k_dispersion#+ k_WestVel
    if AdvectionDiffusion_type=="K_smag":
        print('add random walk to kernel')
        k_dispersion = pset1.Kernel(smagdiff)
        sum_kernels = AdvectionRK4 + k_dispersion + k_periodicBC + k_age

    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      EXECUTE ")
    print ("    #-----------------------------------------------------------------------------------------")

    output_file1 = pset1.ParticleFile(name=f_out1, outputdt=timedelta(hours=6))

    ParcelsRandom.seed(random.randint(0,3000))  # Random seed for reproducibility

    print(k_forback)
    
    pset1.execute(sum_kernels,
                 runtime=timedelta(days=T_DRIFT),
                 dt=timedelta(minutes=10)*k_forback,
                 output_file=output_file1,
                 recovery = {StatusCode.ErrorOutOfBounds: DeleteParticle})
    #output_file1.export()

    print('Check output:', f_out1)


    #ds = xr.open_dataset(f_out1)
    #print(ds)

    #traj_lon=ds.lon.values
    #traj_lat=ds.lat.values
    #traj_time=ds.time.values

    now = datetime.now()
    print(' EXP1 FINISHED -----  ', now.strftime("%m/%d/%Y, %H:%M:%S"))

    
    
    
def run_parcels_Drift(PROD, data_u, var_u, \
                            data_v, var_v, \
                            meshmask,\
                      drifters_lon, drifters_lat,drifters_id, T_DRIFT,\
                       f_out1,
                      radius_cloud=0.25, seeding_resolution=1/12, 
                      forback='FWD', time_period=False,hour = 0,
                      nb_repeats=None,
                      AdvectionDiffusion_type = "AdvectionRK4",
                      T_delta = 60,
                      HSampling = 6):
    import gc
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, StatusCode, Variable, Field,AdvectionDiffusionM1
    
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
                       'data': list(data_u)},
                 'V': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': list(data_v)}}#,

    print ('filenames[U][data]', filenames['U']['data'])
    print ('filenames[V][data]', filenames['V']['data'])
    print('var_u=',var_u)
    print('var_v=',var_v)
    variables = {'U': var_u,
                 'V': var_v}#,
   
    if PROD == 'GLOBC' or PROD == 'SMOC' :
        dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                      'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}
    elif "FCST" in PROD or "NWCT" in PROD :
        dimensions = {'U': {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter'},
                      'V': {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter'}}

    else:
        dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                      'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}
    #else:             
    #    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
    #                  'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}#,
    gc.collect()

    period=1  # set length of dataset (in days), for cycling through data set, e.g. period =2, dataset= [JD1,JD2,JD1,JD2,...]
    #if only one timestep:
    if PROD == 'GLOBC' or PROD == 'PSY4' :
        fieldset1 = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, time_periodic=time_period)#,deferred_load=True,field_chunksize=400 ) allow_time_extrapolation=True,
    elif "FCST" in PROD  or "NWCT" in PROD :
        print(f'{filenames=}')
        print(f'{variables=}')
        print(f'{dimensions=}')
        print(time_period)
        fieldset1 = FieldSet.from_nemo(filenames, variables, dimensions,
                                       allow_time_extrapolation=True,
                                       time_periodic=time_period
                                       )#,deferred_load=True)#,field_chunksize=400 ) 
        print('Create fieldset OK')
        e1u = Field.from_netcdf(
            filenames=meshmask,
            variable="e1u",
            dimensions={"lon": "nav_lon", "lat": "nav_lat"},
            interp_method="nearest",
        )
        e1v = Field.from_netcdf(
            filenames=meshmask,
            variable="e1v",
            dimensions={"lon": "nav_lon", "lat": "nav_lat"},
            interp_method="nearest",
        )
        fieldset1.add_field(e1u)
        fieldset1.add_field(e1v)
    else:
        print(f'{filenames=}')
        print(f'{variables=}')
        print(f'{dimensions=}')
        print(time_period)
        fieldset1 = FieldSet.from_nemo(filenames, variables, dimensions,
                                       allow_time_extrapolation=True,
                                       time_periodic=time_period,
                                       )#,deferred_load=True,field_chunksize=400 ) 
        print('Create fieldset OK')
        e1u = Field.from_netcdf(
            filenames=meshmask,
            variable="e1u",
            dimensions={"lon": "glamu", "lat": "gphiu"},
            interp_method="nearest",
        )
        e1v = Field.from_netcdf(
            filenames=meshmask,
            variable="e1v",
            dimensions={"lon": "glamv", "lat": "gphiv"},
            interp_method="nearest",
        )
        fieldset1.add_field(e1u)
        fieldset1.add_field(e1v)
    if AdvectionDiffusion_type=="K_smag": 
        if PROD == "GLOBC" :
            print('# Random Walk SMAGORINSKY     ') 
            x = fieldset1.U.grid.lon
            y = fieldset1.U.grid.lat
            cell_areas_u = Field(name='cell_areas', data=fieldset1.U.cell_areas(), lon=x, lat=y)
            fieldset1.add_field(cell_areas_u)
            fieldset1.add_constant('Cs', 0.1)
        else : 
            print('# Random Walk SMAGORINSKY     ')
            x_u = fieldset1.e1u.grid.lon
            y_u = fieldset1.e1u.grid.lat
            cell_areas = Field(name='cell_areas', data=fieldset1.e1u.data*fieldset1.e1v.data, lon=x_u, lat=y_u)
            fieldset1.add_field(cell_areas)
            Dispersion_field = Field(name='Dispersion_field',data = fieldset1.e1u.data*fieldset1.e1v.data*0,lon = x_u, lat = y_u)
            fieldset1.add_field(Dispersion_field)
            #x_v = fieldset1.e1v.grid.lon
            #y_v = fieldset1.e1v.grid.lat
            #cell_areas_v = Field(name='cell_areas_v', data=fieldset1.e1v.data, lon=x_v, lat=y_v)
            #fieldset1.add_field(cell_areas_v)
            fieldset1.add_constant('Cs', 0.1)
    if AdvectionDiffusion_type == "AdvectionDiffusionM1":
        if PROD == "GLOBC" or "GLO12_FCST" in PROD:
            print("# Diffusion Markov1  ")
            #fieldset1.add_constant("dres", 0.5)
            x = fieldset1.U.grid.lon
            y = fieldset1.U.grid.lat
            fieldset1.add_constant("dres",abs(x[0]-x[1]))
            print(x)
            print(x.shape)
            print(abs(x[0]-x[1]))
            time = fieldset1.U.grid.time
            K_bar = 200
            Kh_zonal =  Field("Kh_zonal",data = np.meshgrid((K_bar/((1000*6378.137)**2) * (180/np.pi)**2 * np.cos(y*np.pi/180)),x)[0].reshape((1,y.shape[0],x.shape[0])), lon = x , lat = y)
            Kh_meridional = Field("Kh_meridional",data = np.meshgrid((K_bar/((1000*6378.137)**2) * (180/np.pi)**2 * np.cos(y*np.pi/180)),x)[0].reshape((1,y.shape[0],x.shape[0])), lon = x , lat = y)
            fieldset1.add_field(Kh_zonal)
            fieldset1.add_field(Kh_meridional)

        
        else:
            print("# Diffusion Markov1  ")
            #fieldset1.add_constant("dres", 0.5)
            x = fieldset1.e1u.grid.lon
            y = fieldset1.e1u.grid.lat
            fieldset1.add_constant("dres",abs(x[0,0]-x[0,1]))
            print(x)
            print(x.shape)
            print(abs(x[0,0]-x[0,1]))
            print(y)
            time = fieldset1.U.grid.time
            K_bar = 200
            Kh_zonal =  Field("Kh_zonal",data = (K_bar/((1000*6378.137)**2) * (180/np.pi)**2 * np.cos(y*np.pi/180)).reshape((1,y.shape[0],x.shape[1])), lon = x , lat = y)
            Kh_meridional = Field("Kh_meridional",data = (K_bar/((1000*6378.137)**2) * (180/np.pi)**2 * np.cos(y*np.pi/180)).reshape((1,y.shape[0],x.shape[1])), lon = x , lat = y)
            fieldset1.add_field(Kh_zonal)
            fieldset1.add_field(Kh_meridional)

    
    now = datetime.now()
    print(now.strftime("%m/%d/%Y, %H:%M:%S"))


    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      Define ParticleSet ")
    print ("#-----------------------------------------------------------------------------------------")

    # Define a new Particle type including extra Variables
    #compiler ="icc" #"icc" # gcc
    #print('user set compiler = ', compiler)
    #distinguish between two cases: seed at drifter location, seed in neighbourhood
    #print(radius_cloud)
    if radius_cloud == 0:  #seed only at given location (not around)
        print('\nradius_cloud:', radius_cloud)
        lon_all = drifters_lon.ravel()
        lat_all = drifters_lat.ravel()
        #label_all=drifters_id.ravel()
        label_all = drifters_id
        #print('label_all,', label_all)
        #print(np.shape(lon_all))
        #print(np.shape(lat_all))
        #print(np.shape(label_all))
    else:
        # Seed around point of release dep on radius_cloud=0.25, seeding_resolution=1/12
        #determine number of circles and their radius:
        print('\nradius_cloud:', radius_cloud)
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
        drifters_cloud_id = np.matlib.repmat(drifters_id, len(dlon_all), 1).transpose()
        print('drifters_cloud_lon = ',drifters_cloud_lon.shape) 
        print('drifters_cloud_id = ',drifters_cloud_id.shape) 
        #sys.exit(1)

        lon_all=drifters_cloud_lon.ravel()
        lat_all=drifters_cloud_lat.ravel()
        label_all=drifters_cloud_id.ravel()

    #lon_all = np.concatenate([lon_releases3,lon_releases4])#,lon_releases3,lon_releases4])
    #lat_all = np.concatenate([lat_releases3,lat_releases4])#,lat_releases3,lat_releases4])
    npart = len(lon_all)
    #dep_all = np.array([15.0 for i in range(npart)])
    if hour==12:
        time_pos = (np.arange(1, npart+1)/np.arange(1, npart+1)) * delta(hours=12).total_seconds()
    elif hour==0:
        time_pos = np.arange(0, npart)*0
    print('    total number of releases = ', lon_all.shape[0])
    print('    lon releases = ', lon_all[:10])
    print('    lat releases = ', lat_all[:10])
    print('    label releases = ', label_all[:10])

    def Age(fieldset, particle, time):
        particle.age = particle.age + math.fabs(particle.dt)
    
    class SampleParticle(JITParticle):         # Define a new particle class
        distcoast = Variable('distcoast', dtype=np.float32)  # set initialising to 0 (as can take too long)
        age = Variable('age', initial = 0.)
        label = Variable('label', to_write='once', initial=0.,dtype=np.float32)

    
    print("    #-----------------------------------------------------------------------------------------")
    print('    #    Create ParticleSet')
    print("    #-----------------------------------------------------------------------------------------")

    #Sample Curl along traj
    lon_1 = lon_all
    lat_1 = lat_all
    time_1 = time_pos
    label_1 = label_all
    if nb_repeats!= None:
        for i in range (nb_repeats):
            lon_all = np.concatenate((lon_all, lon_1), axis=None) 
            lat_all = np.concatenate((lat_all, lat_1), axis=None)
            label_all = np.concatenate((label_all, label_1), axis=None)
            time_pos=np.concatenate((time_pos, time_1), axis=None)

    pset1 = ParticleSet(fieldset=fieldset1, pclass=SampleParticle,
                       lon=lon_all, lat=lat_all, time=time_pos) #, depth=dep_all)
    for i in range(len(label_all)): pset1[i].label = label_all[i]
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
        elif particle.lon < -180:
            particle.lon += 360

    def smagdiff(particle, fieldset, time):

        dx = 0.01
        # gradients are computed by using a local central difference.
        updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon+dx]
        umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon-dx]
        updy, vpdy = fieldset.UV[time, particle.depth, particle.lat+dx, particle.lon]
        umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat-dx, particle.lon]

        dudx = (updx - umdx) / (2*dx)
        dudy = (updy - umdy) / (2*dx)
        
        dvdx = (vpdx - vmdx) / (2*dx)
        dvdy = (vpdy - vmdy) / (2*dx)
        A = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
        #A_v = fieldset.cell_areas_v[time, 0, particle.lat, particle.lon]

        sq_deg_to_sq_m = (1852*60)**2*math.cos(particle.lat*math.pi/180)
        A = A / sq_deg_to_sq_m
        #A_v = A_v / sq_deg_to_sq_m

        Kh = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5*(dudy + dvdx)**2 + dvdy**2)
        #Kh_v = fieldset.Cs * A_v * math.sqrt(dudx**2 + 0.5*(dudy + dvdx)**2 + dvdy**2)
        #fieldset.Dispersion_field += K_h
        dlat = ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Kh) 
        dlon = ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Kh) 

        #dlat = fieldset.rng[int(ParcelsRandom.normalvariate(0, 10))] * math.sqrt(2*math.fabs(particle.dt)* Kh) 
        #dlon = fieldset.rng[int(ParcelsRandom.normalvariate(0, 10))] * math.sqrt(2*math.fabs(particle.dt)* Kh) 

        particle.lat += dlat
        particle.lon += dlon
    def DeleteParticle(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    k_age = pset1.Kernel(Age)
    k_periodicBC = pset1.Kernel(periodicBC)
    k_delete = pset1.Kernel(DeleteParticle)
    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC


    
    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC
    if AdvectionDiffusion_type=="AdvectionRK4":
        sum_kernels = AdvectionRK4  + k_periodicBC + k_age + k_delete#+ k_dispersion#+ k_WestVel
        #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC
    if AdvectionDiffusion_type=="AdvectionDiffusionM1":
        sum_kernels = AdvectionDiffusionM1 + k_periodicBC + k_age + k_delete#+ k_dispersion#+ k_WestVel

    if AdvectionDiffusion_type=="K_smag":
        print('add random walk to kernel')
        k_dispersion = pset1.Kernel(smagdiff)
        sum_kernels = AdvectionRK4  + k_periodicBC + k_age + k_dispersion+ k_delete

    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      EXECUTE ")
    print ("    #-----------------------------------------------------------------------------------------")

    output_file1 = pset1.ParticleFile(name=f_out1, outputdt=timedelta(hours=HSampling))
    
    ParcelsRandom.seed(random.randint(0,3000))  # Random seed for reproducibility
    pset1.execute(sum_kernels,
                 runtime=timedelta(days=T_DRIFT),
                 dt=timedelta(minutes=T_delta)*k_forback,
                 output_file=output_file1)

    #output_file1.export()

    print('Check output:', f_out1)
    #ds = xr.open_dataset(f_out1)
    #print(ds)

    #traj_lon=ds.lon.values
    #traj_lat=ds.lat.values
    #traj_time=ds.time.values

    now = datetime.now()
    print(' EXP1 FINISHED -----  ', now.strftime("%m/%d/%Y, %H:%M:%S"))

    #return traj_lon, traj_lat, traj_time
    
    
def run_parcels_Drift_with_repeat(PROD, data_u, var_u,
                             data_v, var_v, meshmask,
                             data_stokes, ustokes, v_stokes,
                             drifters_lon, drifters_lat, drifters_id, T_DRIFT,
                             f_out1,
                             radius_cloud=0.25, seeding_resolution=1/12,
                             dt_output=24 , repeatdt=None, nb_repeats=None, K_disp=None, time_period=False):
    import gc
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, StatusCode, Variable, Field, DiffusionUniformKh, AdvectionDiffusionM1, AdvectionDiffusionEM
    import math

    forback='FWD'
    print('execute run_parcels_Drift_with_repeat...')
    print('configuration of Parcels run:')
    print('MODEL=', PROD)
    print('drifter at location= (', drifters_lon,';', drifters_lat,')')
    print('Spatial noise:')
    print('release particles within: ', radius_cloud,' degrees around location, with a ', seeding_resolution, 'degrees resolution')
    print('temporal noise:')
    print('number of repeated releases:', nb_repeats)
    print('spaced out by repeatdt=',repeatdt)
    print('convert to hours=',repeatdt.total_seconds())
    print('starting date ', data_u[0])
    if forback=='FWD':
        k_forback = 1 #coef to x to dt to tell parcels to run in forward (+ve dt), or bwd mode (-ve dt)
        print('forwards in time')
    else:
        print('backwards in time')
        k_forback=-1
    #sys.exit(1)
    print('#-----------------------------------------------------------------------------------------')
    print('#      A. CREATE FIELDSET:          ')
    print('#-----------------------------------------------------------------------------------------')
    print('#      1. U,V          ')

    filenames = {'U': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': data_u},
                 'V': {'lon': meshmask,
                       'lat': meshmask,
                       'depth': meshmask,
                       'data': data_v}}#,

    #print (filenames['U']['data'])
    #print('var_u=',var_u)
    #print('var_v=',var_v)

    variables = {'U': var_u,
                 'V': var_v}#,

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                      'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}#,
    gc.collect()
    period=1  # set length of dataset (in days), for cycling through data set, e.g. period =2, dataset= [JD1,JD2,JD1,JD2,...]
    fieldset_nemo = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True, time_periodic=time_period)#,deferred_load=True,field_chunksize=400 ) allow_time_extrapolation=True,


    print('#      2. STOKES          ')
    #wstokes=1
    if data_stokes is not None:
        stokesfiles = data_stokes
        stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
        stokesvariables = {'U': 'vsdx', 'V': 'vsdy'}
        fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions, allow_time_extrapolation=True, time_periodic=time_period)
        #fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

        fieldset1 = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U,
                                      V=fieldset_nemo.V+fieldset_stokes.V)
    else:
        fieldset1 = fieldset_nemo

    print('#-----------------------------------------------------------------------------------------')
    print('#      B. ADD FIELDS:          ')
    print('#-----------------------------------------------------------------------------------------')
    print('#      1. DIST TO COAST          ')

    print('#      2. RANDOM WALK          ')
    #create field with ones:
    import xarray as xr
    ds2 = xr.open_dataset(meshmask)
    nav_lon=ds2.nav_lon.values
    nav_lat=ds2.nav_lat.values

    if K_disp:
        K_m = K_disp*np.ones((nav_lon.shape[0], nav_lon.shape[1]))
        K_h = K_disp*np.ones((nav_lon.shape[0], nav_lon.shape[1]))

        Kh_meridional = Field('Kh_meridional', K_m,lon=nav_lon,lat=nav_lat, mesh='spherical')
        Kh_zonal = Field('Kh_zonal', K_h,lon=nav_lon,lat=nav_lat, mesh='spherical')
        fieldset1.add_field(Kh_meridional)
        fieldset1.add_field(Kh_zonal)
    #fieldset1.add_constant('dres', 1/20)


    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      Define ParticleSet ")
    print ("#-----------------------------------------------------------------------------------------")

    # Define a new Particle type including extra Variables
    #compiler ="icc" #"icc" # gcc
    #print('user set compiler = ', compiler)
    #distinguish between two cases: seed at drifter location, seed in neighbourhood
    if radius_cloud == 0:  #seed only at given location (not around)
        #lon_all=drifters_lon.ravel()
        #lat_all=drifters_lat.ravel()
        lon_batch=drifters_lon.ravel()
        lat_batch=drifters_lat.ravel()
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
        print(dlon_all,' new positions around each drifter location')

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
        print('drifters_cloud_lon=', drifters_cloud_lon.shape)
        print(drifters_cloud_lon)
        #sys.exit(1)
        drifters_cloud_lat = np.matlib.repmat(drifters_lat, len(dlon_all), 1).transpose() + dlat_all
        print('drifters_cloud_lon = ',drifters_cloud_lon.shape) 
        #sys.exit(1)

        #assign drifter id to each element of the cloud.
        drifters_cloud_id = np.matlib.repmat(drifters_id, len(dlon_all), 1).transpose()

        lon_batch=drifters_cloud_lon.ravel(order='C')
        lat_batch=drifters_cloud_lat.ravel(order='C')
        label_batch=drifters_cloud_id.ravel(order='C')
    
        print('lon_batch',lon_batch) 
        print('loat_batch',lat_batch) 
        print('label_batch',label_batch) 
    #sys.exit(1)
    #lon_all = np.concatenate([lon_releases3,lon_releases4])#,lon_releases3,lon_releases4])
    #lat_all = np.concatenate([lat_releases3,lat_releases4])#,lat_releases3,lat_releases4])
    npart = len(lon_batch)
    #dep_all = np.array([15.0 for i in range(npart)])
    time_pos = np.arange(0, npart) * 0
    print('    total number of releases = ', lon_batch.shape[0])
    #sys.exit(1)

    print('    total number of particles = ', npart)
    print('    total number of repeats = ', repeatdt)
    #sys.exit(1)

    #So far tracer not initialise (to avoid scipy mode)
    # need to create a kernel

    def Age(fieldset, particle, time):
        particle.age = particle.age + math.fabs(particle.dt)
    
    class SampleParticle(JITParticle):         # Define a new particle class
        distcoast = Variable('distcoast', dtype=np.float32)  # set initialising to 0 (as can take too long)
        age = Variable('age', initial = 0.)
        label = Variable('label', to_write='once', initial=0.,dtype=np.float32)

    #pset1 = ParticleSet(fieldset=fieldset1, pclass=SampleParticle,
    #                   lon=lon_all, lat=lat_all, repeatdt=repeatdt ) #, depth=dep_all)
    pset1 = ParticleSet(fieldset=fieldset1, pclass=SampleParticle,
                       lon=[], lat=[], time=[] ) #, depth=dep_all)
    # initialise drifter id for all elements of clouds
    #pset1.label = id_all
    #for i in range(len(id_all)): pset1[i].label = id_all[i]
         
    print('pset1.lon',pset1.lon) 
    print('pset1.lat',pset1.lat) 
    print('pset1.label',pset1.label) 
    #sys.exit(1)    

    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      KERNELS for experiment")
    print ("    #-----------------------------------------------------------------------------------------")

    ## Create KERNELS
    # Create a periodic boundary condition kernel
    def periodicBC(particle, fieldSet, time):
        if particle.lon > 180:
            particle.lon -= 360
        elif particle.lon < -180:
            particle.lon += 360

    def BrownianMotion2D(particle, fieldset, time):
        """Kernel for simple Brownian particle diffusion in zonal and meridional direction.
        Assumes that fieldset has fields Kh_zonal and Kh_meridional"""
        dtt = particle.dt
        r = 1.
        kh_meridional = fieldset.Kh_meridional[time, particle.depth, particle.lat, particle.lon]
        #particle.lat += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dtt)*kh_meridional/r)
        particle.lat += ParcelsRandom.uniform(-1,1) * math.sqrt(2*math.fabs(dtt)*kh_meridional/r)
        #particle.lat += math.sqrt(2*math.fabs(dtt)*kh_meridional/r)
        kh_zonal = fieldset.Kh_zonal[time, particle.depth, particle.lat, particle.lon]
        #particle.lon += random.uniform(-1., 1.) * math.sqrt(2*math.fabs(dtt)*kh_zonal/r)
        particle.lon += ParcelsRandom.uniform(-1,1) * math.sqrt(2*math.fabs(dtt)*kh_zonal/r)




    k_age = pset1.Kernel(Age)
    k_periodicBC = pset1.Kernel(periodicBC)
    #kernels = AdvectionRK4 + pset1.Kernel(DeleteParticle) #+ periodicBC
    sum_kernels = AdvectionRK4  + k_periodicBC + k_age #+ k_dispersion#+ k_WestVel
    #kernels=AdvectionDiffusionM1 + k_sample_at_t0
    if K_disp:
        print('add random walk to kernel')
        k_dispersion = pset1.Kernel(BrownianMotion2D)
        sum_kernels += k_dispersion


    print ("    #-----------------------------------------------------------------------------------------")
    print ("    #                      EXECUTE ")
    print ("    #-----------------------------------------------------------------------------------------")
    output_file1 = pset1.ParticleFile(name=f_out1, outputdt=timedelta(hours=dt_output))
    
    runtime=(T_DRIFT)*24
    print(type(repeatdt))
    print(repeatdt.total_seconds())
    print('runtime (hrs)', runtime)
    repeatdt_hrs = repeatdt.total_seconds()/3600
    print('repeatdt_hrs', repeatdt_hrs)
    print('release times:', np.arange(0, runtime, repeatdt_hrs))
    #sys.exit(1)

    #Initialise batch1 for release time t0    
    lon_all= []#lon_batch
    lat_all = []#lat_batch
    label_all = []#label_batch
    time_all = []#np.zeros(len(lon_batch))

    #for time in np.arange(0+repeatdt_hrs, nb_repeats*repeatdt_hrs, repeatdt_hrs):
    for time in np.arange(0, nb_repeats*repeatdt_hrs, repeatdt_hrs):
        print('time=',time)
        lon_all = np.concatenate((lon_all, lon_batch), axis=None) 
        lat_all = np.concatenate((lat_all, lat_batch), axis=None)
        label_all = np.concatenate((label_all, label_batch), axis=None)
        time_all=np.concatenate((time_all, np.ones(lon_batch.shape)*time), axis=None)
        print('shape lon:',np.shape(lon_all))
        print('shape lat:',np.shape(lat_all))
        print('shape labl:',np.shape(label_all))
        print('shape time:',np.shape(time_all))


    pset1 = ParticleSet(fieldset=fieldset1, pclass=SampleParticle,lon=lon_all, lat=lat_all, time=time_all) 
    for i in range(len(label_all)): pset1[i].label = label_all[i]

    print('pset1.lon',pset1.lon)
    print('pset1.lat',pset1.lat)
    print('pset1.label',pset1.label)
    print('pset1.time',pset1.time)
    
    pset1.execute(sum_kernels, dt=0) #+ k_delete+k_BC, dt=0)#+k_dist++k_sample,dt=0)
    print(pset1[0])

    pset1.execute(sum_kernels,
                     runtime=timedelta(hours=runtime),
                     dt=timedelta(minutes=10),
                     output_file=output_file1,
                     recovery = {StatusCode.ErrorOutOfBounds: DeleteParticle})


    output_file1.close()
