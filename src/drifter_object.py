#!/usr/bin/env python
# coding: utf-8
# Using the magic encoding
# -*- coding: utf-8 -*-

"""This is a python package for drifter module.

This module is used to defined a drifter class
Author : slawchune September 2018
"""

import numpy as np
import datetime

"""
"""

class drifter:
    #########################################################################################
    # Define a basic drifter class, with basic atributes, basic methods 
    #########################################################################################   
    def __init__(self):
        self.time=[]
        self.lon=[]
        self.lat=[]
        self.depth=[]
        self.u=[]
        self.v=[]
        self.drogue=[]
        self.id=[]
    #########################################################################################
    #########################################################################################
    def filter_time(self, date_min, date_max):
        # This fuction supresses drifter data outside of the time window bounded by
        # [date_min,date_max]
        # time_min and time_max are belived to be datetime objects
        time=np.array(self.time)
        ind=np.squeeze(np.where( (time>date_min) & (time<date_max) ))
        #tmp_delta_time=[abs(o.total_seconds()) for o in (time-date_min)]
        self.filter_index(ind)
    #########################################################################################
    def filter_zone(self, lon_min, lon_max, lat_min, lat_max):
        # This fuction supresses drifter data outside of a spatial box bounded by
        # [lon_min, lon_max, lat_min, lat_max]
        lon=np.array(self.lon) ; lat=np.array(self.lat) 
        ind=np.squeeze(np.where( (lon>=lon_min) & (lon<=lon_max) & (lat>=lat_min) & (lat<=lat_max) ))
        self.filter_index(ind)

    #########################################################################################
    def filter_drogue(self):
        # # This fuction supresses drifter data when the drogue is attached
        drogue=np.array(self.drogue)
        ind=np.squeeze(np.where(drogue == 0))
        self.filter_index(ind)
    #########################################################################################    

    #########################################################################################
    def filter_index(self, ind):
        if (not(ind.size) or (ind.size==1)):
            # print( 'WARNING : data n° '+self.id+' is out of specified zone and supressed')
            self.__init__() # we use built in initialisation to empty all the data
        else:
            self.lon=self.lon[ind] ; self.lat=self.lat[ind]
            self.depth=self.depth[ind]
            self.u=self.u[ind] ; self.v=self.v[ind]
            self.time=self.time[ind]
            self.drogue=self.drogue[ind]
            #return a list of class attribut
            members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            if 'lon_err' in members :
               self.lon_err=self.lon_err[ind] ; self.lat_err=self.lat_err[ind]
               self.u_err=self.u_err[ind] ; self.v_err=self.v_err[ind]
            '''
            # return a list of class attribut
            members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")
            # filter all atributes array with an index
            for m in members:
               setattr(self ,m, self.m[ind] )
            '''
    #########################################################################################


"""
we defined a subclass of drifter for AOML 
with appropriate methods 
"""
class drifter_AOML(drifter):
    #########################################################################################
    def get_drifter(self,file_path):
        # Method to charge a AOML drifter from 
        fichier= open(file_path,'r')
        tmp=fichier.readlines()
        fichier.close
        self.id=tmp[0].split()[0]
        for line in tmp:
            self.time.append( datetime.datetime( int(line.split()[1]), int(line.split()[2]), int(line.split()[3]), int(line.split()[4]) ) )
            lon=((float(line.split()[5])- 180) % 360) - 180 # convert longitude from [0,360] to [-180,180] 
            self.lon.append(lon) ; self.lat.append(float(line.split()[6]))
            self.u.append(float(line.split()[9])) ; self.v.append(float(line.split()[10]))
            # Drogue status in position number 14 in gps data, 15 otherwise
            if ('gps' in file_path):
                self.drogue.append(int(line.split()[14]))
            else: 
                self.drogue.append(int(line.split()[15]))
            if (self.drogue[-1]==1):
                # assuming a drogued buoy travels at 15m of depth
                self.depth.append(15)
            else:
                # assuming a non-drogued buoy stays at surface 
                self.depth.append(0)
    #########################################################################################

"""
subclass for netcdf drifters
ex: file=/scratch/work/lawchunes/Validation_SMOC/PhoD/v1.02/driftertrajGPS_1.02.nc
name=
"""

class drifter_AOML_nc(drifter):
    def __init__(self):
      # add new attributes
      self.u_err=[]  # observational error
      self.v_err=[]
      self.lat_err=[]
      self.lon_err=[]

    #########################################################################################
    def get_drifter(self, id, lon, lat, u, v, drogue, time, lon_err, lat_err, u_err, v_err):
        self.id=id
        self.lon=lon
        self.lat=lat
        self.u=u
        self.v=v
        self.drogue=drogue
        self.time=time
        self.depth=np.where( drogue==1, 20, 0 )
        self.u_err=u_err
        self.v_err=v_err
        self.lat_err=lat_err
        self.lon_err=lon_err

    #########################################################################################

    
"""
we defined a subclass of drifter for PARCELS drifter 
with appropriate methods 
"""

class one_drifter_PARCELS(drifter):

    def __init__(self):
        # add new attributes
        self.model=[]  # model should contain the model name 
        self.date0=[]
        
    def get_drifter(self,file_path, **kwargs):
    # Drifter object consists in a single trajectory, so the goal of the routine
    # is to retrive only the trajectory numbered i_traj in a Parcels netcdf output
    # key word argument: i_traj = value
    #########################################################################################        
        
        import netCDF4
        # Method to charge a Parcels drifter from 
        nc_file= netCDF4.Dataset(file_path,'r',format='NETCDF4')
        # Read dimensions --> NO USE 
        #nb_time=len(nc_file.dimensions['obs'])
        #nb_traj=len(nc_file.dimensions['traj'])
        # Specify if we want to extract a specific particle, otherwise, select
        # the first one
        i_traj=0
        # if we have i_traj=i_traj in argument at function call, 
        # we choose the i_traj particle         
        for key,value in kwargs.items():
            print(key, value)
            if (key=='i_traj'):
                i_traj=value

        #Read variable
        '''
        self.timedelta=nc_file.variables['time'][:].data[i_traj]
        self.lon=nc_file.variables['lon'][:].data[i_traj]
        self.lat=nc_file.variables['lat'][:].data[i_traj]
        self.depth=nc_file.variables['z'][:].data[i_traj]
        time=nc_file.variables['time'][:].data[i_traj]        
        '''
        self.timedelta=nc_file.variables['time'][:][i_traj]
        self.lon=nc_file.variables['lon'][:][i_traj]
        self.lat=nc_file.variables['lat'][:][i_traj]
        self.depth=nc_file.variables['z'][:][i_traj]
        time=nc_file.variables['time'][:][i_traj]
        
        self.timedelta=time-time[0]
        
        try:
            self.id=nc_file.id_buoy
        except:
            pass
        try:
            self.model=nc_file.model
        except:
            pass 
        
        # Subsection to transform time into python time object:
        str_time_units=nc_file.variables['time'].units.split(' ')[0]
        str_time_date0=nc_file.variables['time'].units.split(' ')[2]
        date0_yyyy=int(str_time_date0.split('-')[0])
        date0_mm=int(str_time_date0.split('-')[1]) 
        date0_dd=int(str_time_date0.split('-')[2].split('T')[0])
        date0_hh=int(str_time_date0.split('-')[2].split('T')[1].split(':')[0])
        date0_mn=int(str_time_date0.split('-')[2].split('T')[1].split(':')[1])
        date_0=datetime.datetime(date0_yyyy, date0_mm, date0_dd, date0_hh, date0_mn)
        
        if ('seconds' in str_time_units):
            self.time=[ datetime.timedelta(seconds=j)+date_0 for j in time ]
        elif ('minutes' in str_time_units):
            self.time=[ datetime.timedelta(minutes=j)+date_0 for j in time ]
        elif ('hours' in str_time_units):
            self.time=[ datetime.timedelta(hours=j)+date_0 for j in time ]
        elif ('days' in str_time_units):
            self.time=[ datetime.timedelta(days=j)+date_0 for j in time ]
        
      
        # Subsection fo fill u and v
        self.u=nc_file.variables['U'][:][i_traj]
        self.v=nc_file.variables['V'][:][i_traj]
        #self.u, self.v, R, teta=compute_uvRteta(self.time, self.lon, self.lat)
        # return date0
        self.date0=self.time[0].strftime('%Y%m%d%H%M')
        nc_file.close()
    #########################################################################################

'''
SOME USEFULL FUNCTIONS TO DEAL WITH DRIFTERS
'''

########################################################################################

def distance_on_earth(lat1, long1, lat2, long2):
    lat1=np.radians(lat1) ; lon1=np.radians(long1)
    lat2=np.radians(lat2) ; lon2=np.radians(long2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6373.0
    distance=R*c
    return(distance)
    
########################################################################################       
def compute_uvRteta(date, lon, lat):

        import math
        import numpy as np

        u=np.arange(np.size(date))*float(np.NaN) ; v=np.arange(np.size(date))*float(np.NaN)
        R=np.arange(np.size(date))*float(np.NaN) ; teta=np.arange(np.size(date))*float(np.NaN)

        for i in range(0,np.size(date)-1):
                delta_lon=0 ; delta_lat=0
                #distance
                if (lon[i+1] != lon[i]):
                        delta_lon=distance_on_earth(lat[i],lon[i],lat[i],lon[i+1])*1000.
                if (lat[i+1] != lat[i]):
                        delta_lat=distance_on_earth(lat[i],lon[i],lat[i+1],lon[i])*1000.
                #sign
                if (lon[i+1] < lon[i]):
                        delta_lon=-delta_lon
                if (lat[i+1] < lat[i]):
                        delta_lat=-delta_lat
                u[i]=(delta_lon)/(date[i+1]-date[i]).total_seconds()
                v[i]=(delta_lat)/(date[i+1]-date[i]).total_seconds()
                
        R=np.sqrt(np.power(u,2)+np.power(v,2))
        teta=np.arctan2(v,u)*180.0/math.pi
        return u, v, R, teta
########################################################################################
   

#######################################################################################
# compute angle difference, obs is assumed to be the angle of reference
def angle_difference(simu,obs):
        difference=simu-obs
        for i,values in enumerate(difference): 
            while difference[i] < -180 :
                    difference[i]=difference[i]+360
            while difference[i] > 180 :
                    difference[i]=difference[i]-360
        return difference  
#######################################################################################

#######################################################################################
#compute average of angles
def angle_array_average(angle_array):
    import math
    angle_array_radians=np.radians(angle_array)
    sin_array=np.sin(angle_array_radians)
    cos_array=np.cos(angle_array_radians)
    angle_av=np.arctan2(sin_array.mean(),cos_array.mean())
    angle_av=np.degrees(angle_av)
    return angle_av
#######################################################################################

#######################################################################################


#######################################################################################
def compute_drifter_errors(lon_obs , lat_obs, u_obs, v_obs, 
                           lon_simu, lat_simu, u_simu, v_simu):
    distance_error=distance_on_earth(lat_obs, lon_obs, lat_simu, lon_simu)
    angle_error=angle_difference(np.arctan2(v_simu, u_simu)*180.0/np.pi, \
                                 np.arctan2(v_obs, u_obs)*180.0/np.pi )
    velocity_error=np.sqrt(np.power(u_simu,2)+np.power(v_simu,2)) - np.sqrt(np.power(u_obs,2)+np.power(v_obs,2))

    distance_obs_from_start=distance_on_earth(lat_obs, lon_obs, lat_obs[0], lon_obs [0])
    delta_obs=np.zeros(np.shape(distance_obs_from_start))
    for i in np.arange(len(delta_obs)):
      if (i >= 1):
        delta_obs[i]=distance_obs_from_start[i]-distance_obs_from_start[i-1]
    
    # Liu and weisberg (2011) index 
    treshold_LIU=1
    l_0i=np.cumsum(distance_obs_from_start)

    with np.errstate(divide='ignore', invalid='ignore'):
      L=np.where(np.cumsum(l_0i) != 0, np.cumsum(distance_error)/np.cumsum(l_0i), np.nan )
      #L=np.cumsum(distance_error)/np.cumsum(l_0i)

      L[L<=treshold_LIU]=1-L[L<=treshold_LIU]/treshold_LIU
      L[L>treshold_LIU ]=0

      # Molcard Skill score
      M=np.where(distance_obs_from_start != 0, 1-(distance_error/distance_obs_from_start), np.nan ) 
      #M=1-(distance_error/distance_obs_from_start)

      # Sutherland Skill Score
      num=distance_error * delta_obs
      den=distance_obs_from_start * delta_obs
      S=np.where(np.cumsum(den) !=0 , 1-(np.cumsum(num)/np.cumsum(den)) , np.nan)
      #S=1-(np.cumsum(num)/np.cumsum(den))

    return distance_error, angle_error, velocity_error, L, M, S 
#######################################################################################


#######################################################################################
def compute_vector_corr(u_obs, v_obs, u_simu, v_simu):
    import cmath, math
    # compute complex numbers
    C_obs=u_obs+1j*v_obs
    C_simu=u_simu+1j*v_simu
    Cmatrix=np.corrcoef(C_simu,C_obs)[0,1]
    coeff, phase= cmath.polar(Cmatrix)
    phase=math.degrees(phase)
    return coeff, phase
#######################################################################################
