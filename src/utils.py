#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from numpy import pi
import pandas as pd
import numpy.matlib
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import matplotlib
#matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import time
import netCDF4 as nc
from tqdm import tqdm
import os, glob,sys
import sys, pprint
sys.displayhook = pprint.pprint  # Improves print of variables (e.g. locals(), globals(), dir())) to screen
import shutil
import shlex,subprocess
import time
from configparser import ConfigParser
import xarray as xr
from scipy import interpolate
import getpass
username=getpass.getuser()
print('username: ',username)
import os, glob,sys
from operator import attrgetter
from scipy.ndimage import gaussian_filter
import yaml
#sys.path.append("/home/svangennip/DEV/Drift/LIU_SCORE/src/")
from extras import dist as sw_dist
import yaml
import datetime as dt
from Logger import Logger

def build_runtime_config(args, config_path):
    """
    Build full runtime configuration for Lagrangian workflow.
    """

    # ------------------------------------------
    # 1. Parse CLI args
    # ------------------------------------------
    DATE_START, DATE_END, leadtime, SYSTEM = args
    # ------------------------------------------
    # 2. Load YAML
    # ------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config_info = config["config_base"]
    config_model_params = config['model_params']

    # Add parcel config
    config_parcels = config['parcels']
    # 1. defaults
    scheme = config_info["advection_diffusion_type"]
    base = config_parcels["defaults"].get(scheme)
    if base is None:
        raise ValueError(f"Unknown scheme: {scheme}")
    parcels_dt = base["dt_minutes"]
    sampling = base["sampling_hours"]

    # 2. overrides (if exists)
    overrides = config_parcels.get("overrides", {})
    if SYSTEM in overrides:
        if scheme in overrides[SYSTEM]:
            override = overrides[SYSTEM][scheme]
            parcels_dt = override.get("dt_minutes", parcels_dt)
            sampling = override.get("sampling_hours", sampling)
    # store in config
    config_info["parcels_dt_minutes"] = parcels_dt
    config_info["parcels_sampling_hours"] = sampling
    config_info["scheme"] = scheme
    length_time =int(config_info["drift_time"])
    # ------------------------------------------
    # 3. Inject runtime args
    # ------------------------------------------
    config_info["SYSTEM"] = SYSTEM
    config_info["leadtime"] = leadtime
    config_info["gridtype"] = config_model_params[f'{SYSTEM}_{leadtime}']['grid_type']
    config_info["lon"] = config_model_params[f'{SYSTEM}_{leadtime}']['lon']
    config_info["lat"] = config_model_params[f'{SYSTEM}_{leadtime}']['lat']
    config_info["time"] = config_model_params[f'{SYSTEM}_{leadtime}']['time']
    # ------------------------------------------
    # 4. Dates
    # ------------------------------------------
    STR_DATE_START = DATE_START.strftime("%Y-%m-%d")
    STR_DATE_END = DATE_END.strftime("%Y-%m-%d")
    config_info["DATE_START"] = STR_DATE_START
    config_info["DATE_END"] = STR_DATE_END

    config_info["PERIOD"] = f"{STR_DATE_START} - {STR_DATE_END}"

    start_days = pd.date_range(STR_DATE_START, STR_DATE_END, freq="D")

    DATE_END_PLUS_DRIFT_TIME = DATE_END + dt.timedelta(
        days=int(config_info["drift_time"]) + 1
    )

    # ------------------------------------------
    # 5. Region
    # ------------------------------------------
    config_info["STR_REGION"] = StringRegion(
        config_info["lon_min"],
        config_info["lon_max"],
        config_info["lat_min"],
        config_info["lat_max"],
    )

    # ------------------------------------------
    # 6. Experiment name
    # ------------------------------------------
    EXP_NAME = (
        f"Parcels_Drift-{SYSTEM}_{leadtime}_{scheme}_drift{length_time}_dt{parcels_dt}_"
        f"{DATE_START.strftime('%Y%m%d')}-"
        f"{DATE_END.strftime('%Y%m%d')}"
    )

    config_info["EXP_NAME"] = EXP_NAME

    # ------------------------------------------
    # 7. Output directory
    # ------------------------------------------
    dir_out_data = config_info['dir_data'] + scheme + "/"
    dir_out_imgs = config_info['dir_imgs'] + scheme + "/"
    if not os.path.exists(dir_out_imgs):
        print(f'\ncreate folder : {dir_out_imgs}')
        # Create directory where to store images:
        os.makedirs(dir_out_imgs)
    if not os.path.exists(dir_out_data):
        print(f'\ncreate folder : {dir_out_data}')
        # Create directory where to store results:
        os.makedirs(dir_out_data)

    DIR_WORK = config_info["dir_work"] + scheme + "/"
    PROJECT = config_info["project"]

    config_info["DIR_OUT"] = (
        f"{DIR_WORK}DRIFT_{SYSTEM}/"
        f"{EXP_NAME}/ZONE_{config_info['STR_REGION']}/"
    )

    # ------------------------------------------
    # 8. Output file
    # ------------------------------------------
    pickle_file = os.path.join(
        config_info["DIR_OUT"],
        f"{EXP_NAME}_drift-observations.pkl"
    )
    outfile = (
                f"{dir_out_data}Lagrangian_scores_"
                f"{leadtime}_{SYSTEM}_"
                f"{STR_DATE_START}_"
                f"{STR_DATE_END}.nc"
              )

    # ------------------------------------------
    # 8bis. Logger (NOW INCLUDED ✔)
    # ------------------------------------------
    log = Logger(
        f"Lagrangian_tool {STR_DATE_START} - {STR_DATE_END} {SYSTEM}"
    ).run(int(config_info["loglevel"]))

    # ------------------------------------------
    # 9. Return full runtime object
    # ------------------------------------------
    return {
        "config_info": config_info,
        "start_days": start_days,
        "config_model_params": config_model_params,
        "pickle_file": pickle_file,
        "stats_out": outfile,
        "dir_out_imgs": dir_out_imgs,
        "log": log,
    }

def convert_id_to_char_array(ids, string_length=8):
    import numpy as np

    ids_str = np.asarray(ids).astype(str)

    max_len = max(len(s) for s in ids_str)
    if max_len > string_length:
        raise ValueError(
            f"ID length ({max_len}) exceeds string_length ({string_length})"
        )
    # pad with spaces (NetCDF style)
    ids_padded = np.char.ljust(ids_str, string_length)

    # convert to char array
    #char_array = np.array([list(s) for s in ids_str], dtype='S1')
    # Convert to char array (S1)
    char_array = np.array(
        [np.frombuffer(s.encode('utf-8'), dtype='S1') for s in ids_padded]
    )
    return char_array_to_string_list(char_array)

def char_array_to_string_list(char_array):
    return [
        b''.join(row).decode('utf-8').strip()
        for row in char_array
    ]


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
    print(t_start)
    print(t_end)
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



def DriftersInZoneInPeriod_remote(lon_min, lon_max, lat_min, lat_max, date_start, date_end, dico_name, outfile):
    import datetime 
 
    date_format = '%Y-%m-%dT%H:%M:%SZ'

    str_date1=date_start.strftime('%Y-%m-%d')
    str_date2=date_end.strftime('%Y-%m-%d')
    print("date1 : ",str_date1)
    str_region = StringRegion(lon_min,lon_max,lat_min,lat_max)
    date_start = datetime.datetime.strptime(str_date1+"T12:00:00Z", date_format)
    date_end = datetime.datetime.strptime(str_date2+"T12:00:00Z", date_format)

    T_DRIFT = (date_end-date_start).days

    #Check output dir exists:
    dir_out = os.path.dirname(outfile)
    if not os.path.exists(dir_out):
        print('\ncreate folder : ', dir_out)
        # Create directory where to store results:
        os.makedirs(dir_out)    
    
    

    f = open(dico_name,)
    metadata_drifters = json.load(f)
    print('loading ',dico_name)
    
    list_drifter_in_time_zone = []
    for dft_id in tqdm(metadata_drifters):
        t_start = datetime.datetime.strptime(metadata_drifters[dft_id][0], date_format)
        t_end = datetime.datetime.strptime(metadata_drifters[dft_id][1], date_format)
        if date_start >= t_start and date_end <= t_end :
            list_drifter_in_time_zone.append(dft_id)
    
    print('nb drifter in the time period : ',len(list_drifter_in_time_zone))
    df = pd.DataFrame()
    k = 0
    nb = 0
    diff_nb_hours = int((date_end-date_start).days*24)
    for dft_file in tqdm(list_drifter_in_time_zone):
        data = xr.open_dataset(dft_file)
        ind_start = np.where(data.TIME == np.datetime64(date_start))[0]
        ind_end = np.where(data.TIME == np.datetime64(date_end))[0]
        if (ind_end-ind_start) == diff_nb_hours : 
            if ind_start.shape[0] == 1 and ind_end.shape[0] == 1:
                lon_start = data.LONGITUDE[ind_start]
                lon_end = data.LONGITUDE[ind_end]
                lat_start = data.LATITUDE[ind_start]
                lat_end = data.LATITUDE[ind_end] 
                if (lon_min <= lon_start)*(lon_max >= lon_start)*(lat_min <= lat_start)*(lat_max >= lat_start):
                    if (lon_min <= lon_end)*(lon_max >= lon_end)*(lat_min <= lat_end)*(lat_max >= lat_end):
                        id = int(data.platform_code)
                        date = pd.to_datetime(data.TIME.values[ind_start[0]:ind_end[0]]).date
                        time = pd.to_datetime(data.TIME.values[ind_start[0]:ind_end[0]]).time
                        latitude = data.LATITUDE.values[ind_start[0]:ind_end[0]]
                        longitude = data.LONGITUDE.values[ind_start[0]:ind_end[0]]
                        dic = {'id': id,'date':date,'time':time,'lat':latitude,'lon':longitude}
                        if k ==0:
                            df = pd.DataFrame(dic,index = nb*np.ones(longitude.shape))
                            k=1
                        else : 
                            df = pd.concat([df,pd.DataFrame(dic,index = nb*np.ones(longitude.shape))])
                        nb+=1
    df.to_csv(outfile)

        
def DriftersInZoneInPeriod(lon_min, lon_max, lat_min, lat_max, date_start, date_end, f_drifters_all, outfile):
    str_date1=date_start.strftime('%Y-%m-%d')
    str_date2=date_end.strftime('%Y-%m-%d')
    str_region = StringRegion(lon_min,lon_max,lat_min,lat_max)
    
    T_DRIFT = (date_end-date_start).days

    #Check output dir exists:
    dir_out = os.path.dirname(outfile)
    if not os.path.exists(dir_out):
        print('\ncreate folder : ', dir_out)
        # Create directory where to store results:
        os.makedirs(dir_out)    
    
    print('loading ',f_drifters_all)
    #drifters = pd.read_csv(f_drifters_all, sep='\s+') #when unknown number of spaces separating columns
    drifters = pd.read_csv(f_drifters_all) #when unknown number of spaces separating columns
    print(drifters.head(5))
    #select all data within time frame
    drifters_i = drifters[(drifters.date>=str_date1) & (drifters.date<=str_date2)]
    print(drifters_i.head(5))
    # print('save data on drifters within ', str_date1, ' and ', str_date1,':', outfile2)
    # drifters_i.to_csv(outfile2, index=False)
    if lon_max < lon_min :
        drifters_zone_at_start_all1 = drifters_i[(drifters_i.date==str_date1) & (drifters_i.time=='12:00:00') & \
                              (drifters_i.lon>lon_min)     & (drifters_i.lon<=180) &\
                              (drifters_i.lat>lat_min)     & (drifters_i.lat<=lat_max)]
        drifters_zone_at_start_all2 = drifters_i[(drifters_i.date==str_date1) & (drifters_i.time=='12:00:00') & \
                              (drifters_i.lon>=-180)     & (drifters_i.lon<=lon_max) &\
                              (drifters_i.lat>lat_min)     & (drifters_i.lat<=lat_max)]
        print('drifters_zone_at_start_all2 : \n',drifters_zone_at_start_all2.head(3))
        drifters_zone_at_start_all = pd.concat([drifters_zone_at_start_all1,drifters_zone_at_start_all2])
        print('drifters_zone_at_start_all : \n', drifters_zone_at_start_all.head(3))

    #identify all drifters within target zone on starting date at 12:00:00
    else : 
        drifters_zone_at_start_all = drifters_i[(drifters_i.date==str_date1) & (drifters_i.time=='12:00:00') & \
                              (drifters_i.lon>lon_min)     & (drifters_i.lon<=lon_max) &\
                              (drifters_i.lat>lat_min)     & (drifters_i.lat<=lat_max)]
        print(drifters_zone_at_start_all.head(3))

    list_drifters_at_start=drifters_zone_at_start_all['id'].values
    #print('drifters inside zone on',str_date1,': \n',list_drifters_at_start)
    nb_drifters=len(list_drifters_at_start)
    print('total nb of drifters in zone at start:',nb_drifters)
    #Select all traj points from identified drifters
    drifters_i.set_index("id", inplace=True)  #need to set index to being id
    drifters_zone_traj_at_start = drifters_i.loc[list_drifters_at_start] #select data along id column

    drifters_i=drifters_i.reset_index()  #reset index for both spreadsheets
    drifters_zone_traj_at_start=drifters_zone_traj_at_start.reset_index()

    #Now identify drifters that reach end of period:
    drifters_date_at_end = drifters_zone_traj_at_start[(drifters_zone_traj_at_start['date'] == str_date2) &  \
                                                       (drifters_zone_traj_at_start['time'] == '12:00:00')]
    print('nb drifters at end of period:',len(drifters_date_at_end['id'].values))
    #find id's of drifters that finish
    list_drifters_at_end=drifters_date_at_end['id'].values
    #print('list_drifters_end=\n',list_drifters_at_end)

    #identify id's in both lists
    list_drifters_full_period = match_elements(list_drifters_at_start, list_drifters_at_end)
    #print("match elements: \n", list_drifters_full_period)

    #Select all traj points from identified drifters
    drifters_i.set_index("id", inplace=True)  #need to set index to being id
    drifters_zone_traj = drifters_i.loc[list_drifters_full_period] #select data along id column
    drifters_i=drifters_i.reset_index()
    drifters_zone_traj=drifters_zone_traj.reset_index()

    #Now remove data points before/after 12:00:00 for start/end date
    index = drifters_zone_traj[(drifters_zone_traj['date'] == str_date1) & ((drifters_zone_traj['time'] == '00:00:00') | (drifters_zone_traj['time'] == '06:00:00')) | \
                                  (drifters_zone_traj['time'] == '18:00:00') & (drifters_zone_traj['date'] == str_date2) ].index
    drifters_zone_traj.drop(index, inplace=True) #remove data points
    drifters_zone_traj=drifters_zone_traj.reset_index()
    #print(drifters_zone_traj.head(30))

    #Select start points of drifters that reach end
    drifters_zone_at_start_all.set_index("id", inplace=True)  #need to set index to being id
    drifters_zone_at_start = drifters_zone_at_start_all.loc[list_drifters_full_period] #select data along id column
    drifters_zone_at_start=drifters_zone_at_start.reset_index()
    list_drifters_at_start2=drifters_zone_at_start['id'].values
    #print('drifters inside zone on',str_date1,': \n',list_drifters_at_start)

    print('\nlist drifters in area: \n', list_drifters_at_start2)
    print('\n', drifters_zone_traj)
    print('\n', drifters_zone_traj.shape)

    #print('save data on drifters within [ ', str_date1, ' - ', str_date1,'] and for region ', str_region,' :', outfile2)
    drifters_zone_traj.to_csv(outfile, index=False)


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


## return U, V files within date range:
#def ReturnUVlist(MODEL, date_start, date_end, dir_in=None, leadtime=None):
#
#    print ("#-----------------------------------------------------------------------------------------")
#    print ("#                      READ IN FIELDSET")
#    print ("#-----------------------------------------------------------------------------------------")
#
#    print('script ReturnUVlist')
#    print('model :', MODEL)
#    print('date_start :', date_start)
#    print('date_end :', date_end)
#    print('dir_in :', dir_in)
#    #reconstruct date1
#    DD1=date_start.day
#    MM1=date_start.month
#    YYYY1=date_start.year 
#
#    DD2=date_end.day
#    MM2=date_end.month
#    YYYY2=date_end.year
#
#    config_object = ConfigParser()
#    config_object.read(dir_in+"src/config_products.ini")
#    #config_object.read("/config_products.ini")
#
#
#    if MODEL not in config_object.sections():
#        print(MODEL, "is either a typo or not yet tuned for this program")
#        print('model runs available: ',config_object.sections())
#        sys.exit(1)
#
#    run_info = config_object[MODEL]
#    fname_prefix=run_info["fname_prefix"]
#    #path to files 
#    path_UV=run_info["path"]
#
#    print('path_UV: ', path_UV)
#    print('fname_prefix: ', fname_prefix)
#    #sys.exit(1)
#    
#    #e.g.       /sortref/opr/pgn/PSY4/psy4v3r1/2009/07/PSY4V3R1_1dAV_20090703_20090704_gridU_R20090708.nc
#    #e.g.     /sortref/reanalyse/glorys/phy/GLORYS12V1/RUN5free/PGN/1991/12/GLORYS12V1_1dAV_19911204_19911205_gridU_R19911211.nc
#    # create list of files:
#    print('identify U,V :')
#
#    list_years = range (YYYY1, YYYY2+1, 1)
#    print('years=', list_years)
#    
#    u_files,v_files, ssh_files= [],[],[]  # initialise arryas
#    for i in list_years:
#        print (i)
#        if MODEL == 'GLOBC':
#            # e.g. /dataref/opr/DATA/OCEAN_CURRENTS/REP/L4/MULTIOBS_GLO_PHY_REP_015_004/dataset-uv-rep-daily/2019/dataset-uv-rep-daily_20191009T1200Z_P20200521T0000.nc
#            # nrt_global_allsat_phy_l4_20181224_20181230.nc
#            u_files += sorted(glob.glob(path_UV+'/*/' + fname_prefix+ str(i)+'*.nc'))
#            v_files += sorted(glob.glob(path_UV+'/*/' + fname_prefix+ str(i)+'*.nc'))
#            
#            #provide how file starts to identify it in list
#            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
#            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
#            print('f_input_start=', f_input_start)
#            print('f_input_end=', f_input_end)
#
#        elif MODEL == 'ORCA12_RBB':
#            print('got here')
#            #e.g. /sortref/ose-2/btchonang/GLORYS12V1/GLO12OSSE/NatRun/ORCA12-TRBB36s003b_1d_grid_T_20141120-20141120.nc
#            u_files += sorted(glob.glob(path_UV +'/'+ fname_prefix+'grid_U_'+str(i)+'*.nc'))
#            v_files += sorted(glob.glob(path_UV+'/' + fname_prefix+ 'grid_V_'+str(i)+'*.nc'))
#            
#            #provide how file starts to identify it in list
#            f_input_start = str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
#            f_input_end =str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
#            print('f_input_start=', f_input_start)
#            print('f_input_end=', f_input_end)
#
#        elif MODEL == 'GLO12V4':
#            # GLO12: /sortref/opr/diag/GLO12/GLO12V4/data/Extract_UV/GLO12_ng_1d-m_20200101-20200101_3DU-uo_rtr_R20200108.nc
#            u_files += sorted(glob.glob(path_UV +'/'+ fname_prefix+str(i)+'*3DU-uo_*.nc'))
#            v_files += sorted(glob.glob(path_UV +'/'+ fname_prefix+str(i)+'*3DV-vo_*.nc'))
#            
#            #provide how file starts to identify it in list
#            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
#            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
#            print('f_input_start=', f_input_start)
#            print('f_input_end=', f_input_end)
#        elif MODEL == 'IBI36':
#            print(path_UV+fname_prefix+'*gridU_R*.nc')
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*gridU_R*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*gridV_R*.nc'))
#            f_input_start = fname_prefix + str(YYYY1)+format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') + format(DD2,'02d')
#
#        elif MODEL == 'SMOC' or MODEL == 'SMOC_GLO12':
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) +format(MM2,'02d') + format(DD2,'02d')
#
#        elif MODEL == 'IBI8':
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*_interp_resol8.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*_interp_resol8.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1)+format(MM1,'02d')+ format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2)+format(MM2,'02d')+ format(DD2,'02d')
#
#        elif MODEL == 'GLO8':
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#
#        elif MODEL == 'IBI4' or MODEL == 'IBI12':
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridU*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridV*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#
#        elif MODEL == "PSY4":
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridU*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridV*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#        
#        elif "GLO12_FCST" in MODEL:
#            ## Recode this
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+'fcst*/'+fname_prefix+'*3DU*.nc'))
#            v_files += sorted(glob.glob(path_UV+'fcst*/'+fname_prefix+'*3DV*.nc'))
#            print(u_files)
#            sys.exit(1)
#            #f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            #f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#            f_input_start = 'fcst-1_R' + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d') + '.nc'
#            f_input_end = 'fcst-1_R' + str(YYYY2) + format(MM2,'02d') + format(DD2,'02d') + '.nc'
#            
#        elif "GLO12_NWCT" in MODEL or "GLO12_HDCT" in MODEL:
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*3DU*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*3DV*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#
#        elif "PSY4_FCST" in MODEL:
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridU*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridV*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#
#        elif "PSY4_NWCT" in MODEL:
#            print(path_UV+fname_prefix)
#            u_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridU*.nc'))
#            v_files += sorted(glob.glob(path_UV+fname_prefix+'*_gridV*.nc'))
#            print(u_files)
#            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
#            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
#
#        else :
#            print('Inside else')
#            # e.g. /sortref/opr/pgn/PSY4/psy4v3r1/2006/10/PSY4V3R1_1dAV_20061011_20061012_gridU_R20061018.nc
#            u_files += sorted(glob.glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridU_*.nc'))
#            v_files += sorted(glob.glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridV_*.nc'))
#            
#            
#            #provide how file starts to identify it in list
#            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
#            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
#            print('f_input_start=', f_input_start)
#            print('f_input_end=', f_input_end)
#
#
#    #ssh_files=ssh_files.sort() 
#
#
#    #sys.exit(1)
#    print('f_input_start',f_input_start)
#    print('f_input_end',f_input_end)
#
#    print('UV field, identify start date:')
#    print('nb files in Ulist = ', len(u_files))
#    counter=0
#    for line in u_files:
#        if f_input_start in line:
#           pos1u = counter 
#           print ('    ',f_input_start, ' in line ', pos1u,' :', line)        
#        counter+=1
#
#    #sys.exit(1)
#
#    print('UV field, identify end date')
#    counter=0
#    for line in u_files:
#        print(line)
#        print(f_input_end)
#        if f_input_end in line:
#           pos2u = counter 
#           print ('    ',f_input_end, ' in line ', pos2u,' :', line)        
#        counter+=1
#
#    counter=0
#    for line in v_files:
#        if f_input_start in line:
#           pos1v = counter
#           print ('    ',f_input_start, ' in line ', pos1v,' :', line)
#        counter+=1
#
#    #sys.exit(1)
#
#    print('UV field, identify end date')
#    counter=0
#    for line in v_files:
#        if f_input_end in line:
#           pos2v = counter 
#           print ('    ',f_input_end, ' in line ', pos2v,' :', line) 
#        counter+=1
#
#
#    #sys.exit(1)
#    # clear up paths to previous dayly outputs  from list:
#    if 'pos2u' in locals():
#        del(u_files[pos2u+1:])
#    if 'pos1u' in locals():
#        del(u_files[:pos1u])
#
#    if 'pos2v' in locals():
#        del(v_files[pos2v+1:])
#    if 'pos1v' in locals():
#        del(v_files[:pos1v])
#
#
#    print('nb files in ulist = ', len(u_files))
#    print('nb files in vlist = ', len(v_files))
#    print('nb files in ssh list = ', len(ssh_files))
#
#    return u_files, v_files

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


def SepDistanceEachTimestep(lon_grp1,lat_grp1,lon_grp2,lat_grp2):
    print('SepDistanceEachTimestep...')
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    DEG2NM, NM2KM = 60., 1.8520
    # Angle conversions.
    deg2rad, rad2deg = pi / 180.0,  180.0 / pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
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
    return dist


def TrajDistBetweenTimesteps(lon_grp1,lat_grp1):
    print('TrajDistBetweenTimesteps...')
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
 
def LiuScore(traj_drifter_lon, traj_drifter_lat,traj_lon_mod, traj_lat_mod):
    print('compute Liu Score')
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
    di_SVP_MOD =SepDistanceEachTimestep(traj_drifter_lon_daily[:,1:], traj_drifter_lat_daily[:,1:], traj_lon_mod_daily[:,1:], traj_lat_mod_daily[:,1:])
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
    
def LiuScore2(traj_drifter_lon, traj_drifter_lat,traj_lon_mod, traj_lat_mod):

    dli_drifters = TrajDistBetweenTimesteps(traj_drifter_lon, traj_drifter_lat)

    #2. Calculate separation distance between Model drifter and SVP at each time step:(ignore starting position)
    di_SVP_MOD =SepDistanceEachTimestep(traj_drifter_lon[:,1:], traj_drifter_lat[:,1:], traj_lon_mod[:,1:], traj_lat_mod[:,1:])
    liu_index = []
    den = np.zeros(di_SVP_MOD.shape[0])
    for i in range(di_SVP_MOD.shape[1]):
        den += np.sum(dli_drifters[:,:i], axis=1)
        liu_index.append(np.sum(di_SVP_MOD[:,:i], axis=1)/den)   
    return np.array(liu_index)





def ExtractSurface(model,u_files, v_files,zlevel,var_u, var_v, dir_out) :
    print("Fonction ExtractLayer")

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


