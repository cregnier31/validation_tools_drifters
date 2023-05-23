#!/usr/bin/env python
#-*- coding: utf-8 -*-


# in command line load python and its environment:
# export PATH="/homelocal-px/px-137/svgennip/software/Miniconda_kara/bin:$PATH"
# source activate /homelocal-px/px-137/svgennip/software/Miniconda_kara/envs/env_py35_kara

import numpy as np
import pandas as pd
import numpy.matlib
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import netCDF4 as nc

import os, glob,sys
import sys, pprint
sys.displayhook = pprint.pprint  # Improves print of variables (e.g. locals(), globals(), dir())) to screen
import shutil
import shlex,subprocess
sys.path.append("/home/svangennip/PYTHON/Modules/")
import time
sys.path.append("/home/svangennip/PYTHON/Modules/LIB_SI/")
from NEMOchange import shift360
from tools_SI import StringRegion

from skimage.feature import peak_local_max
from skimage.measure import find_contours, points_in_poly, grid_points_in_poly
from configparser import ConfigParser

import getpass
username=getpass.getuser()
print('username: ',username)
import os, glob,sys
from operator import attrgetter

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
    drifters = pd.read_csv(f_drifters_all, sep='\s+') #when unknown number of spaces separating columns
    print(drifters.head(5))
    #select all data within time frame
    drifters_i = drifters[(drifters.date>=str_date1) & (drifters.date<=str_date2)]
    print(drifters_i.head(5))
    # print('save data on drifters within ', str_date1, ' and ', str_date1,':', outfile2)
    # drifters_i.to_csv(outfile2, index=False)

    #identify all drifters within target zone on starting date at 12:00:00
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
                                (drifters_zone_traj['date'] == str_date2) &  (drifters_zone_traj['time'] == '18:00:00')].index
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








#     print('loading ',f_drifters_all)
#     drifters = pd.read_csv(f_drifters_all, sep='\s+') #when unknown number of spaces separating columns
#     print(drifters.head(5))
#     #select all data within time frame
#     drifters_i = drifters[(drifters.date>=str_date1) & (drifters.date<=str_date2)]
#     print(drifters_i.head(5))
#     print('save data on drifters within ', str_date1, ' and ', str_date1,':', outfile)
#     drifters_i.to_csv(outfile, index=False)

#     #identify all drifters within target zone on starting date at 12:00:00
#     drifters_zone_at_start_all = drifters_i[(drifters_i.date==str_date1) & (drifters_i.time=='12:00:00') & \
#                               (drifters_i.lon>lon_min)     & (drifters_i.lon<=lon_max) &\
#                               (drifters_i.lat>lat_min)     & (drifters_i.lat<=lat_max)]
#     print(drifters_zone_at_start_all.head(3))
    
#     list_drifters_at_start=drifters_zone_at_start_all['id'].values
#     #print('drifters inside zone on',str_date1,': \n',list_drifters_at_start)
#     nb_drifters=len(list_drifters_at_start)
#     print('total nb of drifters in zone at start:',nb_drifters)
#     #Select all traj points from identified drifters
#     drifters_i.set_index("id", inplace=True)  #need to set index to being id
#     drifters_zone_traj_at_start = drifters_i.loc[list_drifters_at_start] #select data along id column
    
#     drifters_i=drifters_i.reset_index()  #reset index for both spreadsheets
#     drifters_zone_traj_at_start=drifters_zone_traj_at_start.reset_index()
    
#     #Now identify drifters that reach end of period:
#     drifters_date_at_end = drifters_zone_traj_at_start[(drifters_zone_traj_at_start['date'] == str_date2) &  \
#                                                        (drifters_zone_traj_at_start['time'] == '12:00:00')]
#     print('nb drifters at end of period:',len(drifters_date_at_end['id'].values))
#     #find id's of drifters that finish
#     list_drifters_at_end=drifters_date_at_end['id'].values
#     #print('list_drifters_end=\n',list_drifters_at_end)
    
#     #identify id's in both lists
#     list_drifters_full_period = match_elements(list_drifters_at_start, list_drifters_at_end)
#     #print("match elements: \n", list_drifters_full_period)
    
#     #Select all traj points from identified drifters
#     drifters_i.set_index("id", inplace=True)  #need to set index to being id
#     drifters_zone_traj = drifters_i.loc[list_drifters_full_period] #select data along id column
#     drifters_i=drifters_i.reset_index()
#     drifters_zone_traj=drifters_zone_traj.reset_index()
    
#     #Now remove data points before/after 12:00:00 for start/end date
#     index = drifters_zone_traj[(drifters_zone_traj['date'] == str_date1) & ((drifters_zone_traj['time'] == '00:00:00') | (drifters_zone_traj['time'] == '06:00:00')) | \
#                                 (drifters_zone_traj['date'] == str_date2) &  (drifters_zone_traj['time'] == '18:00:00')].index
#     drifters_zone_traj.drop(index, inplace=True) #remove data points
#     drifters_zone_traj=drifters_zone_traj.reset_index()
#     #print(drifters_zone_traj.head(30))
    
#     #Select start points of drifters that reach end
#     drifters_zone_at_start_all.set_index("id", inplace=True)  #need to set index to being id
#     drifters_zone_at_start = drifters_zone_at_start_all.loc[list_drifters_full_period] #select data along id column
#     drifters_zone_at_start=drifters_zone_at_start.reset_index()
#     list_drifters_at_start2=drifters_zone_at_start['id'].values
#     #print('drifters inside zone on',str_date1,': \n',list_drifters_at_start)
#     nb_drifters2=len(list_drifters_at_start2)
#     print('total nb of drifters in zone at start that reach end:',nb_drifters2)
#     print('\nlist drifters in area: \n', list_drifters_at_start2)
#     print('\n', drifters_zone_traj)
#     print('\n', drifters_zone_traj.shape)
    


  



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
def ReturnUVlist(MODEL, date_start, date_end, ):

    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      READ IN FIELDSET")
    print ("#-----------------------------------------------------------------------------------------")


    print('model :', MODEL)
    print('date_start :', date_start)
    print('date_end :', date_end)

    #reconstruct date1
    DD1=date_start.day
    MM1=date_start.month
    YYYY1=date_start.year 

    DD2=date_end.day
    MM2=date_end.month
    YYYY2=date_end.year

    config_object = ConfigParser()
    config_object.read("/home/svangennip/DEV/LAVD/src/config_products.ini")


    if MODEL not in config_object.sections():
        print(MODEL, "is either a typo or not yet tuned for this program")
        print('model runs available: ',config_object.sections())
        sys.exit(1)

    run_info = config_object[MODEL]
    path_UV = run_info["path"]
    fname_prefix=run_info["fname_prefix"]
    path_SSH = run_info["path_ssh"]
    fname_prefix_ssh=run_info["fname_prefix_ssh"]
    

    print('path_UV: ', path_UV)
    print('fname_prefix: ', fname_prefix)
    print('fname_prefix_ssh: ', fname_prefix_ssh)
    #sys.exit(1)

    #e.g.       /sortref/opr/pgn/PSY4/psy4v3r1/2009/07/PSY4V3R1_1dAV_20090703_20090704_gridU_R20090708.nc
    #e.g.     /sortref/reanalyse/glorys/phy/GLORYS12V1/RUN5free/PGN/1991/12/GLORYS12V1_1dAV_19911204_19911205_gridU_R19911211.nc
    # create list of files:
    print('identify U,V :')

    list_years = range (YYYY1, YYYY2+1, 1)
    print('years=', list_years)
    u_files,v_files, ssh_files= [],[],[]  # initialise arryas
    for i in list_years:
        print (i)
        #pathA = data_path2+'*/*/PSY4V3R1_1dAV_'+str(i)+'*gridU*.nc'
        #print(pathA)

        if MODEL == 'GLOBC':
            # e.g. /dataref/opr/DATA/OCEAN_CURRENTS/REP/L4/MULTIOBS_GLO_PHY_REP_015_004/dataset-uv-rep-daily/2019/dataset-uv-rep-daily_20191009T1200Z_P20200521T0000.nc
            # nrt_global_allsat_phy_l4_20181224_20181230.nc
            u_files += sorted(glob.glob(path_UV+'/*/' + fname_prefix+ str(i)+'*.nc'))
            v_files += sorted(glob.glob(path_UV+'/*/' + fname_prefix+ str(i)+'*.nc'))
            ssh_files += sorted(glob.glob(path_SSH+'/*/' + fname_prefix_ssh+ str(i)+'*.nc'))
            print(path_SSH+'/*/' + fname_prefix_ssh+ str(i)+'*.nc') 
            
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)

            #provide how ssh file starts to identify it in list
            f_input_start_ssh =fname_prefix_ssh + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end_ssh =fname_prefix_ssh + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start_ssh=', f_input_start_ssh)
            print('f_input_end_ssh=', f_input_end_ssh)
            #sys.exit(1)
        elif MODEL == 'ORCA12_RBB':
            print('got here')
            #e.g. /sortref/ose-2/btchonang/GLORYS12V1/GLO12OSSE/NatRun/ORCA12-TRBB36s003b_1d_grid_T_20141120-20141120.nc
            u_files += sorted(glob.glob(path_UV +'/'+ fname_prefix+'grid_U_'+str(i)+'*.nc'))
            v_files += sorted(glob.glob(path_UV+'/' + fname_prefix+ 'grid_V_'+str(i)+'*.nc'))
            ssh_files += sorted(glob.glob(path_SSH+'/' + fname_prefix_ssh+ 'grid_T_'+ str(i)+'*.nc'))
            
            
            #provide how file starts to identify it in list
            f_input_start = str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)

            #provide how ssh file starts to identify it in list
            f_input_start_ssh =fname_prefix_ssh + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end_ssh =fname_prefix_ssh + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start_ssh=', f_input_start_ssh)
            print('f_input_end_ssh=', f_input_end_ssh)
        else :
            # e.g. /sortref/opr/pgn/PSY4/psy4v3r1/2006/10/PSY4V3R1_1dAV_20061011_20061012_gridU_R20061018.nc
            u_files += sorted(glob.glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridU_*.nc'))
            v_files += sorted(glob.glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridV_*.nc'))
            ssh_files += sorted(glob.glob(path_SSH+'/*/*/' + fname_prefix_ssh+ str(i)+'*grid2D_*.nc'))
            
            
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)

            #provide how ssh file starts to identify it in list
            f_input_start_ssh =fname_prefix_ssh + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end_ssh =fname_prefix_ssh + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start_ssh=', f_input_start_ssh)
            print('f_input_end_ssh=', f_input_end_ssh)



    #ssh_files=ssh_files.sort() 

    
    #sys.exit(1)
    print('f_input_start',f_input_start)
    print('f_input_end',f_input_end)

    print('UV field, identify start date:')
    #print('Ulist = ', u_files)
    counter=0
    for line in u_files:
        if f_input_start in line:
           pos1 = counter 
           print ('    ',f_input_start, ' in line ', pos1,' :', line)        
        counter+=1

    #sys.exit(1)

    print('UV field, identify end date')
    counter=0
    for line in u_files:
        if f_input_end in line:
           pos2 = counter 
           print ('    ',f_input_end, ' in line ', pos2,' :', line)        
        counter+=1

    #sys.exit(1)
    # clear up paths to previous dayly outputs  from list:
    del(u_files[pos2+1:])
    del(u_files[:pos1])

    del(v_files[pos2+1:])
    del(v_files[:pos1])

    #Same for ssh
    print('ssh files, identify start date:')
    #print('ssh_list = ', ssh_files)
    counter=0
    for line in ssh_files:
        if f_input_start_ssh in line:
           pos1a = counter 
           print ('    ',f_input_start_ssh, ' in line ', pos1a,' :', line)        
        counter+=1

    #sys.exit(1)

    print('ssh_files, identify end date')
    counter=0
    for line in ssh_files:
        if f_input_end_ssh in line:
           pos2a = counter 
           print ('    ',f_input_end_ssh, ' in line ', pos2a,' :', line)        
        counter+=1
    

    del(ssh_files[pos2a+1:])
    del(ssh_files[:pos1a])
    

    print('nb files in ulist = ', len(u_files))
    print('nb files in vlist = ', len(v_files))
    print('nb files in ssh list = ', len(ssh_files))

    return u_files, v_files, ssh_files

def SeparationDistance(lon_grp1,lat_grp1, lon_grp2,lat_grp2):
    from numpy import pi
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

    sizey, sizex = np.shape(lon_grp1)
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
    print('\ndist[0,:]\n',dist[0,:])

    return dist

def SepDistanceEachTimestep(lon_grp1,lat_grp1,lon_grp2,lat_grp2):
    print('SepDistanceEachTimestep...')
    from numpy import pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    DEG2NM, NM2KM = 60., 1.8520
    # Angle conversions.
    deg2rad, rad2deg = pi / 180.0,  180.0 / pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    
    print('SVP:\n',np.shape(lon_grp1), np.shape(lat_grp1),'\nMODEL\n',np.shape(lon_grp2),np.shape(lat_grp2))
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
    print('dlon=', dlon.shape)
    print('dlat=', dlat.shape)
    #sys.exit(1)


    if np.any(np.abs(dlon) > 180):
        flag = abs(dlon) > 180
        dlon[flag] = -np.sign(dlon[flag]) * (360 - np.abs(dlon[flag]))

    sizey, sizex = np.shape(lon_grp1)
    mean_lat=(lat_grp2 + lat_grp1)/2.0
    print(mean_lat.shape)
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
    print('\nSVP_lon\n',lon_grp1[0,:])
    print('\nSVP_lat\n',lat_grp1[0,:])
    print('\nMOD_lon\n',lon_grp2[0,:])
    print('\nMOD_lat\n',lat_grp2[0,:])
    print('\ndist\n',dist[0,:])
    
    #sys.exit(1)
    return dist


def TrajDistBetweenTimesteps(lon_grp1,lat_grp1):
    print('TrajDistBetweenTimesteps...')
    from numpy import pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    DEG2NM, NM2KM = 60., 1.8520
    # Angle conversions.
    deg2rad, rad2deg = pi / 180.0,  180.0 / pi
    earth_radius = 6371000.
    # 1 nm = 1.8520 km
    
    print(np.shape(lon_grp1), np.shape(lat_grp1))#,np.shape(lon_grp2),np.shape(lat_grp2))
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
    print('dlon=', dlon.shape)
    print('dlat=', dlat.shape)
    #sys.exit(1)


    if np.any(np.abs(dlon) > 180):
        flag = abs(dlon) > 180
        dlon[flag] = -np.sign(dlon[flag]) * (360 - np.abs(dlon[flag]))

    sizey, sizex = np.shape(lon_grp1)
    mean_lat=(lat_grp1[:,:-1] + lat_grp1[:,1:])/2.0
    print(mean_lat.shape)
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
    print('\ndist\n',dist[0,:])
    #sys.exit(1)
    return dist
    
def LiuScore(traj_drifter_lon, traj_drifter_lat,traj_lon_mod, traj_lat_mod):
    print('compute Liu Score')
    print('drifter 6hourly: SVP', traj_drifter_lon.shape,', MODEL:', traj_lon_mod.shape)
    #Need to only take daily output:
    ind_daily=np.arange(0,traj_drifter_lon.shape[1],4)
    print('select positions:', ind_daily)
    
    traj_drifter_lon_daily=traj_drifter_lon[:,ind_daily]
    traj_drifter_lat_daily=traj_drifter_lat[:,ind_daily]
    traj_lon_mod_daily=traj_lon_mod[:,ind_daily]
    traj_lat_mod_daily=traj_lat_mod[:,ind_daily]
    print(traj_drifter_lon_daily.shape, traj_lon_mod_daily.shape)
    #1. Calculate SVP distance completed between time steps
    dli_drifters = TrajDistBetweenTimesteps(traj_drifter_lon_daily, traj_drifter_lat_daily)

    #2. Calculate separation distance between Model drifter and SVP at each time step:(ignore starting position)
    di_SVP_MOD =SepDistanceEachTimestep(traj_drifter_lon_daily[:,1:], traj_drifter_lat_daily[:,1:], traj_lon_mod_daily[:,1:], traj_lat_mod_daily[:,1:])
    print('dli_drifters.shape, di_SVP_MOD.shape\n', dli_drifters.shape, di_SVP_MOD.shape)
    #3. Compute Liu Score for 1, 3, 5 days
    
    #Compute S1 
    S1=di_SVP_MOD[:,0]/dli_drifters[:,0]
    
    #Compute S3=(d1+d2+d3)/(I01 + I02 + I03) 
    # with I01=dl1, I02=dl1+dl2, I03=dl1+dl2+dl3
    I01 = dli_drifters[:,0]
    I02 = np.sum(dli_drifters[:,:2], axis=1)
    I03 = np.sum(dli_drifters[:,:3], axis=1)
    print(I01.shape)
    print(I02.shape)
    print(I03.shape)
    num_S3   = np.sum(di_SVP_MOD[:,:3], axis=1)
    denom_S3 = I01+I02+I03  
    print(num_S3.shape, denom_S3.shape)
    S3=np.divide(num_S3, denom_S3)
    print(S3.shape)
    #Same for S5:
    I04 = np.sum(dli_drifters[:,:4], axis=1)
    I05 = np.sum(dli_drifters[:,:5], axis=1)
    num_S5 = np.sum(di_SVP_MOD[:,:5], axis=1)
    denom_S5   = I01+I02+I03+I04+I05
    S5=np.divide(num_S5,denom_S5)

    print('\nS1',S1.shape,S1,'\n\nS3', S3.shape, S3, '\n\nS5', S5.shape, S5)
    

    return S1, S3, S5
    #sys.exit(1)
    


def CalcCurlGlobC (u_files, zlevel, dir_out):
    from metpy.units import units
    from scipy.ndimage import gaussian_filter
    from siphon.catalog import TDSCatalog
    from siphon.ncss import NCSS
    import metpy.calc as mpcalc
    print('Calculate Curl GlobCurrent')
    print('for level:', zlevel, ' (1=surface,  2=15m depth)')

    dir_out2= dir_out+'/CURL/GLOBC/'
    if not os.path.exists(dir_out2):
        print('\ncreate temp folder : ', dir_out2)
        # Create directory where to store results:
        os.makedirs(dir_out2)    


    #load list of Globcurrent file:
    #nc_file = '/dataref/opr/DATA/OCEAN_CURRENTS/REP/L4/MULTIOBS_GLO_PHY_REP_015_004/dataset-uv-rep-daily/2019/dataset-uv-rep-daily_20190318T1200Z_P20200520T0000.nc'
    
    curl_files = []
    for line in u_files:  #loop through each uv file
        
        nc_file = line.strip()  #need strip to remove '\n' newline
        print(nc_file)
        # Create outfile name:
        fuv_name = os.path.basename(nc_file)  
        
        fcurl_name = fuv_name.replace("uv", "curl")
        #print(fname_vor)
        nc_out=dir_out2 + fcurl_name
        curl_files.append(nc_out)
        if not os.path.exists(nc_out):
            
            #read in UV file
            fh = Dataset(nc_file, mode='r')
            lat = fh.variables['latitude'][:]
            lon = fh.variables['longitude'][:]
            t = fh.variables['time']#[:]
            t_val=t[:]
            t_units = t.units
            t_calendar = t.calendar 


            U = fh.variables['uo'][0]
            V = fh.variables['vo'][0]
            fh.close()

            #u = units.Quantity(np.ma.array(U, mask=U[U.mask]), units('m/s'))
            u = units.Quantity(np.ma.array(U), units('m/s'))
            v = units.Quantity(np.ma.array(V), units('m/s'))

            #select surface values
            u_surf = u[zlevel-1,:,:]
            v_surf = v[zlevel-1,:,:]
            #u_15 = u[1,:,:]
            #v_15 = v[1,:,:]

            
            # Calcualte dx and dy for calculations
            dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)


            #print('calculate Relative Vorticity...')
            # Planetary Vorticity
            f = mpcalc.coriolis_parameter(np.deg2rad(lat)).to('1/s')
            f1 = np.expand_dims(f, axis=1)

            # Relative Vorticity
            print("u_surf",u_surf.shape,"\nv_surf", v_surf.shape,'\ndx, dy', dx.shape, dy.shape )
            vor = mpcalc.vorticity(u_surf, v_surf, dx = dx, dy = dy) #, dim_order='yx')
            
            # Abosolute Vorticity
            avor = vor + f1

            print('... save as ',nc_out)
            foo = Dataset(nc_out, 'w')
            # Global Attributes
            foo.title = 'Vorticity GlobCurrent '
            foo.institution = 'Mercator-Ocean International'
            #foo.history = 'Created ' + time.ctime(time.time())

            # Create dimensions
            foo.createDimension('time', len(t_val))
            foo.createDimension('y', len(lat))
            foo.createDimension('x', len(lon))

            # Create variables
            nc_times = foo.createVariable('time', float, ('time',), zlib=True)
            nc_latitudes = foo.createVariable('latitude', float, ('y'), zlib=True)
            nc_longitudes = foo.createVariable('longitude', float, ( 'x'), zlib=True)
            nc_vor = foo.createVariable('vor', float, ('y', 'x'), zlib=True, fill_value=0)  #, least_significant_digit=12,
            nc_avor = foo.createVariable('avor', float, ('y', 'x'), zlib=True, fill_value=0)  #, least_significant_digit=12,

            # Variable attributes
            nc_latitudes.units = 'degree_north'
            nc_longitudes.units = 'degree_east'
            nc_vor.units = '1/s'
            nc_avor.units = '1/s'
            nc_times.units = t_units
            nc_times.calendar = t_calendar

            # Write to nc_Variable
            nc_times[:] = t_val
            nc_latitudes[:] = lat
            nc_longitudes[:] = lon

            nc_vor[:]=vor.magnitude
            nc_avor[:]=avor.magnitude

            foo.close()
        

        else:
            print('...',nc_out, " already exists")
                

    print('done')
    return curl_files


def CalcRelVorticity(model,u_files, var_u,v_files, var_v, zlevel, overf, pos_grid, dir_out) :
    print("Fonction CalcRelVorticity")

    dir_work='/data/rd_exchange/svangennip/rd2/svangennip/TOOLS/NEMO/RelVort/'+model+'/'
    if not os.path.exists(dir_work):
        print('\n script cannot compute curl for : ', model)
        sys.exit(1)
    else:
        os.chdir(dir_work) 
        print(os.getcwd())
           

    dir_out2= dir_out+'/CURL/'+model+'/'
    if not os.path.exists(dir_out2):
        print('\ncreate temp folder : ', dir_out2)
        # Create directory where to store results:
        os.makedirs(dir_out2)          
    
    BIN='/home/cbricaud/OUTILS/CDFTOOLS/bin/'
    #PSY4V3R1_1dAV_20100101_20100102_gridU_R20100106.nc
    
    curl_files = []
    for uline, vline in zip(u_files, v_files):  #loop through each uv file
        

        u_file = uline.strip()  #need strip to remove '\n' newline
        v_file = vline.strip()  #need strip to remove '\n' newline
        
        print(u_file)
        print(v_file)
        # Create outfile name:
        fu_name = os.path.basename(u_file)  
        
        fcurl_name = fu_name.replace("gridU", "curl")
        nc_out=dir_out2 + fcurl_name
        
        #print('--->',nc_out)
        curl_files.append(nc_out)
        
        #'''
        #Recupere la namelist
        if not os.path.exists(nc_out):
            #e.g. '/home/cbricaud/OUTILS/CDFTOOLS/bin/cdfcurl -u $fu $var_u -v $fv $var_v -l "12" -surf -overf -o fout 
            if pos_grid == 'F':
                print('Calculate on F grid')
                if overf==1:
                    cmd_exe = '%s/cdfcurl -u %s %s -v %s %s -l "%s" -surf -overf -o %s' %(BIN, u_file, var_u, v_file, var_v,zlevel, nc_out)
                else:
                    cmd_exe = '%s/cdfcurl -u %s %s -v %s %s -l "%s" -surf -o %s' %(BIN, u_file, var_u, v_file, var_v,zlevel, nc_out)
          
            elif pos_grid == 'T':
                print('Calculate on T grid')
                if overf==1:
                    cmd_exe = '%s/cdfcurl -u %s %s -v %s %s -l "%s" -T -surf -overf -o %s' %(BIN, u_file, var_u, v_file, var_v,zlevel, nc_out)
                else:
                    cmd_exe = '%s/cdfcurl -u %s %s -v %s %s -l "%s" -T -surf -o %s' %(BIN, u_file, var_u, v_file, var_v,zlevel, nc_out)

            print(cmd_exe)   
            os.system(cmd_exe)
            print('output=', nc_out,'\n')
        else:
            print('...',nc_out, " already exists")        
            #'''
    return curl_files  

def ExtractSurface(model,u_files, v_files,z_level, dir_out) :
    print("Fonction CalcRelVorticity")

           

    dir_out2= dir_out+'/UV/'+model+'/'
    if not os.path.exists(dir_out2):
        print('\ncreate temp folder : ', dir_out2)
        # Create directory where to store results:
        os.makedirs(dir_out2)          
    
    
    u_files_surf, v_files_surf  = [], []
    for uline, vline in zip(u_files, v_files):  #loop through each uv file
        

        u_file = uline.strip()  #need strip to remove '\n' newline
        v_file = vline.strip()  #need strip to remove '\n' newline
        
        print(u_file)
        print(v_file)
        # Create outfile name:
        fu_name = os.path.basename(u_file)  
        fv_name = os.path.basename(v_file)  
        
        u_file_surf=dir_out2 + fu_name
        v_file_surf=dir_out2 + fv_name
        
        #print('--->',nc_out)
        u_files_surf.append(u_file_surf)
        v_files_surf.append(v_file_surf)
        
        
        #'''
        print('extract u_surf')
        depth='deptht'
        if not os.path.exists(u_file_surf):
            # e.g. ncwa -a deptht -d deptht,10 infile.nc file-depth10.nc 
            # Example to select only 10th level of depth dim (resulting file won't keep this dimension):
            #  If no -a option is precised, ncwa is used to average data. By default all dimensions will be averaged to give only a float value.
            
            cmd_exe1 = 'ncwa -F -a %s -d %s,%s %s %s ' %(depth,depth,z_level, u_file, u_file_surf)
            print(cmd_exe1)   
            os.system(cmd_exe1)
        else:
            print('...',u_file_surf, " already exists")        
            #'''
        #sys.exit(1)
        print('extract v_surf')
        if not os.path.exists(v_file_surf):
            cmd_exe2 = 'ncwa -F -a %s -d %s,%s %s %s ' %(depth,depth,z_level, v_file, v_file_surf)
            print(cmd_exe2)   
            os.system(cmd_exe2)
        else:
            print('...',v_file_surf, " already exists")        
            #'''



    return u_files_surf, v_files_surf  


