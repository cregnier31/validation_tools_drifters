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
import netCDF4
import scipy.io
import drifter_object
from os import listdir
from os.path import isfile, join


#-----------------------------------------------------------------------------------
def daterange_deltah(datedeb, datefin, delta_h):
    # nombre de pas d'heure à faire
    #hfin=int((datefin - datedeb).total_seconds()/3600)
    hfin=int( ((datefin - datedeb).days*86400+(datefin - datedeb).seconds)/3600)
    return (datedeb + datetime.timedelta(hours=h) for h in range(0,hfin,delta_h))
#------------------------------------------------------------------------------------


def read_all_drifters_Phod(date_min, date_max, lon_min, lon_max, lat_min, lat_max, ln_drogue_off) :

    nb_hours_by_day_data=24
    DIR_DRIFTER_DATA='/scratch/work/regnierc/data/UV_Drifters/PhOD/DRIFTERS_PHOD_2.01.1/'
    file_total=[f for f in listdir(DIR_DRIFTER_DATA) if isfile(join(DIR_DRIFTER_DATA, f))]
    print(file_total[0:10])

    '''
    LOAD ALL WANTED DRIFTERS INTO A DICTIONNARY
    '''

    Drifters={}

    cc=0
    for ff, file_path in enumerate(file_total):
        #if (ff % 100 ==0):
        print(str(ff)+'/'+str(len(file_total))+' buoys treated')
        print(DIR_DRIFTER_DATA+'/'+file_path)
        nc_file= netCDF4.Dataset(DIR_DRIFTER_DATA+'/'+file_path,'r',format='NETCDF4')
        # Retreiving some min/max info on dates
        str_time_units=nc_file.variables['time'].units.split(' ')[0]
        str_time_date0=nc_file.variables['time'].units.split(' ')[2]
        drogue_lost_date=nc_file.variables['drogue_lost_date'][0]
        end_date=nc_file.variables['end_date'][0]
        #start_date=nc_file.variables['start_date'][0]
        start_date=nc_file.variables['deploy_date'][0]
        if start_date < 0 or  end_date < 0:
            #print('we skip this buoy if there is no data to load', file_path )
            # ---> we skip this buoy if there is no data to load
            nc_file.close()
            continue

        drogue_time_units=nc_file.variables['drogue_lost_date'].units.split(' ')[0]
        drogue_time_date0=nc_file.variables['drogue_lost_date'].units.split(' ')[2]
        date0_yyyy=int(drogue_time_date0.split('-')[0]) ; date0_mm=int(drogue_time_date0.split('-')[1]) ; date0_dd=int(drogue_time_date0.split('-')[2].split('T')[0]) ; date0_hh=0 ; date0_mn=0; 
        date_0_drogue=datetime.datetime(date0_yyyy, date0_mm, date0_dd, date0_hh, date0_mn)
        if (not drogue_lost_date.mask):
           if ('seconds' in drogue_time_units):
             date_drogue_lost=date_0_drogue+datetime.timedelta(seconds=int(drogue_lost_date))
           elif ('hours' in drogue_time_units):
             date_drogue_lost=date_0_drogue+datetime.timedelta(hours=int(drogue_lost_date))
           elif ('days' in drogue_time_units):
             date_drogue_lost=date_0+datetime.timedelta(days=int(drogue_lost_date))
        else:
          date_drogue_lost=0

        # if start_date or end_date are not marked
        if ('seconds' in drogue_time_units):
            if (not end_date.mask):
                date_end_date=date_0_drogue+datetime.timedelta(seconds=int(end_date))
            if (not start_date.mask):
                try:
                    date_start_date=date_0_drogue+datetime.timedelta(seconds=int(start_date))
                except:
                    #print('we skip this buoy if there is no data to load', file_path )
                    # ---> we skip this buoy if there is no data to load
                    nc_file.close()
                    continue
        elif ('hours' in drogue_time_units):
            if (not end_date.mask):
                date_end_date=date_0_drogue+datetime.timedelta(hours=int(end_date))
            if (not start_date.mask):
                date_start_date=date_0_drogue+datetime.timedelta(hours=int(start_date))
        elif ('days' in drogue_time_units):
            if (not end_date.mask):
                date_end_date=date_0+datetime.timedelta(days=int(end_date))
            if (not start_date.mask):
                date_start_date=date_0+datetime.timedelta(days=int(start_date))
        if (end_date.mask):
            date_end_date=date_max
        if (start_date.mask):
            date_start_date=date_min

        str_time_units=nc_file.variables['time'].units.split(' ')[0]
        str_time_date0=nc_file.variables['time'].units.split(' ')[2]

        #---------------------------------------------------------------
        # check if there is some data do load before doing anything else
        freq_trace=1 #hourly
        date_string_data=[]
        for datenow in daterange_deltah(date_start_date, date_end_date, freq_trace):
          date_string_data.append(datenow.strftime("%Y%m%d%H%M"))
        date_string_wanted=[]
        for datenow in daterange_deltah(date_min, date_max, freq_trace):
          date_string_wanted.append(datenow.strftime("%Y%m%d%H%M"))
        intersection = list(set(date_string_wanted) & set(date_string_data))
        if (not intersection ):
          #print('we skip this buoy if there is no data to load', file_path )
          # ---> we skip this buoy if there is no data to load 
          nc_file.close()
          continue
        #---------------------------------------------------------------

        print('buoy ok')
        id_buoy=np.squeeze(nc_file.variables['ID'][:])
        time=np.squeeze(nc_file.variables['time'][:])
        lon=np.squeeze(nc_file.variables['longitude'][:])
        lat=np.squeeze(nc_file.variables['latitude'][:])
        lon_err=np.squeeze(nc_file.variables['err_lon'][:])
        lat_err=np.squeeze(nc_file.variables['err_lat'][:])
        u=np.squeeze(nc_file.variables['ve'][:])
        v=np.squeeze(nc_file.variables['vn'][:])
        u_err=np.squeeze(nc_file.variables['err_ve'][:])
        v_err=np.squeeze(nc_file.variables['err_vn'][:])
        nc_file.close()

        lon=( (lon- 180) % 360) - 180
        # Subsection to transform time into python time object:
        date0_yyyy=int(str_time_date0.split('-')[0])
        date0_mm=int(str_time_date0.split('-')[1])
        date0_dd=int(str_time_date0.split('-')[2].split('T')[0])
#       date0_hh=int(str_time_date0.split('-')[2].split('T')[1].split(':')[0])
#       date0_mn=int(str_time_date0.split('-')[2].split('T')[1].split(':')[1])
        date0_hh=0 ; date0_mn=0
        date_0=datetime.datetime(date0_yyyy, date0_mm, date0_dd, date0_hh, date0_mn)


        if ('seconds' in str_time_units):
            date=[date_0+datetime.timedelta(seconds=int(i)) for i in time]
        elif ('hours' in str_time_units):
            date=[date_0+datetime.timedelta(hours=int(i)) for i in time]
        elif ('days' in str_time_units):
            date=[date_0+datetime.timedelta(days=int(i)) for i in time]
        date=np.array(date)
        # fill drogue array for 2.01_experimental
        drogue=np.ones_like(date)
        if (not drogue_lost_date.mask):
            drogue=np.where(date<date_drogue_lost , 1, 0)

        # we filter time period between min/max  straight away to save time in the forthcoming loop
        #print ('****')
        #print (file_path)
        #print('max date='+str(max(date))+' ; min date='+str(min(date)))
        ind=np.squeeze(np.where( (date>date_min) & (date<date_max) ))
        time=time[ind] ; lon=lon[ind] ; lat=lat[ind] ; u=u[ind] ; v=v[ind] ; drogue=drogue[ind] ; date=date[ind] ; u_err=u_err[ind] ; v_err=v_err[ind] ; lon_err=lon_err[ind] ; lat_err=lat_err[ind]
        id_buoy=id_buoy[ind]

        total_id=np.unique(id_buoy)
        total_id = total_id[np.isfinite(total_id)]

        # from this place, we have to unroll buoy id
        #cc=0
        for id in total_id:

            ind=np.where(id_buoy==id)

            sub_id=id_buoy[ind].astype(int)[0]
            sub_lon=lon[ind]
            sub_lat=lat[ind]
            sub_u=u[ind]
            sub_v=v[ind]
            sub_drogue=drogue[ind].astype(int)
            sub_time=time[ind]
            sub_lon_err=lon_err[ind]
            sub_lat_err=lat_err[ind]
            sub_u_err=u_err[ind]
            sub_v_err=v_err[ind]

            if ('seconds' in str_time_units):
                 sub_time=[ datetime.timedelta(seconds=int(j))+date_0 for j in  sub_time[:] ]
            elif ('minutes' in str_time_units):
                 sub_time=[ datetime.timedelta(minutes=int(j))+date_0 for j in  sub_time[:] ]
            elif ('hours' in str_time_units):
                 sub_time=[ datetime.timedelta(hours=int(j))+date_0 for j in  sub_time[:] ]
            elif ('days' in str_time_units):
                 sub_time=[ datetime.timedelta(days=int(j))+date_0 for j in  sub_time[:] ]
            sub_time=np.array(sub_time)

            # build a drifter object for each drifter:
            d=drifter_object.drifter_AOML_nc()
            d.get_drifter(sub_id, sub_lon, sub_lat, sub_u, sub_v, sub_drogue, sub_time, sub_lon_err, sub_lat_err, sub_u_err, sub_v_err)
            if (d.id):
                d.filter_time(date_min, date_max) # filter for the correct time period
                if (d.id):
                    d.filter_zone(lon_min, lon_max, lat_min, lat_max) # filter for the correct space period
                    if (ln_drogue_off): # filter buoys with drogues
                        d.filter_drogue()
                    if (d.id != ''  and len(d.time)>=nb_hours_by_day_data*7): #only keep trajectories lasting longer than aweek
                        Drifters[d.id]=d
                        cc+=1
                        if (cc%100==0):
                            print('{}/{} buoys retained'.format(cc,len(total_id)))

    return Drifters
