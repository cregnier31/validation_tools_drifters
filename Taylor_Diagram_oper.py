#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import os, glob,sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("config.ini")
config_info = config_object['CONFIG_BASE']
MODELS = config_info["MODELS"].split(',')
lon_min = int(config_info["lon_min"])
lon_max = int(config_info["lon_max"])
lat_min = int(config_info["lat_min"])
lat_max = int(config_info["lat_max"])
DATE1 = config_info["DATE_start"]
DATE2 = config_info["DATE_end"]
PROJECT = config_info["PROJECT"]
EXP_NAME = config_info["EXP_NAME"]
username = config_info["username"]
dir_work = config_info["dir_work"]
dir_in = config_info["dir_in"]
print("MODELS : ",MODELS)
print("Region : ",lon_min,", ",lon_max,", ",lat_min,", ",lat_max)
print("From : ",DATE1," to ",DATE2)

sys.path.insert(0, dir_in+"src/")
from Drift_Tools import ReturnUVlist, CalcCurlGlobC, CalcRelVorticity, ExtractSurface, match_elements, StringRegion, DispDirAndLength,  SeparationDistance, TrajDistBetweenTimesteps, SepDistanceEachTimestep, LiuScore, shift360, DriftersInZoneInPeriod, coord2index

str_region = StringRegion(lon_min,lon_max,lat_min,lat_max)
dir_out=dir_work+"DRIFT/"+PROJECT+'/'+EXP_NAME+'/ZONE_'+str_region+'/'


print("Dir out : ",dir_out)


DD1=np.int32(DATE1[:2])
MM1=np.int32(DATE1[2:4])
YYYY1=np.int32(DATE1[4:])
DD2=np.int32(DATE2[:2])
MM2=np.int32(DATE2[2:4])
YYYY2=np.int32(DATE2[4:])
#print(DD1,'-',MM1,'-',YYYY1)
date_start = datetime.date(YYYY1, MM1, DD1)
str_date1=date_start.strftime('%Y-%m-%d')
date_start_min1 = date_start-datetime.timedelta(days=1)
date_end=datetime.date(YYYY2, MM2, DD2)
str_date2=date_end.strftime('%Y-%m-%d')
date_end_plus1 = date_end+datetime.timedelta(days=1)
T_DRIFT = (date_end-date_start).days


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

for MODEL in MODELS:
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
    
    if MODEL in ['PSY4V3R1','IBI36','GLO12V4']:
        u_files_mod_3D, v_files_mod_3D = ReturnUVlist(MODEL, date_start_min1, date_end_plus1)
        u_files, v_files = ExtractSurface(MODEL,u_files_mod_3D, v_files_mod_3D,zlevel,var_u_mod, var_v_mod, dir_work+'MODEL_OUTPUT/'+MODEL+'/UV/15m/')
        u_files_mod.append(u_files)
        v_files_mod.append(v_files)
    else : 
        u_files, v_files = ReturnUVlist(MODEL, date_start_min1, date_end_plus1)
        u_files_mod.append(u_files)
        v_files_mod.append(v_files)




# retrieve GDP drifters trajectories
dir_in_svp='/data/rd_exchange/svangennip/DATA/GlobalDriftingProgram//YEAR/'

fname_in='Drifters_'+str(YYYY1)+'.csv'
f_drifters_all=dir_in_svp+fname_in


fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'.csv'
dir_drifters = dir_out+'/DRIFTERS/'

if not os.path.exists(dir_drifters):
    print('\ncreate folder : ', dir_drifters)
    # Create directory where to store results:
    os.makedirs(dir_drifters)
outfile2 = dir_drifters+fout_driftersZone

if not os.path.isfile(outfile2):
    DriftersInZoneInPeriod(lon_min, lon_max, lat_min, lat_max, date_start_min1, date_end_plus1, f_drifters_all, outfile2)
else:
    print(outfile2, 'already exists')

print('loading ',outfile2)
drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
print(drifters_zone_traj)
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
    print('remove drifter ',spurious_drifters[i])
    ind1 = drifters_zone_traj[drifters_zone_traj['id'] == np.array(spurious_drifters[i])].index
    drifters_zone_traj.drop(ind1, inplace=True) #remove data points
#drifters_zone_traj=drifters_zone_traj.reset_index()  
nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
        
print('total nb of drifters in zone at start that reach end:',nb_drifters2)
#Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
nb_points = T_DRIFT*4+1+8
traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
traj_drifter_date = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )

#
#
#     CALCUL FOR SVP DRIFTERS
#
#



# Calc One day speed of SVP
R = 6378.137
lon_uv_svp = np.array([np.mean(traj_drifter_lon[:,i*4:i*4+4],axis = 1) for i in range(int(len(traj_drifter_lon[0])/4))]).T
lat_uv_svp = np.array([np.mean(traj_drifter_lat[:,i*4:i*4+4],axis = 1) for i in range(int(len(traj_drifter_lat[0])/4))]).T
u_svp = ((np.radians(lon_uv_svp[:,1:]-lon_uv_svp[:,:-1]))*np.cos(np.radians(lat_uv_svp[:,:-1]))* R)*(1000/(24*3600))
v_svp = ((np.radians(lat_uv_svp[:,1:]-lat_uv_svp[:,:-1] ))*R)*(1000/(24*3600))


date_uv_svp = traj_drifter_date[:,::4][:,1:-1]




# retrieve speed on svp location for the MODEL (method = nearest)


U_model = []
V_model = []
indice_date = [77,73,116]
for u_files,v_files,var_u,var_v,MODEL,ind_date  in zip(u_files_mod,v_files_mod,var_u_mod,var_v_mod,MODELS,indice_date):
    u_mod= []
    v_mod= []
    for u_file,v_file in zip(u_files,v_files):
        print(u_file[ind_date:ind_date+8])
        str_date = u_file[ind_date:ind_date+8]
        str_date = str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:]
        u = xr.open_dataset(u_file)
        v = xr.open_dataset(v_file)
        uo = u[var_u].values
        vo = v[var_v].values

        ind_svp = np.where(date_uv_svp==str_date)

        if MODEL in ['PSY4V3R1','GLO12V4','IBI36']:
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


                u_mod.append(uo[0,0][ind_modu[:,0],ind_modu[:,1]])
                v_mod.append(vo[0,0][ind_modv[:,0],ind_modv[:,1]])
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
                    u_int.append(u.interp(longitude = lo,latitude = la,method='nearest').uo.values[0][1])# zlevel = 15m
                    v_int.append(v.interp(longitude = lo,latitude = la,method='nearest').vo.values[0][1])# zlevel = 15m
                u_mod.append(u_int)
                v_mod.append(v_int)
    U_model.append(np.array(u_mod))
    V_model.append(np.array(v_mod))

U_model = np.array(U_model)
V_model = np.array(V_model)



# Calcul (norm std and corr coeff R) for the Taylor diagram
u_corrcoef = np.array([[np.corrcoef(U_model.T[i,:,j],u_svp[i])[0,1] for i in range(len(u_svp))] for j in range (len(MODELS))])
v_corrcoef = np.array([[np.corrcoef(V_model.T[i,:,j],v_svp[i])[0,1] for i in range(len(v_svp))] for j in range (len(MODELS))])

u_std = np.array([[np.std(U_model.T[i,:,j]) for i in range(len(u_svp))] for j in range (len(MODELS))])
v_std = np.array([[np.std(V_model.T[i,:,j]) for i in range(len(v_svp))] for j in range (len(MODELS))])
print(u_std.shape)
print(u_corrcoef.shape)






# rmse curve 
def corr_coeff_for_rms(rms,std_r,std_t):
    return (std_r**2+std_t**2-rms**2)/(2*std_r*std_t)
r0 = 1
def taylor_skill_score(s,std_n):
    return (s*(std_n+1/std_n)**2*(1+r0))/(4)-1


##### FIGURES

dir_plot = dir_out+'PLOT/'
if not os.path.exists(dir_plot):
    print('\ncreate folder : ', dir_plot)
    # Create directory where to store results:
    os.makedirs(dir_plot)

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

ax.plot(np.linspace(0,np.pi/2,100),std_ru*np.ones(100),'--',c = 'cyan',label = 'std observed')
ax.plot(0,std_ru, '*',c= 'cyan',ms = 15, clip_on=False, zorder=100)
ax.plot(np.zeros(std_drifters_svp_u.shape),std_drifters_svp_u, '*',c= 'k',ms = 8, clip_on=False, zorder=100,label = 'std drifters')

plt.legend()
ax.text(np.pi/6,0.165,'Correlation',fontsize = 12,rotation = -np.arccos(np.pi/6)*180/np.pi)

ax.set_rmax(0.16)
ax.set_rmin(0)
#ax.set_title('Taylor Diagram for zonal velocity U')
ax.text(np.pi/1.8,0.06,'STD ($m.s^{-1}$)',fontsize = 12,rotation =90)

plt.savefig(dir_plot+'Taylor_diagram_U_SVP_15m_'+str_date1+'_'+str_date2+'.png',dpi = 600)





fig, ax = plt.subplots(figsize = (8,8),subplot_kw={'projection': 'polar'})
std_rv = np.mean(np.std(v_svp,axis = 1))
std_drifters_svp_v = np.std(v_svp,axis = 1)

for i in np.arange(0.02,0.15,0.02):
    r = np.linspace(0.001,0.16,100)
    theta = -corr_coeff_for_rms(i,std_rv,r)*np.pi/2+np.pi/2
    r = np.delete(r,np.where((theta<-np.pi)+(theta > np.pi))[0])
    theta = np.delete(theta,np.where((theta<-np.pi)+(theta > np.pi))[0])
    line, = ax.plot(theta,r,c = 'k',alpha = 0.5)
    if i == 0.02:
        line.set_label('centered RMS')
    if theta[np.argmin(abs(r-0.1))]< np.pi/2 and theta[np.argmin(abs(r-0.1))]> 0 : 
        t = ax.text(theta[np.argmin(abs(r-0.1))],0.1, str(round(i,2)), fontsize=11)
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
    elif theta[np.argmin(abs(r-0.1))]< 0: 
        t = ax.text(theta[np.argmin(abs(r-std_rv))],std_rv, str(round(i,2)), fontsize=11)
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
    ax.plot(-v_corrcoef[i]*np.pi/2+np.pi/2, v_std[i],'o',c = colors[i],alpha = 0.3,ms = 4)
    ax.plot(np.nanmean(-v_corrcoef[i]*np.pi/2+np.pi/2), np.nanmean(v_std[i]),'o',c = colors[i],ms = 10,label = 'MEAN '+MODELS[i])

ax.plot(np.linspace(0,np.pi/2,100),std_rv*np.ones(100),'--',c = 'cyan')
ax.plot(0,std_rv, '*',c= 'cyan',ms = 15, clip_on=False, zorder=100,label = 'std drifters')
ax.plot(np.zeros(std_drifters_svp_v.shape),std_drifters_svp_v, '*',c= 'k',ms = 8, clip_on=False, zorder=100,label = 'std drifters')

plt.legend()
ax.text(np.pi/5,0.165,'Correlation',fontsize = 12,rotation = -np.arccos(np.pi/6)*180/np.pi)
ax.set_rmax(0.16)
ax.set_rmin(0)
#ax.set_title('Taylor Diagram for meridional velocity V')
ax.text(np.pi/1.8,0.06,'STD ($m.s^{-1}$)',fontsize = 12,rotation =90)

plt.savefig(dir_plot+'Taylor_diagram_V_SVP_15m_'+str_date1+'_'+str_date2+'.png',dpi = 600)



