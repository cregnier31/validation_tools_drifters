#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Copernicus Marine User Support Team"
__copyright__ = "(C) 2022 E.U. Copernicus Marine Service Information"
__credits__ = ["E.U. Copernicus Marine Service Information"]
__license__ = "MIT License - You must cite this source"
__version__ = "202104"
__maintainer__ = "D. Bazin, E. DiMedio, C. Giordan"
__email__ = "servicedesk dot cmems at mercator hyphen ocean dot eu"

# Import necessary libraries

from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import xarray as xr
import numpy as np
import pandas as pd
from Drift_Tools import DriftersInZoneInPeriod_remote, resample_traj
import datetime
import ftputil
import os, glob,sys
from tqdm import tqdm
import json
from tqdm import tqdm

def copernicusmarine_datastore(dataset, username, password):
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    return data_store





def Retrieve_data_from_copernicusmarine_datastore(MODELS,USERNAME,PASSWORD,depth,lon_min,lon_max,
                                                                lat_min,lat_max,str_date1,str_date2):
    """
    return data in the time period and in the zone for MODELS
    """        
    data = []
    for dataset in MODELS:
        data_store = xr.open_dataset(copernicusmarine_datastore(dataset, USERNAME, PASSWORD))
        
        try :
            data.append(data_store.sel(time = slice(str_date1+" 12:00:00",str_date2+" 12:00:00"),
                                       longitude = slice(lon_min,lon_max),
                                       latitude = slice(lat_min,lat_max)).sel(depth = depth,
                                                                              method = 'nearest'))
        except KeyError:
            data.append(data_store.sel(time = slice(str_date1+" 12:00:00",str_date2+" 12:00:00"),
                                       lon = slice(lon_min,lon_max),
                                       lat = slice(lat_min,lat_max)).sel(depth = depth,method = 'nearest'))

    colors = np.array(['b','r','g','k'])[:len(MODELS)]
    return data,colors


















def Rsetrieve_drifter_from_copernicusmarine_datastore(user,password,DIR_WORK,dir_out,str_date1,str_date2,T_DRIFT,str_region,lon_min,lon_max,lat_min,lat_max,date_start,date_end,tay_diag= False):
    """
    """
    product_name = 'INSITU_GLO_PHY_UV_DISCRETE_NRT_013_048' #type aimed In Situ product
    dataset_name = 'cmems_obs-ins_glo_phy-cur_nrt_drifter_irr'
    host = 'nrt.cmems-du.eu' #type aimed host (nrt.cmems-du.eu or my.cmems-du)
    index_file = 'index_history.txt' #type aimed index file 

    output_directory = DIR_WORK+'DATA/DRIFTER48/' #path

    date_format = "%Y-%m-%dT%H:%M:%SZ" 
    ini = datetime.datetime.strptime(str_date1+'T00:00:00Z', date_format)
    end = datetime.datetime.strptime(str_date2+'T23:59:59Z', date_format)

    #connect to CMEMS FTP
    with ftputil.FTPHost(host, user, password) as ftp_host: 

        #open the index file to read
        with ftp_host.open("Core"+'/'+product_name+'/'+ dataset_name + '/' + index_file, "r") as indexfile:

            #read the index file as a comma-separate-value file
            index = np.genfromtxt(indexfile, skip_header=6, unpack=False, delimiter=',', dtype=None, names=['catalog_id', 'file_name','geospatial_lat_min', 'geospatial_lat_max', 'geospatial_lon_min','geospatial_lon_max','time_coverage_start', 'time_coverage_end', 'provider', 'date_update', 'data_mode', 'parameters'])

            #loop over the lines/netCDFs and download the most sutable ones for you
            new = 0
            for netCDF in index:

                #getting ftplink, filepath and filename
                ftplink = netCDF['file_name']
                filepath = '/'.join(ftplink.decode().split('/')[3:len(ftplink.decode().split('/'))])
                ncdf_file_name = ftplink[ftplink.decode().rfind('/')+1:]

                #download netCDF if meeting selection criteria
                time_start = datetime.datetime.strptime(netCDF['time_coverage_start'].decode(), date_format)
                time_end = datetime.datetime.strptime(netCDF['time_coverage_end'].decode(), date_format)

                #download netCDF if meeting selection criteria
                if time_end > ini and time_start < end:
                    if ftp_host.path.isfile(filepath):
                        if not os.path.exists(output_directory+filepath.split('/')[-1]):
                            cwd = os.getcwd()
                            os.chdir(output_directory)
                            ftp_host.download(filepath, ncdf_file_name)  # remote, local
                            os.chdir(cwd)
                            if 'MO_TS_DC' in filepath[-1]:
                                print("Wrong type of drifter ! removed : ",filepath.split('/')[-1])
                                os.remove(output_directory+filepath.split('/')[-1])
                            else :
                                print(output_directory+filepath.split('/')[-1])
                                suspicious_drifter = resample_traj(output_directory+filepath.split('/')[-1])
                                new = 1

                            if suspicious_drifter :
                                print("Remove drifter : ",filepath.split('/')[-1])
                                os.remove(output_directory+filepath.split('/')[-1])
                        else:
                            print(output_directory+filepath.split('/')[-1],' already exist')



    file_list = sorted(glob.glob(output_directory+'*.nc'))
    dico_name = output_directory+'metadata_drifters.txt'
    if new == 1 or not os.path.exists(dico_name):
        dico = {}
        for file in tqdm(file_list): 
            ds = xr.open_dataset(file)
            t_start = ds.time_coverage_start
            t_end = ds.time_coverage_end
            dico[file] = t_start, t_end
            
        with open(dico_name, 'w') as file:
            print("Write dictionnary")
            file.write(json.dumps(dico)) # use `json.loads` to do the reverse
        
        


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
        DriftersInZoneInPeriod_remote(lon_min, lon_max, lat_min, lat_max, date_start, date_end, dico_name, outfile2)
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
        if not nb_entries >= (T_DRIFT*24):
            print(drifter_ids[i],': not enough entries, only ', nb_entries)
            spurious_drifters.append(drifter_ids[i])

    #ind1 = drifters_zone_traj['id']==spurious_drifters).index
    for i in range(len(spurious_drifters)):
        print('remove drifter ',spurious_drifters[i])
        ind1 = drifters_zone_traj[drifters_zone_traj['id'] == np.array(spurious_drifters[i])].index
        drifters_zone_traj.drop(ind1, inplace=True) #remove data points
    drifters_zone_traj=drifters_zone_traj.reset_index()  
    nb_drifters2=len(pd.unique(drifters_zone_traj['id']))

    print('total nb of drifters in zone at start that reach end:',nb_drifters2)
    #Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
    nb_points = T_DRIFT*24    # hourly
    traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
    traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
    traj_drifter_date = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
    traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
    traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )  
    
    return traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id


def Calculate_model_traj_remote(MODELS,dir_out,nb_points,traj_drifter_lon,traj_drifter_lat,traj_drifter_id,date_start,DRIFT_TIME,data,AdvectionDiffusion_type,nb_repeats,cloud_stat = False):
    from Drift_exec import run_parcels_Drift, run_parcels_Drift_remote
    for i in range(len(MODELS)):
        dir_work_mod = dir_out+'/EXP/'+MODELS[i]+'/'
        print('nb points : ',nb_points)
        for j in tqdm(np.arange(0,nb_points-DRIFT_TIME*24,24)): # one sim per day
            traj_drifter_lon_start = traj_drifter_lon[:,j]
            traj_drifter_lat_start = traj_drifter_lat[:,j]
            traj_drifter_id_start = traj_drifter_id[:,j]
            date_sim = date_start + datetime.timedelta(days=int(j//24))
            str_date_sim = date_sim.strftime('%Y-%m-%d')
            date_end_sim = date_sim + datetime.timedelta(days=DRIFT_TIME+1)
            str_date_end_sim = date_end_sim.strftime('%Y-%m-%d')
            f_traj1 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc.zarr'
            f_traj2 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_and_around.zarr'
            f_traj3 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_smag.zarr'
            f_traj4 = dir_work_mod + "Trajectories_"+MODELS[i]+'_'+str_date_sim+'_'+str(DRIFT_TIME)+'_at_SVPloc_M1.zarr'

#            if not (os.path.isfile(f_traj1) & os.path.isfile(f_traj2)) :
            print('#run parcels with just the SVP locations')
            var_u = 'uo'
            var_v = 'vo'

            if cloud_stat:
                run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_sim+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                             f_traj2, radius_cloud=0.25, seeding_resolution=1/12, \
                             time_period=False)
            if AdvectionDiffusion_type=="Smag":
                run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_sim+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                             f_traj3, radius_cloud=0, seeding_resolution=0, \
                             time_period=False,AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = nb_repeats)
            if AdvectionDiffusion_type=="AdvectionDiffusionM1":
                run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_sim+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                             f_traj4, radius_cloud=0, seeding_resolution=0, \
                             time_period=False,AdvectionDiffusion_type=AdvectionDiffusion_type,nb_repeats = nb_repeats)

            if True:
                print(data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_sim+" 12:00:00")))
                run_parcels_Drift_remote(MODELS[i],data[i].sel(time = slice(str_date_sim+" 12:00:00",str_date_end_sim+" 12:00:00")),var_u,var_v,traj_drifter_lon_start, traj_drifter_lat_start,traj_drifter_id_start, DRIFT_TIME,\
                             f_traj1, radius_cloud=0, seeding_resolution=0, \
                             time_period=False)
            print(' done...')



def Calculate_speed_at_svp_location_remote(lon_uv_svp,lat_uv_svp,traj_drifter_date,traj_drifter_time,traj_drifter_id,data):
    """
    """


    date_uv_svp = traj_drifter_date[:,::24][:,1:] + " 12:00:00"

    uv_svp_loc = []


    for da in data  :
              
        try:
            uv_svp_loc.append(da.sel(longitude = lon_uv_svp[:,:-1].flatten(),latitude = lat_uv_svp[:,:-1].flatten(),time = date_uv_svp.flatten(),method = "nearest"))
        except KeyError:
            uv_svp_loc.append(da.sel(lon = lon_uv_svp[:,:-1].flatten(),lat = lat_uv_svp[:,:-1].flatten(),time = date_uv_svp.flatten(),method = "nearest"))


    # reshape
    lon_model =[]
    lat_model =[]
    time_model =[]
    U_model = []
    V_model = []

    for da in uv_svp_loc:
        try:
                       
            lon_model.append(da.longitude.values.reshape(date_uv_svp.shape))
            lat_model.append(da.latitude.values.reshape(date_uv_svp.shape))
            U_model.append(np.array([da.uo.values[i,i,i] for i in range (da.longitude.values.shape[0])]).reshape(date_uv_svp.shape))
            V_model.append(np.array([da.vo.values[i,i,i] for i in range (da.longitude.values.shape[0])]).reshape(date_uv_svp.shape))

        except AttributeError:
             lon_model.append(da.lon.values.reshape(date_uv_svp.shape))
             lat_model.append(da.lat.values.reshape(date_uv_svp.shape))
             U_model.append(np.array([da.uo.values[i,i,i] for i in range (da.lon.values.shape[0])]).reshape(date_uv_svp.shape))
             V_model.append(np.array([da.vo.values[i,i,i] for i in range (da.lon.values.shape[0])]).reshape(date_uv_svp.shape))


        time_model.append(da.time.values.reshape(date_uv_svp.shape))

    lon_model = np.array(lon_model)
    lat_model = np.array(lat_model)
    time_model =np.array(time_model)
    U_model =np.array(U_model)
    V_model = np.array(V_model)
    
    return U_model, V_model









def Retrieve_drifter_pandas(user,password,DIR_WORK,dir_out,str_date1,str_date2,str_region,date_start,date_end):
    yyear = str_date1[0:4]
    index_file = 'index_history.txt' #type aimed index file
    new = 1
    output_directory = "/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/" + yyear + "/"

    file_start = "/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/"     + yyear + "/GL_TS_DC_"+str_date1[:4]+str_date1[5:7]+str_date1[8:10]+"_FILTR.nc"
    file_end = "/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/" +     yyear + "/GL_TS_DC_"+str_date2[:4]+str_date2[5:7]+str_date2[8:10]+"_FILTR.nc"


    file_list = np.array(sorted(glob.glob(output_directory+'*.nc')))
    print(file_start)
    print(file_end)
    print(file_list)
    print(output_directory)
    files_in  = file_list[(file_list>= file_start)*(file_list<=file_end)]

    
    TIME = []
    DATE = []
    LATITUDE = []
    LONGITUDE = []
    ID = []
    u = []
    v = []
    u_qc = []
    v_qc = []
    
    u_fltr = []
    v_fltr = []
    u_qc_fltr = []
    v_qc_fltr = []


    u_ws = []
    v_ws = []
    u_ws_qc = []
    v_ws_qc = []

    u_ws_fltr  = []
    v_ws_fltr  = []
    u_ws_qc_fltr  = []
    v_ws_qc_fltr  = []



    current_test = []
    current_test_qc = []
    index_first = 1
    nb = 0
    for fichier in files_in:
        print(fichier)
        # Lire le fichier NetCDF avec xarray
        ncdata = xr.open_dataset(fichier)
        variables = ncdata.data_vars
        if True:
            if 'EWCT' and 'NSCT' in variables.keys():
                u = ncdata['EWCT'].values.reshape(-1)
                v = ncdata['NSCT'].values.reshape(-1)
                u_qc = ncdata['EWCT_QC'].values.reshape(-1)
                v_qc = ncdata['NSCT_QC'].values.reshape(-1)
            else:
                u = None; v = None
                u_qc = None; v_qc = None

            if 'EWCT_FILTR' and 'NSCT_FILTR' in variables.keys():
                u_fltr = ncdata['EWCT_FILTR'].values.reshape(-1)
                v_fltr = ncdata['NSCT_FILTR'].values.reshape(-1)
                u_qc_fltr = ncdata['EWCT_FILTR_QC'].values.reshape(-1)
                v_qc_fltr = ncdata['NSCT_FILTR_QC'].values.reshape(-1)
            else:
                u_fltr = None; v_fltr = None
                u_qc_fltr = None; v_qc_fltr = None

        
            if 'EWCT_WS' and 'NSCT_WS' in variables.keys():
                u_ws = ncdata['EWCT_WS'].values.reshape(-1)
                v_ws = ncdata['NSCT_WS'].values.reshape(-1)
                u_ws_qc = ncdata['EWCT_WS_QC'].values.reshape(-1)
                v_ws_qc = ncdata['NSCT_WS_QC'].values.reshape(-1)
            else:
                u_ws = None; v_ws = None
                u_ws_qc = None; v_ws_qc = None
        
            if 'EWCT_WS_FILTR' and 'NSCT_WS_FILTR' in variables.keys():
                u_ws_fltr = ncdata['EWCT_WS_FILTR'].values.reshape(-1)
                v_ws_fltr = ncdata['NSCT_WS_FILTR'].values.reshape(-1)
            else:
                u_ws_fltr = None ; v_ws_fltr = None
            if 'EWCT_WS_FILTR_QC' and 'NSCT_WS_FILTR_QC' in variables.keys():
                u_ws_qc_fltr = ncdata['EWCT_WS_FILTR_QC'].values.reshape(-1)
                v_ws_qc_fltr = ncdata['NSCT_WS_FILTR_QC'].values.reshape(-1)
            else:
                u_ws_qc_fltr = None; v_ws_qc_fltr = None



            if 'CURRENT_TEST' in variables.keys():
                current_test = ncdata['CURRENT_TEST'].values.reshape(-1)
            else:
                current_test = None; current_test_qc = None

            if 'CURRENT_TEST_QC' in variables.keys():
                current_test_qc = ncdata['CURRENT_TEST_QC'].values.reshape(-1)
            else:
                current_test_qc = None

            LONGITUDE = ncdata['LONGITUDE'].values.reshape(-1) 
            LATITUDE = ncdata['LATITUDE'].values.reshape(-1) 
            DATE = pd.to_datetime(ncdata['TIME'].values).date.reshape(-1) 
            TIME = pd.to_datetime(ncdata['TIME'].values).time.reshape(-1) 
            ID = np.int32(ncdata.PLATFORM_CODE.values).reshape(-1) 
            #print(pd.to_datetime(TIME).date)    
            dic = {'id':ID,'date':DATE,'time':TIME,'lat':LATITUDE,'lon':LONGITUDE,'EWCT':u,'NSCT':v,'EWCT_QC':u_qc,'NSCT_QC':v_qc,'EWCT_FILTR':u_fltr,'NSCT_FILTR':v_fltr,'EWCT_QC_FILTR':u_qc_fltr,'NSCT_QC_FILTR':v_qc_fltr,'EWCT_WS':u_ws,'NSCT_WS':v_ws,'EWCT_WS_QC':u_ws_qc,'NSCT_WS_QC':v_ws_qc,'EWCT_WS_FILTR':u_ws_fltr,'NSCT_WS_FILTR':v_ws_fltr,'EWCT_WS_QC_FILTR':u_ws_qc_fltr,'NSCT_WS_QC_FILTR':v_ws_qc_fltr,'CURRENT_TEST':current_test,'CURRENT_TEST_QC':current_test_qc}
            print(dic['id'])
            print(dic['EWCT'])

            if index_first ==1:
                df = pd.DataFrame(dic,index = np.int32(nb*np.ones(LONGITUDE.shape)))
                nb +=1
                index_first=0
            else :
                df = pd.concat([df,pd.DataFrame(dic, index = np.int32(nb*np.ones(LONGITUDE.shape))  )])
                nb += 1

    pd.DataFrame(df).to_csv('Q2_Dict.csv')

def Retrieve_drifter_from_copernicusmarine_datastore_fidji1(user,password,DIR_WORK,dir_out,str_date1,str_date2,T_DRIFT,str_region,lon_min,lon_max,lat_min,lat_max,date_start,date_end,tay_diag= False):
    """
    """
    yyear = str_date1[0:4]
    index_file = 'index_history.txt' #type aimed index file 
    new = 1
    #output_directory = DIR_WORK+'DATA/DRIFTER48/' #path
    output_directory = "/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/" + yyear + "/"



    if tay_diag :
        fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'_taylor_diag.csv'
    else :
        fout_driftersZone = 'Drifters_'+str_region+'_'+str_date1+'_'+str_date2+'1.csv'
    dir_drifters = dir_out+'/DRIFTERS/'

    if  not os.path.exists(dir_drifters):
        print('\ncreate folder : ', dir_drifters)
        # Create directory where to store results:
        os.makedirs(dir_drifters)
    outfile2 = dir_drifters+fout_driftersZone
    
    if True:#eot os.path.isfile(outfile2):           
        file_start = "/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/" + yyear + "/GL_TS_DC_"+str_date1[:4]+str_date1[5:7]+str_date1[8:10]+"_FILTR.nc"
        file_end = "/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/" + yyear + "/GL_TS_DC_"+str_date2[:4]+str_date2[5:7]+str_date2[8:10]+"_FILTR.nc"
        print(file_start)
        print(file_end)
        
        file_list = np.array(sorted(glob.glob(output_directory+'*.nc')))
        files_in  = file_list[(file_list>= file_start)*(file_list<=file_end)]
        platform_code = []
        platform_code_unique = []
        TIME = []
        LATITUDE = []
        LONGITUDE = []
        for file in files_in:
            ds = xr.open_dataset(file)
    
            platform_code_unique.append(np.int32(pd.unique(ds.PLATFORM_CODE.values)))
            platform_code.append(np.int32(ds.PLATFORM_CODE.values))
            TIME.append(ds.TIME.values)
            LATITUDE.append(ds.LATITUDE.values)
            LONGITUDE.append(ds.LONGITUDE.values)



        #print([platform_code_unique[i] - platform_code[i] for i in range(len(platform_code_unique))])
        
        ID = platform_code_unique[0]
     
        ID_old = ID
        for day_code in platform_code_unique[1:]:
            for code in ID:
                if False:#code not in day_code:
                    ID = np.delete(ID, np.where(ID == code))
                    print("delete : ",code)    

        

        df = pd.DataFrame()
        k = 0
        nb = 0

        lllat = []
        lllon = []
        for id in tqdm(ID):
            for t in range(len(TIME)-1):
                indice = np.where(platform_code[t]==id)
                #print(indice)
                #print(len(indice[0]))
                if len(indice[0]) >0:
                    lon_start = LONGITUDE[t][indice][0]

                    lon_end = LONGITUDE[t][indice][-1]
                    lat_start = LATITUDE[t][indice][0]
                    lat_end = LATITUDE[t][indice][-1]
                    lllon.append(lon_start)
                    lllon.append(lon_end)
                    lllat.append(lat_end)
                    lllat.append(lat_start)

                    if False:#not (lon_min<=lon_start)*(lon_max>=lon_start)*(lat_min <= lat_start)*(lat_max >= lat_start)*(lon_min <= lon_end)*(lon_max >= lon_end)*(lat_min <= lat_end)*(lat_max >= lat_end):
                        print("Not in the zone : ",id)
                        ID = np.delete(ID, np.where(ID == id))
                        #break
                    else:
                        if k ==0:
                            dic = {'id':platform_code[t][indice],'date':pd.to_datetime(TIME[t][indice]).date,'time':pd.to_datetime(TIME[t][indice]).time,'lat':LATITUDE[t][indice],'lon':LONGITUDE[t][indice]}

                            df = pd.DataFrame(dic,index = np.int32(nb*np.ones(LONGITUDE[t][indice].shape)))
                            k=1
                            print(df)                 
                        else :
                            dic = {'id':platform_code[t][indice],'date':pd.to_datetime(TIME[t][indice]).date,'time':pd.to_datetime(TIME[t][indice]).time,'lat':LATITUDE[t][indice],'lon':LONGITUDE[t][indice]}

                            df = pd.concat([df,pd.DataFrame(dic,index = np.int32(nb*np.ones(LONGITUDE[t][indice].shape)))])
                            nb+=1
                            #print(df)
        print(sorted(lllat)[0:20])
        print(sorted(lllat)[-20:-1])
        
        print(sorted(lllon)[0:20])
        print(sorted(lllon)[-20:-1])


        for ii in range(nb):
            print(df['lat'][ii])
            print(df['lon'][ii])
            #time.sleep(2)
        
        print(np.min(df['lat']))
        print(np.max(df['lat']))
        print(np.min(df['lon']))
        print(np.max(df['lon']))

        df.to_csv(outfile2)
        
    else:
        print(outfile2, 'already exists')
        
    
    print('loading ',outfile2)
    drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
    print(drifters_zone_traj)
    nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))
    #remove drifters with missing entries (should be nb_days*4+1)
    #Check each drifter has n days * 4 entries
    drifter_ids = pd.unique(drifters_zone_traj['id'])
    spurious_drifters=[]
    for i in range(nb_drifters_temp):
        nb_entries = len(drifters_zone_traj[drifters_zone_traj.id == drifter_ids[i]])
        print(nb_entries)
        print(drifters_zone_traj[drifters_zone_traj.id == drifter_ids[i]])

        if not nb_entries >= (T_DRIFT*24):
            print(drifter_ids[i],': not enough entries, only ', nb_entries)
            spurious_drifters.append(drifter_ids[i])
    
    print(f'{drifters_zone_traj=}')
    
    for i in range(len(spurious_drifters)):
        print('remove drifter ',spurious_drifters[i])
        ind1 = drifters_zone_traj[drifters_zone_traj['id'] == np.array(spurious_drifters[i])].index
        drifters_zone_traj.drop(ind1, inplace=True) #remove data points
    
    drifters_zone_traj=drifters_zone_traj.reset_index()  
    print(f'{drifters_zone_traj}')

    nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
    print(f'{nb_drifters2=}')
    #exit()
    #print(np.nanmin(np.array(nb_drifters2['lat'])))

    print('total nb of drifters in zone at start that reach end:',nb_drifters2)
    #Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
    nb_points = T_DRIFT*24   # hourly

    traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
    traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
    traj_drifter_date  = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
    traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
    traj_drifter_id    = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )   


    return traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id
