from configparser import ConfigParser
from glob import glob
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta

def Retrieve_data_local(config_object,
                        MODEL,
                        date_start,
                        date_end_plus_DRIFT_TIME,
                        dir_out,
                        dir_in,
                        dir_work,
                        depth,
                        log,
                        leadtime=None):
    #config_object = ConfigParser()
    #config_object.read(dir_in+"cfg/config_products.ini")
    path_UV = []
    fname_prefix = []
    var_u_mod = []
    var_v_mod = []
    meshmask = []
    dir_work_mod = []
    #u_files_mod = []
    #v_files_mod = []
    colors = []

    log.debug("#-----------------------------------------------------------------------------------------")
    log.debug("#                      RETRIEVE U,V, SSH FIELDS")
    log.debug("#-----------------------------------------------------------------------------------------")

    available_models = list(config_object.keys())
    if MODEL not in available_models:
        log.error(f'{MODEL} is either a typo or not yet tuned for this program')
        log.error(f'model runs available: {available_models}')
    log.debug(f"Model {MODEL}")
    log.debug ("*******************************************************************")
    run_info = config_object[MODEL]
    path_UV.append(run_info["path"])
    fname_prefix.append(run_info["fname_prefix"])
    var_u_mod.append(run_info["var_u"])
    var_v_mod.append(run_info["var_v"])
    meshmask.append(run_info["meshmask"])
    colors.append(run_info["color"])
    #dir_work_mod.append(dir_out+'/EXP/'+MODEL+'/')
    #if not os.path.exists(dir_work_mod[-1]):
    #    print('\ncreate folder : ', dir_work_mod[-1])
    #    # Create directory where to store results:
    #    os.makedirs(dir_work_mod[-1])

    if MODEL in ['PSY4V3R1','IBI36','GLO12V4']:
        zlevel = 11
        u_files_mod_3D, v_files_mod_3D = ReturnUVlist(config_object, MODEL, date_start, date_end_plus_DRIFT_TIME, leadtime=leadtime)
        u_files, v_files = ExtractSurface(MODEL,u_files_mod_3D, v_files_mod_3D,zlevel,var_u_mod[i], var_v_mod[i], dir_work+'MODEL_OUTPUT/'+MODEL+'/UV/15m/')
        u_files_mod.append(u_files)
        v_files_mod.append(v_files)
    else :
        u_files_mod, v_files_mod = ReturnUVlist(config_object, MODEL, date_start,
                                                date_end_plus_DRIFT_TIME,
                                                log, dir_in, leadtime=leadtime)

    log.debug(f'path_UV: {path_UV[0]}')
    log.debug(f'fname_prefix: {fname_prefix[0]}')
    return u_files_mod,v_files_mod,var_u_mod,var_v_mod,fname_prefix,meshmask


def ReturnUVlist(config_object, MODEL, date_start, date_end, log, dir_in=None, leadtime=None):

    # return U, V files within date range:

    log.debug("#-----------------------------------------------------------------------------------------")
    log.debug("#                      READ IN FIELDSET")
    log.debug("#-----------------------------------------------------------------------------------------")

    log.debug('script ReturnUVlist')
    log.debug(f'model : {MODEL}')
    log.debug(f'date_start : {date_start}')
    log.debug(f'date_end : {date_end}')
    log.debug(f'dir_in : {dir_in}')
    #reconstruct date1
    dateval = date_start.strftime("%Y%m%d")
    DD1=date_start.day
    MM1=date_start.month
    YYYY1=date_start.year

    DD2=date_end.day
    MM2=date_end.month
    YYYY2=date_end.year

    #config_object = ConfigParser()
    #config_object.read(dir_in+"src/config_products.ini")


    available_models = list(config_object.keys())
    if MODEL not in available_models:
        log.error(f'{MODEL} is either a typo or not yet tuned for this program')
        log.error(f'model runs available: {available_models}')
        sys.exit(1)

    run_info = config_object[MODEL]
    fname_prefix=run_info["fname_prefix"]
    #path to files
    path_UV=run_info["path"]

    log.debug(f'path_UV: {path_UV})')
    log.debug(f'fname_prefix: {fname_prefix}')
    log.debug('identify U,V :')
    log.debug(f'MODEL: {MODEL}')
    log.debug(f'YYYY1: {YYYY1}')
    log.debug(f'YYYY2: {YYYY2}')
    log.debug(f'Start date : {date_start}')
    log.debug('-------------------------')
    list_years = range (YYYY1, YYYY2+1, 1)
    log.debug(f'years = {list_years}')

    u_files,v_files, ssh_files= [],[],[]  # initialise arryas
    for i in list_years:
        if MODEL == 'GLOBCURR_HDCT':
            # e.g. /dataref/opr/DATA/OCEAN_CURRENTS/REP/L4/MULTIOBS_GLO_PHY_REP_015_004/dataset-uv-rep-daily/2019/dataset-uv-rep-daily_20191009T1200Z_P20200521T0000.nc
            # nrt_global_allsat_phy_l4_20181224_20181230.nc
           # u_files += sorted(glob(path_UV+'/' + fname_prefix+ str(i)+'*.nc'))
           # v_files += sorted(glob(path_UV+'/' + fname_prefix+ str(i)+'*.nc'))
            print(f"{fname_prefix}{i}*.nc")
            print(f"{path_UV}/")
            files = list(Path(f"{path_UV}/").glob(f"{fname_prefix}{i}*.nc"))
            files_by_day = defaultdict(list)
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            date_start  = str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            date_end = str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            date_list = []
            # convert strings → date objects
            d = date(int(date_start[0:4]), int(date_start[4:6]), int(date_start[6:8]))
            end = date(int(date_end[0:4]), int(date_end[4:6]), int(date_end[6:8]))
            while d <= end:
                date_list.append(d.strftime("%Y%m%d"))
                d += timedelta(days=1)
        elif MODEL == 'ORCA12_RBB':
            #e.g. /sortref/ose-2/btchonang/GLORYS12V1/GLO12OSSE/NatRun/ORCA12-TRBB36s003b_1d_grid_T_20141120-20141120.nc
            u_files += sorted(glob(path_UV +'/'+ fname_prefix+'grid_U_'+str(i)+'*.nc'))
            v_files += sorted(glob(path_UV+'/' + fname_prefix+ 'grid_V_'+str(i)+'*.nc'))
            #provide how file starts to identify it in list
            f_input_start = str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')

        elif MODEL == 'GLO12V4' or MODEL == 'GLO12V4_TEST5' or MODEL == 'GLO12V4_TEST8':
            # GLO12: /sortref/opr/diag/GLO12/GLO12V4/data/Extract_UV/GLO12_ng_1d-m_20200101-20200101_3DU-uo_rtr_R20200108.nc
            u_files += sorted(glob(path_UV +'/'+ fname_prefix+str(i)+'*3DU-uo_*.nc'))
            v_files += sorted(glob(path_UV +'/'+ fname_prefix+str(i)+'*3DV-vo_*.nc'))
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            log.debug(f'f_input_start= {f_input_start}')
            log.debug(f'f_input_end= {f_input_end}')
        elif MODEL == 'IBI36':
            u_files += sorted(glob(path_UV+fname_prefix+'*gridU_R*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*gridV_R*.nc'))
            f_input_start = fname_prefix + str(YYYY1)+format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') + format(DD2,'02d')

        elif MODEL == 'SMOC_HDCT':
            log.debug(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) +format(MM2,'02d') + format(DD2,'02d')
        elif MODEL == 'IBI8':
            u_files += sorted(glob(path_UV+fname_prefix+'*_interp_resol8.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_interp_resol8.nc'))
            f_input_start = fname_prefix + str(YYYY1)+format(MM1,'02d')+ format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2)+format(MM2,'02d')+ format(DD2,'02d')

        elif MODEL == 'GLO8':
            u_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif MODEL == 'IBI4' or MODEL == 'IBI12':
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif MODEL == "PSY4":
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
        elif "GLO12_FCST" in MODEL:
            # Recode this
            log.debug(path_UV+'fcst*/'+fname_prefix+'*3DU*.nc')
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
            u_files += (
                        sorted(glob(f'{path_UV}nwct/{fname_prefix}*3DU*R{dateval}.nc')) +
                        sorted(glob(f'{path_UV}fcst*/{fname_prefix}*3DU*R{dateval}.nc'))
                      )

            v_files += (
                        sorted(glob(f'{path_UV}nwct/{fname_prefix}*3DV*R{dateval}.nc')) +
                        sorted(glob(f'{path_UV}fcst*/{fname_prefix}*3DV*R{dateval}.nc'))
                      )
            #u_files += sorted(glob(f'{path_UV}*/{fname_prefix}*3DU*R{dateval}.nc'))
            #v_files += sorted(glob(f'{path_UV}*/{fname_prefix}*3DV*R{dateval}.nc'))
            #f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            #f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
            #f_input_start = 'fcst-1_R' + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d') + '.nc'
            #f_input_end = 'fcst-1_R' + str(YYYY2) + format(MM2,'02d') + format(DD2,'02d') + '.nc'
            log.debug(f_input_start)
            log.debug(f_input_end)
        elif "GLO12_NWCT" in MODEL or "GLO12_HDCT" in MODEL or "GLO12_TEST5_HDCT" in MODEL or "GLO12_TEST8_HDCT" in MODEL:
            u_files += sorted(glob(path_UV+fname_prefix+'*3DU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*3DV*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif "PSY4_FCST" in MODEL:
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif "PSY4_NWCT" in MODEL:
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        else :
            # e.g. /sortref/opr/pgn/PSY4/psy4v3r1/2006/10/PSY4V3R1_1dAV_20061011_20061012_gridU_R20061018.nc
            u_files += sorted(glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridU_*.nc'))
            v_files += sorted(glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridV_*.nc'))
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')


    #ssh_files=ssh_files.sort()

    if leadtime == 'HDCT' and MODEL != 'GLOBCURR_HDCT':
        log.debug(f'f_input_start {f_input_start}')
        log.debug(f'f_input_end {f_input_end}')
        log.debug('UV field, identify start date:')
        log.debug(f'nb files in Ulist =  {len(u_files)}')
        log.debug(f'nb files in Vlist =  {len(v_files)}')
        counter=0
        for line in u_files:
            if f_input_start in line:
               pos1u = counter
               log.debug(f'    {f_input_end}  in line {pos1u} : {line}')
            counter+=1

        log.debug(f'U field, identify end date')
        counter=0
        for line in u_files:
            if f_input_end in line:
               pos2u = counter
               log.debug(f'    {f_input_end}  in line {pos2u} : {line}')
            counter+=1

        counter=0
        log.debug(f'U field, identify end date')
        for line in v_files:
            if f_input_start in line:
               pos1v = counter
               log.debug(f'    {f_input_start}  in line {pos1v} : {line}')
            counter+=1
        log.debug(f'V field identify end date')

        #sys.exit(1)

        log.debug('UV field, identify end date')
        counter=0
        for line in v_files:
            if f_input_end in line:
               pos2v = counter
               log.debug(f'    {f_input_end}  in line {pos2v} : {line}')
            counter+=1
        log.debug(f'V field, identify end date')


        # clear up paths to previous dayly outputs  from list:
        if 'pos2u' in locals():
            del(u_files[pos2u+1:])
        if 'pos1u' in locals():
            del(u_files[:pos1u])

        if 'pos2v' in locals():
            del(v_files[pos2v+1:])
        if 'pos1v' in locals():
            del(v_files[:pos1v])

    else:
        print('case globcurrent')
        day_files = defaultdict(list)
        for f in files:
            name = f.name
            try:
                left, right = name.split("_", 1)
                day = right[:8]  # YYYYMMDD
                p_date = right.split("_P")[1][:8]
                #files_by_day[day].append((p_date, f))
                # keep only in range
                if date_start <= day <= date_end:
                    day_files[day].append((p_date, f))
            except Exception as e:
                continue
        # keep ONLY ONE file per day (latest P)
        one_file_per_day = [
            max(lst, key=lambda x: x[0])[1]
            for lst in day_files.values()
        ]
        # optional: sort by day
        one_file_per_day = sorted(one_file_per_day)
        u_files = one_file_per_day
        v_files = one_file_per_day
    log.debug(f'nb files in ulist = {len(u_files)}')
    log.debug(f'nb files in vlist =  {len(v_files)}')
    log.debug(f'nb files in ssh list = {len(ssh_files)}')
    #log.debug(u_files)
    log.debug('--------------------')
    return u_files, v_files
