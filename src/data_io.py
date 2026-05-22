from configparser import ConfigParser
from glob import glob
import os
import sys

def Retrieve_data_local(MODEL, 
                        date_start,
                        date_end_plus_DRIFT_TIME,
                        dir_out,
                        dir_in,
                        dir_work,
                        depth,
                        leadtime=None):
    config_object = ConfigParser()
    config_object.read(dir_in+"src/config_products_fidji.ini")
    path_UV = []
    fname_prefix = []
    var_u_mod = []
    var_v_mod = []
    meshmask = []
    dir_work_mod = []
    #u_files_mod = []
    #v_files_mod = []
    colors = []

    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      RETRIEVE U,V, SSH FIELDS")
    print ("#-----------------------------------------------------------------------------------------")

    if MODEL not in config_object.sections():
        print(MODEL, "is either a typo or not yet tuned for this program")
        print('model runs available: ',config_object.sections())
    print(MODEL)
    print(config_object)
    print ("*******************************************************************")
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
        print("OK1")
        u_files_mod_3D, v_files_mod_3D = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME, leadtime=leadtime)
        u_files, v_files = ExtractSurface(MODEL,u_files_mod_3D, v_files_mod_3D,zlevel,var_u_mod[i], var_v_mod[i], dir_work+'MODEL_OUTPUT/'+MODEL+'/UV/15m/')
        u_files_mod.append(u_files)
        v_files_mod.append(v_files)
    else : 
        print("OK2")
        print(MODEL, type(date_start), type(date_end_plus_DRIFT_TIME))
        u_files_mod, v_files_mod = ReturnUVlist(MODEL, date_start, date_end_plus_DRIFT_TIME,dir_in, leadtime=leadtime)
        #u_files_mod.append(u_files)
        #v_files_mod.append(v_files)

    
    print('path_UV: ', path_UV[0])
    print('fname_prefix: ', fname_prefix[0])
    return u_files_mod,v_files_mod,var_u_mod,var_v_mod,fname_prefix,meshmask


def ReturnUVlist(MODEL, date_start, date_end, dir_in=None, leadtime=None):

    # return U, V files within date range:

    print ("#-----------------------------------------------------------------------------------------")
    print ("#                      READ IN FIELDSET")
    print ("#-----------------------------------------------------------------------------------------")

    print('script ReturnUVlist')
    print('model :', MODEL)
    print('date_start :', date_start)
    print('date_end :', date_end)
    print('dir_in :', dir_in)
    #reconstruct date1
    dateval = date_start.strftime("%Y%m%d")
    DD1=date_start.day
    MM1=date_start.month
    YYYY1=date_start.year 

    DD2=date_end.day
    MM2=date_end.month
    YYYY2=date_end.year

    config_object = ConfigParser()
    config_object.read(dir_in+"src/config_products.ini")


    if MODEL not in config_object.sections():
        print(MODEL, "is either a typo or not yet tuned for this program")
        print('model runs available: ',config_object.sections())
        sys.exit(1)

    run_info = config_object[MODEL]
    fname_prefix=run_info["fname_prefix"]
    #path to files 
    path_UV=run_info["path"]

    print('path_UV: ', path_UV)
    print('fname_prefix: ', fname_prefix)
    print('identify U,V :')
    print(f'MODEL: {MODEL}')
    print(f'YYYY1: {YYYY1}')
    print(f'YYYY2: {YYYY2}')
    print(f'Start date : {date_start}')
    print('-------------------------')
    list_years = range (YYYY1, YYYY2+1, 1)
    print('years=', list_years)
    
    u_files,v_files, ssh_files= [],[],[]  # initialise arryas
    for i in list_years:
        print (i)
        if MODEL == 'GLOBC':
            # e.g. /dataref/opr/DATA/OCEAN_CURRENTS/REP/L4/MULTIOBS_GLO_PHY_REP_015_004/dataset-uv-rep-daily/2019/dataset-uv-rep-daily_20191009T1200Z_P20200521T0000.nc
            # nrt_global_allsat_phy_l4_20181224_20181230.nc
            u_files += sorted(glob(path_UV+'/*/' + fname_prefix+ str(i)+'*.nc'))
            v_files += sorted(glob(path_UV+'/*/' + fname_prefix+ str(i)+'*.nc'))
            
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)

        elif MODEL == 'ORCA12_RBB':
            print('got here')
            #e.g. /sortref/ose-2/btchonang/GLORYS12V1/GLO12OSSE/NatRun/ORCA12-TRBB36s003b_1d_grid_T_20141120-20141120.nc
            u_files += sorted(glob(path_UV +'/'+ fname_prefix+'grid_U_'+str(i)+'*.nc'))
            v_files += sorted(glob(path_UV+'/' + fname_prefix+ 'grid_V_'+str(i)+'*.nc'))
            
            #provide how file starts to identify it in list
            f_input_start = str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)

        elif MODEL == 'GLO12V4':
            # GLO12: /sortref/opr/diag/GLO12/GLO12V4/data/Extract_UV/GLO12_ng_1d-m_20200101-20200101_3DU-uo_rtr_R20200108.nc
            u_files += sorted(glob(path_UV +'/'+ fname_prefix+str(i)+'*3DU-uo_*.nc'))
            v_files += sorted(glob(path_UV +'/'+ fname_prefix+str(i)+'*3DV-vo_*.nc'))
            
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)
        elif MODEL == 'IBI36':
            print(path_UV+fname_prefix+'*gridU_R*.nc')
            u_files += sorted(glob(path_UV+fname_prefix+'*gridU_R*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*gridV_R*.nc'))
            f_input_start = fname_prefix + str(YYYY1)+format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') + format(DD2,'02d')

        elif MODEL == 'SMOC' or MODEL == 'SMOC_GLO12':
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) +format(MM2,'02d') + format(DD2,'02d')

        elif MODEL == 'IBI8':
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*_interp_resol8.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_interp_resol8.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1)+format(MM1,'02d')+ format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2)+format(MM2,'02d')+ format(DD2,'02d')

        elif MODEL == 'GLO8':
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif MODEL == 'IBI4' or MODEL == 'IBI12':
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif MODEL == "PSY4":
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
        
        elif "GLO12_FCST" in MODEL:
            ## Recode this
            #print(path_UV+'fcst*/'+fname_prefix+'*3DU*.nc')
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
            print('inside GLO12_FCST')
            #f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            #f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')
            #f_input_start = 'fcst-1_R' + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d') + '.nc'
            #f_input_end = 'fcst-1_R' + str(YYYY2) + format(MM2,'02d') + format(DD2,'02d') + '.nc'
            print(f_input_start)
            print(f_input_end)
            
        elif "GLO12_NWCT" in MODEL or "GLO12_HDCT" in MODEL:
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*3DU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*3DV*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif "PSY4_FCST" in MODEL:
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        elif "PSY4_NWCT" in MODEL:
            print(path_UV+fname_prefix)
            u_files += sorted(glob(path_UV+fname_prefix+'*_gridU*.nc'))
            v_files += sorted(glob(path_UV+fname_prefix+'*_gridV*.nc'))
            print(u_files)
            f_input_start = fname_prefix + str(YYYY1) + format(MM1,'02d') + format(DD1,'02d')
            f_input_end = fname_prefix + str(YYYY2) + format(MM2,'02d') +format(DD2,'02d')

        else :
            print('Inside else')
            # e.g. /sortref/opr/pgn/PSY4/psy4v3r1/2006/10/PSY4V3R1_1dAV_20061011_20061012_gridU_R20061018.nc
            u_files += sorted(glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridU_*.nc'))
            v_files += sorted(glob(path_UV+'/*/*/' + fname_prefix+ str(i)+'*gridV_*.nc'))
            
            
            #provide how file starts to identify it in list
            f_input_start =fname_prefix + str(YYYY1)+format(MM1,'02d')+format(DD1,'02d')
            f_input_end =fname_prefix + str(YYYY2)+format(MM2,'02d')+format(DD2,'02d')
            print('f_input_start=', f_input_start)
            print('f_input_end=', f_input_end)


    #ssh_files=ssh_files.sort() 

    if leadtime == 'HDCT':
        print('f_input_start',f_input_start)
        print('f_input_end',f_input_end)

        print('UV field, identify start date:')
        print('nb files in Ulist = ', len(u_files))
        counter=0
        for line in u_files:
            if f_input_start in line:
               pos1u = counter 
               print ('    ',f_input_start, ' in line ', pos1u,' :', line)        
            counter+=1

        
        print('UV field, identify end date')
        counter=0
        for line in u_files:
            print(line)
            print(f_input_end)
            if f_input_end in line:
               pos2u = counter
               print ('    ',f_input_end, ' in line ', pos2u,' :', line)
            counter+=1

        counter=0
        for line in v_files:
            if f_input_start in line:
               pos1v = counter
               print ('    ',f_input_start, ' in line ', pos1v,' :', line)
            counter+=1

        #sys.exit(1)

        print('UV field, identify end date')
        counter=0
        for line in v_files:
            if f_input_end in line:
               pos2v = counter
               print ('    ',f_input_end, ' in line ', pos2v,' :', line)
            counter+=1


        #sys.exit(1)
        # clear up paths to previous dayly outputs  from list:
        if 'pos2u' in locals():
            del(u_files[pos2u+1:])
        if 'pos1u' in locals():
            del(u_files[:pos1u])

        if 'pos2v' in locals():
            del(v_files[pos2v+1:])
        if 'pos1v' in locals():
            del(v_files[:pos1v])


    print('nb files in ulist = ', len(u_files))
    print('nb files in vlist = ', len(v_files))
    print('nb files in ssh list = ', len(ssh_files))
    print(u_files)
    print('--------------------')
    return u_files, v_files
