from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))
from read_drifters_nc import read_all_drifters_Phod
import datetime

version_drifter_database='2.01.1' # version of PhOD. Available choices are '1.02' or '1.04' or '2.01_experimental'
ln_drogue_off=True # select only buoys without drogues (lost drogue) 
'''
CHOICE OF EXPERIMENT
'''

freq_launch=3*24 # launch frequency (in hours)
total_time_drift=freq_launch # in hours

# select the global zone
lon_min=-180
lon_max=180
lat_min=-90
lat_max=90

experience_name = "Phod_database"
if (ln_drogue_off):
    experience_name=experience_name+'_undrogued'
else:
    experience_name=experience_name+'_drogued'
date_min=datetime.datetime(2024, 1, 1, 1, 0)
date_max=datetime.datetime(2025, 1, 1, 1, 0)

DIR_pickle_out = "../out/"
'''----------------------------
'''

'''
S T A R T
'''

'''
LOAD ALL WANTED DRIFTERS INTO A DICTIONNARY
'''
Drifters={}

date_max_string=str(date_max.year).zfill(4)+str(date_max.month).zfill(2)+str(date_max.day).zfill(2) 
date_min_string=str(date_min.year).zfill(4)+str(date_min.month).zfill(2)+str(date_min.day).zfill(2)

drifter_file=DIR_pickle_out+"/"+experience_name+"_drifters_"+date_min_string+'_'+date_max_string+".p"

if  (not os.path.exists(drifter_file)):
    print('Prepare driter database for '+experience_name)
    Drifters = read_all_drifters_Phod(date_min, date_max, lon_min, lon_max, lat_min, lat_max, ln_drogue_off)
    if (Drifters):
        print('save drifters database in '+drifter_file)
        with open(drifter_file, 'wb') as file:
            pickle.dump( Drifters, file, protocol=pickle.HIGHEST_PROTOCOL )
    else:
        print('No drifters were found for you dates, please enlarge period')
        sys.exit()
else:
    print('load drifters database from '+ drifter_file)
    with open(drifter_file, 'rb') as file:
        Drifters= pickle.load(file)
