import sys 
import gc
import os
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from configparser import ConfigParser
from data_io import Retrieve_data_local
from trajectory_io import Retrieve_traj_files, Retrieve_drifters_traj,\
                     Traj_filter_parcels, retrieve_drifters_for_one_day_traj
from parcels_tools import run_parcels_Drift
#from spatial_binning import bin_spatial, bin_spatial2
#from lagrangian_metrics import SeparationDistance,\
#                        SeparationDistance_havers,\
#                        SeparationDistance_havers2,\
#                        LiuScore2
from Logger import Logger
from utils import convert_id_to_char_array

def process_one_day(day, config_dict):
    """
    Process one initialization day for Lagrangian validation.

    This function:
    ---------------
    1. Retrieves observed drifter trajectories for a given day
    2. Creates initial particle positions from drifters
    3. Runs Parcels trajectory simulations
    4. Filters suspicious or invalid trajectories
    5. Computes separation distance and Liu skill score
    6. Computes daily averaged diagnostics
    7. Spatially bins statistics on a regular lon/lat grid
    8. Returns aggregated statistics for later multi-day accumulation

    Parameters
    ----------
    day : datetime
        Initialization day for drifter selection and trajectory launch

    day : leadtime
        Leadtime value for analysis or forecast fields HDCT or FCST

    DIR_OUT : str
        Output directory for temporary and Parcels trajectory files

    LON_MIN, LON_MAX : float
        Longitude bounds for drifter spatial selection

    LAT_MIN, LAT_MAX : float
        Latitude bounds for drifter spatial selection

    MODEL : str
         Ocean model configuration to process

    DIR_IN : str
        Input directory containing model velocity fields

    DIR_WORK : str
        Working directory for intermediate files

    DEPTH : float
        Initial particle depth

    ADVECTION_DIFFUSION_TYPE : str
        Parcels kernel configuration
        (e.g. advection only, diffusion, smagorinsky)

    NB_REPEATS : int or None
        Number of repeated launches per drifter position

    STR_REGION : str
        Name of the validation region

    DRIFT_TIME : int
        Total drift duration in days


    Returns
    -------
    dict
        Dictionary containing:

        - 'sum_sep' : accumulated separation distance
        - 'count_sep' : number of valid separation samples
        - 'sum_liu' : accumulated Liu skill score
        - 'count_liu' : number of valid Liu samples
    """

    import datetime
    import os
    import numpy as np

    # ======================================================
    # Get back the parameters from config file
    # ======================================================
    config_info  = config_dict['config_info']
    DIR_OUT = config_info["DIR_OUT"]
    LON_MIN = config_info["lon_min"]
    LON_MAX = config_info["lon_max"]
    LAT_MIN = config_info["lat_min"]
    LAT_MAX = config_info["lat_max"]
    MODEL = config_info["SYSTEM"]
    leadtime = config_info["leadtime"]
    DIR_IN = config_info["dir_in"]
    DIR_WORK = config_info["dir_work"]
    DRIFT_PATH = config_info["drift_path"]
    DEPTH = config_info["depth"]
    ADVECTION_DIFFUSION_TYPE = config_info["advection_diffusion_type"]
    NB_REPEATS = config_info["nb_repeats"]
    STR_REGION = config_info["STR_REGION"]
    DRIFT_TIME = config_info["drift_time"]
    loglevel = config_info["loglevel"]
    # Create Logger
    log = Logger("lagrangian",
                ).run(loglevel)

    # ======================================================
    # Output paths
    # ======================================================
    str_date_sim = day.strftime('%Y%m%d')
    srt_date_end = day + datetime.timedelta(days=int(DRIFT_TIME))
    srt_date_end = srt_date_end.strftime('%Y%m%d')
    # factor = 4  # 4 × 6h = 1 day

    dir_work_mod = f"{DIR_OUT}{str_date_sim}"
    if not os.path.exists(dir_work_mod):
        os.makedirs(dir_work_mod)

    # ======================================================
    # Retrieve drifters trajectories
    # ======================================================
    log.debug(f"Retrieve drifters for one day {day}")
    result = retrieve_drifters_for_one_day_traj(
        day,
        DRIFT_TIME,
        LON_MIN,
        LON_MAX,
        LAT_MIN,
        LAT_MAX,
        STR_REGION,
        str_date_sim,
        srt_date_end,
        DRIFT_PATH,
        DIR_OUT,
        log
        )
    #if trajs_obs is None:
    if result is None:
        log.info(f"{day.date()} → no valid drifters")
        return None
    else:
        trajs_obs, outfile2 = result

    log.debug(f"Number of valid drifters {len(trajs_obs)}")

    # ========================================================
    # Compute Lagrangian dispersion of the model with Parcels
    # ========================================================
    f_traj = f"{dir_work_mod}/Trajectories_{MODEL}_{str_date_sim}_{srt_date_end}_at_SVPloc_smag.zarr"
    if not os.path.exists(f_traj):
        #return 'skipped'
        log.debug(f"\nProcessing drift with parcel for {day}")
        # ======================================================
        # Initial conditions
        # ======================================================
        lon0 = np.array([traj['lon'][0] for traj in trajs_obs])
        lat0 = np.array([traj['lat'][0] for traj in trajs_obs])
        ids = [traj['id'] for traj in trajs_obs]
        #with open("/home/smer/regnierc/code/validation_tools_drifters/drifter_ids_model.txt", "w") as f:
        #    for did in ids:
        #        f.write(f"{did}\n")
        log.debug(f'Min {min(lon0)}')
        log.debug(f'Max {max(lon0)}')
        id_char = convert_id_to_char_array(ids, string_length=8)
        log.debug(np.shape(lon0))
        log.debug(np.shape(lat0))
        log.debug(np.shape(ids))
        # ======================================================
        # Retrieve model data
        # ======================================================
        DATE_END_PLUS_DRIFT_TIME = day + datetime.timedelta(days=DRIFT_TIME)

        u_files_mod_temp, v_files_mod_temp, var_u_mod, var_v_mod, fname_prefix, meshmask = \
            Retrieve_data_local(
                config_dict['config_model_params'],
                f'{MODEL}_{leadtime}',
                day,
                DATE_END_PLUS_DRIFT_TIME,
                DIR_OUT, DIR_IN, DIR_WORK,
                DEPTH, log, leadtime=leadtime
            )
        # ======================================================
        # Run Parcels
        # ======================================================
        run_parcels_Drift(
            config_info,
            MODEL,
            u_files_mod_temp, var_u_mod,
            v_files_mod_temp, var_v_mod,
            meshmask,
            lon0, lat0, id_char,
            DRIFT_TIME,
            f_traj,
            log,
            radius_cloud=0,
            seeding_resolution=0,
            time_period=False,
            nb_repeats=NB_REPEATS
        )
        del lon0, lat0, ids, id_char
        del u_files_mod_temp, v_files_mod_temp
        del meshmask
    else:
        log.info(f"\nExisting parcels trajectory {f_traj}")
    gc.collect()
    return outfile2
