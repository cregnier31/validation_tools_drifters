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
#from parcels_tools import run_parcels_Drift
from lagrangian_engine import run_lagrangian

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

    config_dict: is a dictionnary that contains the following parameters
        leadtime (int): Leadtime value for analysis or forecast fields HDCT or FCST
        DIR_OUT (str): Output directory for temporary and Parcels trajectory files
        LON_MIN, LON_MAX (float): Longitude bounds for drifter spatial selection
        LAT_MIN, LAT_MAX (float): Latitude bounds for drifter spatial selection
        MODEL (str): Ocean model configuration to process
        DIR_IN (str): Input directory containing model velocity fields
        DIR_WORK(str): Working directory for intermediate files
        DEPTH (float): Initial particle depth
        ADVECTION_DIFFUSION_TYPE (str) Parcels kernel configuration
            (e.g. advection only, diffusion, smagorinsky)
        NB_REPEATS (int or None): Number of repeated launches per drifter position
        STR_REGION (str): Name of the validation region
        DRIFT_TIME (int): Total drift duration in days

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
    DRIFT_PATH = config_info["path_data"]
    DEPTH = config_info["depth"]
    ADVECTION_DIFFUSION_TYPE = config_info["advection_diffusion_type"]
    NB_REPEATS = config_info["nb_repeats"]
    STR_REGION = config_info["STR_REGION"]
    DRIFT_TIME = config_info["drift_time"]
    obs_type = config_info["obs_type"]
    loglevel = config_info["loglevel"]
    engine = config_info["engine"]
    print(f"{engine=}")
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
        obs_type,
        DRIFT_PATH,
        DIR_OUT,
        log,
        start_hour=12 if MODEL.upper() == "GLOENS" and leadtime == "FCST" else 0,
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
    # Isolate opt-in geometry experiments from existing trajectory files.
    # The suffix also flows through subsequent validation and plotting paths.
    geometry_tag = os.environ.get("GLOENS_OUTPUT_TAG", "").strip()
    if geometry_tag and MODEL.upper() == "GLOENS":
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", geometry_tag):
            raise ValueError("GLOENS_OUTPUT_TAG must contain only letters, digits, _ or -")
        f_traj = f_traj[:-5] + f"_{geometry_tag}.zarr"
        log.warning("GLOENS isolated trajectory output tag: %s", geometry_tag)
    # Parcels 4 writes Parquet, while legacy Parcels 3 writes Zarr.
    # Reuse existing output rather than attempting to recreate it.
    f_traj_parquet = os.path.splitext(f_traj)[0] + ".parquet"
    is_gloens_parquet = (
        config_info.get("input_format", "netcdf").lower() == "zarr"
        and config_info.get("gridtype") == "irregular"
    )
    existing_traj = (
        f_traj_parquet if is_gloens_parquet and os.path.exists(f_traj_parquet)
        else f_traj if os.path.exists(f_traj)
        else None
    )
    if existing_traj is not None and config_info.get("remove_old_particules", False):
        # Remove only the exact trajectory artifact for this experiment/cycle.
        # Never touch cached velocities, observation inputs, or other cycles.
        import shutil
        if os.path.isdir(existing_traj):
            shutil.rmtree(existing_traj)
        else:
            os.remove(existing_traj)
        log.warning("Removed old trajectory output: %s", existing_traj)
        existing_traj = None
    if existing_traj is None:
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

        # Zarr transport is independent of ensemble selection and model identity.
        if config_info.get("input_format", "netcdf").lower() == "zarr":
            if engine.lower() != "parcels":
                raise NotImplementedError("Zarr velocity inputs currently require Parcels")
            from forecast_inputs import prepare_velocity_forecast, discover_gloens_stores
            if config_info.get("zarr_bucket"):
                log.debug("discover_gloens_stores")
                u_store, v_store, storage_options = discover_gloens_stores(
                    day,
                    bucket=config_info["zarr_bucket"],
                    endpoint_url=config_info["zarr_endpoint_url"],
                    prefix=config_info["zarr_prefix"],
                    log=log
                )
                log.debug(f'{u_store=} {v_store=}')
            else:
                u_store = config_info["zarr_u_store"]
                v_store = config_info.get("zarr_v_store") or u_store
                storage_options = None
            with prepare_velocity_forecast(
                # Daily fields are at noon: N samples span only N-1 days.
                # Load two additional timestamps so a full N-day RK4 run
                # has an interpolation guard beyond its final endpoint.
                u_store, v_store, day, days=int(DRIFT_TIME) + 2,
                storage_options=storage_options,
                mode=config_info["zarr_velocity_mode"],
                member=config_info["zarr_ensemble_member"],
                u_name=config_info["zarr_u_var"],
                v_name=config_info["zarr_v_var"],
                ensemble_dim=config_info["zarr_ensemble_dim"],
                depth=float(DEPTH),
                u_depth=config_info["zarr_u_depth"],
                v_depth=config_info["zarr_v_depth"],
                log=log,
                cache_dir=os.path.join(DIR_WORK, "zarr_ensemble_mean_cache"),
            ) as velocities:
                if config_info["gridtype"] == "irregular":
                    from gloens_parcels import run_gloens_parcels
                    log.debug(f'Run gloens_parcels')
                    run_gloens_parcels(
                        velocities, config_info["zarr_meshmask"],
                        day, lon0, lat0, f_traj, drift_days=DRIFT_TIME,
                        u_name=config_info["zarr_u_var"],
                        v_name=config_info["zarr_v_var"],
                        depth=float(DEPTH),
                        scheme=config_info["scheme"],
                        dt_minutes=config_info["parcels_dt_minutes"],
                        sampling_hours=config_info["parcels_sampling_hours"],
                        log=log,
                        cache_dir=os.path.join(DIR_WORK, "zarr_ensemble_mean_cache"),
                    )
                else:
                    from parcels_tools import run_parcels_Drift_remote
                    run_parcels_Drift_remote(
                        MODEL, velocities,
                        config_info["zarr_u_var"],
                        config_info["zarr_v_var"],
                        lon0, lat0, id_char, DRIFT_TIME, f_traj,
                        radius_cloud=0, seeding_resolution=0,
                        time_period=False, nb_repeats=NB_REPEATS,
                        AdvectionDiffusion_type=ADVECTION_DIFFUSION_TYPE,
                    )
        else:
            (u_files_mod_temp, v_files_mod_temp, var_u_mod, var_v_mod, fname_prefix, meshmask) = Retrieve_data_local(
                    config_dict['config_model_params'],
                    f'{MODEL}_{leadtime}',
                    day,
                    DATE_END_PLUS_DRIFT_TIME,
                    DIR_OUT, DIR_IN, DIR_WORK,
                    DEPTH, log, leadtime=leadtime
                )
            run_lagrangian(
                engine,
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
            del u_files_mod_temp, v_files_mod_temp, meshmask
        del lon0, lat0, ids, id_char
    else:
        log.info(f"\nExisting parcels trajectory {existing_traj}; skipping advection")
    gc.collect()
    return outfile2
