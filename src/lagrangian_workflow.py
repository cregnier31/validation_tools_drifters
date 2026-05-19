import sys 
import os
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from configparser import ConfigParser
from data_io import Retrieve_data_local
from trajectory_io import Retrieve_traj_files, Retrieve_drifters_traj,\
                     Traj_filter_parcels, retrieve_drifters_for_one_day_traj
from parcels_tools import run_parcels_Drift
from spatial_binning import bin_spatial, bin_spatial2
from lagrangian_metrics import SeparationDistance,\
                        SeparationDistance_havers,\
                        LiuScore2
from utils import convert_id_to_char_array

def process_one_day(day, leadtime, DIR_OUT,
                    T_DRIFT, LON_MIN,
                    LON_MAX,LAT_MIN,LAT_MAX,
                    MODEL,DIR_IN, DIR_WORK,
                    DEPTH,ADVECTION_DIFFUSION_TYPE,
                    NB_REPEATS, STR_REGION,
                    COPERNICUS_DRIFTER, DRIFT_TIME,
                    resolution=2,
                    ):
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

    T_DRIFT : int
        Total drift duration in days

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

    COPERNICUS_DRIFTER : str
        Path or identifier for drifter observations

    DRIFT_TIME : int
        Temporal sampling interval of drifter observations (hours)
    
    resolution : int
        Resolution of output grid for the statistics

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
    # Output paths
    # ======================================================
    str_date_sim = day.strftime('%Y%m%d')
    srt_date_end = day + datetime.timedelta(days=int(T_DRIFT))
    srt_date_end = srt_date_end.strftime('%Y%m%d')
    factor = 4  # 4 × 6h = 1 day
    resolution = resolution

    dir_work_mod = f"{DIR_OUT}{str_date_sim}"
    if not os.path.exists(dir_work_mod):
        os.makedirs(dir_work_mod)

    print(f"Retrieve drifters for one day {day}")
    trajs_obs, outfile2 = retrieve_drifters_for_one_day_traj(
        day,
        T_DRIFT,
        LON_MIN,
        LON_MAX,
        LAT_MIN,
        LAT_MAX,
        STR_REGION,
        str_date_sim,
        srt_date_end,
        DIR_OUT
        )
    
    if trajs_obs is None:
        print(f"{day.date()} → no valid drifters")
        return None

    print(f"Number of valid drifters {len(trajs_obs)}")

    f_traj = f"{dir_work_mod}/Trajectories_{MODEL}_{str_date_sim}_{srt_date_end}_at_SVPloc_smag.zarr"
    if not os.path.exists(f_traj):
        #return 'skipped'
        print(f"\nProcessing drift with parcel for {day}")
        # ======================================================
        # Initial conditions
        # ======================================================
        lon0 = np.array([traj['lon'][0] for traj in trajs_obs])
        lat0 = np.array([traj['lat'][0] for traj in trajs_obs])
        ids = [traj['id'] for traj in trajs_obs]
        #with open("/home/smer/regnierc/code/validation_tools_drifters/drifter_ids_model.txt", "w") as f:
        #    for did in ids:
        #        f.write(f"{did}\n")
        print(f'Min {min(lon0)}')
        print(f'Max {max(lon0)}')
        id_char = convert_id_to_char_array(ids, string_length=8)
    
        print(np.shape(lon0))
        print(np.shape(lat0))
        print(np.shape(ids))
 
        # ======================================================
        # Retrieve model data
        # ======================================================
        DATE_END_PLUS_DRIFT_TIME = day + datetime.timedelta(days=T_DRIFT)
    
        u_files_mod_temp, v_files_mod_temp, var_u_mod, var_v_mod, fname_prefix, meshmask = \
            Retrieve_data_local(
                f'{MODEL}_{leadtime}',
                day,
                DATE_END_PLUS_DRIFT_TIME,
                DIR_OUT, DIR_IN, DIR_WORK, DEPTH, leadtime=leadtime
            )
        # ======================================================
        # Run Parcels
        # ======================================================
        run_parcels_Drift(
            MODEL,
            u_files_mod_temp, var_u_mod,
            v_files_mod_temp, var_v_mod,
            meshmask,
            lon0, lat0, id_char,
            T_DRIFT,
            f_traj,
            radius_cloud=0,
            seeding_resolution=0,
            time_period=False,
            AdvectionDiffusion_type=ADVECTION_DIFFUSION_TYPE,
            nb_repeats=NB_REPEATS
        )
        
    # --------------------------------------------
    # Retrieve Drifters trajectories
    # --------------------------------------------
    print('--------------------------------------------')
    print('Retrieve drifters trajectories')
    traj_drifter_lon_temp,traj_drifter_lat_temp,traj_drifter_date_temp,\
        traj_drifter_time_temp,traj_drifter_id_temp,\
        nb_drifters_temp = Retrieve_drifters_traj(outfile2,T_DRIFT,COPERNICUS_DRIFTER)
    print(f'Taille drifters obs {np.shape(traj_drifter_lon_temp)}')
    # --------------------------------------------
    # Retrieve Model trajectories
    # --------------------------------------------
    print('Retrieve model trajectories')
    print('--------------------------------------------')
    traj_lon, traj_lat,traj_time,traj_IDS = Retrieve_traj_files(MODEL,dir_work_mod)
    #print(nb_drifters_temp, len(traj_IDS.squeeze()))
    if nb_drifters_temp  != len(traj_IDS.squeeze()):
        drifter_traj_ids = np.unique(traj_IDS)
        traj_drifter_lon  = []
        traj_drifter_lat  = []
        traj_drifter_date = []
        traj_drifter_time = []
        traj_drifter_id   = []
        for mm in range(nb_drifters_temp):
            if traj_drifter_id_temp[mm,0] in drifter_traj_ids:
                traj_drifter_lon.append(traj_drifter_lon_temp[mm,:])
                traj_drifter_lat.append(traj_drifter_lat_temp[mm,:])
                traj_drifter_date.append(traj_drifter_date_temp[mm,:])
                traj_drifter_time.append(traj_drifter_time_temp[mm,:])
                traj_drifter_id.append(traj_drifter_id_temp[mm,:])
        traj_drifter_lon = np.array(traj_drifter_lon)
        traj_drifter_lat = np.array(traj_drifter_lat)
        traj_drifter_date = np.array(traj_drifter_date)
        traj_drifter_time = np.array(traj_drifter_time)
        traj_drifter_id = np.array(traj_drifter_id)
    else:
        print('inside else')
        traj_drifter_lon  = traj_drifter_lon_temp
        traj_drifter_lat  = traj_drifter_lat_temp
        traj_drifter_date = traj_drifter_date_temp
        traj_drifter_time = traj_drifter_time_temp
        traj_drifter_id   = traj_drifter_id_temp
    
    print('------------------OK------------------------')
    print(f'Taille drifters observation {np.shape(traj_drifter_lon)}')
    # print(traj_drifter_lon))
    print('--------------------------------')
    print(f'Taille drifters Modele Parcel raw {np.shape(traj_lon)}')
    if(np.prod(np.shape(traj_drifter_lon)) >0):
        # Reduce the oversampled trajectories of observations to compare with model trajectories        
        drifter_lon,drifter_lat,drifter_date,drifter_time, Tm, Nd, Td = Traj_filter_parcels(DRIFT_TIME,traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time, traj_lon,traj_lat)
        print(f"Dimensions observations subsampled raw : {np.shape(drifter_lon)}")
        print(f'Dimensions model Parcels : {np.shape(traj_lon)}')
        # Change to 0-360
        traj_lon = ((traj_lon + 180) % 360) - 180
        drifter_lon = ((drifter_lon + 180) % 360) - 180
        # ---------------------------------------------------------
        # Select suspicious trajectories by fixing a big theshold
        # ---------------------------------------------------------
        threshold = 300.00  # km
        dist_first_day = SeparationDistance_havers(
            drifter_lon[:, :, :, :4],
            drifter_lat[:, :, :, :4],
            traj_lon[:, :, :, :4],
            traj_lat[:, :, :, :4]
        )
        maxdist = np.nanmax(dist_first_day, axis=-1)
        bad_traj = np.squeeze(maxdist > threshold)
        nb_bad = np.sum(bad_traj)
        print(f"Number of suspicious trajectories: {nb_bad}")
        # ------------------------------------------------------    
        # Filter only if suspicious trajectories exist
        # ------------------------------------------------------    
        if nb_bad > 0:
            #mask = dist0 > threshold
            good_idx = np.where(~bad_traj)[0]
            print(f"Number of suspicious trajectories: {np.sum(bad_traj)}")
            print("Number of good trajectories:", len(good_idx))
            print(np.shape(good_idx))
            print(np.shape(drifter_lon))
            print(np.shape(drifter_lat))
            print('---------------------------------')
            
            traj_lon = traj_lon[:, :, good_idx, :]
            traj_lat = traj_lat[:, :, good_idx, :]
            drifter_lon = drifter_lon[:, :, good_idx, :]
            drifter_lat = drifter_lat[:, :, good_idx, :]
            
            print(np.shape(drifter_lon))
            print(np.shape(drifter_lat))
            print('********************************************')
            ## ---------------------------------------------------------
            ## Build dataframe
            ## ---------------------------------------------------------
            #df = pd.DataFrame({
            #    "lon_drift": lon_drift0,
            #    "lat_drift": lat_drift0,
            #    "lon_model": lon_model0,
            #    "lat_model": lat_model0,
            #    "distance_km": dist0,
            #})
            #
            ## optional trajectory id
            #df["trajectory"] = np.where(mask)[0]
            #
            ## reorder columns
            #df = df[
            #    [
            #        "trajectory",
            #        "lon_drift",
            #        "lat_drift",
            #        "lon_model",
            #        "lat_model",
            #        "distance_km",
            #    ]
            #]
            #
            ## ---------------------------------------------------------
            ## Save CSV
            ## ---------------------------------------------------------
            #outfile = "suspicious_trajectories_first_step.csv"
            #
            #df.to_csv(outfile, index=False)
            #print(f"Saved {outfile}")
        # --------------------------------------------------------------------
        # Compute statistical scores ; Separation distance and Liu Skill score
        # --------------------------------------------------------------------
        sep_dist = SeparationDistance_havers(drifter_lon,drifter_lat,traj_lon,traj_lat)
        # remove extra dims → (1019, 40)
        lon2d = traj_lon[0, 0, :, :]
        lat2d = traj_lat[0, 0, :, :]
        Nd, Tm = lon2d.shape
        Ndays = Tm // factor
        sep_dist_daily = sep_dist.reshape(Nd, Ndays, factor).mean(axis=2)
        # reshape + mean → (1019, 10)
        traj_lon_daily_tmp = lon2d[:, :Ndays*factor].reshape(Nd, Ndays, factor).mean(axis=2)
        traj_lat_daily_tmp = lat2d[:, :Ndays*factor].reshape(Nd, Ndays, factor).mean(axis=2)
        # put back original structure → (1, 1, 1019, 10)
        traj_lon_daily = traj_lon_daily_tmp[np.newaxis, np.newaxis, :, :]
        traj_lat_daily = traj_lat_daily_tmp[np.newaxis, np.newaxis, :, :]
        #print(sep_dist)
        #print(drifter_time)
        #print(np.shape(sep_dist_daily))
        #print(sep_dist_daily)
        #print(traj_lon.shape[0])
    
        liu_score = np.array([[LiuScore2(drifter_lon[i,j],drifter_lat[i,j],traj_lon[i,j],traj_lat[i,j]).T \
                    for j in range(traj_lon.shape[1])] for i in range (traj_lon.shape[0])])
        # Add missing value for the first step
        liu_score = np.insert(liu_score, 0, np.nan, axis=-1)
        Nt = liu_score.shape[-1]
        
        # reshape
        newshape = liu_score.shape[:-1] + (Nt // factor, factor)

        liu_daily = np.nanmean(
            liu_score.reshape(newshape),
            axis=-1
        )
        #Nt = (liu_score.shape[-1] // factor) * factor
        #liu_trim = liu_score[..., :Nt]
        ## reshape
        #newshape = liu_trim.shape[:-1] + (Nt // factor, factor)
        #liu_daily = liu_trim.reshape(newshape).mean(axis=-1)
        ## reshape daily
        #liu_score_daily = liu_score.reshape(Nd, Ndays, factor).mean(axis=2)
        #print(liu_score)
        #print(np.shape(liu_score))
        #print(np.shape(liu_daily))
        #print(liu_daily)
        #print('----------------------') 
        # Compute skill score with the assumption of n=1
        skill_score = 1-liu_daily
        #print(skill_score)
        
    if True:# np.prod(np.shape(np.where(np.isnan(traj_lon)))):
        nans = np.unique(np.where(np.isnan(traj_lon))[2])
        if len(nans)>0:
            traj_lon = np.delete(traj_lon,nans,axis = 2)
            traj_lat = np.delete(traj_lat,nans,axis = 2)
            drifter_lon = np.delete(drifter_lon,nans,axis = 2)
            drifter_lat = np.delete(drifter_lat,nans,axis = 2)
        nans = np.unique(np.where(np.isnan(drifter_lon))[2])
        if len(nans)>0:
            traj_lon = np.delete(traj_lon,nans,axis = 2)
            traj_lat = np.delete(traj_lat,nans,axis = 2)
            drifter_lon = np.delete(drifter_lon,nans,axis = 2)
            drifter_lat = np.delete(drifter_lat,nans,axis = 2)
    
    
    corr_lon = np.array([[[pearsonr(drifter_lon[i,j,k],traj_lon[i,j,k])[0] \
        for k in range(traj_lon.shape[2])] for j in range(traj_lon.shape[1])] for i in range(traj_lon.shape[0])])
    corr_lat = np.array([[[pearsonr(drifter_lat[i,j,k],traj_lat[i,j,k])[0] \
        for k in range(traj_lat.shape[2])] for j in range(traj_lat.shape[1])] for i in range(traj_lat.shape[0])])
    
    
    if np.prod(np.shape(sep_dist))>0:
        prop_sep_dist = np.count_nonzero(sep_dist[:,:,:,-1]<50 ,axis =(1,2) )/ (sep_dist.shape[1]*sep_dist.shape[2])
        prop_skill_score = np.count_nonzero(skill_score[:,:,:,-1]>0 ,axis =(1,2) )/ (skill_score.shape[1]*skill_score.shape[2])
        prop_corr_lon = np.count_nonzero(corr_lon>0.7 ,axis =(1,2) )/ (corr_lon.shape[1]*corr_lon.shape[2])
        prop_corr_lat = np.count_nonzero(corr_lat>0.7 ,axis =(1,2) )/ (corr_lat.shape[1]*corr_lat.shape[2])
    
    # -----------------------------------------------------
    # Spatially bins statistics on a regular lon/lat grid
    # -----------------------------------------------------
    mean_sep, count_sep = bin_spatial(sep_dist_daily, traj_lon_daily.squeeze(), traj_lat_daily.squeeze(), res=resolution)
    sum_sep = mean_sep * count_sep
    lui_score_bin, count_liu_score = bin_spatial(skill_score.squeeze(), traj_lon_daily.squeeze(), traj_lat_daily.squeeze(), res=resolution)
    sum_liu = lui_score_bin * count_liu_score
    print(f"Done {day}")
    return {
    "sum_sep": sum_sep,
    "count_sep": count_sep,
    "sum_liu": sum_liu,
    "count_liu": count_liu_score
}
