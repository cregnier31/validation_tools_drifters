import sys 
import os
import numpy as np
import xarray as xr
from trajectory_io import Retrieve_traj_files, Retrieve_traj_parquet, Retrieve_drifters_traj,\
                     Traj_filter_parcels
from spatial_binning import bin_spatial, bin_spatial_weighted
from lagrangian_metrics import SeparationDistance,\
                        SeparationDistance_havers,\
                        SeparationDistance_havers2,\
                        LiuScore2
from Logger import Logger
import datetime


def read_traj_compute_scores(day,
                             config_info,
                             filename_drift,
                             filtered=True,
                             resolution=2):
    """

    For each day retrieve drifter and model trajectories and compute the
    following scores Lagrangian distance, Liu score...

    Parameters
    ----------
    resolution : int
        Resolution of output grid for the statistics
    filtered = True # logical filter out suspicious drifters
    """
    # Create Logger
    loglevel = config_info["loglevel"]
    log = Logger("lagrangian_postprocess",
                ).run(loglevel)
    MODEL = config_info["SYSTEM"]
    dir_out_parcels = config_info["DIR_OUT"]
    leadtime = config_info["leadtime"]
    DIR_WORK = config_info["dir_work"]
    #DRIFT_PATH = config_info["drift_path"]
    #DRIFT_PATH = config_info["path_data"]
    #COPERNICUS_DRIFTER = config_info["copernicus_drifter"]
    DRIFTER_type = config_info["obs_type"]
    T_DRIFT = config_info["drift_time"]
    sampling = config_info["parcels_sampling_hours"]

    # ======================================================
    # Test if the output file exist
    # ======================================================
    if filename_drift is None:
        log.error("Skipped missing drifters")
        return None

    # ======================================================
    # Output paths
    # ======================================================
    str_date_sim = day.strftime('%Y%m%d')
    srt_date_end = day + datetime.timedelta(days=int(T_DRIFT))
    srt_date_end = srt_date_end.strftime('%Y%m%d')
    dir_work_mod = f"{dir_out_parcels}{str_date_sim}"
    threshold = 500.00  # threshold in km to filter out suspicious drifters
    save = False # Save suspicious drifter paths
    if sampling == 1:
        factor = 1
    elif sampling == 6:
        factor = 4
    elif sampling == 12:
        factor = 2
    #if MODEL == "SMOC":
    #    factor = 24
    #else:
    #    factor = 4  # 4 × 6h = 1 day

    # --------------------------------------------
    # Retrieve Drifters trajectories
    # --------------------------------------------
    log.debug('--------------------------------------------')
    log.debug('Retrieve drifters trajectories')
    traj_drifter_lon_temp,traj_drifter_lat_temp,traj_drifter_date_temp,\
        traj_drifter_time_temp,traj_drifter_id_temp,\
        nb_drifters_temp = Retrieve_drifters_traj(filename_drift,\
        T_DRIFT, log, DRIFTER_type)
    log.debug(f'Size drifters obs {np.shape(traj_drifter_lon_temp)}')
    # --------------------------------------------
    # Retrieve Model trajectories
    # --------------------------------------------
    log.debug('Retrieve model trajectories')
    log.debug('--------------------------------------------')
    parquet_files = __import__("glob").glob(
        os.path.join(dir_work_mod, f"*Trajectories_{MODEL}_*at_SVPloc*.parquet"))
    if parquet_files:
        # Observed rows retain the same sorted drifter order used at seeding.
        observed_ids = traj_drifter_id_temp[:, 0]
        traj_lon, traj_lat, traj_time, traj_IDS, particle_indices = Retrieve_traj_parquet(
            MODEL, dir_work_mod, log, observed_ids)
        # Align observations to particle order even if some particles are absent.
        traj_drifter_lon_temp = traj_drifter_lon_temp[particle_indices]
        traj_drifter_lat_temp = traj_drifter_lat_temp[particle_indices]
        traj_drifter_date_temp = traj_drifter_date_temp[particle_indices]
        traj_drifter_time_temp = traj_drifter_time_temp[particle_indices]
        traj_drifter_id_temp = traj_drifter_id_temp[particle_indices]
        nb_drifters_temp = len(particle_indices)
    else:
        traj_lon, traj_lat,traj_time,traj_IDS = Retrieve_traj_files(MODEL,dir_work_mod, log)
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
        traj_drifter_lon  = traj_drifter_lon_temp
        traj_drifter_lat  = traj_drifter_lat_temp
        traj_drifter_date = traj_drifter_date_temp
        traj_drifter_time = traj_drifter_time_temp
        traj_drifter_id   = traj_drifter_id_temp

    log.debug(f'Taille drifters observation {np.shape(traj_drifter_lon)} {day}')
    log.debug('--------------------------------')
    log.debug(f'Taille drifters Modele Parcel raw {np.shape(traj_lon)} {day}')
    if(np.prod(np.shape(traj_drifter_lon)) >0):
        # Reduce the oversampled trajectories of observations to compare with model trajectories
        log.debug(f'Reduce the oversampled trajectories of observations')
        length_drift = T_DRIFT * factor
        drifter_lon,drifter_lat,drifter_date,drifter_time, Tm, Nd, Td, traj_lon, traj_lat = \
                Traj_filter_parcels(int(T_DRIFT),
                                    traj_drifter_lon,
                                    traj_drifter_lat,
                                    traj_drifter_date,
                                    traj_drifter_time,
                                    traj_lon,
                                    traj_lat,
                                    length_drift,
                                    log)
        log.debug(f"Dimensions observations subsampled raw : {np.shape(drifter_lon)}")
        log.debug(f'Dimensions model Parcels : {np.shape(traj_lon)}')
        if filtered:
            # Change to 0-360
            traj_lon = ((traj_lon + 180) % 360) - 180
            drifter_lon = ((drifter_lon + 180) % 360) - 180
            log.debug(drifter_lon.shape)
            log.debug(traj_lon.shape)
            # ---------------------------------------------------------
            # Select suspicious trajectories by fixing a big theshold
            # ---------------------------------------------------------
            dist_first_day = SeparationDistance_havers2(
                drifter_lon[:, :, :, :length_drift],
                drifter_lat[:, :, :, :length_drift],
                traj_lon[:, :, :, :length_drift],
                traj_lat[:, :, :, :length_drift]
            )
            maxdist = np.nanmax(dist_first_day, axis=-1)
            bad_traj = np.squeeze(maxdist > threshold)
            nb_bad = np.sum(bad_traj)
            log.debug(f"Number of suspicious trajectories: {nb_bad}")
            # --------------------------------------------------------
            # Filter only if suspicious trajectories exist
            # -------------------------------------------------------- 
            if nb_bad > 0:
                good_idx = np.where(~bad_traj)[0]
                bad_idx = np.where(bad_traj)[0]
                log.debug(f"Number of suspicious trajectories: {np.sum(bad_traj)}")
                log.debug(f"Number of good trajectories: {len(good_idx)}")
                log.debug(np.shape(good_idx))
                log.debug(np.shape(drifter_lon))
                log.debug(np.shape(drifter_lat))
                log.debug('********************************************')
                if save:
                    # ---------------------------------------------------------
                    # Select suspicious indices
                    # ---------------------------------------------------------
                    lon_drift_bad = drifter_lon[:, :, bad_idx, :]
                    lat_drift_bad = drifter_lat[:, :, bad_idx, :]
                    lon_model_bad = traj_lon[:, :, bad_idx, :]
                    lat_model_bad = traj_lat[:, :, bad_idx, :]
                    dist_bad = dist_first_day[:, :, bad_idx, :]
                    # time array (assumes same shape)
                    time_bad = traj_time[:, :, bad_idx, :] if 'traj_time' in globals() else None

                    # ---------------------------------------------------------
                    # Build CSV rows
                    # ---------------------------------------------------------
                    rows = []

                    ntraj = lon_drift_bad.shape[2]
                    nt = lon_drift_bad.shape[-1]

                    for t in range(nt):
                        for i in range(ntraj):

                            rows.append({
                                "trajectory": int(bad_idx[i]),
                                "time_index": t,

                                "lon_drift": lon_drift_bad[0, 0, i, t],
                                "lat_drift": lat_drift_bad[0, 0, i, t],

                                "lon_model": lon_model_bad[0, 0, i, t],
                                "lat_model": lat_model_bad[0, 0, i, t],
                                "distance_km": dist_bad[0, 0, i, t],

                            })

                    # ---------------------------------------------------------
                    # Save CSV
                    # ---------------------------------------------------------
                    df_bad = pd.DataFrame(rows)

                    outfile_csv = f"data/suspicious_trajectories_{str_date_sim}.csv"
                    df_bad.to_csv(outfile_csv, index=False)

                    log.debug(f"Saved: {outfile_csv}")
                traj_lon = traj_lon[:, :, good_idx, :]
                traj_lat = traj_lat[:, :, good_idx, :]
                drifter_lon = drifter_lon[:, :, good_idx, :]
                drifter_lat = drifter_lat[:, :, good_idx, :]
        # --------------------------------------------------------------------
        # Compute statistical scores ; Separation distance and Liu Skill score
        # --------------------------------------------------------------------
        log.debug("Compute Separation distance")
        sep_dist = SeparationDistance_havers2(drifter_lon,drifter_lat,traj_lon,traj_lat)
        # remove extra dims → (1019, 40)
        lon2d = traj_lon[0, 0, :, :]
        lat2d = traj_lat[0, 0, :, :]
        Nd, Tm = lon2d.shape
        Ndays = Tm // factor
        log.info(f"Number of days {Ndays}")
        sep_dist_daily = sep_dist.reshape(Nd, Ndays, factor).mean(axis=2)
        # reshape + mean
        traj_lon_daily_tmp = lon2d[:, :Ndays*factor].reshape(Nd, Ndays, factor).mean(axis=2)
        traj_lat_daily_tmp = lat2d[:, :Ndays*factor].reshape(Nd, Ndays, factor).mean(axis=2)
        # put back original structure → For example (1, 1, 1019, 10)
        traj_lon_daily = traj_lon_daily_tmp[np.newaxis, np.newaxis, :, :]
        traj_lat_daily = traj_lat_daily_tmp[np.newaxis, np.newaxis, :, :]
        # Debug
        log.debug(sep_dist)
        log.debug(drifter_time)
        log.debug(np.shape(sep_dist_daily))
        log.debug(sep_dist_daily)
        log.debug(traj_lon.shape[0])

        log.debug("Compute Liu score")
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
        # Compute skill score with the assumption of n=1
        #skill_score = 1-liu_daily
        skill_score = liu_daily

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

    # ---------------------------------------------------------
    # Remove singleton dimensions
    # ---------------------------------------------------------
    lon_drift = np.squeeze(drifter_lon)   # (ntraj, ntime)
    lon_model = np.squeeze(traj_lon)

    lat_drift = np.squeeze(drifter_lat)
    lat_model = np.squeeze(traj_lat)

    ntraj, ntime = lon_drift.shape

    # ---------------------------------------------------------
    # Compute correlation
    # ---------------------------------------------------------
    corr_lon = np.full((ntraj, ntime), np.nan)
    corr_lat = np.full((ntraj, ntime), np.nan)
    # ---------------------------------------------------------
    # Progressive correlation
    # ---------------------------------------------------------
    for k in range(ntraj):

        for t in range(2, ntime + 1):
            # need at least 2 points for correlation
            try:
                corr_lon[k, t-1] = pearsonr(
                    lon_drift[k, :t],
                    lon_model[k, :t]
                )[0]
                corr_lat[k, t-1] = pearsonr(
                    lat_drift[k, :t],
                    lat_model[k, :t]
                )[0]
            except:
                pass

    #corr_lon = np.array([[[pearsonr(drifter_lon[i,j,k],traj_lon[i,j,k])[0] \
    #    for k in range(traj_lon.shape[2])] for j in range(traj_lon.shape[1])] for i in range(traj_lon.shape[0])])
    #corr_lat = np.array([[[pearsonr(drifter_lat[i,j,k],traj_lat[i,j,k])[0] \
    #    for k in range(traj_lat.shape[2])] for j in range(traj_lat.shape[1])] for i in range(traj_lat.shape[0])])
    if np.prod(np.shape(sep_dist))>0:
        prop_sep_dist = np.count_nonzero(sep_dist[:,:,:,-1]<50 ,axis =(1,2) )/ (sep_dist.shape[1]*sep_dist.shape[2])
        prop_skill_score = np.count_nonzero(skill_score[:,:,:,-1]>0 ,axis =(1,2) )/ (skill_score.shape[1]*skill_score.shape[2])
        #prop_corr_lon = np.count_nonzero(corr_lon>0.7 ,axis =(1,2) )/ (corr_lon.shape[1]*corr_lon.shape[2])
        #prop_corr_lat = np.count_nonzero(corr_lat>0.7 ,axis =(1,2) )/ (corr_lat.shape[1]*corr_lat.shape[2])
    # -----------------------------------------------------
    # Spatially bins statistics on a regular lon/lat grid
    # -----------------------------------------------------
    log.debug("Compute spatially bins statistics on a regular lon/lat grid")
    mean_sep, count_sep = bin_spatial(sep_dist_daily, traj_lon_daily.squeeze(), traj_lat_daily.squeeze(), res=resolution)
    sum_sep = mean_sep * count_sep
    lui_score_bin, count_liu_score = bin_spatial(skill_score.squeeze(), traj_lon_daily.squeeze(), traj_lat_daily.squeeze(), res=resolution)
    sum_liu = lui_score_bin * count_liu_score
    #corr_lon_bin, count_corr_lon = bin_spatial(corr_lon, traj_lon_daily.squeeze(), traj_lat_daily.squeeze(), res=resolution)
    #sum_corr_lon = corr_lon_bin, * count_corr_lon
    # corr_lat_bin, count_corr_lat = bin_spatial(corr_lat, traj_lon_daily.squeeze(), traj_lat_daily.squeeze(), res=resolution)
    #sum_corr_lat = corr_lat_bin, * count_corr_lat

    log.info(f"Done {day}")
    return {
    "sum_sep": sum_sep,
    "count_sep": count_sep,
    "sum_liu": sum_liu,
    "count_liu": count_liu_score,
    #"sum_corr_lon": sum_corr_lon,
    #"count_corr_lon": count_corr_lon,
    #"sum_corr_lat": sum_corr_lat,
    #"count_corr_lat": count_corr_lat,
    "sep_dist_daily": sep_dist_daily,
    "liu_skill_daily": skill_score.squeeze()
    }

def aggregate_and_save_results(results,
                               name_out,
                               log,
                               res=2):
    """
    Aggregate parallel daily results and save final NetCDF.

    Parameters
    ----------
    results : list
        Output list from Parallel(process_one_day)

    name_out : str
        Name of the output file

    res : float
        Spatial resolution in degrees
    """

    # ======================================================
    # Initialize
    # ======================================================

    sum_sep_total = None
    count_sep_total = None

    sum_liu_total = None
    count_liu_total = None

    #sum_corr_lon_total = None
    #count_corr_lon_total = None
    #sum_corr_lat_total = None
    #count_corr_lat_total = None

    # ======================================================
    # Aggregate results
    # ======================================================
    for ires, res_day in enumerate(results):

        log.info(f"\nProcessing result {ires+1}/{len(results)}")

        if res_day is None:
            log.error("Skipped None result")
            continue

        sum_sep = res_day['sum_sep']
        count_sep = res_day['count_sep']

        sum_liu = res_day['sum_liu']
        count_liu = res_day['count_liu']

        #sum_corr_lon = res_day['sum_corr_lon']

        #count_corr_lon = res_day['count_corr_lon']
        #sum_corr_lat = res_day['sum_corr_lat']
        #count_corr_lat = res_day['count_corr_lat']

        log.info(f"Mean daily separation: {np.nanmean(sum_sep)}")

        # --------------------------------------------------
        # First initialization
        # --------------------------------------------------

        if sum_sep_total is None:

            sum_sep_total = sum_sep.copy()
            count_sep_total = count_sep.copy()

            sum_liu_total = sum_liu.copy()
            count_liu_total = count_liu.copy()

            #sum_corr_lon_total = sum_corr_lon.copy()
            #count_corr_lon_total = count_corr_lon.copy()
            #sum_corr_lat_total = sum_corr_lat.copy()
            #count_corr_lat_total = count_corr_lat.copy()
        # --------------------------------------------------
        # Accumulate
        # --------------------------------------------------

        else:

            sum_sep_total += sum_sep
            count_sep_total += count_sep

            sum_liu_total += sum_liu
            count_liu_total += count_liu
            #sum_corr_lon_total += sum_corr_lon
            #count_corr_lon_total += count_corr_lon
            #sum_corr_lat_total += sum_corr_lat
            #count_corr_lat_total += count_corr_lat

    # ======================================================
    # Final means
    # ======================================================

    mean_sep_total = sum_sep_total / count_sep_total
    mean_sep_total[count_sep_total == 0] = np.nan

    mean_liu_total = sum_liu_total / count_liu_total
    mean_liu_total[count_liu_total == 0] = np.nan

    #sum_corr_lon_total = sum_corr_lon_total / count_corr_lon_total
    #sum_corr_lat_total = sum_corr_lat_total / count_corr_lat_total
    log.info(f"Final separation shape: {np.shape(mean_sep_total)}")
    log.info(f"Final Liu shape: {np.shape(mean_liu_total)}")

    # ======================================================
    # Save NetCDF
    # ======================================================

    outfile = name_out
    save_netcdf(
        mean_sep_total,
        count_sep_total,
        mean_liu_total,
        outfile,
        log,
        res=res
    )

    log.info(f"\nSaved: {outfile}")

    return (
        mean_sep_total,
        count_sep_total,
        mean_liu_total,
        count_liu_total
    )

# =========================================================
# 7. SAVE TO NETCDF
# =========================================================
def save_netcdf2(mean_sep, count_sep, mean_liu, outfile, res=2):

    import numpy as np
    import xarray as xr

    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    #ndays = mean_sep.shape[2]
    ndays = mean_sep.shape[0]
    leadtime = np.arange(1, ndays + 1)

    ds = xr.Dataset(
        {
            #"mean_separation_distance": (
            #    ("lon", "lat", "leadtime"),
            #    mean_sep
            #),
            #"liu_skill_score": (
            #    ("lon", "lat", "leadtime"),
            #    mean_liu
            #),
            #"count": (
            #    ("lon", "lat", "leadtime"),
            #    count_sep
            "mean_separation_distance": (
                ("leadtime", "lon", "lat"),
                mean_sep
            ),
            "liu_skill_score": (
                ("leadtime", "lon", "lat"),
                mean_liu
            ),
            "count": (
                ("leadtime", "lon", "lat"),
                count_sep
            ),
        },
        coords={
            "lon": lon_centers,
            "lat": lat_centers,
            "leadtime": leadtime,
        }
    )

    ds["mean_separation_distance"].attrs = {
        "units": "km",
        "long_name": "Mean separation distance"
    }

    ds["liu_skill_score"].attrs = {
        "units": "",
        "long_name": "Liu skill score : perfect score is 1"
    }

    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds.data_vars
    }

    ds.to_netcdf(outfile, encoding=encoding)

    #print(f"Saved: {outfile}")




def save_netcdf(mean_sep, count_sep, mean_liu, outfile, log, res=2):

    lon_bins = np.arange(-180, 180 + res, res)
    #lon_bins = np.arange(-180, 181, res)
   # lat_bins = np.arange(-90, 91, res)
    lat_bins = np.arange(-90, 90 + res, res)

    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    ndays = mean_sep.shape[2]
    leadtime = np.arange(1, ndays + 1)

    ds = xr.Dataset(
        {
            "mean_separation_distance": (("lat", "lon", "leadtime"), mean_sep),
            "liu_skill_score": (("lat", "lon", "leadtime"), mean_liu),
            "count": (("lat", "lon", "leadtime"), count_sep),
        },
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
            "leadtime": leadtime,
        }
    )

    ds["mean_separation_distance"].attrs = {
        "units": "km",
        "long_name": "Mean separation distance"
    }
    ds["liu_skill_score"].attrs = {
        "units": "",
        "long_name": "Liu skill score : perfect score is 1"
    }

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}

    ds.to_netcdf(outfile, encoding=encoding)

    log.info(f"Saved: {outfile}")
