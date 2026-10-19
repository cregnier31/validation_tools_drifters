#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os,glob
import time
import sys
import xarray as xr
import pandas as pd
import numpy as np


def retrieve_drifters_for_one_day_traj(
    day,
    T_DRIFT,
    lon_min,
    lon_max,
    lat_min, lat_max,
    str_region,
    str_date1,
    str_date2,
    obs_type,
    drift_path,
    dir_out,
    log
):
    yyear = day.strftime("%Y")

    base_dir = ( drift_path + yyear + "/" )
    end_day = day + pd.Timedelta(days=T_DRIFT)
    nb_points = T_DRIFT * 24

    fout_driftersZone = f"Drifters_{str_region}_{str_date1}_{str_date2}.csv"
    dir_drifters = dir_out+'/DRIFTERS/'

    if not os.path.exists(dir_drifters):
        log.debug(f'\ncreate folder {dir_drifters}')
        # Create directory where to store results:
        os.makedirs(dir_drifters, exist_ok=True)
    outfile2 = dir_drifters+fout_driftersZone
    print(obs_type.upper())
    if obs_type.upper() == "COPERNICUS_DAILY":
        # ==========================================================
        # 1. Select files
        # ==========================================================
        log.debug('Select files')
        file_list = sorted(glob.glob(base_dir + "*/*.nc") + glob.glob(base_dir + "*.nc"))
        def extract_date(f):
            return pd.to_datetime(os.path.basename(f).split("_")[3], format="%Y%m%d")

        files_in = [f for f in file_list if day <= extract_date(f) <= end_day]

        log.debug(f'Select files {files_in}')
        if len(files_in) == 0:
            log.debug('Files are missing')
            return None

        # ==========================================================
        # 2. Load data
        # ==========================================================
        log.debug('Load data')
        df_list = []

        for file in files_in:
            ds = xr.open_dataset(file)

            df_tmp = pd.DataFrame({
                'id': ds.PLATFORM_CODE.values.astype(np.int32),
                'time': pd.to_datetime(ds.TIME.values),
                'lat': ds.LATITUDE.values,
                'lon': ds.LONGITUDE.values
            })

            df_list.append(df_tmp)

        df = pd.concat(df_list, ignore_index=True)

        # ==========================================================
        # 3. Spatial filtering
        # ==========================================================
        log.debug('Spatial filtering')
        df = df[
            (df.lon >= lon_min) & (df.lon <= lon_max) &
            (df.lat >= lat_min) & (df.lat <= lat_max)
        ]
        ## ==========================================================
        ## 4. Remove invalid coordinates
        ## ==========================================================
        log.debug('Removing invalid coordinates')
        valid = (
            np.isfinite(df.lon) &
            np.isfinite(df.lat) &
            (np.abs(df.lat) <= 90) &
            (np.abs(df.lon) <= 360)
        )
        log.debug(f"Invalid positions removed: {np.sum(~valid)}")
        df = df[valid]
        # ==========================================================
        # 4. Sort + prepare time
        # ==========================================================

        df = df.sort_values(['id', 'time'])
        df['date'] = df['time'].dt.floor('D')
        # Keep a fixed hourly grid, including missing days as NaN. Do not
        # interpolate observations: downstream statistics must mask gaps.
        clock = pd.date_range(start=day, periods=nb_points, freq="h")
        trajs = []
        rejected_no_start = 0
        for drifter_id, g in df.groupby("id", sort=True):
            g = g[(g.time >= day) & (g.time < end_day)]
            # Reject ambiguous duplicate observations rather than choosing one.
            if g.time.duplicated().any():
                log.debug("Skipping drifter %s: duplicate observation timestamps", drifter_id)
                continue
            g = g.set_index("time").reindex(clock)
            if not np.isfinite(g.lon.iloc[0]) or not np.isfinite(g.lat.iloc[0]):
                rejected_no_start += 1
                continue
            n_valid = int((np.isfinite(g.lon) & np.isfinite(g.lat)).sum())
            trajs.append({
                "id": drifter_id,
                "lon": g.lon.to_numpy(dtype=float),
                "lat": g.lat.to_numpy(dtype=float),
                "time": clock.to_numpy(),
                "date": clock.floor("D").to_numpy(),
            })
        log.info(
            "Observation hourly alignment %s: %d seeded drifters, "
            "%d without valid initialization, %d missing hourly position pairs",
            day.date(), len(trajs), rejected_no_start,
            sum(int((~np.isfinite(t["lon"]) | ~np.isfinite(t["lat"])).sum())
                for t in trajs),
        )

        if len(trajs) == 0:
            return None

        # ==========================================================
        # 6. Save CSV
        # ==========================================================

        # ==========================================================
        # Save CLEAN trajectories (not raw df)
        # ==========================================================
        df_trajs_list = []
        for traj in trajs:
            n = len(traj['lon'])
            df_tmp = pd.DataFrame({
                'id': np.repeat(traj['id'], n),
                'time': traj['time'],
                'date': traj['date'],
                'lat': traj['lat'],
                'lon': traj['lon']
            })
            df_trajs_list.append(df_tmp)
        df_trajs = pd.concat(df_trajs_list, ignore_index=True)
        df_trajs.to_csv(outfile2, index=False)
        log.debug(f"Saved trajectories CSV: {outfile2}")
        log.debug(f"{day.date()} → {len(trajs)} valid trajectories")
        log.debug("--------------------------------")
        return trajs, outfile2
    elif obs_type.upper() == "AOML":
        print("Case2")
        return retrieve_drifters_noaa(
                day,
                T_DRIFT,
                lon_min,
                lon_max,
                lat_min, lat_max,
                str_region,
                str_date1,
                str_date2,
                obs_type,
                drift_path,
                dir_out,
                log
            )
    else:
        log.error(f"{obs_type} not known")
        sys.exit(1)


def retrieve_drifters_noaa(day,
                           T_DRIFT,
                           lon_min,
                           lon_max,
                           lat_min, lat_max,
                           str_region,
                           str_date1,
                           str_date2,
                           obs_type,
                           drift_path,
                           dir_out,
                           log
                        ):

    print("retrieve_drifters_noaa")
    yyear = day.strftime("%Y")

    base_dir = ( drift_path + yyear + "/" )
    end_day = day + pd.Timedelta(days=T_DRIFT)
    nb_points = T_DRIFT * 24

    fout_driftersZone = f"Drifters_{str_region}_{str_date1}_{str_date2}.csv"
    dir_drifters = dir_out+'/DRIFTERS/'

    if not os.path.exists(dir_drifters):
        log.debug(f'\ncreate folder {dir_drifters}')
        # Create directory where to store results:
        os.makedirs(dir_drifters, exist_ok=True)
    outfile2 = dir_drifters+fout_driftersZone
    print(drift_path)
    file_csv = glob.glob(f"{drift_path}*csv")[0]
    print(file_csv)
    # ---------------------------------------------------------
    # Read CSV
    # ---------------------------------------------------------
    log.info(f"Reading NOAA file {drift_path}")
    df = pd.read_csv(
        file_csv,
        parse_dates=["time"],
        low_memory=False,
    )
    # Remove timezone information
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.tz_localize(None)
    )

    # ---------------------------------------------------------
    # Time selection
    #---------------------------------------------------------

    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=DRIFT_TIME)

    df = df[
            (df.time >= start)
           &
            (df.time <= end)
           ]

    log.info(f"{len(df)} observations after time filtering")
    print(df)
    #print(file_csv)
    sys.exit(1)

def Retrieve_traj_files(MODEL,dir_out, log):
    traj_files_temp = [] 
    traj_files_int = []
    #traj_files_int.append(glob.glob(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc.zarr'))
    files = glob.glob(dir_out + '/' + '*Trajectories_' + MODEL + '_*at_SVPloc*.zarr')
    #if files:
    traj_files_int.append(glob.glob(dir_out+'/'+'*Trajectories_'+MODEL+'_*at_SVPloc*.zarr'))
    #    #files = glob.glob(dir_out + '/' + '*Trajectories_' + MODEL + '_*at_SVPloc*.zarr')
    #    print(files[0])
    #    traj_files_int.append(files[0])
    #print(dir_out+'EXP/'+MODEL+'/'+'Trajectories_'+MODEL+'_*at_SVPloc.zarr')
    traj_files_temp.append(traj_files_int)
    log.debug(f"Trajectories {traj_files_temp}")
    traj_files =  np.sort([inner for outer in traj_files_temp for inner in outer])
    traj_lon_temp = []
    traj_lat_temp = []
    traj_time_temp = []
    traj_IDS_temp = []

    #traj_lon1  = []
    #traj_lat1  = []
    #traj_time1 = []
    #for i in range(len(MODELS)):
    traj_lon_int = []
    traj_lat_int = []
    traj_time_int = []
    traj_IDS_int = []
    #traj_date_int = []
    list_traj = traj_files[0]
    #for file in traj_files[i]:
    for file in list_traj:
        log.info(f"Trajectories zarr file {file}")
        ds = xr.open_zarr(file)
        traj_time_int.append(ds.time.values)
        traj_lon_int.append(ds.lon.values)
        traj_lat_int.append(ds.lat.values)
        traj_IDS_int.append(ds['label'].values)
        break
    traj_lon_temp.append(traj_lon_int)
    traj_lat_temp.append(traj_lat_int)
    traj_time_temp.append(traj_time_int)
    traj_IDS_temp.append(traj_IDS_int)
    #traj_lon1.append(traj_lon_int)
    #traj_lat1.append(traj_lat_int)
    #traj_time1.append(traj_time_int)

    taille_max = max(len(x) for x in traj_lon_temp)
    traj_lon = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_lon_temp])

    taille_max = max(len(x) for x in traj_lat_temp)
    traj_lat = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_lat_temp])
    taille_max = max(len(x) for x in traj_time_temp)
    traj_time = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_time_temp])

    taille_max = max(len(x) for x in traj_IDS_temp)
    traj_IDS = np.array([x + [x[-1]] * (taille_max - len(x)) for x in traj_IDS_temp])


    #return np.array(traj_lon1), np.array(traj_lat1),np.array(traj_time1)
    return traj_lon,traj_lat,traj_time,traj_IDS


def Retrieve_drifters_traj(outfile2, T_DRIFT, log, remote = False,DRIFTER_type="COPERNICUS_DAILY"):
    log.debug(f"loading drifters_traj {outfile2}")
    drifters_zone_traj = pd.read_csv(outfile2) #when unknown number of spaces separating columns
    log.debug(drifters_zone_traj.head(5))
    #print(drifters_zone_traj.dtypes)

    # CSV is already on the exact hourly clock; preserve missing values.
    # Pivot by ID/time rather than relying on contiguous rows or padding
    # missing observations with the final known position.
    if DRIFTER_type.upper() == "COPERNICUS_DAILY":
        ids = np.sort(drifters_zone_traj["id"].unique())
        times = pd.date_range(
            start=pd.to_datetime(drifters_zone_traj["time"]).min(),
            periods=int(T_DRIFT) * 24, freq="h"
        )
        df = drifters_zone_traj.copy()
        df["time"] = pd.to_datetime(df["time"])
        if df.duplicated(["id", "time"]).any():
            raise ValueError("Duplicate observation ID/timestamp pairs")
        lon = df.pivot(index="id", columns="time", values="lon").reindex(index=ids, columns=times)
        lat = df.pivot(index="id", columns="time", values="lat").reindex(index=ids, columns=times)
        lon_values = lon.to_numpy(dtype=float)
        lat_values = lat.to_numpy(dtype=float)
        time_values = np.broadcast_to(times.to_numpy(), lon_values.shape).copy()
        date_values = np.broadcast_to(times.floor("D").to_numpy(), lon_values.shape).copy()
        id_values = np.broadcast_to(ids[:, None], lon_values.shape).copy()
        return lon_values, lat_values, date_values, time_values, id_values, len(ids)
    nb_drifters_temp=len(pd.unique(drifters_zone_traj['id']))
    #remove drifters with missing entries (should be nb_days*4+1)
    #Check each drifter has n days * 4 entries
    drifter_ids = pd.unique(drifters_zone_traj['id'])
    spurious_drifters=[]

    for i in range(nb_drifters_temp):
        nb_entries = len(drifters_zone_traj[drifters_zone_traj.id == drifter_ids[i]])
        #print(nb_entries, i)
       # if not nb_entries >= ((T_DRIFT+1)*24):
        if not nb_entries >= ((T_DRIFT)*24):
            log.error(drifter_ids[i],': not enough entries, only ', nb_entries)
            spurious_drifters.append(drifter_ids[i])

    if False:
        #drifters_zone_traj_old = drifters_zone_traj
        #ind1 = drifters_zone_traj['id']==spurious_drifters).index
        for i in range(len(spurious_drifters)):
            log.debug(f'remove drifter {spurious_drifters[i]}')
            ind1 = drifters_zone_traj[drifters_zone_traj['id'] == np.array(spurious_drifters[i])].index
            #print(drifters_zone_traj['id'] == np.array(spurious_drifters[i]))
            drifters_zone_traj.drop(ind1, inplace=True) #remove data points
        drifters_zone_traj=drifters_zone_traj.reset_index()
        nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
        log.debug(f'{nb_drifters_temp} - {nb_drifters2} = {nb_drifters_temp - nb_drifters2}'   )
    else:
        nb_drifters2=len(pd.unique(drifters_zone_traj['id']))
    #print( f'{aa}  - {len( drifters_zone_traj)} =  {aa  - len( drifters_zone_traj)}')
    log.debug(f'total nb of drifters in zone at start that reach end: {nb_drifters2}')

    #Sort drifter info in indiv. variables traj_lon, traj_lat, traj_id, etc...
    if remote or DRIFTER_type == "COPERNICUS_DAILY":
        nb_points = T_DRIFT*24   # hourly
        #nb_points = (T_DRIFT+1)*24   # hourly
    else :
        nb_points = T_DRIFT*4 + 1    # hourly

    #print(nb_points)
    log.debug(np.shape(drifters_zone_traj))
    #print(drifters_zone_traj)
    #print(type(drifters_zone_traj))
    #print(np.shape(drifters_zone_traj.time.values))
    #print(np.shape(drifters_zone_traj.date.values))
    #print(nb_drifters2*nb_points)

    #for i in range(nb_points):
    #    print(drifters_zone_traj.id.values[i])
    #    print(drifters_zone_traj.date.values[i])
    #    print(drifters_zone_traj.time.values[i])
    #
    #print(drifters_zone_traj.id.values[nb_points])
    #print(drifters_zone_traj.date.values[nb_points])
    #print(drifters_zone_traj.id.values[nb_points +24])
    #print(drifters_zone_traj.date.values[nb_points+24])
    #print(drifters_zone_traj.id.values[nb_points+72])
    #print(drifters_zone_traj.date.values[nb_points+72])
    ll =0
    while drifters_zone_traj.id.values[ll] == drifters_zone_traj.id.values[0]:
        ll += 1
    #print(f'{ll=}')
    #print(f'{nb_points = }')
    #print(f'{ll*nb_drifters2}')
    #print(np.shape(drifters_zone_traj.date.values))
    


    


    if False:
        print('Case1')
        traj_drifter_lon   = np.array([drifters_zone_traj.lon.values[ll,0:nb_points] for ll in range(nb_drifters2)])#      np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
        traj_drifter_lat   = np.array([drifters_zone_traj.lat.values[ll,0:nb_points] for ll in range(nb_drifters2)])#      np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
        traj_drifter_date  = np.array([drifters_zone_traj.date.values[ll,0:nb_points] for ll in range(nb_drifters2)])# np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
        traj_drifter_time  = np.array([drifters_zone_traj.time.values[ll,0:nb_points] for ll in range(nb_drifters2)])#    np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
        traj_drifter_id    = np.array([drifters_zone_traj.id.values[ll,0:nb_points] for ll in range(nb_drifters2)])# np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )
    elif False:
        print('Case2')
        traj_drifter_lon   = np.reshape(drifters_zone_traj.lon.values,(nb_drifters2, nb_points) )
        traj_drifter_lat   = np.reshape(drifters_zone_traj.lat.values,(nb_drifters2, nb_points) )
        traj_drifter_date  = np.reshape(drifters_zone_traj.date.values,(nb_drifters2, nb_points) )
        traj_drifter_time  = np.reshape(drifters_zone_traj.time.values,(nb_drifters2, nb_points) )
        traj_drifter_id  = np.reshape(drifters_zone_traj.id.values,(nb_drifters2, nb_points) )
    else:
        print('Case3')
        list_lon_temp = []
        list_lat_temp = []
        list_date_temp = []
        list_time_temp = []
        list_id_temp = []
        IDS = pd.unique(drifters_zone_traj['id'])    
        for ll in range(len(IDS)):
            indexx = np.where(drifters_zone_traj.id.values == IDS[ll])
            #print(indexx[0])
            #print(indexx[0][0])

            list_lon_temp.append(drifters_zone_traj.lon.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_lat_temp.append(drifters_zone_traj.lat.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_date_temp.append(drifters_zone_traj.date.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_time_temp.append(drifters_zone_traj.time.values[indexx[0][0]:indexx[0][0]+nb_points])
            list_id_temp.append(drifters_zone_traj.id.values[indexx[0][0]:indexx[0][0]+nb_drifters2])
        
        taille_max = max(len(x) for x in list_lon_temp)
        #print(taille_max)
        #print([len(x) for x in list_lon_temp])

        list_lon = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_lon_temp])
        taille_max = max(len(x) for x in list_lat_temp)
        list_lat = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_lat_temp])
        taille_max = max(len(x) for x in list_date_temp)
        list_date = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_date_temp])
        taille_max = max(len(x) for x in list_time_temp)
        list_time = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_time_temp])

        taille_max = max(len(x) for x in list_id_temp)
        list_id = np.array([list(x) + [x[-1]] * (taille_max - len(x)) for x in list_id_temp])



        traj_drifter_lon = np.array(list_lon)
        traj_drifter_lat= np.array(list_lat)
        traj_drifter_date = np.array(list_date)
        traj_drifter_time = np.array(list_time)
        traj_drifter_id = np.array( list_id)

    #print(np.shape(drifters_zone_traj.date.values))
    #print(np.shape(traj_drifter_date))
    #print(f'{nb_drifters2=}')
    #print(f'{nb_points=}')
    #print(T_DRIFT)

    return traj_drifter_lon,traj_drifter_lat,traj_drifter_date,traj_drifter_time,traj_drifter_id,nb_drifters2



def Traj_filter_parcels(
    DRIFT_TIME,
    traj_drifter_lon,
    traj_drifter_lat,
    traj_drifter_date,
    traj_drifter_time,
    traj_lon,
    traj_lat,
    length_drift, 
    log
):
    """
    Align drifter trajectories onto model trajectory time grid.

    Parameters
    ----------
    traj_drifter_lon : (Nd, Td)
    traj_drifter_lat : (Nd, Td)
    traj_drifter_time : (Nd, Td) datetime64
    traj_lon : (Nd, Tm)
    traj_time_model : (Tm,) datetime64

    Returns

    Parameters
    ----------
    traj_drifter_lon : (Nd, Td)
    traj_drifter_lat : (Nd, Td)
    traj_drifter_time : (Nd, Td) datetime64
    traj_drifter_date : (Nd, Td) str or datetime
    traj_lon : (Ni, Nj, Nd, Tm)
    traj_lat : (Ni, Nj, Nd, Tm)
    traj_time_model : (Tm,) datetime64
    length_drift : max length of model trajectory
    Returns
    -------
    drifter_lon : (Nd, Tm)
    drifter_lat : (Nd, Tm)

    """

    log.debug(f'Dimension lon {traj_lon.shape}')
    Nd, Td = traj_drifter_lon.shape

    Tm = traj_lon.shape[-1]

    # ----------------------------------------------------
    # Trim model value if it's greater that observation value
    # ----------------------------------------------------
    if Tm > length_drift:
        log.warning(
            f"Model time dimension ({Tm}) > drifter ({Td} and length_drift {length_drift}) -> trimming model"
        )
        #traj_lon = traj_lon[..., :Td]
        #traj_lat = traj_lat[..., :Td]
        traj_lon = traj_lon[..., :length_drift]
        traj_lat = traj_lat[..., :length_drift]
        Tm = traj_lon.shape[-1]
    # Forecast may cover 27 days while observations cover 28. Compare
    # only the common period; never reshape 672 observed hours into 648.
    if Tm <= 0 or Tm > Td:
        raise ValueError(f"Incompatible trajectory lengths: observed={Td}, model={Tm}")
    factor = Td // Tm
    if factor < 1:
        raise ValueError("Model has more samples than observations")
    common = Tm * factor
    if common != Td:
        log.info("Trimming observed trajectories from %d to %d samples to match model", Td, common)
        traj_drifter_lon = traj_drifter_lon[:, :common]
        traj_drifter_lat = traj_drifter_lat[:, :common]
        traj_drifter_date = traj_drifter_date[:, :common]
        traj_drifter_time = traj_drifter_time[:, :common]
        Td = common
    # --- Compute resampling factor automatically
    log.info(f"Sampling factor {factor}")
    log.debug(f"Sampling factor {factor}")
    #if Td % Tm != 0:
    #    log.error(f"Shape model {traj_lon.shape} /= Shape drifter {Nd} {Td}")
    #    raise ValueError(f"Drifter time dimension not divisible by model time dimension {traj_lon.shape}")

    # --- Reshape and average
    log.info(f"Nd Tm {Nd} {Tm}")
    lon_reshaped = traj_drifter_lon.reshape(Nd, Tm, factor)
    lat_reshaped = traj_drifter_lat.reshape(Nd, Tm, factor)

    drifter_lon_2d = lon_reshaped.mean(axis=2)
    drifter_lat_2d = lat_reshaped.mean(axis=2)

    # --- Time & date: take central value in each window
    mid = factor // 2

    time_reshaped = traj_drifter_time.reshape(Nd, Tm, factor)
    date_reshaped = traj_drifter_date.reshape(Nd, Tm, factor)

    drifter_time_2d = time_reshaped[:, :, mid]
    drifter_date_2d = date_reshaped[:, :, mid]
    # --- Now match original output shape
    # Your original code had extra dims (i,j,:,:)
    # So we broadcast the result

    drifter_lon = np.broadcast_to(drifter_lon_2d, traj_lon.shape)
    drifter_lat = np.broadcast_to(drifter_lat_2d, traj_lat.shape)
    drifter_time = np.broadcast_to(drifter_time_2d, traj_lat.shape)
    drifter_date = np.broadcast_to(drifter_date_2d, traj_lat.shape)

    return drifter_lon, drifter_lat, drifter_date, drifter_time,  Tm, Nd, Td, traj_lon, traj_lat


def Retrieve_traj_parquet(MODEL, dir_out, log, observed_ids):
    """Read Parcels 4 particle records as legacy (1, 1, particle, time) arrays.

    Parcels 4 particle_id is the zero-based initialization index, not the
    observed platform code. observed_ids must be in the original seeding order.
    """
    paths = sorted(glob.glob(os.path.join(
        dir_out, f"*Trajectories_{MODEL}_*at_SVPloc*.parquet")))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Parcels 4 Parquet trajectory in {dir_out}; found {paths}")
    path = paths[0]
    df = pd.read_parquet(path, columns=["t", "x", "y", "particle_id"])
    if df.empty:
        raise ValueError(f"Empty Parcels trajectory output: {path}")
    expected = np.asarray(observed_ids)
    if len(np.unique(expected)) != len(expected):
        raise ValueError("Observed drifter IDs are not unique")
    df = df.sort_values(["particle_id", "t"])
    particle_ids = np.sort(df.particle_id.unique())
    if np.any(particle_ids < 0) or np.any(particle_ids >= len(expected)):
        raise ValueError("Parcels particle_id is outside the original drifter seed range")
    # Recovered/deleted particles have shorter records. Align on the common
    # output clock, leaving missing positions as NaN (never interpolate them).
    # The first valid sample may be missing for a particle, so use the union.
    if df[["t", "x", "y"]].isna().any().any():
        log.warning("Parcels Parquet contains missing time/position values")
    df = df.dropna(subset=["t"])
    if df.empty:
        raise ValueError("No valid Parcels output timestamps")
    if df.duplicated(["particle_id", "t"]).any():
        raise ValueError("Duplicate Parcels particle/time records")
    clock = np.sort(df.t.unique())
    ntime = len(clock)
    if ntime < 1:
        raise ValueError("Empty Parcels output time axis")
    # A pivot makes the missing records explicit and preserves the seeded IDs.
    lon2d = df.pivot(index="particle_id", columns="t", values="x")
    lat2d = df.pivot(index="particle_id", columns="t", values="y")
    lon2d = lon2d.reindex(index=particle_ids, columns=clock)
    lat2d = lat2d.reindex(index=particle_ids, columns=clock)
    lon = lon2d.to_numpy(dtype=float)[None, None, :, :]
    lat = lat2d.to_numpy(dtype=float)[None, None, :, :]
    time_values = np.broadcast_to(clock, (1, 1, len(particle_ids), ntime))
    ids = expected[particle_ids][None, None, :]
    missing = np.count_nonzero(~np.isfinite(lon) | ~np.isfinite(lat))
    short = int((df.groupby("particle_id").size() < ntime).sum())
    log.info(
        "Read Parcels 4 Parquet %s: %d particles, %d time samples, "
        "%d incomplete particles, %d missing/nonfinite position pairs",
        path, len(particle_ids), ntime, short, missing,
    )
    return lon, lat, time_values, ids, particle_ids
