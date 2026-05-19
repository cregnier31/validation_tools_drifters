import sys 
import os
import numpy as np
import xarray as xr

def aggregate_and_save_results(results,
                               name_out,
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

    # ======================================================
    # Aggregate results
    # ======================================================

    for ires, res_day in enumerate(results):

        print(f"\nProcessing result {ires+1}/{len(results)}")

        if res_day is None:
            print("Skipped None result")
            continue

        sum_sep = res_day['sum_sep']
        count_sep = res_day['count_sep']

        sum_liu = res_day['sum_liu']
        count_liu = res_day['count_liu']

        print(type(sum_sep))
        print("Mean daily separation:",
              np.nanmean(sum_sep))

        # --------------------------------------------------
        # First initialization
        # --------------------------------------------------

        if sum_sep_total is None:

            sum_sep_total = sum_sep.copy()
            count_sep_total = count_sep.copy()

            sum_liu_total = sum_liu.copy()
            count_liu_total = count_liu.copy()

        # --------------------------------------------------
        # Accumulate
        # --------------------------------------------------

        else:

            sum_sep_total += sum_sep
            count_sep_total += count_sep

            sum_liu_total += sum_liu
            count_liu_total += count_liu

    # ======================================================
    # Final means
    # ======================================================

    mean_sep_total = sum_sep_total / count_sep_total
    mean_sep_total[count_sep_total == 0] = np.nan

    mean_liu_total = sum_liu_total / count_liu_total
    mean_liu_total[count_liu_total == 0] = np.nan

    print("\nFinal separation shape:",
          np.shape(mean_sep_total))

    print("Final Liu shape:",
          np.shape(mean_liu_total))

    # ======================================================
    # Save NetCDF
    # ======================================================

    outfile = name_out
    save_netcdf(
        mean_sep_total,
        count_sep_total,
        mean_liu_total,
        outfile,
        res=res
    )

    print(f"\nSaved: {outfile}")

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

    print(np.shape(mean_sep))
    print(np.shape(count_sep))
    print(np.shape(mean_liu))
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

    print(f"Saved: {outfile}")




def save_netcdf(mean_sep, count_sep, mean_liu, outfile, res=2):

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

    print(f"Saved: {outfile}")
