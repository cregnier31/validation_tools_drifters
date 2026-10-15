import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ==========================================================
# 1. AREA-WEIGHTED GLOBAL MEAN
# ==========================================================
def compute_global_weighted_timeseries(ds,
                                      var,
                                      count_var="count",
                                      lat_name="lat"):
    """
    Compute global weighted mean over (lat, lon) for each leadtime.
    Uses cosine latitude weighting.
    """

    data = ds[var].values          # (lat, lon, leadtime)
    count = ds[count_var].values

    lat = ds[lat_name].values

    # cosine latitude weights (area correction)
    w_lat = np.cos(np.deg2rad(lat))[:, None, None]  # (lat,1,1)

    weighted_data = data * count * w_lat
    weighted_count = count * w_lat

    num = np.nansum(weighted_data, axis=(0, 1))
    den = np.nansum(weighted_count, axis=(0, 1))

    return num / np.where(den == 0, np.nan, den)


# ==========================================================
# 2. LOAD ONE MODEL
# ==========================================================
def load_model_timeseries(nc_file, var):
    ds = xr.open_dataset(nc_file)
    return compute_global_weighted_timeseries(ds, var)


# ==========================================================
# 3. LOAD ALL MODELS
# ==========================================================
def load_all_models(file_dict, var):
    """
    file_dict = {
        "REF": "ref.nc",
        "GLO12": "model.nc",
        ...
    }
    """
    results = {}

    for model, f in file_dict.items():
        results[model] = load_model_timeseries(f, var)

    return results


# ==========================================================
# 4. BUILD SCORE DATAFRAME
# ==========================================================
def build_score_dataframe(results):
    df = pd.DataFrame(results)   # columns = models
    #df.index = leadtimes
    #return df.T                  # rows = models
    # keep raw leadtime count automatically
    return df

# ==========================================================
# 5. SCORECARD PLOT (REFERENCE BASED)
# ==========================================================
def plot_scorecard(ax,
                   df,
                   leadtimes,
                   title="Scorecard",
                   unit=""):

    ax.axis("off")

    arr = df.values
    ref = arr[0]   # first model = reference

    diff = arr - ref

    max_diff = np.nanmax(np.abs(diff[1:])) if arr.shape[0] > 1 else 1.0

    n_models, n_leads = arr.shape

    # table text
    cell_text = [[f"{v:.3f}" for v in row] for row in arr]

    # colors
    cell_colors = []

    for i in range(n_models):
        row_colors = []

        for j in range(n_leads):

            if i == 0:
                row_colors.append((0.9, 0.9, 0.9, 1.0))  # reference
                continue

            d = diff[i, j]
            norm = abs(d / max_diff) if max_diff > 0 else 0

            # IMPORTANT:
            # lower value than reference = better → BLUE
            if d < 0:
                row_colors.append((0.0, 0.4, 1.0, norm))
            else:
                row_colors.append((1.0, 0.0, 0.0, norm))

        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=list(df.index),
        colLabels=[str(l) for l in leadtimes],
        cellColours=cell_colors,
        cellLoc="center",
        loc="center"
    )

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1)
        cell.set_text_props(weight="bold")

    ax.text(
        0.5, 1.05,
        f"{title} {f'[{unit}]' if unit else ''}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=12,
        weight="bold"
    )

    return df


# ==========================================================
# 6. FULL WORKFLOW
# ==========================================================
def run_scorecard_pipeline(file_dict,
                           var="mean_separation_distance",
                           leadtimes=[1, 3, 5, 7, 9],
                           title="Separation distance",
                           unit="km",
                           output_img="scorecard.png"):

    # 1. load models
    results = load_all_models(file_dict, var)
    # 2. dataframe (models × leadtime)
    df = build_score_dataframe(results)
    leadtimes = df.columns

    #, leadtimes)
    # 3. plot
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_scorecard(
        ax,
        df,
        leadtimes,
        title=title,
        unit=unit
    )
    # save image
    fig.savefig(
                output_img,
                dpi=300,
                bbox_inches="tight"
                )
    plt.tight_layout()
    plt.show()

    return df


# ==========================================================
# 7. EXAMPLE USAGE
# ==========================================================
DIR_KSMAG = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/data/K_smag/"
DIR_RK4 = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/data/AdvectionRK4/"
dir_out = '../../imgs/'
file_img = f"{dir_out}/scorecard_multimodel_2024.png"
file_dict = {
    "GLO12": f"{DIR_KSMAG}Lagrangian_scores_HDCT_GLO12_2024-01-04_2024-12-31.nc",
    "GLO12_TEST5": f"{DIR_KSMAG}Lagrangian_scores_HDCT_GLO12_TEST5_2024-01-04_2024-12-31.nc",
    "SMOC": f"{DIR_RK4}Lagrangian_scores_HDCT_SMOC_2024-01-04_2024-12-28.nc",
    "GLOBCURR": f"{DIR_RK4}Lagrangian_scores_HDCT_GLOBCURR_2024-01-04_2024-12-21.nc"
}

df = run_scorecard_pipeline(
    file_dict,
    var="mean_separation_distance",
    leadtimes=[1, 3, 5, 7, 9],
    title="Global weighted separation distance",
    unit="km",
    output_img=file_img,

)
