import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import sys

# ==========================================================
# Area weighted mean
# ==========================================================

def area_weighted_mean(da):
    """
    Compute global mean weighted by latitude area
    da(lat, lon, leadtime)
    """

    # remove invalid values after computation
    print(np.nanmax(da))
    da = da.where(np.isfinite(da))
    da = da.where(da <= 10000)

    lat = da.lat

    weights = np.cos(np.deg2rad(lat))

    weights = xr.DataArray(
        weights,
        coords={"lat": lat},
        dims=["lat"]
    )

    return (
        da.weighted(weights)
        .mean(dim=("lat", "lon"))
    )


# ==========================================================
# Read one model file
# ==========================================================

def read_score(
        filename,
        variable="mean_separation_distance"
):

    ds = xr.open_dataset(filename)

    score = ds[variable]

    global_score = area_weighted_mean(score)

    return global_score.values



# ==========================================================
# Build dataframe
# ==========================================================

def build_separation_dataframe(
        files,
        leadtimes, variable
):

    data = {}

    for model, filename in files.items():

        values = read_score(filename, variable)

        # keep requested leadtimes only
        indices = np.array(leadtimes) - 1
        values = values[indices]

        #values = values[:len(leadtimes)]

        data[model] = values


    df = pd.DataFrame(
        data,
        index=leadtimes
    ).T


    return df



# ==========================================================
# Scorecard plot
# ==========================================================

def plot_separation_scorecard(
        df,
        reference,
        outfile,
        title="Mean separation distance"
):
    models = list(df.index)
    leads = list(df.columns)
    arr = df.values
    if  title == "Mean separation distance":
        var_long_name = f'{title} [km]'
    else:
        var_long_name = f'{title}'

    # reference values
    ref = df.loc[reference].values
    n_vars = 1
    n_rows = 1
    # improvement
    delta = (arr - ref) / ref * 100
    fig, ax = plt.subplots(
        figsize=(6, 3),
        dpi=150
    )
    ax.axis("off")
    table_text = []
    colors = []
    #vmax = np.nanmax(np.abs(delta[1:]))
    vmax = 20
    norm = Normalize(
        vmin=-vmax,
        vmax=vmax
    )
    cmap = cm.get_cmap("bwr")
    for i,model in enumerate(models):
        row_text=[]
        row_color=[]
        for j,lead in enumerate(leads):
            value = arr[i,j]
            row_text.append(
                f"{value:.3f}"
            )
            if model == reference:
                row_color.append(
                    (0.85,0.85,0.85,1)
                )
            else:
                row_color.append(
                    cmap(
                        norm(delta[i,j])
                    )
                )
        table_text.append(row_text)
        colors.append(row_color)

    table=ax.table(
        cellText=table_text,
        rowLabels=models,
        colLabels=[
            f"{x}d" for x in leads
        ],
        cellColours=colors,
        loc="center",
        cellLoc="center"
    )

#    table.auto_set_font_size(False)
#    table.set_fontsize(11)

    #for cell in table.get_celld().values():
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(1)
        cell.set_height(0.26)
        cell.set_edgecolor("white")
        cell.get_text().set_weight("bold")
        if row == 0:
            #cell.set_facecolor("purple")
            cell.get_text().set_color("purple")
            cell.get_text().set_weight("bold")

        if col == -1:
            txt = str(cell.get_text().get_text())
            if txt == "GLO12":
                cell.set_facecolor((0.0, 1.0, 1.0))  # physics
            else:
                cell.set_facecolor((0.7, 1.0, 0.7))  # AI models
                rgba = cell.get_facecolor()  # Update color after override

    ax.text(0.5, -0.20, "Lead time [days]", transform=ax.transAxes,
            ha='center', va='top', fontsize=12, color="purple")
    ax.text(0.4, 1.15, var_long_name, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10, weight='bold',
            color='red')

    #fig.suptitle(
    #    title,
    #    fontsize=14
    #    #pad=2
    #)
    # colorbar for comparison
    sm = cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.08,
        pad=0.3
    )
    cbar.set_label(
        "Δ separation distance vs reference (%)"
    )
    plt.tight_layout(
                rect=[0, 0.05, 1, 0.95]
                )
    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==========================================================
# Example
# ==========================================================

DIR_KSMAG = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/data/K_smag/"
DIR_RK4 = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/data/AdvectionRK4/"
dir_out = '../../imgs/'
files = {
        "GLO12": f"{DIR_KSMAG}Lagrangian_scores_weighted_HDCT_GLO12_2024-01-04_2024-12-28.nc",
        "GLO12_TEST5": f"{DIR_KSMAG}Lagrangian_scores_weighted_HDCT_GLO12_TEST5_2024-01-04_2024-12-28.nc",
        "SMOC": f"{DIR_RK4}Lagrangian_scores_weighted_HDCT_SMOC_2024-01-04_2024-12-28.nc",
        "GLOBCURR": f"{DIR_RK4}Lagrangian_scores_weighted_HDCT_GLOBCURR_2024-01-04_2024-12-28.nc"
}

leadtimes=[1,3,5,7,10]

varname = "mean_separation_distance"
#varname = "liu_skill_score"
title="Mean separation distance"
#title="Liu score"
file_img = f"{dir_out}/scorecard_multimodel_2024_{varname}.png"
#varname = "mean_separation_distance"
df = build_separation_dataframe(
    files,
    leadtimes,
    varname
)


print(df)


plot_separation_scorecard(
    df,
    reference="GLO12",
    outfile=file_img,
    title=title
)
