import os

import numpy as np
import pandas as pd
import xarray as xr
import math
from matplotlib.cm import ScalarMappable
#import geopandas as gpd

from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import json
from shapely.geometry import shape

# =============================================================================
# Compute weighted mean inside one region
# =============================================================================

def regional_weighted_mean(da, mask):

    weights = xr.DataArray(
        np.cos(np.deg2rad(da.lat)),
        coords={"lat": da.lat},
        dims=["lat"],
    )

    mask = xr.DataArray(
        mask,
        coords={"lat": da.lat, "lon": da.lon},
        dims=("lat", "lon"),
    )

    da = da.where(mask)

    da = da.where(np.isfinite(da))

    return da.weighted(weights).mean(("lat", "lon"))


# =============================================================================
# Create masks from geojson
# =============================================================================

def build_region_masks(geojson_file, lon, lat):

    with open(geojson_file) as f:
        geo = json.load(f)

    #regions = gpd.read_file(geojson_file)


    lon2d, lat2d = np.meshgrid(lon, lat)

    masks = {}

    for feature in geo["features"]:
    #for _, row in regions.iterrows():

        #region_name = row["NAME"]
        #polygon = row.geometry

        region_name = feature["properties"]["NAME"]
        polygon = shape(feature["geometry"])

        mask = np.zeros(lon2d.shape, dtype=bool)

        for j in range(lat2d.shape[0]):
            for i in range(lon2d.shape[1]):

                mask[j, i] = polygon.contains(
                    Point(lon2d[j, i], lat2d[j, i])
                )

        masks[region_name] = mask

    return masks


# =============================================================================
# Read all models for one region
# =============================================================================

def build_region_dataframe(
    files,
    masks,
    region_name,
    leadtimes,
    variable,
):

    region_mask = masks[region_name]

    results = {}

    for model, filename in files.items():

        ds = xr.open_dataset(filename)

        score = ds[variable]

        values = regional_weighted_mean(
            score,
            region_mask
        ).values

        idx = [d - 1 for d in leadtimes]

        values = values[idx]

        results[model] = values

    df = pd.DataFrame(
        results,
        index=leadtimes
    ).T

    return df


# =============================================================================
# Plot scorecard
# =============================================================================

def plot_scorecard(
    ax,
    df,
    title,
    norm,
    cmap,
    reference="GLO12",
):

    models = df.index.tolist()
    leads = df.columns.tolist()

    ref = df.loc[reference].values
    improvement = np.zeros_like(df.values)

    for i in range(df.shape[0]):
        #improvement[i] = 100.0 * (ref - df.values[i]) / ref
        improvement[i] = 100.0 * (df.values[i] - ref ) / ref
    delta = df.values - ref


    # ------------------------------------------------------
    # Cell colours
    # ------------------------------------------------------

    colors = []

    for imod in range(len(models)):

        row = []

        for ilead in range(len(leads)):

            if models[imod] == reference:

                row.append((0.85, 0.85, 0.85, 1))

            else:

                #row.append(
                #    cmap(
                #        norm(delta[imod, ilead])
                #    )
                #)
                row.append(cmap(norm(improvement[imod, ilead])))

        colors.append(row)

    #table_text = [
    #    [f"{v:.1f}" for v in row]
    #    for row in df.values
    #]
    #print(table_text)
    table_text = []

    for i, model in enumerate(df.index):
        if model == reference:
            table_text.append(
                [f"{v:.2f}" for v in df.loc[model]]
            )
        else:
            table_text.append(
                [f"{v:.1f}" for v in improvement[i]]
            )
    #fig, ax = plt.subplots(
    #    figsize=(8, 4),
    #    dpi=150,
    #)

    ax.axis("off")

    table = ax.table(
        cellText=table_text,
        rowLabels=models,
        colLabels=[f"{d} d" for d in leads],
        cellColours=colors,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.4, 1.8)

    for (row, col), cell in table.get_celld().items():

        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)

        # Column headers
        if row == 0:

            cell.set_facecolor("mediumpurple")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")

        # Row labels
        if col == -1:

            if cell.get_text().get_text() == reference:
                cell.set_facecolor("lightgrey")
            else:
                cell.set_facecolor((0.80, 1.00, 0.80))

            cell.get_text().set_weight("bold")

        # Reference row
        if row > 0 and models[row-1] == reference:

            cell.get_text().set_color("black")
            cell.get_text().set_weight("bold")

        # Other rows
        elif row > 0:


            rgba = cell.get_facecolor()

            luminance = (
                0.299 * rgba[0]
                + 0.587 * rgba[1]
                + 0.114 * rgba[2]
            )

            cell.get_text().set_color(
                "white" if luminance < 0.55 else "black"
            )

            cell.get_text().set_weight("bold")

    # ------------------------------------------------------
    # Region title
    # ------------------------------------------------------

    ax.set_title(
        title,
        fontsize=12,
        weight="bold",
        pad=6,
    )

    return table



#    # purple headers
#
#    for c in range(len(leads)):
#        table[(0, c)].set_facecolor("mediumpurple")
#        table[(0, c)].get_text().set_color("white")
#        table[(0, c)].get_text().set_weight("bold")
#
#    ax.set_title(
#        title,
#        fontsize=14,
#        weight="bold",
#        pad=8,
#    )
#
#    sm = plt.cm.ScalarMappable(
#        norm=norm,
#        cmap=cmap,
#    )
#
#    cbar = fig.colorbar(
#        sm,
#        ax=ax,
#        orientation="horizontal",
#        pad=0.02,
#        fraction=0.05,
#    )
#
#    cbar.set_label(
#        f"Difference relative to {reference}"
#    )

    #plt.savefig(
    #    outfile,
    #    dpi=200,
    #    bbox_inches="tight",
    #)

    plt.close()


# =============================================================================
# Main
# =============================================================================

DIR_KSMAG = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/data/K_smag/"
DIR_RK4 = "/home/ext/mr/smer/regnierc/SAVE/codes/validation_tools_drifters/data/AdvectionRK4/"

DIR_OUT = "../../imgs"

GEOJSON = "../../statics/mask_GLOBAL_CMEMS_all_regions.geojson"

files = {

    "GLO12":
    f"{DIR_KSMAG}Lagrangian_scores_weighted_HDCT_GLO12_2024-01-04_2024-12-28.nc",

    "GLO12_TEST5":
    f"{DIR_KSMAG}Lagrangian_scores_weighted_HDCT_GLO12_TEST5_2024-01-04_2024-12-28.nc",

    "SMOC":
    f"{DIR_RK4}Lagrangian_scores_weighted_HDCT_SMOC_2024-01-04_2024-12-28.nc",

    "GLOBCURR":
    f"{DIR_RK4}Lagrangian_scores_weighted_HDCT_GLOBCURR_2024-01-04_2024-12-28.nc",

}

leadtimes = [1, 3, 5, 7, 10]

variable = "liu_skill_score"
title = "Liu score"
variable = "mean_separation_distance"
title="Mean separation distance"

# -------------------------------------------------------------------------
# Read first file to build masks
# -------------------------------------------------------------------------

ds = xr.open_dataset(next(iter(files.values())))

masks = build_region_masks(
    GEOJSON,
    ds.lon.values,
    ds.lat.values,
)
# -------------------------------------------------------------------------
# Loop over all regions
# -------------------------------------------------------------------------

nregions = len(masks)

ncols = 4
nrows = math.ceil(nregions / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(6*ncols, 2.8*nrows),
    dpi=150,
)

axes = axes.ravel()
norm = TwoSlopeNorm(
    vcenter=0,
    vmin=-20,
    vmax=20
)
cmap = plt.get_cmap("bwr")

for ax, region in zip(axes, masks.keys()):
#for region in masks.keys():

    print(region)

    df = build_region_dataframe(
        files,
        masks,
        region,
        leadtimes,
        variable,
    )

    print(df)
    ref = df.loc["GLO12"].values

    delta = df.values - ref

    plot_scorecard(
        ax,
        df,
        title=f"{region}",
        norm=norm,
        cmap=cmap,
        reference="GLO12",
    )

# Hide unused axes
for ax in axes[nregions:]:
    ax.axis("off")

sm = ScalarMappable(
    cmap="bwr",   # blue = improvement, red = degradation
    norm=norm
)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes.tolist(),
    orientation="horizontal",
    shrink=0.9,
    extend="both",
    fraction=0.025,
    pad=0.3,
)

cbar.set_label(
    "% improvement relative to GLO12",
    fontsize=14
)
cbar.ax.tick_params(labelsize='large')

fig.suptitle(
    title,
    fontsize=18,
    weight="bold"
)

plt.tight_layout(rect=[0, 0.15, 1, 0.96])

outfile = os.path.join(
    DIR_OUT,
    f"scorecard_{variable}_allregions_improvement.png"
)
plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)
