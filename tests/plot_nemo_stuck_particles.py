#!/usr/bin/env python3
"""Plot NEMO surface ocean mask and four stationary GLOENS drifters.

Reads the existing meshmask only; writes a PNG next to February 29 diagnostics.
"""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

STUCK = {
    "3201769": (-72.08799743652344, 30.03499984741211),
    "4102665": (-66.23899841308594, 36.525001525878906),
    "6203615": (-78.36599731445312, 30.718000411987305),
    "6203642": (-77.36100006103516, 32.047000885009766),
}

def surface(da):
    for dim in list(da.dims):
        if dim not in ("y", "x"):
            da = da.isel({dim: 0})
    return np.asarray(da.values)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshmask", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    ap.add_argument("--cycle", default="20240229")
    ap.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    ap.add_argument("--extent", nargs=4, type=float, metavar=("WEST","EAST","SOUTH","NORTH"),
                    default=[-82, -62, 27, 41])
    ap.add_argument("--global", dest="global_view", action="store_true", help="Show full global iORCA025 mask")
    args = ap.parse_args()
    with xr.open_dataset(args.meshmask, engine="netcdf4") as ds:
        lon_name = "glamt" if "glamt" in ds else "nav_lon"
        lat_name = "gphit" if "gphit" in ds else "nav_lat"
        if "tmask" not in ds:
            ap.error("meshmask has no tmask")
        lon = surface(ds[lon_name])
        lat = surface(ds[lat_name])
        wet = surface(ds["tmask"])
    west,east,south,north = args.extent
    region = np.ones(lon.shape, dtype=bool) if args.global_view else ((lon >= west-1) & (lon <= east+1) & (lat >= south-1) & (lat <= north+1))
    yy,xx = np.where(region)
    if len(yy)==0:
        ap.error("No NEMO cells in requested extent")
    ys = slice(max(0,yy.min()-2), min(lon.shape[0],yy.max()+3))
    xs = slice(max(0,xx.min()-2), min(lon.shape[1],xx.max()+3))
    lon,lat,wet = lon[ys,xs],lat[ys,xs],wet[ys,xs]
    fig,ax = plt.subplots(figsize=(17,8) if args.global_view else (12,9),layout="constrained")
    cmap = ListedColormap(["#c8b99e","#a9d8ed"])
    # NEMO curvilinear coordinates can be masked/non-finite at folded or
    # boundary cells. Matplotlib rejects any non-finite X/Y in pcolormesh.
    # Fill coordinates for geometry only, and mask the affected data cells.
    valid_xy = np.isfinite(lon) & np.isfinite(lat)
    if not np.all(valid_xy):
        from scipy.ndimage import distance_transform_edt
        nearest_idx = distance_transform_edt(~valid_xy, return_distances=False,
                                             return_indices=True)
        lon = lon[tuple(nearest_idx)]
        lat = lat[tuple(nearest_idx)]
    plotted = np.ma.masked_where(~valid_xy | ~np.isfinite(wet), wet)
    ax.pcolormesh(lon, lat, plotted,
                  cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
    for identifier,(x,y) in STUCK.items():
        ax.scatter(x,y,s=95,c="#c62828",edgecolors="white",linewidths=1.3,zorder=5)
        ax.annotate(identifier,(x,y),xytext=(7,8),textcoords="offset points",
                    fontsize=10,fontweight="bold",color="#8b1414",zorder=6,
                    bbox=dict(facecolor="white",alpha=.78,edgecolor="none",pad=1))
    if args.global_view:
        ax.set_xlim(-180,180);ax.set_ylim(-85,90)
    else:
        ax.set_xlim(west,east);ax.set_ylim(south,north)
    ax.set_xlabel("Longitude (°E)");ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"NEMO T-point surface mask and frozen drifters | GLOENS {args.cycle}")
    ax.grid(alpha=.2)
    ax.legend(handles=[
        Line2D([0],[0],marker="s",linestyle="none",markerfacecolor="#c8b99e",markeredgecolor="none",markersize=12,label="T-mask = 0 (land)"),
        Line2D([0],[0],marker="s",linestyle="none",markerfacecolor="#a9d8ed",markeredgecolor="none",markersize=12,label="T-mask = 1 (ocean)"),
        Line2D([0],[0],marker="o",linestyle="none",markerfacecolor="#c62828",markeredgecolor="white",markersize=10,label="Frozen model particle"),
    ],loc="upper right")
    root = Path(args.root)
    matches = list(root.glob(f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift10_dt5_{args.cycle}-{args.cycle}/ZONE_*/{args.cycle}"))
    out = (matches[0] / "gulf_stream_extension") if len(matches)==1 else Path.cwd()
    out.mkdir(parents=True,exist_ok=True)
    path = out / ("nemo_tmask_global_stuck_particles.png" if args.global_view else "nemo_tmask_stuck_particles.png")
    fig.savefig(path,dpi=200)
    plt.close(fig)
    print("Saved:",path)

if __name__=="__main__":
    main()
