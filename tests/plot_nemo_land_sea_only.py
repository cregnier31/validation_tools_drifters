#!/usr/bin/env python3
"""Plot only the surface NEMO tmask as a binary land/sea map."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset
from scipy.spatial import cKDTree

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshmask", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--extent", nargs=4, type=float, metavar=("WEST","EAST","SOUTH","NORTH"))
    args = ap.parse_args()
    path = Path(args.meshmask)
    with Dataset(path) as ds:
        t = ds.variables["tmask"]
        # NEMO meshmask normally has (time_counter, nav_lev, y, x).
        mask = np.asarray(np.ma.filled(t[0, 0, :, :], 0))
        names = list(ds.variables)
        lon_name = next((n for n in ("glamt", "nav_lon", "longitude") if n in names), None)
        lat_name = next((n for n in ("gphit", "nav_lat", "latitude") if n in names), None)
        if not lon_name or not lat_name:
            raise ValueError("No T-grid lon/lat coordinates found (glamt/gphit or nav_lon/nav_lat)")
        def surface_xy(var):
            data = var[:]
            while data.ndim > 2:
                data = data[0]
            return np.asarray(np.ma.filled(data, np.nan), dtype=float)
        lon = surface_xy(ds.variables[lon_name])
        lat = surface_xy(ds.variables[lat_name])
    if mask.shape != lon.shape or mask.shape != lat.shape:
        raise ValueError(f"Shape mismatch: mask={mask.shape}, lon={lon.shape}, lat={lat.shape}")
    vals, counts = np.unique(mask, return_counts=True)
    print("Original surface tmask value counts:", dict(zip([int(x) for x in vals], [int(x) for x in counts])))
    sea = mask == 1
    valid = np.isfinite(lon) & np.isfinite(lat)
    print(f"Ocean cells (tmask==1): {np.count_nonzero(sea):,}")
    print(f"Land/other cells (tmask!=1): {np.count_nonzero(~sea):,}")
    print(f"Invalid lon/lat coordinates: {np.count_nonzero(~valid):,}")
    # Scatter the cell centers instead of pcolormesh: robust to the NEMO
    # tripolar fold, masked coordinates, and dateline discontinuity.
    if args.extent:
        west, east, south, north = args.extent
        valid &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    # Rasterize onto a regular geographic grid. This avoids the empty spaces
    # and stripe artifacts caused by drawing grid centers as square markers.
    if args.extent:
        west, east, south, north = args.extent
        resolution = 0.05
    else:
        west, east, south, north = -180., 180., -85., 90.
        resolution = 0.20
    xx = np.arange(west + resolution / 2, east, resolution)
    yy = np.arange(south + resolution / 2, north, resolution)
    gx, gy = np.meshgrid(xx, yy)
    # Unit-sphere Cartesian coordinates correctly handle the dateline and
    # high latitudes, unlike Euclidean distances in lon/lat.
    def sphere_xyz(lon_deg, lat_deg):
        lo = np.deg2rad(lon_deg)
        la = np.deg2rad(lat_deg)
        return np.column_stack((np.cos(la) * np.cos(lo),
                                np.cos(la) * np.sin(lo),
                                np.sin(la)))
    tree = cKDTree(sphere_xyz(lon[valid], lat[valid]))
    _, idx = tree.query(sphere_xyz(gx.ravel(), gy.ravel()), workers=-1)
    raster = sea[valid][idx].reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    cmap = ListedColormap(["#d5c7ac", "#2674b8"])
    ax.imshow(raster.astype(np.uint8), origin="lower",
              extent=(west, east, south, north), interpolation="nearest",
              cmap=cmap, vmin=-0.5, vmax=1.5, aspect="auto")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#d5c7ac", label="Land / tmask = 0"),
                       Patch(facecolor="#2674b8", label="Ocean / tmask = 1")],
              loc="lower left")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title(f"NEMO surface land/sea tmask (nearest-neighbor raster) — {path.name}")
    output = Path(args.output) if args.output else path.with_name(path.stem + "_land_sea.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
