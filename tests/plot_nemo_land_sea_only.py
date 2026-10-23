#!/usr/bin/env python3
"""Plot only the surface NEMO tmask as a binary land/sea map."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from netCDF4 import Dataset

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
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    # Render land first, ocean second; no particle markers or annotations.
    for state, color, label in ((False, "#d5c7ac", "Land / tmask != 1"),
                                (True, "#2674b8", "Ocean / tmask = 1")):
        sel = valid & (sea == state)
        ax.scatter(lon[sel], lat[sel], s=0.7 if args.extent is None else 4,
                   c=color, marker="s", linewidths=0, rasterized=True, label=label)
    if args.extent:
        ax.set_xlim(west, east); ax.set_ylim(south, north)
    else:
        ax.set_xlim(-180, 180); ax.set_ylim(-85, 90)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title(f"NEMO surface land/sea tmask — {path.name}")
    ax.legend(loc="lower left", markerscale=6)
    ax.set_aspect("auto")
    output = Path(args.output) if args.output else path.with_name(path.stem + "_land_sea.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
