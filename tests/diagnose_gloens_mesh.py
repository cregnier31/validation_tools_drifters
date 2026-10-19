#!/usr/bin/env python3
"""Read-only diagnostic of NEMO meshmask geometry and GLOENS failure positions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshmask", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path,
                        help="Existing *_grid_failure_particles.csv")
    parser.add_argument("--output", type=Path, default=Path("nemo_grid_diagnostic.png"))
    args = parser.parse_args()

    with xr.open_dataset(args.meshmask) as grid:
        names = ("glamt", "gphit", "glamu", "gphiu",
                 "glamv", "gphiv", "glamf", "gphif")
        for name in names:
            if name in grid:
                v = grid[name]
                print(f"{name}: shape={v.shape}, dims={v.dims}")
            else:
                print(f"{name}: MISSING")
        lon = grid["glamf"]
        lat = grid["gphif"]
        if "time_counter" in lon.dims:
            lon = lon.isel(time_counter=0)
            lat = lat.isel(time_counter=0)
        lon = np.asarray(lon.values, dtype=float).squeeze()
        lat = np.asarray(lat.values, dtype=float).squeeze()

    if lon.ndim != 2 or lat.shape != lon.shape:
        raise ValueError(f"Expected matching 2D F-point coordinates: {lon.shape}, {lat.shape}")
    invalid = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lat) > 90)
    zero = np.isclose(lon, 0) & np.isclose(lat, 0)
    # Inspect adjacent F-point longitude jumps; this is a diagnostic,
    # not a proof that the corresponding Parcels faces are degenerate.
    jump_x = np.abs(np.diff(lon, axis=1)) > 180
    jump_y = np.abs(np.diff(lon, axis=0)) > 180
    print(f"F-point invalid coordinates: {invalid.sum()}")
    print(f"F-point (0,0) coordinates: {zero.sum()}")
    print(f"Adjacent longitude jumps >180 degrees: x={jump_x.sum()}, y={jump_y.sum()}")
    for j, i in ((0, 356), (0, 526), (0, 934), (0, 1019), (1, 356)):
        if j < lon.shape[0] and i < lon.shape[1]:
            print(f"Parcels warning example j={j} i={i}: "
                  f"F-point lon={lon[j,i]:.6f} lat={lat[j,i]:.6f}")

    frame = pd.read_csv(args.failures)
    suspects = frame[frame["suspect"].astype(str).str.lower().eq("true")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True)
    # Subsample coordinates for responsive plotting.
    stride = max(1, int(np.ceil(lon.size / 150000)))
    flatlon, flatlat = lon.ravel()[::stride], lat.ravel()[::stride]
    axes[0].scatter(flatlon, flatlat, s=0.1, c="0.65", alpha=0.15,
                    label="F-point coordinates (sample)")
    axes[0].scatter(suspects["seed_lon"], suspects["seed_lat"],
                    s=20, c="crimson", marker="x", label="Suspect seeds")
    axes[0].set_title("Suspect particle positions versus NEMO F points")
    bad = invalid | zero
    axes[1].scatter(lon[bad], lat[bad], s=2, c="orange",
                    label="Invalid or (0,0) F points")
    axes[1].scatter(suspects["seed_lon"], suspects["seed_lat"],
                    s=20, c="crimson", marker="x", label="Suspect seeds")
    axes[1].set_title("Invalid/zero mesh coordinates and suspect particles")
    for ax in axes:
        ax.set(xlim=(-180, 180), ylim=(-90, 90), ylabel="Latitude")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left")
    axes[-1].set_xlabel("Longitude")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
