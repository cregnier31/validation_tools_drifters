#!/usr/bin/env python3
"""Run 10 observed drifters through 28-day GLOENS mean velocities and plot tracks.

Experimental: requires sufficient RAM to materialize 28 global ensemble means.
The daily mean fields run from day 0 noon to day 27 noon; boundary velocity
extrapolation is explicit for the midnight launch and final 36 hours.
"""
import argparse
from datetime import timedelta
import logging
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forecast_inputs import discover_gloens_stores, prepare_ensemble_forecast
from trajectory_io import retrieve_drifters_for_one_day_traj


def horizontal_coordinates(ds, var):
    """Locate 2-D geographical coordinates on a native NEMO grid."""
    da = ds[var]
    spatial = da.dims[-2:]
    candidates = [
        ("nav_lon", "nav_lat"), ("longitude", "latitude"),
        ("lon", "lat"), ("glamf", "gphif"),
    ]
    for xname, yname in candidates:
        if xname in ds and yname in ds:
            lon, lat = ds[xname], ds[yname]
            if lon.ndim == lat.ndim == 2 and lon.dims == lat.dims == spatial:
                return lon.values.astype("float32"), lat.values.astype("float32")
            if lon.ndim == lat.ndim == 1 and lon.dims == (spatial[1],) and lat.dims == (spatial[0],):
                return np.meshgrid(lon.values.astype("float32"), lat.values.astype("float32"))
    raise ValueError(
        f"Cannot identify geographical coordinates for {var}: spatial dims {spatial}, "
        f"available variables {list(ds.variables)}. Inspect native NEMO navigation coordinates."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20251106")
    parser.add_argument("--particles", type=int, default=0, help="0 means all available validated drifters")
    parser.add_argument("--output", default=None)
    parser.add_argument("--meshmask", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc",
                        help="NEMO iORCA025 meshmask containing glamu/gphiu and glamv/gphiv")
    args = parser.parse_args()
    if args.particles < 0:
        raise SystemExit("--particles must be positive")
    day = pd.Timestamp(args.date)
    end = day + pd.Timedelta(days=28)
    out = Path(args.output or f"outputs/gloens_10_drifters_{args.date}")
    out.mkdir(parents=True, exist_ok=True)

    with (ROOT / "cfg/config_drifters.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg["config_base"]
    with tempfile.TemporaryDirectory() as tmp:
        result = retrieve_drifters_for_one_day_traj(
            day, 28, base["lon_min"], base["lon_max"], base["lat_min"],
            base["lat_max"], "global", args.date, end.strftime("%Y%m%d"),
            "COPERNICUS_DAILY", cfg["path_data"]["COPERNICUS_DAILY"],
            tmp + "/", logging.getLogger("gloens-comparison"))
        if result is None:
            raise RuntimeError("No complete 28-day drifter trajectories")
        observed, _ = result
    observed = sorted(observed, key=lambda t: int(t["id"]))[:args.particles] if args.particles else sorted(observed, key=lambda t: int(t["id"]))
    if not observed:
        raise RuntimeError("No valid drifters")

    u_store, v_store, options = discover_gloens_stores(day)
    from parcels import AdvectionRK4, FieldSet, JITParticle, ParticleSet
    with prepare_ensemble_forecast(u_store, v_store, day, days=28, storage_options=options) as ds:
        # Preserve native NEMO U and V horizontal positions; do not assume
        # that their staggered grids are interchangeable.
        mesh_path = Path(args.meshmask)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"NEMO meshmask not found: {mesh_path}")
        with xr.open_dataset(mesh_path, decode_times=False) as mesh:
            # NEMO C-grid: U lives at glamu/gphiu, V at glamv/gphiv.
            def mesh_coord(name):
                a = mesh[name]
                if "time_counter" in a.dims:
                    a = a.isel(time_counter=0)
                if a.dims != ("y", "x"):
                    raise ValueError(f"{name} has unexpected dimensions {a.dims}")
                return a.values.astype("float32")
            u_lon, u_lat = mesh_coord("glamu"), mesh_coord("gphiu")
            v_lon, v_lat = mesh_coord("glamv"), mesh_coord("gphiv")
            if u_lon.shape != tuple(ds["uo"].shape[-2:]) or v_lon.shape != tuple(ds["vo"].shape[-2:]):
                raise ValueError("NEMO meshmask U/V shapes do not match GLOENS velocity grids")
            print(f"Using NEMO C-grid coordinates from {mesh_path}")
        u = ds["uo"].transpose("time", *ds["uo"].dims[-2:]).values.astype("float32")
        v = ds["vo"].transpose("time", *ds["vo"].dims[-2:]).values.astype("float32")
        if u.shape[0] != 28 or v.shape[0] != 28:
            raise ValueError("Expected 28 daily velocity fields")
        if not np.isfinite(u).any() or not np.isfinite(v).any():
            raise ValueError("No finite forecast velocities")
        times = (pd.to_datetime(ds.time.values) - day).total_seconds().to_numpy()
        # Velocity is in m/s. Parcels spherical mesh converts to degrees/s.
        fieldset = FieldSet.from_data(
            {"U": u, "V": v},
            {"U": {"lon": u_lon, "lat": u_lat, "time": times},
             "V": {"lon": v_lon, "lat": v_lat, "time": times}},
            mesh="spherical", allow_time_extrapolation=True)
        del u, v

        pset = ParticleSet(
            fieldset=fieldset, pclass=JITParticle,
            lon=[float(t["lon"][0]) for t in observed],
            lat=[float(t["lat"][0]) for t in observed],
            time=np.zeros(len(observed), dtype="float64"))
        zarr_path = out / "model_trajectories.zarr"
        if zarr_path.exists():
            raise FileExistsError(f"Refusing to overwrite {zarr_path}")
        pfile = pset.ParticleFile(name=str(zarr_path), outputdt=timedelta(hours=1))
        print("WARNING: velocity extrapolation is enabled before first noon field and after last noon field.")
        pset.execute(AdvectionRK4, runtime=timedelta(days=28),
                     dt=timedelta(minutes=30), output_file=pfile)

    with xr.open_zarr(zarr_path, decode_timedelta=True) as simulated:
        model_lon = simulated["lon"].values.copy()
        model_lat = simulated["lat"].values.copy()
        model_time = simulated["time"].values.copy()
    if model_lon.shape[0] != len(observed):
        raise AssertionError(f"Expected {len(observed)} simulated trajectories; got {model_lon.shape}")
    if not np.isfinite(model_lon).any():
        raise AssertionError("All simulated coordinates are invalid")

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for i, obs in enumerate(observed):
        color = colors[i]
        ax.plot(obs["lon"], obs["lat"], color=color, linewidth=1.7,
                label="Observed drifters" if i == 0 else None)
        ax.plot(model_lon[i], model_lat[i], color=color, linestyle="--",
                linewidth=1.5, label="GLOENS ensemble mean" if i == 0 else None)
        ax.scatter(obs["lon"][0], obs["lat"][0], color=color, s=16, zorder=3)
    ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
           title=f"GLOENS ensemble mean vs {len(observed)} observed drifters | {args.date} | 28 days")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    png = out / "comparison_2d.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(f"PASS: saved {len(observed)} model trajectories to {zarr_path}")
    print(f"PASS: saved 2D observed/model comparison to {png}")
    print("NOTE: This is an experimental comparison with extrapolated boundary velocities.")


if __name__ == "__main__":
    main()
