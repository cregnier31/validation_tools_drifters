#!/usr/bin/env python3
"""Read-only first-step diagnostic for frozen GLOENS drifters (20240229).

Compares nearest native NEMO U/V grid values and masks with existing Parcels
trajectory motion. This is NOT a Parcels-interpolated velocity measurement.
Never writes to or modifies the velocity cache.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

TARGETS = ["3201769", "4102665", "6203615", "6203642"]


def nearest(grid, lon, lat):
    lon = np.asarray(grid[0], dtype=float)
    lat = np.asarray(grid[1], dtype=float)
    dist2 = (((lon - float(grid[2]) + 180) % 360 - 180) *
             np.cos(np.deg2rad(float(grid[3]))))**2 + (lat - float(grid[3]))**2
    if not np.isfinite(dist2).any():
        return None
    return tuple(int(v) for v in np.unravel_index(np.nanargmin(dist2), dist2.shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshmask", required=True, help="Path to existing NEMO meshmask NetCDF")
    ap.add_argument("--velocity-zarr", help="Existing cached ensemble-mean Zarr store (optional)")
    ap.add_argument("--u-var", default="uo")
    ap.add_argument("--v-var", default="vo")
    ap.add_argument("--cycle", default="20240229")
    ap.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    args = ap.parse_args()
    root = Path(args.root)
    cycle_dirs = list(root.glob(
        f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift10_dt5_{args.cycle}-{args.cycle}/ZONE_*/{args.cycle}"))
    if len(cycle_dirs) != 1:
        ap.error(f"Expected one cycle directory, got {len(cycle_dirs)}")
    output = cycle_dirs[0] / "gulf_stream_extension"
    output.mkdir(parents=True, exist_ok=True)
    observations = list((cycle_dirs[0].parent / "DRIFTERS").glob("Drifters_*.csv"))
    if len(observations) != 1:
        ap.error("Expected one observations CSV")
    obs = pd.read_csv(observations[0], usecols=["id", "time", "lon", "lat"])
    obs.id = obs.id.astype(str)
    obs.time = pd.to_datetime(obs.time)
    start = pd.Timestamp(args.cycle) + pd.Timedelta(hours=12)
    seeds = obs[obs.time == start].drop_duplicates("id").set_index("id")
    missing = set(TARGETS) - set(seeds.index)
    if missing:
        ap.error(f"Missing initial observed drifter IDs: {sorted(missing)}")
    rows = []
    with xr.open_dataset(args.meshmask, engine="netcdf4") as mask:
        coords = {}
        for name in ["glamu", "gphiu", "glamv", "gphiv", "glamf", "gphif", "glamt", "gphit"]:
            if name in mask:
                a = mask[name]
                for dim in a.dims:
                    if dim not in ("y", "x"):
                        a = a.isel({dim: 0})
                coords[name] = np.asarray(a.values)
        for ident in TARGETS:
            seed = seeds.loc[ident]
            lon, lat = float(seed.lon), float(seed.lat)
            for loc, pair in [("U", ("glamu", "gphiu")), ("V", ("glamv", "gphiv")),
                              ("F", ("glamf", "gphif")), ("T", ("glamt", "gphit"))]:
                if not all(n in coords for n in pair):
                    continue
                idx = nearest((coords[pair[0]], coords[pair[1]], lon, lat))
                if idx is None:
                    continue
                y, x = idx
                row = dict(id=ident, seed_lon=lon, seed_lat=lat, grid=loc,
                           y=y, x=x, grid_lon=float(coords[pair[0]][idx]),
                           grid_lat=float(coords[pair[1]][idx]))
                for name in ("umask", "vmask", "tmask", "fmask"):
                    if name in mask:
                        a = mask[name]
                        try:
                            row[name] = float(a.isel({d: 0 for d in a.dims if d not in ("y", "x")}).isel(y=y, x=x).values)
                        except (KeyError, IndexError, ValueError):
                            pass
                rows.append(row)
    df = pd.DataFrame(rows)
    if args.velocity_zarr:
        # Read-only opening; select first time and surface level before any
        # spatial indexing. Native staggered U/V values are NOT Parcels UV.
        with xr.open_zarr(args.velocity_zarr, consolidated=None) as vel:
            for component, var in (("U", args.u_var), ("V", args.v_var)):
                if var not in vel:
                    raise SystemExit(f"{var} missing from {args.velocity_zarr}")
                da = vel[var]
                selection = {d: 0 for d in da.dims if d not in ("y", "x")}
                surface = da.isel(selection)
                for row in rows:
                    if row["grid"] != component:
                        continue
                    try:
                        row["native_velocity_first_time"] = float(
                            surface.isel(y=row["y"], x=row["x"]).values)
                    except (KeyError, IndexError, ValueError) as exc:
                        row["native_velocity_error"] = str(exc)
        df = pd.DataFrame(rows)
    filename = output / "coastal_native_grid_audit.csv"
    df.to_csv(filename, index=False)
    print(df.to_string(index=False))
    print("Saved", filename)
    print("Note: nearest native-grid values are NOT Parcels-interpolated velocities.")
    print("A nonzero native velocity and frozen track would warrant FieldSet/kernel probing.")


if __name__ == "__main__":
    main()
