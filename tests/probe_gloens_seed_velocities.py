#!/usr/bin/env python3
"""Read-only native-vs-Parcels velocity probe for four frozen GLOENS drifters.

Requires an EXISTING velocity Zarr store with uo/vo and time,y,x dimensions.
Does not run advection or write into the velocity cache.
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

SEEDS = {
    "3201769": (-72.087997, 30.035000),
    "4102665": (-66.238998, 36.525002),
    "6203615": (-78.365997, 30.718000),
    "6203642": (-77.361000, 32.047001),
}

def xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack((np.cos(lat)*np.cos(lon),
                            np.cos(lat)*np.sin(lon), np.sin(lat)))

def coords(ds, prefix):
    a, b = ds["glam"+prefix], ds["gphi"+prefix]
    for dim in a.dims:
        if dim not in ("y", "x"):
            a = a.isel({dim: 0})
            b = b.isel({dim: 0})
    return np.asarray(a), np.asarray(b)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--velocity-zarr", required=True)
    ap.add_argument("--meshmask", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    ap.add_argument("--u-var", default="uo")
    ap.add_argument("--v-var", default="vo")
    ap.add_argument("--depth", type=float, default=0.)
    ap.add_argument("--output", default="gloens_seed_velocity_probe.csv")
    args = ap.parse_args()
    import parcels
    with xr.open_zarr(args.velocity_zarr, consolidated=None) as velocity, xr.open_dataset(args.meshmask, engine="netcdf4") as grid:
        surfaces = {}
        for component, name in (("U", args.u_var), ("V", args.v_var)):
            arr = velocity[name]
            dims = [d for d in arr.dims if d not in ("time", "y", "x")]
            if len(dims) != 1:
                raise ValueError(f"{name}: expected one depth dimension, got {arr.dims}")
            arr = arr.sel({dims[0]: args.depth}, method="nearest").reset_coords(drop=True)
            surfaces[component] = arr
            print(f"{component}: {arr.dims}, time={arr.time.values[0]}, depth requested={args.depth}")
        ds = xr.Dataset(surfaces)
        fcoords = grid[["glamf", "gphif"]]
        if "time_counter" in fcoords.dims:
            fcoords = fcoords.isel(time_counter=0, drop=True)
        sgrid = parcels.convert.nemo_to_sgrid(fields={"U": ds["U"], "V": ds["V"]}, coords=fcoords)
        fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid)
        print("Parcels version:", parcels.__version__)
        for comp in ("U", "V"):
            field = getattr(fieldset, comp)
            print(f"Field {comp}: interp_method={getattr(field, 'interp_method', None)}")
        grids = {}
        for comp, suffix in (("U", "u"), ("V", "v")):
            lo, la = coords(grid, suffix)
            valid = np.isfinite(lo) & np.isfinite(la)
            tree = cKDTree(xyz(lo[valid], la[valid]))
            indices = np.flatnonzero(valid)
            grids[comp] = (tree, indices, lo.shape)
        rows = []
        for ident, (lon, lat) in SEEDS.items():
            row = {"id": ident, "lon": lon, "lat": lat}
            for comp in ("U", "V"):
                tree, indices, shape = grids[comp]
                _, index = tree.query(xyz(np.array([lon]), np.array([lat]))[0])
                y, x = np.unravel_index(indices[index], shape)
                arr = surfaces[comp]
                vals = np.asarray(arr.isel(y=slice(max(0,y-1),y+2), x=slice(max(0,x-1),x+2), time=0).values)
                row[f"{comp}_native_y"] = int(y)
                row[f"{comp}_native_x"] = int(x)
                row[f"{comp}_native_nearest"] = float(arr.isel(time=0,y=y,x=x).values)
                row[f"{comp}_native_3x3"] = np.array2string(vals, precision=5)
            # Parcels 4 sampling interfaces differ by version. Report errors
            # rather than silently treating unsupported calls as zero velocity.
            t = pd.Timestamp(ds.time.values[0]).to_pydatetime()
            for comp in ("U", "V"):
                field = getattr(fieldset, comp)
                try:
                    row[f"{comp}_parcels"] = float(field.eval(t, 0., lat, lon))
                except Exception as exc:
                    row[f"{comp}_parcels_error"] = f"{type(exc).__name__}: {exc}"
            print("\n", ident, row)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print("Saved", args.output)
        print("Native velocities and Parcels interpolation are different diagnostics.")
        print("A sampling API error does not mean the physical velocity is zero.")

if __name__ == "__main__":
    main()
