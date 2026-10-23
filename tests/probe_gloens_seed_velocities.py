#!/usr/bin/env python3
"""Read-only native-vs-Parcels velocity probe for four frozen GLOENS drifters.

Requires existing read-only NumPy ensemble-mean velocity cache files.
Does not run advection or write into the velocity cache.
"""
import argparse
import json
from pathlib import Path
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
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--cycle", default="20240229")
    ap.add_argument("--meshmask", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    ap.add_argument("--u-var", default="uo")
    ap.add_argument("--v-var", default="vo")
    ap.add_argument("--depth", type=float, default=0.)
    ap.add_argument("--output", default="gloens_seed_velocity_probe.csv")
    args = ap.parse_args()
    import parcels
    cache = Path(args.cache_dir)
    surfaces = {}
    metadata = {}
    for component, name in (("U", args.u_var), ("V", args.v_var)):
        matches = sorted(cache.glob(f"FCST_R{args.cycle}_*_{name}_ensmean_28days.json"))
        if len(matches) != 1:
            ap.error(f"Expected one {name} metadata file, found {len(matches)}")
        meta = json.loads(matches[0].read_text())
        array = np.load(matches[0].with_suffix(".npy"), mmap_mode="r")
        if tuple(meta["shape"]) != array.shape:
            raise ValueError(f"{name} shape mismatch: {meta['shape']} != {array.shape}")
        metadata[component] = (meta, array)
        print(f"{component}: shape={array.shape}, dtype={array.dtype}, ensemble_size={meta.get('ensemble_size')}")
    with xr.open_dataset(args.meshmask, engine="netcdf4") as grid:
        fcoords = grid[["glamf", "gphif"]]
        if "time_counter" in fcoords.dims:
            fcoords = fcoords.isel(time_counter=0, drop=True)
        for component in ("U", "V"):
            meta, array = metadata[component]
            if args.depth != 0:
                raise ValueError("Depth coordinates absent in cache metadata; use --depth 0")
            surfaces[component] = xr.DataArray(array[:, 0, :, :], dims=("time", "y", "x"),
                coords={"time": pd.to_datetime(meta["times"])})
        ds = xr.Dataset(surfaces)
        sgrid = parcels.convert.nemo_to_sgrid(fields={"U": ds["U"], "V": ds["V"]}, coords=fcoords)
        # Compare native and SGRID values at the exact same array indices.
        # This is a value-preservation check, not proof of spatial alignment.
        print("\\n=== Native versus converted SGRID values ===")
        locations = [
            ("U", 653, 859), ("V", 653, 860),
            ("U", 685, 883), ("V", 685, 883),
            ("U", 657, 834), ("V", 656, 835),
            ("U", 663, 838), ("V", 663, 839),
        ]
        for component, y, x in locations:
            original = ds[component]
            converted = sgrid[component]
            print(f"{component} at y={y}, x={x}: "
                  f"native dims={original.dims}, shape={original.shape}; "
                  f"SGRID dims={converted.dims}, shape={converted.shape}")
            try:
                native_value = float(original.isel(time=0, y=y, x=x).values)
                spatial_dims = ("y_center", "x") if component == "U" else ("y", "x_center")
                expected_dims = {"depth", "time", *spatial_dims}
                if not expected_dims.issubset(converted.dims):
                    print(f"  native={native_value:.8f}; "
                          f"unexpected SGRID dimensions: {converted.dims}")
                    continue
                converted_value = float(converted.isel({
                    "depth": 0, "time": 0, spatial_dims[0]: y, spatial_dims[1]: x
                }).values)
                print(f"  native={native_value:.8f}; "
                      f"SGRID_same_index={converted_value:.8f}; "
                      f"difference={converted_value-native_value:+.8f}")
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                print(f"  comparison unavailable: {type(exc).__name__}: {exc}")
        print("Note: same-index comparison alone does not establish "
              "geographic grid alignment.\\n")
        print("\\n=== SGRID coordinate metadata ===")
        for component in ("U", "V"):
            da = sgrid[component]
            print(component, "attrs=", dict(da.attrs))
            for cname in da.coords:
                coord = da.coords[cname]
                print(" ", cname, "dims=", coord.dims, "shape=", coord.shape,
                      "attrs=", dict(coord.attrs))
        print("SGRID dataset attrs=", dict(sgrid.attrs))
        fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid)
        print("\\n=== Parcels field grid metadata ===")
        for component in ("U", "V"):
            field = getattr(fieldset, component)
            g = field.grid
            for attr in ("lon", "lat", "time"):
                val = getattr(g, attr, None)
                if val is not None:
                    print(component, attr, "shape=", np.shape(val),
                          "dtype=", getattr(np.asarray(val), "dtype", None))
            print(component, "field data shape=", np.shape(getattr(field, "data", None)))
        print("\\n=== Parcels versus native grid coordinates at seed cells ===")
        for component, y, x in [
            ("U", 653, 859), ("V", 653, 860),
            ("U", 685, 883), ("V", 685, 883),
            ("U", 657, 834), ("V", 656, 835),
            ("U", 663, 838), ("V", 663, 839),
        ]:
            g = getattr(fieldset, component).grid
            print(f"{component} y={y} x={x}")
            for suffix in ("u", "v", "f"):
                native_lon, native_lat = coords(grid, suffix)
                print(f"  NEMO {suffix.upper()}: "
                      f"lon={native_lon[y,x]:.6f} lat={native_lat[y,x]:.6f}")
            for attr in ("lon", "lat"):
                a = np.asarray(getattr(g, attr))
                if a.ndim == 2 and y < a.shape[0] and x < a.shape[1]:
                    print(f"  Parcels {attr}: {float(a[y,x]):.6f}")
                else:
                    print(f"  Parcels {attr}: unexpected shape {a.shape}")
        # Discover the installed Parcels 4 API rather than guessing constructors.
        import inspect
        print("\\n=== Installed Parcels fieldset API discovery ===")
        print("Parcels version:", parcels.__version__)
        print("FieldSet constructors:")
        for name in sorted(n for n in dir(parcels.FieldSet)
                           if n.startswith("from_")):
            method = getattr(parcels.FieldSet, name)
            try:
                signature = str(inspect.signature(method))
            except (TypeError, ValueError):
                signature = "(signature unavailable)"
            print(f"  {name}{signature}")
        print("Field constructors:")
        for name in sorted(n for n in dir(parcels.Field)
                           if n.startswith("from_")):
            method = getattr(parcels.Field, name)
            try:
                signature = str(inspect.signature(method))
            except (TypeError, ValueError):
                signature = "(signature unavailable)"
            print(f"  {name}{signature}")
        print("UV type:", type(fieldset.UV))
        print("UV __getitem__ signature:", inspect.signature(type(fieldset.UV).__getitem__))
        print("U eval signature:", inspect.signature(fieldset.U.eval))
        print("V eval signature:", inspect.signature(fieldset.V.eval))
        print("U grid numeric time endpoints:",
              float(fieldset.U.grid.time[0]), float(fieldset.U.grid.time[-1]))
        print("U grid time_origin:", repr(getattr(fieldset.U.grid, "time_origin", None)))
        print("UV __getitem__ implementation:")
        try:
            print(inspect.getsource(type(fieldset.UV).__getitem__))
        except (OSError, TypeError) as exc:
            print("Source unavailable:", exc)
        print("U eval implementation:")
        try:
            print(inspect.getsource(fieldset.U.eval))
        except (OSError, TypeError) as exc:
            print("Source unavailable:", exc)
        print("FieldSet signature:", inspect.signature(parcels.FieldSet))
        print("Field signature:", inspect.signature(parcels.Field))
        print("Direct comparison deferred until an installed constructor "
              "and its required inputs are verified.")
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
            # Sample with the same particle-based API used by the production
            # RK4 kernel, avoiding datetime arguments to Field.eval().
            try:
                pset = parcels.ParticleSet(
                    fieldset=fieldset, pclass=parcels.Particle,
                    x=np.array([lon]), y=np.array([lat]),
                    t=pd.Timestamp(ds.time.values[0]).to_pydatetime())
                print(f"\\n=== Sampling state for {ident} ===")
                print("particle t:", repr(pset.t), "z:", repr(pset.z))
                print("particle x:", repr(pset.x), "y:", repr(pset.y))
                for attr in ("state", "ei", "xi", "yi", "zi", "ti"):
                    if hasattr(pset, attr):
                        print(f"particle {attr}:", repr(getattr(pset, attr)))
                try:
                    direct_uv = fieldset.UV.eval(pset.t, pset.z, pset.y, pset.x)
                    print("UV.eval without particles:", repr(direct_uv))
                except Exception as exc:
                    print("UV.eval without particles error:",
                          type(exc).__name__, str(exc))
                try:
                    direct_u = fieldset.U.eval(pset.t, pset.z, pset.y, pset.x)
                    direct_v = fieldset.V.eval(pset.t, pset.z, pset.y, pset.x)
                    print("U/V.eval without particles:", repr(direct_u), repr(direct_v))
                except Exception as exc:
                    print("U/V.eval without particles error:",
                          type(exc).__name__, str(exc))
                u_sample, v_sample = fieldset.UV[pset]
                print("UV[pset]:", repr(u_sample), repr(v_sample))
                if hasattr(pset, "state"):
                    print("particle state after UV[pset]:", repr(pset.state))
                row["U_parcels"] = float(np.asarray(u_sample).ravel()[0])
                row["V_parcels"] = float(np.asarray(v_sample).ravel()[0])
            except Exception as exc:
                row["UV_parcels_error"] = f"{type(exc).__name__}: {exc}"
            print("\n", ident, row)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print("Saved", args.output)
        print("Native velocities and Parcels interpolation are different diagnostics.")
        print("A sampling API error does not mean the physical velocity is zero.")

if __name__ == "__main__":
    main()
