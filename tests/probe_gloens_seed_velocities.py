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
from scipy.ndimage import distance_transform_edt

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
        # Read-only CF-decoding audit: reopen the same file with decoding
        # disabled to inspect on-disk values and variable attributes.
        print("\\n=== Raw NetCDF coordinate and mask audit (decode_cf=False) ===")
        with xr.open_dataset(args.meshmask, engine="netcdf4",
                             decode_cf=False) as raw_grid:
            for name in ("glamf", "gphif", "glamt", "gphit",
                         "glamu", "gphiu", "glamv", "gphiv",
                         "tmask", "umask", "vmask", "fmask"):
                da = raw_grid[name]
                for dim in da.dims:
                    if dim not in ("y", "x"):
                        da = da.isel({dim: 0})
                a = np.asarray(da.values)
                print(f"RAW {name}: dtype={a.dtype}, attrs={dict(raw_grid[name].attrs)}")
                print(f"RAW {name}: min={np.nanmin(a)}, max={np.nanmax(a)}, "
                      f"zeros={int(np.count_nonzero(a == 0))}")
                print(f"RAW {name}: sample [652,713:715]={a[652,713:715].tolist()}")
                if name.startswith("gphi"):
                    print(f"RAW {name}: lat outside [-90,90]="
                          f"{int(np.count_nonzero(np.abs(a) > 90))}")
                if name.endswith("mask"):
                    values, counts = np.unique(a, return_counts=True)
                    print(f"RAW {name}: mask values="
                          f"{dict(zip(values.tolist(), counts.tolist()))}")
        # Audit whether zero F coordinates are associated with masks and
        # whether other staggered coordinates retain information there.
        print("\\n=== Meshmask coordinate/mask audit ===")
        flon, flat = coords(grid, "f")
        zero_f = (flon == 0) & (flat == 0)
        print("F zero-coordinate points:", int(zero_f.sum()))
        for maskname in ("tmask", "umask", "vmask", "fmask"):
            mask = grid[maskname]
            for dim in mask.dims:
                if dim not in ("y", "x"):
                    mask = mask.isel({dim: 0})
            wet = np.asarray(mask.values) != 0
            print(maskname, "surface wet:", int(wet.sum()),
                  "F zero among wet:", int((zero_f & wet).sum()),
                  "F zero among dry:", int((zero_f & ~wet).sum()))
        for suffix in ("t", "u", "v"):
            lo, la = coords(grid, suffix)
            usable = np.isfinite(lo) & np.isfinite(la) & ~((lo == 0) & (la == 0))
            print(f"{suffix.upper()} valid coordinates at F-zero points:",
                  int((usable & zero_f).sum()))
        for jy, ix in ((652, 713), (682, 679), (659, 713),
                       (653, 859), (685, 883), (657, 834), (663, 838)):
            print("Meshmask neighborhood", (jy, ix))
            for suffix in ("f", "t", "u", "v"):
                lo, la = coords(grid, suffix)
                print(suffix.upper(), "lon:",
                      lo[jy:jy+2, ix:ix+2].tolist(),
                      "lat:", la[jy:jy+2, ix:ix+2].tolist())
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
        # DIAGNOSTIC ONLY: replace invalid zero-coordinate F points in an
        # in-memory coordinate copy with nearest finite, nonzero F coordinates.
        # This is NOT a physically correct mesh repair and must never be used
        # for production advection. It tests whether zero-coordinate geometry
        # is responsible for selecting remote, zero-velocity cells.
        print("\\n=== Diagnostic in-memory zero-coordinate geometry test ===")
        f_lon = np.asarray(fcoords["glamf"].values)
        f_lat = np.asarray(fcoords["gphif"].values)
        invalid = (~np.isfinite(f_lon) | ~np.isfinite(f_lat) |
                   ((f_lon == 0) & (f_lat == 0)))
        print("Invalid F-grid coordinate points:", int(invalid.sum()))
        corrected_fieldset = None
        if invalid.any() and (~invalid).any():
            nearest = distance_transform_edt(
                invalid, return_distances=False, return_indices=True)
            lon_fixed = f_lon.copy()
            lat_fixed = f_lat.copy()
            lon_fixed[invalid] = f_lon[tuple(a[invalid] for a in nearest)]
            lat_fixed[invalid] = f_lat[tuple(a[invalid] for a in nearest)]
            fixed_coords = fcoords.copy(deep=True)
            fixed_coords["glamf"] = xr.DataArray(
                lon_fixed, dims=fcoords["glamf"].dims,
                attrs=fcoords["glamf"].attrs)
            fixed_coords["gphif"] = xr.DataArray(
                lat_fixed, dims=fcoords["gphif"].dims,
                attrs=fcoords["gphif"].attrs)
            try:
                fixed_sgrid = parcels.convert.nemo_to_sgrid(
                    fields={"U": ds["U"], "V": ds["V"]},
                    coords=fixed_coords)
                corrected_fieldset = parcels.FieldSet.from_sgrid_conventions(
                    fixed_sgrid)
                print("Diagnostic fieldset constructed (not production-safe)")
            except Exception as exc:
                print("Diagnostic fieldset construction failed:",
                      type(exc).__name__, str(exc))
        else:
            print("No replaceable invalid coordinates; skipping test")
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
        print("U interpolation implementation:")
        try:
            print(inspect.getsource(type(fieldset.U.interp_method).interp))
        except (OSError, TypeError, AttributeError) as exc:
            print("Interpolation source unavailable:", exc)
        print("U/V field data units:", repr(getattr(fieldset.U, "units", None)),
              repr(getattr(fieldset.V, "units", None)))
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
                for attr in ("state", "ei"):
                    try:
                        print(f"particle {attr}:", repr(getattr(pset, attr)))
                    except (AttributeError, KeyError) as exc:
                        print(f"particle {attr} unavailable: {exc}")
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
                # Compare the actual Parcels field data with native cached
                # values, and inspect the cell index selected by interpolation.
                for comp in ("U", "V"):
                    fld = getattr(fieldset, comp)
                    gy = row[f"{comp}_native_y"]
                    gx = row[f"{comp}_native_x"]
                    try:
                        print(f"{comp} field.data[0,0,{gy},{gx}]:",
                              repr(float(fld.data[0, 0, gy, gx])))
                    except Exception as exc:
                        print(f"{comp} field data lookup error:",
                              type(exc).__name__, str(exc))
                    print(f"{comp} interp:", repr(fld.interp_method))
                print("particle ei after sampling:", repr(pset.ei))
                # Inspect Parcels' own spatial/temporal lookup, rather than
                # guessing which cell was selected from the native grid.
                try:
                    from parcels._core.field import _get_positions
                    for comp in ("U", "V"):
                        fld = getattr(fieldset, comp)
                        pp, gp = _get_positions(
                            fld, pset.t, np.atleast_1d(pset.z),
                            np.atleast_1d(pset.y), np.atleast_1d(pset.x),
                            None, None)
                        print(f"{comp} lookup grid positions:", repr(gp))
                        print(f"{comp} lookup particle positions:", repr(pp))
                        jy = int(np.asarray(gp["Y"]["index"]).ravel()[0])
                        ix = int(np.asarray(gp["X"]["index"]).ravel()[0])
                        surface = np.asarray(fld.data.isel(time=0, depth=0).values)
                        print(f"{comp} lookup 2x2 field values at ({jy},{ix}):",
                              repr(surface[jy:jy+2, ix:ix+2]))
                        print(f"{comp} native nearest cell:",
                              (row[f"{comp}_native_y"], row[f"{comp}_native_x"]))
                        try:
                            lon_grid = np.asarray(fld.grid.lon)
                            lat_grid = np.asarray(fld.grid.lat)
                            print(f"{comp} lookup grid coordinate shapes:",
                                  lon_grid.shape, lat_grid.shape)
                            if lon_grid.ndim == 2 and lat_grid.ndim == 2:
                                print(f"{comp} lookup 2x2 lon:",
                                      repr(lon_grid[jy:jy+2, ix:ix+2]))
                                print(f"{comp} lookup 2x2 lat:",
                                      repr(lat_grid[jy:jy+2, ix:ix+2]))
                        except Exception as exc:
                            print(f"{comp} lookup grid coordinates unavailable:",
                                  type(exc).__name__, str(exc))
                        axis_map = fld.grid.get_axis_dim_mapping(fld.data.dims)
                        print(f"{comp} axis mapping:", repr(axis_map),
                              "data dims:", repr(fld.data.dims))
                except Exception as exc:
                    print("Spatial lookup diagnostic error:",
                          type(exc).__name__, str(exc))
                if corrected_fieldset is not None:
                    try:
                        fixed_uv = corrected_fieldset.UV.eval(
                            pset.t, pset.z, pset.y, pset.x)
                        print("DIAGNOSTIC corrected-coordinate UV:",
                              repr(fixed_uv))
                        # Geographic velocity units are degrees/s on a
                        # spherical mesh; convert for comparison to native m/s.
                        u_deg = float(np.asarray(fixed_uv[0]).ravel()[0])
                        v_deg = float(np.asarray(fixed_uv[1]).ravel()[0])
                        lat_rad = np.deg2rad(lat)
                        u_ms = u_deg * 111320.0 * np.cos(lat_rad)
                        v_ms = v_deg * 111320.0
                        print("DIAGNOSTIC corrected-coordinate UV approx m/s:",
                              u_ms, v_ms)
                        from parcels._core.field import _get_positions
                        for component in ("U", "V"):
                            fld = getattr(corrected_fieldset, component)
                            _, gp_fixed = _get_positions(
                                fld, pset.t, np.atleast_1d(pset.z),
                                np.atleast_1d(pset.y), np.atleast_1d(pset.x),
                                None, None)
                            jy = int(np.asarray(gp_fixed["Y"]["index"]).ravel()[0])
                            ix = int(np.asarray(gp_fixed["X"]["index"]).ravel()[0])
                            print(f"DIAGNOSTIC {component} corrected lookup:",
                                  (jy, ix), "native nearest:",
                                  (row[f"{component}_native_y"],
                                   row[f"{component}_native_x"]))
                            print(f"DIAGNOSTIC {component} corrected lookup weights:",
                                  float(np.asarray(gp_fixed["Y"]["bcoord"]).ravel()[0]),
                                  float(np.asarray(gp_fixed["X"]["bcoord"]).ravel()[0]))
                            print(f"DIAGNOSTIC {component} corrected 2x2 values:",
                                  np.asarray(fld.data.isel(time=0, depth=0,
                                      **{("y_center" if component == "U" else "y"): slice(jy,jy+2),
                                         ("x" if component == "U" else "x_center"): slice(ix,ix+2)}).values).tolist())
                    except Exception as exc:
                        print("DIAGNOSTIC corrected-coordinate error:",
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
