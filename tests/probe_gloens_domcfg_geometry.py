#!/usr/bin/env python3
"""Read-only A/B Parcels 4 velocity and spatial-lookup probe for NEMO geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from parcels._core.field import _get_positions
import parcels

SEEDS = {
    "3201769": (-72.087997, 30.035000),
    "4102665": (-66.238998, 36.525002),
    "6203615": (-78.365997, 30.718000),
    "6203642": (-77.361000, 32.047001),
}

def xyz(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack((np.cos(la)*np.cos(lo),
                            np.cos(la)*np.sin(lo), np.sin(la)))

def read_grid(path):
    with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as nc:
        arrays = {}
        for suffix in ("f", "u", "v"):
            for stem in ("glam", "gphi"):
                a = nc[stem+suffix]
                for dim in a.dims:
                    if dim not in ("y", "x"):
                        a = a.isel({dim: 0})
                arrays[stem+suffix] = np.array(a.values, copy=True)
    return arrays

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle", default="20240229")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--meshmask", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    ap.add_argument("--geometry-source", default="/scratch/cregnier/static/iORCA025_domcfg_v3.1.nc")
    ap.add_argument("--output", default="/scratch/cregnier/static/gloens_domcfg_geometry_probe.csv")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    fields = {}
    for comp, var in (("U", "uo"), ("V", "vo")):
        matches = sorted(cache.glob(f"FCST_R{args.cycle}_*_{var}_ensmean_28days.json"))
        if len(matches) != 1:
            ap.error(f"Expected exactly one {var} metadata file, got {len(matches)}")
        meta = json.loads(matches[0].read_text())
        array = np.load(matches[0].with_suffix(".npy"), mmap_mode="r")
        if tuple(meta["shape"]) != array.shape:
            raise ValueError(f"Cache shape mismatch for {var}")
        fields[comp] = xr.DataArray(array[:, 0, :, :],
             dims=("time", "y", "x"),
             coords={"time": pd.to_datetime(meta["times"])})
    ds = xr.Dataset(fields)
    results = []
    for label, path in (("original_meshmask", args.meshmask),
                        ("new_domcfg", args.geometry_source)):
        print(f"\\n=== GEOMETRY {label}: {path} ===", flush=True)
        grid = read_grid(path)
        lon, lat = grid["glamf"], grid["gphif"]
        if lon.shape != tuple(ds["U"].shape[-2:]):
            raise ValueError(f"{label}: geometry shape {lon.shape} differs from velocities {ds['U'].shape[-2:]}")
        zeros = (lon == 0) & (lat == 0)
        print(f"{label}: F paired-zero={int(zeros.sum())}, "
              f"nonfinite={int((~np.isfinite(lon) | ~np.isfinite(lat)).sum())}",
              flush=True)
        fcoords = xr.Dataset({
            "glamf": (("y","x"), lon),
            "gphif": (("y","x"), lat),
        })
        print(f"{label}: constructing SGRID and Parcels fieldset", flush=True)
        sgrid = parcels.convert.nemo_to_sgrid(
            fields={"U": ds["U"], "V": ds["V"]}, coords=fcoords)
        fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid)
        print(f"{label}: fieldset constructed", flush=True)
        trees = {}
        for comp, suffix in (("U", "u"), ("V", "v")):
            lo, la = grid["glam"+suffix], grid["gphi"+suffix]
            valid = np.isfinite(lo) & np.isfinite(la) & ~((lo == 0) & (la == 0))
            trees[comp] = (cKDTree(xyz(lo[valid],la[valid])),
                           np.flatnonzero(valid), lo.shape)
        for ident, (seed_lon, seed_lat) in SEEDS.items():
            row = {"geometry": label, "id": ident, "lon": seed_lon,
                   "lat": seed_lat, "F_paired_zero": int(zeros.sum())}
            try:
                pset = parcels.ParticleSet(
                    fieldset=fieldset, pclass=parcels.Particle,
                    x=np.array([seed_lon]), y=np.array([seed_lat]),
                    t=pd.Timestamp(ds.time.values[0]).to_pydatetime())
                uv = fieldset.UV[pset]
                u_deg = float(np.asarray(uv[0]).ravel()[0])
                v_deg = float(np.asarray(uv[1]).ravel()[0])
                row.update(U_degrees_per_second=u_deg, V_degrees_per_second=v_deg,
                    U_approx_ms=u_deg*111320*np.cos(np.deg2rad(seed_lat)),
                    V_approx_ms=v_deg*111320)
                for comp in ("U", "V"):
                    tree, indices, shape = trees[comp]
                    nearest = int(tree.query(xyz(np.array([seed_lon]),
                                                    np.array([seed_lat])))[1][0])
                    ny, nx = np.unravel_index(indices[nearest], shape)
                    row[f"{comp}_native_j"] = int(ny)
                    row[f"{comp}_native_i"] = int(nx)
                    row[f"{comp}_native_ms"] = float(ds[comp].isel(
                        time=0, y=int(ny), x=int(nx)).values)
                    fld = getattr(fieldset, comp)
                    _, gp = _get_positions(
                        fld, pset.t, np.atleast_1d(pset.z),
                        np.atleast_1d(pset.y), np.atleast_1d(pset.x),
                        None, None)
                    jy = int(np.asarray(gp["Y"]["index"]).ravel()[0])
                    ix = int(np.asarray(gp["X"]["index"]).ravel()[0])
                    row[f"{comp}_lookup_j"] = jy
                    row[f"{comp}_lookup_i"] = ix
                    row[f"{comp}_lookup_distance_cells"] = max(
                        abs(jy-int(ny)), abs(ix-int(nx)))
                    row[f"{comp}_lookup_values"] = str(np.asarray(
                        fld.data.isel(time=0, depth=0).values)[jy:jy+2,ix:ix+2].tolist())
                print(f"{label} {ident}: UV approx m/s="
                      f"({row['U_approx_ms']:.6f}, {row['V_approx_ms']:.6f}), "
                      f"U lookup=({row['U_lookup_j']},{row['U_lookup_i']}), "
                      f"V lookup=({row['V_lookup_j']},{row['V_lookup_i']}), "
                      f"native U=({row['U_native_j']},{row['U_native_i']}), "
                      f"native V=({row['V_native_j']},{row['V_native_i']})",
                      flush=True)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{label} {ident}: ERROR {row['error']}", flush=True)
            results.append(row)
        del fieldset, sgrid
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\\nSaved {args.output}", flush=True)
    print("PASS criterion: domcfg lookup near native U/V cells and nonzero "
          "plausible velocities for all four seeds. This does NOT validate trajectories.",
          flush=True)

if __name__ == "__main__":
    main()
