#!/usr/bin/env python3
"""Read-only comparison of two NEMO mesh files; no Parcels dependency."""
import argparse
import hashlib
import numpy as np
import xarray as xr

VARS = ("glamf", "gphif", "glamt", "gphit", "glamu", "gphiu",
        "glamv", "gphiv", "tmask", "umask", "vmask", "fmask")
COORDS = ("f", "t", "u", "v")

def read_2d(path, name):
    with xr.open_dataset(path, engine="netcdf4", decode_cf=False,
                         mask_and_scale=False, cache=False) as ds:
        a = ds[name]
        for dim in a.dims:
            if dim not in ("y", "x"):
                a = a.isel({dim: 0})
        return np.array(a.values, copy=True)

def summarize(path, label):
    print(f"\\n=== {label}: {path} ===", flush=True)
    with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
        print("dimensions:", dict(ds.sizes), flush=True)
        print("global attributes:", dict(ds.attrs), flush=True)
    arrays = {}
    for name in VARS:
        a = read_2d(path, name)
        arrays[name] = a
        digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
        print(f"{label} {name}: shape={a.shape} dtype={a.dtype} "
              f"min={np.nanmin(a):.6g} max={np.nanmax(a):.6g} "
              f"zero={np.count_nonzero(a == 0)} sha256_16={digest}", flush=True)
        if name.endswith("mask"):
            binary = (a == 0) | (a == 1)
            print(f"{label} {name}: eq0={np.count_nonzero(a == 0)} "
                  f"eq1={np.count_nonzero(a == 1)} "
                  f"nonbinary={np.count_nonzero(~binary)}", flush=True)
    for suffix in COORDS:
        lon, lat = arrays["glam"+suffix], arrays["gphi"+suffix]
        zero = (lon == 0) & (lat == 0)
        invalid = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lat) > 90)
        print(f"{label} {suffix.upper()} coordinates: paired_zero={zero.sum()} "
              f"invalid_numeric={invalid.sum()}", flush=True)
        if suffix+"mask" in arrays:
            mask = arrays[suffix+"mask"]
            print(f"{label} {suffix.upper()} paired_zero mask_eq1="
                  f"{np.count_nonzero(zero & (mask == 1))} "
                  f"mask_eq0={np.count_nonzero(zero & (mask == 0))} "
                  f"mask_nonbinary={np.count_nonzero(zero & (mask != 0) & (mask != 1))}",
                  flush=True)
        # A rough discontinuity metric, not a mesh validity test. Longitude
        # differences are wrapped to [-180,180] to respect the dateline.
        for axis, axis_name in ((0, "j"), (1, "i")):
            dl = np.diff(lon, axis=axis)
            dl = (dl + 180) % 360 - 180
            dp = np.diff(lat, axis=axis)
            valid_pair = np.take(~zero & ~invalid, range(lon.shape[axis]-1), axis=axis) & np.take(~zero & ~invalid, range(1,lon.shape[axis]), axis=axis)
            jumps = valid_pair & ((np.abs(dl) > 5) | (np.abs(dp) > 5))
            print(f"{label} {suffix.upper()} {axis_name}-neighbor jumps_gt_5deg="
                  f"{int(jumps.sum())}", flush=True)
    return arrays

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    parser.add_argument("--reference", default="/scratch/cregnier/static/allmeshes.nc")
    args = parser.parse_args()
    current = summarize(args.current, "CURRENT")
    reference = summarize(args.reference, "REFERENCE")
    print("\\n=== Cross-file comparison ===", flush=True)
    for name in VARS:
        a, b = current[name], reference[name]
        print(f"{name}: shapes={a.shape} vs {b.shape}; "
              f"same_index_comparison={'possible' if a.shape == b.shape else 'INVALID (different grids)'}",
              flush=True)
    print("Reference-grid success with an older Parcels version does not "
          "establish compatibility with Parcels 4 spatial hashing.", flush=True)

if __name__ == "__main__":
    main()
