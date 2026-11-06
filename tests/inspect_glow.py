#!/usr/bin/env python3
"""Inspect GLOW forecast Zarr metadata and local grid mask."""

import argparse
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr


ENDPOINT = "https://s3.waw3-1.cloudferro.com"
BUCKET = "oceanbench-bucket"
PREFIX = "dev/ml-forecast-outputs/glowcascade_v5_ring"


def describe_dataset(ds):
    print("\n=== DATASET ===")
    print(ds)
    print("\n=== VELOCITY VARIABLES ===")
    for name in ("uo", "vo"):
        if name not in ds:
            print(f"WARNING: {name} not found")
            continue
        da = ds[name]
        print(f"\n{name}")
        print("Dimensions:", da.dims)
        print("Shape:", da.shape)
        print("Attributes:", dict(da.attrs))
    print("\n=== COORDINATES ===")
    for name, coord in ds.coords.items():
        print(f"\n{name}")
        print("Dimensions:", coord.dims)
        print("Shape:", coord.shape)
        print("Attributes:", dict(coord.attrs))
        if coord.ndim == 1:
            values = coord.values
            print("First values:", values[:5])
            print("Last values:", values[-5:])


def find_nearest_depth(ds, target_depth):
    print("\n=== DEPTH SELECTION ===")
    candidates = [
        name for name in ds.coords
        if "depth" in name.lower() or name.lower() in ("z", "lev", "level")
    ]
    if not candidates:
        print("No obvious depth coordinate found.")
        return
    for name in candidates:
        coord = ds[name]
        if coord.ndim != 1:
            print(f"{name}: not a 1D coordinate")
            continue
        try:
            values = np.asarray(coord.values, dtype=float)
        except (TypeError, ValueError):
            continue
        finite = np.isfinite(values)
        if not finite.any():
            continue
        valid_indices = np.flatnonzero(finite)
        nearest = valid_indices[np.argmin(np.abs(values[finite] - target_depth))]
        print(f"Coordinate: {name}")
        print(f"Available depths: {values}")
        print(f"Requested depth: {target_depth} m")
        print(f"Nearest depth: {values[nearest]} m")
        print(f"Selected index: {nearest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20240925")
    parser.add_argument("--depth", type=float, default=15.0)
    parser.add_argument("--mask", default="static/mask_GLOW.nc")
    args = parser.parse_args()
    store = f"{BUCKET}/{PREFIX}/{args.date}.zarr"
    print("S3 endpoint:", ENDPOINT)
    print("Zarr store:", f"s3://{store}")
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": ENDPOINT},
    )
    mapper = s3fs.S3Map(root=store, s3=fs, check=False)
    ds = xr.open_zarr(mapper, consolidated=False, chunks={})
    try:
        describe_dataset(ds)
        find_nearest_depth(ds, args.depth)
    finally:
        ds.close()
    print("\n=== STATIC MASK ===")
    mask_path = Path(args.mask)
    if not mask_path.is_file():
        print(f"Mask not found: {mask_path.resolve()}")
        return
    with xr.open_dataset(mask_path) as mask:
        print(mask)
        for name, da in mask.data_vars.items():
            print(f"{name}: dimensions={da.dims}, shape={da.shape}")


if __name__ == "__main__":
    main()
