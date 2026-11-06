#!/usr/bin/env python3
"""Read-only smoke test of the CLASS1 GLOENS S3 Zarr forecast inputs.

Run from repository root:
    python tests/smoke_gloens_zarr.py --run-date 20251106

Requires xarray, zarr, dask, s3fs and access to the CloudFerro endpoint.
Does NOT launch Parcels or read a complete global ensemble into memory.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_inputs import discover_gloens_stores, open_velocity_dataset


def inspect(store, variable, depth_name, options):
    print(f"\nOpening {variable}: {store}", flush=True)
    with open_velocity_dataset(store, storage_options=options) as ds:
        print("Dimensions:", dict(ds.sizes))
        print("Variables:", list(ds.data_vars))
        if variable not in ds:
            raise AssertionError(f"Expected {variable!r} in {store}")
        da = ds[variable]
        print(f"{variable} dimensions:", da.dims)
        for dim in ("ens", "time"):
            if dim not in da.dims or da.sizes[dim] == 0:
                raise AssertionError(f"{variable}: missing/empty {dim} dimension")
        if depth_name in da.dims:
            da = da.isel({depth_name: 0})
        # Select a single member, one forecast time and one grid point
        # before requesting data from the object store.
        indexers = {"ens": 0, "time": 0}
        for dim in da.dims:
            if dim not in indexers:
                indexers[dim] = 0
        sample = da.isel(indexers).compute()
        print(f"{variable} first-member sample:", sample.values.item())
        print(f"{variable} forecast first time:", ds["time"].values[0])
        print(f"{variable} forecast last time:", ds["time"].values[-1])
        return ds.sizes["time"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True, help="Forecast cycle YYYYMMDD")
    parser.add_argument("--bucket", default="MOISICEEF")
    parser.add_argument("--endpoint", default="https://s3.waw3-1.cloudferro.com")
    args = parser.parse_args()

    u, v, options = discover_gloens_stores(
        args.run_date, bucket=args.bucket, endpoint_url=args.endpoint
    )
    print("Discovered U:", u)
    print("Discovered V:", v)
    n_u = inspect(u, "uo", "depthu", options)
    n_v = inspect(v, "vo", "depthv", options)
    if n_u != n_v:
        raise AssertionError(f"U/V time length mismatch: {n_u} vs {n_v}")
    print("\nPASS: both GLOENS Zarr stores opened and sampled successfully.")


if __name__ == "__main__":
    main()
