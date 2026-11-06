#!/usr/bin/env python3
"""Generate GLO12 -> 1/4-degree conservative weights (separate xESMF env).

Example:
python scripts/generate_glo12_025_weights.py \
  --store s3://oceanbench-bucket/public/GLO12/20240103.zarr \
  --endpoint-url https://s3.waw3-1.cloudferro.com \
  --cache-dir /scratch/cregnier/WORK_lagrangian/AdvectionRK4/glo12_025_weights
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glo12_coarsening import generate_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="Local or s3:// GLO12 Zarr store")
    parser.add_argument("--endpoint-url", default="https://s3.waw3-1.cloudferro.com")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--u-var", default="uo")
    args = parser.parse_args()
    import xarray as xr
    if args.store.startswith("s3://"):
        import fsspec
        mapper = fsspec.get_mapper(
            args.store, anon=True,
            client_kwargs={"endpoint_url": args.endpoint_url})
        ds = xr.open_zarr(mapper, consolidated=None)
    else:
        ds = xr.open_zarr(args.store, consolidated=None)
    try:
        print("Generating conservative weights from", args.store, flush=True)
        path = generate_weights(ds[args.u_var], args.cache_dir)
        print("Weights ready:", path, flush=True)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
