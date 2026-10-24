#!/usr/bin/env python3
"""Copy a NEMO meshmask NetCDF, normalizing all four masks to exactly 0 or 1.

All non-mask variables and metadata are copied by netCDF4. The original file
is never modified. Writes to a temporary file then atomically renames.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

MASKS = ("tmask", "umask", "vmask", "fmask")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    parser.add_argument("--output", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1_binary.nc")
    args = parser.parse_args()
    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    if src == dst:
        parser.error("Input and output must be different files")
    if not src.is_file():
        parser.error(f"Input does not exist: {src}")
    if dst.exists():
        parser.error(f"Output already exists; refusing to overwrite: {dst}")
    if not dst.parent.is_dir():
        parser.error(f"Output directory does not exist: {dst.parent}")
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    if tmp.exists():
        parser.error(f"Temporary file already exists: {tmp}")

    try:
        with Dataset(src, "r") as source, Dataset(tmp, "w", format=source.data_model) as target:
            target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
            for name, dim in source.dimensions.items():
                target.createDimension(name, None if dim.isunlimited() else len(dim))
            missing = set(MASKS) - set(source.variables)
            if missing:
                raise ValueError(f"Missing required masks: {sorted(missing)}")
            for name, var in source.variables.items():
                kwargs = {}
                # Retain filters and chunking where supported.
                if source.data_model == "NETCDF4":
                    filters = var.filters() or {}
                    for key in ("zlib", "complevel", "shuffle", "fletcher32"):
                        if key in filters and filters[key] is not None:
                            kwargs[key] = filters[key]
                    chunks = var.chunking()
                    if isinstance(chunks, list):
                        kwargs["chunksizes"] = tuple(chunks)
                    elif chunks == "contiguous":
                        kwargs["contiguous"] = True
                fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
                if fill is not None:
                    kwargs["fill_value"] = fill
                new = target.createVariable(name, var.datatype, var.dimensions, **kwargs)
                new.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs() if attr != "_FillValue"})
                # Chunk along the first dimension to avoid loading huge 4D arrays.
                if not var.shape:
                    new.assignValue(var.getValue())
                    continue
                for i in range(var.shape[0]):
                    slab = var[i, ...]
                    if name in MASKS:
                        raw = np.asarray(np.ma.filled(slab, 0))
                        slab = np.asarray(raw == 1, dtype=var.dtype)
                    new[i, ...] = slab
                if name in MASKS:
                    print(f"Normalized {name}: shape={var.shape}", flush=True)
            target.sync()
        with Dataset(tmp, "r") as check:
            for name in MASKS:
                var = check.variables[name]
                counts = {0: 0, 1: 0}
                for i in range(var.shape[0]):
                    values = np.asarray(np.ma.filled(var[i, ...], 0))
                    invalid = np.count_nonzero((values != 0) & (values != 1))
                    if invalid:
                        raise ValueError(f"Validation failed: {name} has {invalid} nonbinary values")
                    counts[0] += int(np.count_nonzero(values == 0))
                    counts[1] += int(np.count_nonzero(values == 1))
                print(f"{name}: zero={counts[0]:,}, one={counts[1]:,}")
            print(f"Variables preserved: {len(check.variables)}")
        if dst.exists():
            raise FileExistsError(f"Output appeared during processing: {dst}")
        os.rename(tmp, dst)
        print(f"Created: {dst}")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


if __name__ == "__main__":
    main()
