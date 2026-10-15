"""Input adapters for Lagrangian validation.

Keep forecast storage independent from particle integration. Zarr stores may
be local or accessible via fsspec (including s3:// URLs).
"""
from __future__ import annotations

from pathlib import Path
import re

import xarray as xr


def open_velocity_dataset(source, *, storage_options=None, chunks=None):
    """Open a velocity dataset without loading all ensemble data into memory.

    NetCDF paths and local/remote Zarr stores are supported. The caller owns
    the returned dataset and should close it after the particle run.
    """
    source = str(source)
    if source.rstrip("/").endswith(".zarr"):
        if source.startswith(("s3://", "gs://", "https://", "http://")):
            import fsspec
            mapper = fsspec.get_mapper(source, **(storage_options or {}))
            return xr.open_zarr(mapper, consolidated=False, chunks=chunks or {})
        return xr.open_zarr(source, consolidated=False, chunks=chunks or {})
    if storage_options:
        raise ValueError("storage_options are only supported for Zarr inputs")
    return xr.open_dataset(source, chunks=chunks)


def surface_ensemble_mean(ds, *, u_name="uo", v_name="vo",
                          ensemble_dim="ens", depth=0.0,
                          depth_dims=("depthu", "depthv")):
    """Return an xarray Dataset with mean U and V on their native grids.

    Mean the velocity components *before* particle advection. The mean
    particle trajectory is generally not the trajectory of the mean flow.
    """
    fields = {}
    for name, dim in ((u_name, depth_dims[0]), (v_name, depth_dims[1])):
        if name not in ds:
            raise KeyError(f"Velocity variable {name!r} not found")
        data = ds[name]
        if dim in data.dims:
            data = data.sel({dim: depth}, method="nearest")
        if ensemble_dim not in data.dims:
            raise ValueError(f"{name}: missing ensemble dimension {ensemble_dim!r}")
        fields[name] = data.mean(dim=ensemble_dim, skipna=True)
    return xr.Dataset(fields, attrs=dict(ds.attrs))


def forecast_window(ds, *, start, days=28, time_dim="time"):
    """Select one forecast cycle including the initial position at lead zero.

    Validate that the requested window is fully represented; never silently
    run a shorter trajectory because lead times are missing.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    if time_dim not in ds.coords:
        raise KeyError(f"Missing time coordinate {time_dim!r}")
    import numpy as np
    begin = np.datetime64(start)
    end = begin + np.timedelta64(int(days), "D")
    selected = ds.sel({time_dim: slice(begin, end)})
    times = selected[time_dim].values
    if len(times) < 2 or times[0] > begin or times[-1] < end:
        raise ValueError(f"Incomplete {days}-day window from {start}")
    return selected


def forecast_run_date(path):
    """Parse the cycle date in CLASS1-style *_RYYYYMMDD.zarr names."""
    match = re.search(r"_R(\d{8})\.zarr/?$", str(path))
    if not match:
        raise ValueError(f"Cannot identify forecast cycle in {path!r}")
    from datetime import datetime
    return datetime.strptime(match.group(1), "%Y%m%d").date()
