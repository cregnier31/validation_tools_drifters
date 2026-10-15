"""Lazy ensemble forecast velocity inputs for the Lagrangian workflow.

Adapted from class1-comparisons: forecast-cycle Zarr stores, xarray,
fsspec/S3 access and ensemble means of uo/vo prior to advection.
"""
from __future__ import annotations

from datetime import datetime
import re

import xarray as xr


def forecast_run_date(store):
    match = re.search(r"_R(\d{8})\.zarr/?$", str(store))
    if match is None:
        raise ValueError(f"No forecast cycle date in {store!r}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def open_velocity_dataset(source, *, storage_options=None, chunks=None):
    """Open local NetCDF or local/remote Zarr lazily; caller closes dataset."""
    source = str(source)
    if source.rstrip("/").endswith(".zarr"):
        if source.startswith(("s3://", "gs://", "https://", "http://")):
            import fsspec
            source = fsspec.get_mapper(source, **(storage_options or {}))
        return xr.open_zarr(source, consolidated=False, chunks=chunks or {})
    if storage_options:
        raise ValueError("storage_options only applies to Zarr")
    return xr.open_dataset(source, chunks=chunks)


def ensemble_mean_velocity(u_ds, v_ds=None, *, u_name="uo", v_name="vo",
                           ensemble_dim="ens", depth=0.0,
                           u_depth="depthu", v_depth="depthv",
                           time_dim="time"):
    """Produce mean velocities with no ensemble dimension.

    U/V stores can be separate. Native staggered spatial dimensions are
    retained; regridding/rotation must be handled by the Parcels grid adapter.
    """
    if v_ds is None:
        v_ds = u_ds
    fields = []
    for ds, var, zdim in ((u_ds, u_name, u_depth), (v_ds, v_name, v_depth)):
        if var not in ds:
            raise KeyError(f"Missing velocity {var!r}")
        field = ds[var]
        if zdim in field.dims:
            field = field.sel({zdim: depth}, method="nearest")
        if ensemble_dim not in field.dims:
            raise ValueError(f"{var!r} has no {ensemble_dim!r} dimension")
        if time_dim not in field.dims:
            raise ValueError(f"{var!r} has no {time_dim!r} dimension")
        fields.append(field.mean(ensemble_dim, skipna=True))
    if not fields[0][time_dim].identical(fields[1][time_dim]):
        raise ValueError("U and V forecast valid times do not match")
    return xr.Dataset({u_name: fields[0], v_name: fields[1]})


def select_forecast_window(ds, start, *, days=28, time_dim="time"):
    """Require lead zero and a complete time span through day 28."""
    import numpy as np
    if days <= 0:
        raise ValueError("days must be positive")
    start = np.datetime64(start)
    end = start + np.timedelta64(days, "D")
    selected = ds.sel({time_dim: slice(start, end)})
    times = selected[time_dim].values
    if len(times) < 2 or times[0] != start or times[-1] != end:
        raise ValueError(f"Incomplete {days}-day forecast window starting {start}")
    if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Forecast times must be strictly increasing")
    return selected


def prepare_ensemble_forecast(u_store, v_store, start, *, days=28,
                              storage_options=None, u_name="uo", v_name="vo",
                              ensemble_dim="ens", depth=0.0,
                              u_depth="depthu", v_depth="depthv"):
    """Context manager yielding 28-day ensemble mean U/V dataset."""
    from contextlib import contextmanager

    @contextmanager
    def opened():
        with open_velocity_dataset(u_store, storage_options=storage_options) as u_ds:
            if str(u_store) == str(v_store):
                mean = ensemble_mean_velocity(
                    u_ds, u_name=u_name, v_name=v_name, ensemble_dim=ensemble_dim,
                    depth=depth, u_depth=u_depth, v_depth=v_depth)
                yield select_forecast_window(mean, start, days=days)
            else:
                with open_velocity_dataset(v_store, storage_options=storage_options) as v_ds:
                    mean = ensemble_mean_velocity(
                        u_ds, v_ds, u_name=u_name, v_name=v_name,
                        ensemble_dim=ensemble_dim, depth=depth,
                        u_depth=u_depth, v_depth=v_depth)
                    yield select_forecast_window(mean, start, days=days)
    return opened()
