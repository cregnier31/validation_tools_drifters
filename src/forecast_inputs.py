"""Lazy ensemble forecast velocity inputs for the Lagrangian workflow.

Adapted from class1-comparisons: forecast-cycle Zarr stores, xarray,
fsspec/S3 access and ensemble means of uo/vo prior to advection.
"""
from __future__ import annotations

from datetime import datetime
import re
import time
import logging
import sys
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
            import s3fs
            if source.startswith("s3://"):
                source = s3fs.S3Map(root=source, s3=s3fs.S3FileSystem(**(storage_options or {})), check=False)
            else:
                source = fsspec.get_mapper(source, **(storage_options or {}))
        return xr.open_zarr(source, consolidated=False, chunks=chunks or {})
    if storage_options:
        raise ValueError("storage_options only applies to Zarr")
    return xr.open_dataset(source, chunks=chunks)


def select_velocity(u_ds, v_ds=None, *, mode="mean", member=None,
                    u_name="uo", v_name="vo", ensemble_dim="ens",
                    depth=None, u_depth="depthu", v_depth="depthv",
                    time_dim="time", log=None):
    """Preserve all vertical levels; reduce U and V independently."""
    if mode not in ("mean", "member", "deterministic"):
        raise ValueError(f"Unsupported velocity mode: {mode}")
    log = log or logging.getLogger(__name__)
    if v_ds is None:
        v_ds = u_ds

    def process(ds, var):
        started = time.monotonic()
        log.info("Zarr %s: selecting mode=%s, input dims=%s shape=%s", var, mode, ds[var].dims, ds[var].shape)
        field = ds[var]
        if time_dim not in field.dims:
            raise ValueError(f"{var} missing time dimension {time_dim}")
        has_ensemble = ensemble_dim in field.dims
        if mode == "mean":
            if not has_ensemble:
                raise ValueError(f"{var} has no ensemble dimension {ensemble_dim} for mean")
            log.info("Zarr %s: building lazy ensemble mean across %s (%d members)", var, ensemble_dim, field.sizes[ensemble_dim])
            field = field.mean(ensemble_dim, skipna=True)
            log.info("Zarr %s: mean graph built in %.1fs; actual S3 reads occur when materialized", var, time.monotonic() - started)
        elif mode == "member":
            if not has_ensemble:
                raise ValueError(f"{var} has no ensemble dimension {ensemble_dim} for member")
            if member is None:
                raise ValueError("ensemble_member is required for velocity_mode: member")
            field = field.sel({ensemble_dim: member})
        elif has_ensemble:
            raise ValueError(f"{var} contains {ensemble_dim}; select mean or member mode")
        log.info("Zarr %s: selection ready, dims=%s shape=%s elapsed=%.1fs", var, field.dims, field.shape, time.monotonic() - started)
        return field

    # Each component is an independent lazy reduction. No vertical slicing.
    u = process(u_ds, u_name)
    v = process(v_ds, v_name)
    import numpy as np
    if not np.array_equal(u[time_dim].values, v[time_dim].values):
        raise ValueError("U and V forecast valid times do not match")
    return xr.Dataset({u_name: u, v_name: v})


def ensemble_mean_velocity(u_ds, v_ds=None, **kwargs):
    """Backward-compatible wrapper for ensemble mean selection."""
    return select_velocity(u_ds, v_ds, mode="mean", **kwargs)


def select_forecast_window(ds, start, *, days=28, time_dim="time"):
    """Require 28 daily forecast samples (lead days 0 through 27)."""
    import numpy as np
    if days <= 0:
        raise ValueError("days must be positive")
    start = np.datetime64(start, "D")
    # Daily mean GLOENS fields are stamped at 12:00 UTC, not midnight.
    # Select by calendar date and preserve their native timestamps.
    dates = ds[time_dim].values.astype("datetime64[D]")
    indices = np.flatnonzero((dates >= start) & (dates < start + np.timedelta64(days, "D")))
    selected = ds.isel({time_dim: indices})
    times = selected[time_dim].values.astype("datetime64[ns]")
    expected_dates = start + np.arange(days).astype("timedelta64[D]")
    if len(times) != days or not np.array_equal(times.astype("datetime64[D]"), expected_dates):
        raise ValueError(
            f"Incomplete {days}-day forecast window starting {start}; "
            f"available time count={ds.sizes.get(time_dim, 0)}, "
            f"first={ds[time_dim].values[0] if ds.sizes.get(time_dim, 0) else None}, "
            f"last={ds[time_dim].values[-1] if ds.sizes.get(time_dim, 0) else None}, "
            f"selected count={len(times)}"
        )
    if np.any(np.diff(times) != np.timedelta64(1, "D")):
        raise ValueError("Forecast times must be strictly daily with a consistent time of day")
    return selected


def prepare_velocity_forecast(u_store, v_store, start, *, days=28,
                              storage_options=None, mode="mean", member=None,
                              u_name="uo", v_name="vo", ensemble_dim="ens",
                              depth=0.0, u_depth="depthu", v_depth="depthv", log=None):
    """Context manager for deterministic, ensemble member, or ensemble mean U/V."""
    from contextlib import contextmanager

    log = log or logging.getLogger(__name__)

    @contextmanager
    def opened():
        log.info("Zarr: opening U store %s", u_store)
        with open_velocity_dataset(u_store, storage_options=storage_options) as u_ds:
            log.info("Zarr: U store opened")
            if str(u_store) == str(v_store):
                selected = select_velocity(
                    u_ds, mode=mode, member=member, u_name=u_name, v_name=v_name,
                    ensemble_dim=ensemble_dim, depth=depth,
                    u_depth=u_depth, v_depth=v_depth, log=log)
                log.info("Zarr: selecting %d-day window", days)
                yield select_forecast_window(selected, start, days=days)
            else:
                log.info("Zarr: opening V store %s", v_store)
                with open_velocity_dataset(v_store, storage_options=storage_options) as v_ds:
                    log.info("Zarr: V store opened")
                    selected = select_velocity(
                        u_ds, v_ds, mode=mode, member=member,
                        u_name=u_name, v_name=v_name, ensemble_dim=ensemble_dim,
                        depth=depth, u_depth=u_depth, v_depth=v_depth, log=log)
                    yield select_forecast_window(selected, start, days=days)
    return opened()


def prepare_ensemble_forecast(u_store, v_store, start, **kwargs):
    """Backward-compatible ensemble-mean forecast context manager."""
    return prepare_velocity_forecast(u_store, v_store, start, mode="mean", **kwargs)


def discover_gloens_stores(run_date, *, bucket, endpoint_url,
                           prefix, storage_options=None, log=None):
    """Find matching U/V forecast stores for exactly one forecast cycle.

    CLASS1 names: glo4-ens50_ng_1d-m_YYYYMMDD-YYYYMMDD_
                  3DU-uo_fcst_RYYYYMMDD.zarr (and 3DV-vo).
    Only returns matching run cycles and matching valid-time periods.
    """
    import s3fs
    from datetime import datetime

    date = run_date.strftime("%Y%m%d") if hasattr(run_date, "strftime") else str(run_date).replace("-", "")
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"Expected a YYYYMMDD run date; got {run_date!r}")
    options = {"anon": True, "client_kwargs": {"endpoint_url": endpoint_url}}
    options.update(storage_options or {})
    fs = s3fs.S3FileSystem(**options)
    candidates = fs.ls(bucket)
    found = {}
    for path in candidates:
        name = path.rsplit("/", 1)[-1]
        match = re.fullmatch(
            re.escape(prefix) + r"(\d{8})-(\d{8})_(3D[UV]-[uv]o)_fcst_R(\d{8})\.zarr",
            name,
        )
        if not match or match.group(4) != date or match.group(1) != date:
            continue
        key = match.group(3)
        if key in found:
            raise ValueError(f"Multiple GLOENS {key} stores for run {date}")
        found[key] = (f"s3://{bucket}/{name}", match.group(2))
        log.debug(f"Found {found[key]}")
    if not {"3DU-uo", "3DV-vo"} <= found.keys():
        raise FileNotFoundError(f"Missing GLOENS U/V Zarr forecast stores for run {date} in s3://{bucket}")
    if found["3DU-uo"][1] != found["3DV-vo"][1]:
        raise ValueError(f"GLOENS U/V forecast coverage differs for run {date}")
    # Validate advertised end date >= 28 days from the forecast cycle.
    advertised_end = datetime.strptime(found["3DU-uo"][1], "%Y%m%d").date()
    cycle = datetime.strptime(date, "%Y%m%d").date()
    if (advertised_end - cycle).days < 27:
        raise ValueError(f"GLOENS forecast for {date} advertises fewer than 28 daily fields")
    return found["3DU-uo"][0], found["3DV-vo"][0], options
