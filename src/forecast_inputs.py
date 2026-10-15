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
            import s3fs
            if source.startswith("s3://"):
                source = s3fs.S3Map(root=source, s3=s3fs.S3FileSystem(**(storage_options or {})), check=False)
            else:
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
    import numpy as np
    # xarray can attach different scalar depth coordinates to U and V time arrays.
    # Compare actual timestamps, not unrelated coordinates or attributes.
    if not np.array_equal(fields[0][time_dim].values, fields[1][time_dim].values):
        raise ValueError("U and V forecast valid times do not match")
    return xr.Dataset({u_name: fields[0], v_name: fields[1]})


def select_forecast_window(ds, start, *, days=28, time_dim="time"):
    """Require 28 daily forecast samples (lead days 0 through 27)."""
    import numpy as np
    if days <= 0:
        raise ValueError("days must be positive")
    start = np.datetime64(start, "ns")
    end = start + np.timedelta64(days - 1, "D")
    selected = ds.sel({time_dim: slice(start, end)})
    times = selected[time_dim].values
    if len(times) != days or times[0].astype("datetime64[ns]") != start or times[-1].astype("datetime64[ns]") != end:
        raise ValueError(
            f"Incomplete {days}-day forecast window starting {start}; "
            f"available time count={ds.sizes.get(time_dim, 0)}, "
            f"first={ds[time_dim].values[0] if ds.sizes.get(time_dim, 0) else None}, "
            f"last={ds[time_dim].values[-1] if ds.sizes.get(time_dim, 0) else None}, "
            f"selected count={len(times)}, selected first={times[0] if len(times) else None}, "
            f"selected last={times[-1] if len(times) else None}"
        )
    if np.any(np.diff(times) != np.timedelta64(1, "D")):
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


# CLASS1 GLOENS operational Zarr location; same settings as
# class1-comparisons/cfg/config_def_GLOENS.cfg.
GLOENS_BUCKET = "MOISICEEF"
GLOENS_ENDPOINT_URL = "https://s3.waw3-1.cloudferro.com"
GLOENS_PREFIX = "glo4-ens50_ng_1d-m_"


def discover_gloens_stores(run_date, *, bucket=GLOENS_BUCKET,
                           endpoint_url=GLOENS_ENDPOINT_URL,
                           storage_options=None, prefix=GLOENS_PREFIX):
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
