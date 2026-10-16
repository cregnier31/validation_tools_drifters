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


def cached_leadtime_mean(u_ds, v_ds, start, *, days, cache_dir,
                         u_store, v_store, u_name="uo", v_name="vo",
                         ensemble_dim="ens", time_dim="time", log=None):
    """Reduce one forecast lead day at a time into reusable float32 .npy files.

    Files are written atomically and validated by an input/configuration fingerprint.
    Only one (ensemble, depth, y, x) slab is evaluated at a time.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import numpy as np

    log = log or logging.getLogger(__name__)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(dict(u=str(u_store), v=str(v_store),
                               start=str(start), days=int(days),
                               ensemble_dim=ensemble_dim, u_name=u_name,
                               v_name=v_name), sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    run_match = re.search(r"_R(\\d{8})\\.zarr/?$", str(u_store))
    run_date = run_match.group(1) if run_match else str(start).replace("-", "")[:8]
    start_date = np.datetime64(start, "D")
    first_date = str(start_date).replace("-", "")
    last_date = str(start_date + np.timedelta64(int(days) - 1, "D")).replace("-", "")
    # Human-readable identity; full input fingerprint stays in JSON metadata.
    filename_prefix = f"FCST_R{run_date}_{first_date}-{last_date}"
    output = {}
    for ds, var in ((u_ds, u_name), (v_ds, v_name)):
        source = select_forecast_window(ds[[var]], start, days=days)[var]
        if ensemble_dim not in source.dims:
            raise ValueError(f"{var} missing ensemble dimension {ensemble_dim}")
        dims = (time_dim,) + tuple(d for d in source.dims if d not in (ensemble_dim, time_dim))
        shape = tuple(source.sizes[d] for d in dims)
        cache_path = root / f"{filename_prefix}_{var}_ensmean_{days}days.npy"
        meta_path = root / f"{filename_prefix}_{var}_ensmean_{days}days.json"
        meta = dict(identity=identity, var=var, dims=list(dims), shape=list(shape),
                    times=[str(t) for t in source[time_dim].values],
                    ensemble_size=source.sizes[ensemble_dim])
        valid = False
        if not cache_path.exists():
            log.info("Zarr %s: cache miss: NumPy file missing: %s", var, cache_path)
        if not meta_path.exists():
            log.info("Zarr %s: cache miss: metadata JSON missing: %s", var, meta_path)
        if cache_path.exists() and meta_path.exists():
            try:
                stored_meta = json.loads(meta_path.read_text())
                for field in sorted(set(meta) | set(stored_meta)):
                    expected = meta.get(field, "<missing>")
                    actual = stored_meta.get(field, "<missing>")
                    if expected != actual:
                        if field == "identity":
                            try:
                                expected_identity = json.loads(expected)
                                actual_identity = json.loads(actual)
                                for k in sorted(set(expected_identity) | set(actual_identity)):
                                    if expected_identity.get(k) != actual_identity.get(k):
                                        log.warning("Zarr %s: cache identity mismatch %s: cached=%r requested=%r",
                                                    var, k, actual_identity.get(k), expected_identity.get(k))
                            except (TypeError, ValueError):
                                log.warning("Zarr %s: cache identity mismatch: cached=%r requested=%r",
                                            var, actual, expected)
                        else:
                            log.warning("Zarr %s: cache metadata mismatch %s: cached=%r requested=%r",
                                        var, field, actual, expected)
                if stored_meta == meta:
                    arr = np.load(cache_path, mmap_mode="r")
                    valid = arr.shape == shape and arr.dtype == np.float32
                    if arr.shape != shape:
                        log.warning("Zarr %s: cache shape mismatch: cached=%s requested=%s",
                                    var, arr.shape, shape)
                    if arr.dtype != np.float32:
                        log.warning("Zarr %s: cache dtype mismatch: cached=%s requested=float32",
                                    var, arr.dtype)
                    del arr
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                log.warning("Zarr %s: cannot validate cache %s: %s", var, cache_path, exc)
        if not valid:
            if cache_path.exists() or meta_path.exists():
                log.warning("Zarr %s: cache invalid; see preceding cache-miss/mismatch details: %s", var, cache_path)
            tmp = root / f"{filename_prefix}_{var}_ensmean_{days}days.partial.npy"
            progress = root / f"{filename_prefix}_{var}_ensmean_{days}days.progress.json"
            completed = 0
            if tmp.exists() and progress.exists():
                try:
                    checkpoint = json.loads(progress.read_text())
                    if checkpoint.get("meta") == meta:
                        prior = np.load(tmp, mmap_mode="r")
                        if prior.shape == shape and prior.dtype == np.float32:
                            completed = int(checkpoint["completed"])
                            if not 0 <= completed <= days:
                                completed = 0
                        del prior
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    completed = 0
            log.info("Zarr %s: %s mean cache %s from lead %d/%d", var,
                     "resuming" if completed else "building", cache_path, completed, days)
            target = (np.load(tmp, mmap_mode="r+") if completed
                      else np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=shape))
            try:
                for i in range(completed, days):
                    started = time.monotonic()
                    slab = source.isel({time_dim: i}).mean(ensemble_dim, skipna=True)
                    # Preserve source dimension order in the cache.
                    slab = slab.transpose(*(d for d in dims if d != time_dim))
                    target[i] = np.asarray(slab.compute().values, dtype=np.float32)
                    target.flush()
                    checkpoint_tmp = progress.with_suffix(".json.tmp")
                    checkpoint_tmp.write_text(json.dumps({"meta": meta, "completed": i + 1}))
                    os.replace(checkpoint_tmp, progress)
                    log.info("Zarr %s: ensemble mean lead %d/%d completed in %.1fs",
                             var, i + 1, days, time.monotonic() - started)
                del target
                os.replace(tmp, cache_path)
                meta_tmp = meta_path.with_suffix(".json.tmp")
                meta_tmp.write_text(json.dumps(meta))
                os.replace(meta_tmp, meta_path)
                progress.unlink(missing_ok=True)
            except BaseException:
                del target
                # Keep completed lead days for the next Slurm run.
                raise
        else:
            log.info("Zarr %s: reusing cached ensemble mean %s", var, cache_path)
        coords = {d: source[d] for d in dims if d in source.coords}
        output[var] = xr.DataArray(np.load(cache_path, mmap_mode="r"),
                                   dims=dims, coords=coords, name=var)
    return xr.Dataset(output)


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
                              depth=0.0, u_depth="depthu", v_depth="depthv", log=None, cache_dir=None):
    """Context manager for deterministic, ensemble member, or ensemble mean U/V."""
    from contextlib import contextmanager

    log = log or logging.getLogger(__name__)

    @contextmanager
    def opened():
        log.info("Zarr: opening U store %s", u_store)
        with open_velocity_dataset(u_store, storage_options=storage_options) as u_ds:
            log.info("Zarr: U store opened")
            if str(u_store) == str(v_store):
                selected = (cached_leadtime_mean(u_ds, u_ds, start, days=days, cache_dir=cache_dir, u_store=u_store, v_store=v_store, u_name=u_name, v_name=v_name, ensemble_dim=ensemble_dim, log=log) if mode == "mean" and cache_dir else select_velocity(
                    u_ds, mode=mode, member=member, u_name=u_name, v_name=v_name,
                    ensemble_dim=ensemble_dim, depth=depth,
                    u_depth=u_depth, v_depth=v_depth, log=log))
                log.info("Zarr: selecting %d-day window", days)
                yield select_forecast_window(selected, start, days=days)
            else:
                log.info("Zarr: opening V store %s", v_store)
                with open_velocity_dataset(v_store, storage_options=storage_options) as v_ds:
                    log.info("Zarr: V store opened")
                    selected = (cached_leadtime_mean(u_ds, v_ds, start, days=days, cache_dir=cache_dir, u_store=u_store, v_store=v_store, u_name=u_name, v_name=v_name, ensemble_dim=ensemble_dim, log=log) if mode == "mean" and cache_dir else select_velocity(
                        u_ds, v_ds, mode=mode, member=member,
                        u_name=u_name, v_name=v_name, ensemble_dim=ensemble_dim,
                        depth=depth, u_depth=u_depth, v_depth=v_depth, log=log))
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
