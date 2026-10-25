"""GLOW regular-grid Zarr advection using Parcels 4.

GLOW has daily, collocated uo/vo on a 0.25-degree global grid.
"""
from datetime import timedelta
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def run_glow_parcels(velocities, day, lon0, lat0, output_path, *,
                     drift_days=7, u_name="uo", v_name="vo", depth=15.0,
                     scheme="AdvectionRK4", dt_minutes=30, sampling_hours=1,
                     meshmask=None, log=None):
    import parcels

    log = log or logging.getLogger(__name__)
    if int(parcels.__version__.split(".")[0]) < 4:
        raise RuntimeError("GLOW runner requires Parcels 4")
    if scheme not in ("AdvectionRK4", "AdvectionEE"):
        raise NotImplementedError(f"GLOW Parcels 4 scheme {scheme!r} is not implemented")
    if dt_minutes <= 0 or sampling_hours <= 0 or drift_days <= 0:
        raise ValueError("Positive dt, output interval, and drift days required")

    u = velocities[u_name]
    v = velocities[v_name]
    for name, arr in ((u_name, u), (v_name, v)):
        if "depth" not in arr.dims:
            raise ValueError(f"GLOW {name} missing depth dimension: {arr.dims}")
    u = u.sel(depth=float(depth), method="nearest")
    v = v.sel(depth=float(depth), method="nearest")
    log.info("GLOW requested depth %.2f m, selected U %.3f m and V %.3f m",
             depth, float(u.depth), float(v.depth))
    # Drop scalar depth coordinates so xarray does not attempt depth alignment.
    u = u.reset_coords(drop=True).transpose("time", "lat", "lon")
    v = v.reset_coords(drop=True).transpose("time", "lat", "lon")
    if u.shape != v.shape:
        raise ValueError(f"GLOW U/V shape mismatch: {u.shape}, {v.shape}")
    times = pd.to_datetime(u.time.values)
    expected_start = pd.Timestamp(day).normalize()
    if len(times) < 2 or times[0] != expected_start:
        raise ValueError(f"GLOW cycle must start at midnight {expected_start}; got {times[0] if len(times) else 'none'}")
    if not np.all(np.diff(times.values) > np.timedelta64(0, "ns")):
        raise ValueError("GLOW time coordinates must be strictly increasing")
    required = timedelta(days=drift_days).total_seconds()
    dt_seconds = timedelta(minutes=dt_minutes).total_seconds()
    available = (times[-1] - times[0]).total_seconds()
    if required % dt_seconds:
        raise ValueError("GLOW drift duration must be divisible by the RK4 timestep")
    # GLOW supplies N daily fields for N requested forecast days. The last
    # native field is at day N-1, so extend its velocity unchanged through
    # day N+1 to provide a full-day tail and an RK4 interpolation guard.
    # This is persistence extrapolation, not additional forecast information.
    if available < required + dt_seconds:
        if len(times) != drift_days or not np.all(
                np.diff(times.values) == np.timedelta64(1, "D")):
            raise ValueError(
                f"GLOW forecast spans {available/86400:.2f} days; need "
                f"{drift_days} days plus one RK4 timestep. Expected "
                f"{drift_days} consecutive daily fields for persistence extension."
            )
        extension_times = [
            times[-1] + pd.Timedelta(days=1),
            times[-1] + pd.Timedelta(days=2),
        ]
        u = xr.concat([u, *[u.isel(time=-1).expand_dims(time=[t])
                            for t in extension_times]], dim="time")
        v = xr.concat([v, *[v.isel(time=-1).expand_dims(time=[t])
                            for t in extension_times]], dim="time")
        log.warning(
            "GLOW: extending last daily U/V field at %s unchanged to %s "
            "for %d-day integration (persistence extrapolation)",
            times[-1], extension_times[-1], drift_days)

    if meshmask:
        # GLOW mask has different vertical coordinates from uo/vo. Do not
        # align it by velocity depth index or silently infer its value coding.
        log.info("GLOW mask %s not applied: velocity NaNs determine invalid cells; "
                 "mask depth coordinates differ from velocity depths", meshmask)

    # Parcels 4 requires SGRID metadata; legacy from_data/from_xarray_dataset
    # constructors do not exist. GLOW velocities are collocated A-grid fields.
    # The Copernicus Marine converter supports this regular-grid layout.
    u = u.rename({"lat": "latitude", "lon": "longitude"})
    v = v.rename({"lat": "latitude", "lon": "longitude"})
    for arr in (u, v):
        arr.latitude.attrs.update({"axis": "Y", "units": "degrees_north"})
        arr.longitude.attrs.update({"axis": "X", "units": "degrees_east"})
        arr.time.attrs["axis"] = "T"
    sgrid = parcels.convert.copernicusmarine_to_sgrid(
        fields={"U": u, "V": v}
    )
    log.info("GLOW SGRID converted: %s", dict(sgrid.sizes))
    fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
    if hasattr(fieldset, "describe"):
        fieldset.describe()
    seed_lon = np.asarray(lon0, dtype=np.float64)
    seed_lat = np.asarray(lat0, dtype=np.float64)
    if seed_lon.shape != seed_lat.shape:
        raise ValueError("GLOW seed longitude and latitude shapes differ")
    seed_lon = ((seed_lon + 180.0) % 360.0) - 180.0
    log.info("GLOW: initializing %d particles at %s", len(seed_lon), times[0])
    pset = parcels.ParticleSet(
        fieldset=fieldset, pclass=parcels.Particle,
        x=seed_lon, y=seed_lat, t=times[0].to_pydatetime(),
    )
    output_path = Path(output_path).with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing GLOW trajectory: {output_path}")
    pfile = parcels.ParticleFile(
        str(output_path), outputdt=timedelta(hours=sampling_hours))
    kernel = (parcels.kernels.AdvectionRK4 if scheme == "AdvectionRK4"
              else parcels.kernels.AdvectionEE)
    if os.environ.get("GLOENS_DELETE_FAILED_PARTICLES", "0").strip() == "1":
        from gloens_periodic_kernel import DeleteFailedParticles
        kernel = [kernel, DeleteFailedParticles]
        log.warning("GLOW recovery enabled: invalid particles will be deleted")
    log.info("GLOW executing %s for %d days at %d-minute dt",
             scheme, drift_days, dt_minutes)
    pset.execute(kernel, runtime=timedelta(days=drift_days),
                 dt=timedelta(minutes=dt_minutes), output_file=pfile)
    result = pd.read_parquet(output_path, columns=["t", "particle_id", "x", "y"])
    clock = result["t"].dropna()
    if clock.nunique() < 2:
        raise RuntimeError("GLOW produced fewer than two trajectory timestamps")
    log.info("GLOW wrote %d records, %d particles, %d timestamps to %s",
             len(result), result["particle_id"].nunique(),
             clock.nunique(), output_path)
    return str(output_path)
