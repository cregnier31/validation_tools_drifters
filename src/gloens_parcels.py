"""GLOENS NEMO curvilinear C-grid advection (Parcels 3 and 4)."""
from datetime import timedelta
import math
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr



def _diagnose_particle_failure(pset, seed_lon, seed_lat, log):
    """Report individual Parcels 4 particle states after an execution failure.

    Read-only: does not delete particles, retry integration, or change fields.
    """
    try:
        states = np.asarray(pset.state).ravel()
        x = np.asarray(pset.x).ravel()
        y = np.asarray(pset.y).ravel()
        log.error("GLOENS particle diagnostics: %d particles; state counts=%s",
                  len(states),
                  {str(code): int(np.count_nonzero(states == code))
                   for code in np.unique(states)})
        # Parcels status 0 means success; other states indicate failures.
        bad = np.flatnonzero(states != 0)
        if not len(bad):
            log.error("GLOENS particle diagnostics: no nonzero states available; "
                      "grid search may fail before states are updated")
        for idx in bad[:100]:
            log.error("GLOENS suspect particle index=%d state=%s "
                      "seed=(%.6f, %.6f) current=(%.6f, %.6f) "
                      "near_dateline=%s near_zero_zero=%s",
                      int(idx), states[idx],
                      seed_lon[idx], seed_lat[idx], x[idx], y[idx],
                      abs(x[idx]) >= 170,
                      abs(x[idx]) < 0.001 and abs(y[idx]) < 0.001)
        if len(bad) > 100:
            log.error("GLOENS particle diagnostics: %d more suspect particles omitted",
                      len(bad) - 100)
    except Exception:
        log.exception("Could not read Parcels particle states after failure")


def run_gloens_parcels(velocities, meshmask, day, lon0, lat0, output_path, drift_days=28, u_name="uo", v_name="vo", depth=0.0, scheme="AdvectionRK4", dt_minutes=30, sampling_hours=1, log=None, cache_dir=None):
    import parcels
    from parcels import FieldSet, ParticleSet
    parcels_v4 = int(parcels.__version__.split(".")[0]) >= 4

    meshmask = Path(meshmask)
    if not meshmask.is_file():
        raise FileNotFoundError(f"GLOENS NEMO meshmask missing: {meshmask}")
    log = log if log is not None else logging.getLogger(__name__)
    # Match the working NEMO 2D workflow: select the nearest model level
    # before constructing Parcels fields. The 75-level source cache is untouched.
    u = velocities[u_name]
    v = velocities[v_name]
    u_z = [d for d in u.dims if d not in ("time", "y", "x")]
    v_z = [d for d in v.dims if d not in ("time", "y", "x")]
    if len(u_z) != 1 or len(v_z) != 1:
        raise ValueError(f"Expected one depth axis per component: {u_z}, {v_z}")
    u = u.sel({u_z[0]: float(depth)}, method="nearest")
    v = v.sel({v_z[0]: float(depth)}, method="nearest")
    log.info("GLOENS 2D: requested depth %.2f m; U=%.3f m, V=%.3f m",
             depth, float(u[u_z[0]]), float(v[v_z[0]]))
    u = u.reset_coords(drop=True)
    v = v.reset_coords(drop=True)
    # The disk-backed ensemble-mean arrays are sliced at one depth only.
    ds = xr.Dataset({u_name: u, v_name: v})
    with xr.open_dataset(meshmask, engine="netcdf4") as grid:
        for name in ("glamf", "gphif"):
            coord = grid[name]
            if "time_counter" in coord.dims:
                coord = coord.isel(time_counter=0, drop=True)
            ds = ds.assign_coords({name: coord.load()})
    started = time.monotonic()
    log.debug("GLOENS: constructing 2D F-point xarray FieldSet at %.2f m", depth)
    if parcels_v4:
        # Parcels 4 uses SGRID metadata and xarray-native NEMO conversion.
        with xr.open_dataset(meshmask, engine="netcdf4") as grid:
            grid_coords = grid[["glamf", "gphif"]]
            if "time_counter" in grid_coords.dims:
                grid_coords = grid_coords.isel(time_counter=0, drop=True)
            sgrid = parcels.convert.nemo_to_sgrid(
                fields={"U": ds[u_name], "V": ds[v_name]},
                coords=grid_coords
            )
            fieldset = FieldSet.from_sgrid_conventions(sgrid)
    else:
        fieldset = FieldSet.from_xarray_dataset(
            ds, {"U": u_name, "V": v_name},
            {"lon": "glamf", "lat": "gphif", "time": "time"},
            mesh="spherical", allow_time_extrapolation=True,
        )
    log.debug("GLOENS: 2D FieldSet constructed in %.1fs", time.monotonic() - started)
    if not parcels_v4 and (fieldset.U.grid.zdim != 1 or fieldset.V.grid.zdim != 1):
        raise ValueError("Expected a 2D (single-level) GLOENS FieldSet")
    # Parcels 3.1.4 CurvilinearGrid does not support periodic halos.
    if scheme not in ("AdvectionRK4", "AdvectionEE", "K_smag"):
        raise ValueError(f"Unknown GLOENS Parcels scheme: {scheme}")
    if scheme == "K_smag" and parcels_v4:
        raise NotImplementedError("K_smag kernel requires migration to the Parcels 4 vectorized API")
    if scheme == "K_smag":
        from parcels import Field, ParcelsRandom
        fieldset.add_field(Field(name="cell_areas", data=fieldset.U.cell_areas(),
                                 lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat))
        fieldset.add_constant("Cs", 0.1)

        def smagdiff(particle, fieldset, time):
            dx = 0.01
            updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon + dx]
            umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon - dx]
            updy, vpdy = fieldset.UV[time, particle.depth, particle.lat + dx, particle.lon]
            umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat - dx, particle.lon]
            dudx = (updx - umdx) / (2 * dx)
            dudy = (updy - umdy) / (2 * dx)
            dvdx = (vpdx - vmdx) / (2 * dx)
            dvdy = (vpdy - vmdy) / (2 * dx)
            area = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
            area /= (1852 * 60) ** 2 * math.cos(particle.lat * math.pi / 180)
            kh = fieldset.Cs * area * math.sqrt(dudx ** 2 + 0.5 * (dudy + dvdx) ** 2 + dvdy ** 2)
            sigma = math.sqrt(2 * math.fabs(particle.dt) * kh)
            particle.lat += ParcelsRandom.normalvariate(0., 1.) * sigma
            particle.lon += ParcelsRandom.normalvariate(0., 1.) * sigma

    started = time.monotonic()
    seed_lon = np.asarray(lon0, dtype=np.float64)
    seed_lat = np.asarray(lat0, dtype=np.float64)
    log.info("GLOENS seed coordinates: n=%d, lon=%s, lat=%s",
             len(seed_lon), seed_lon.tolist(), seed_lat.tolist())
    log.warning("GLOENS time alignment check: model begins %s, drifter selection begins %s",
                pd.Timestamp(ds.time.values[0]), pd.Timestamp(day))
    log.debug("GLOENS: initializing %d particles", len(lon0))
    try:
        if parcels_v4:
            pset = ParticleSet(fieldset=fieldset, pclass=parcels.Particle,
                               x=np.asarray(lon0, dtype=np.float64),
                               y=np.asarray(lat0, dtype=np.float64),
                               t=pd.Timestamp(ds.time.values[0]).to_pydatetime())
        else:
            pset = ParticleSet(fieldset=fieldset, pclass=parcels.JITParticle,
                               lon=np.asarray(lon0, dtype=np.float64),
                               lat=np.asarray(lat0, dtype=np.float64),
                               depth=np.zeros(len(lon0), dtype=np.float32),
                               time=np.zeros(len(lon0), dtype=np.float64))
    except Exception:
        log.exception("GLOENS failure during ParticleSet initialization; see seed coordinates above")
        raise
    log.debug("GLOENS: particles initialized in %.1fs", time.monotonic() - started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if parcels_v4:
        if output_path.suffix != ".parquet":
            output_path = output_path.with_suffix(".parquet")
        pfile = parcels.ParticleFile(str(output_path), outputdt=timedelta(hours=sampling_hours))
        kernel = parcels.kernels.AdvectionEE if scheme == "AdvectionEE" else parcels.kernels.AdvectionRK4
    else:
        pfile = pset.ParticleFile(name=str(output_path), outputdt=timedelta(hours=sampling_hours))
        kernel = parcels.AdvectionEE if scheme == "AdvectionEE" else parcels.AdvectionRK4
    if scheme == "K_smag":
        kernel = kernel + pset.Kernel(smagdiff)
    log.info("GLOENS scheme=%s, dt=%s minutes, sampling=%s hours", scheme, dt_minutes, sampling_hours)
    # Forecast has 28 daily timestamps, but only 27 days between them.
    times = ds.time.values.astype("datetime64[ns]")
    if len(times) < 2 or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Expected at least two increasing GLOENS forecast times")
    dt_seconds = timedelta(minutes=dt_minutes).total_seconds()
    if dt_seconds <= 0:
        raise ValueError("dt_minutes must be positive")
    available_seconds = (times[-1] - times[0]) / np.timedelta64(1, "s")
    # Stop one timestep before the last sample to avoid an out-of-range RK4 lookup.
    steps = int(max(0, min(timedelta(days=drift_days).total_seconds(),
                           available_seconds - dt_seconds)) // dt_seconds)
    if steps == 0:
        raise ValueError("Forecast time coverage is shorter than two timesteps")
    runtime = timedelta(seconds=steps * dt_seconds)
    log.info("GLOENS available %s to %s; requested %s days; running %s",
             times[0], times[-1], drift_days, runtime)
    started = time.monotonic()
    try:
        pset.execute(kernel, runtime=runtime,
                     dt=timedelta(minutes=dt_minutes), output_file=pfile)
    except Exception:
        log.exception("GLOENS failure during pset.execute; model time range %s to %s",
                      times[0], times[-1])
        _diagnose_particle_failure(pset, seed_lon, seed_lat, log)
        raise
    log.debug("GLOENS: Parcels execution finished in %.1fs", time.monotonic() - started)
    return str(output_path)
