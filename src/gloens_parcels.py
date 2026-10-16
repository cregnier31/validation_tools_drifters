"""Parcels 3.x advection on GLOENS NEMO curvilinear C-grid forecast velocities."""
from datetime import timedelta
import math
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def run_gloens_parcels(velocities, meshmask, day, lon0, lat0, output_path, drift_days=28, u_name="uo", v_name="vo", depth=0.0, scheme="AdvectionRK4", dt_minutes=30, sampling_hours=1, log=None):
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, AdvectionEE, Field, ParcelsRandom

    meshmask = Path(meshmask)
    if not meshmask.is_file():
        raise FileNotFoundError(f"GLOENS NEMO meshmask missing: {meshmask}")
    log = log if log is not None else logging.getLogger(__name__)
    started = time.monotonic()
    log.debug("GLOENS: reading meshmask coordinates from %s", meshmask)
    with xr.open_dataset(meshmask, decode_times=False) as mesh:
        def coords(name):
            da = mesh[name]
            if "time_counter" in da.dims:
                da = da.isel(time_counter=0)
            return np.asarray(da.transpose("y", "x").values, dtype=np.float32)
        ulon, ulat = coords("glamu"), coords("gphiu")
        vlon, vlat = coords("glamv"), coords("gphiv")

    log.debug("GLOENS: meshmask loaded in %.1fs; U grid=%s V grid=%s", time.monotonic() - started, ulon.shape, vlon.shape)

    def materialize_component(var, name):
        field = velocities[var]
        vertical = [dim for dim in field.dims if dim not in ("time", "y", "x")]
        if len(vertical) > 1:
            raise ValueError(f"{name} has unsupported extra dimensions: {vertical}")
        dims = ("time", *vertical, "y", "x")
        started = time.monotonic()
        log.debug("GLOENS: materializing %s dims=%s shape=%s dtype=%s", name, dims, field.shape, field.dtype)
        # Parcels replaces NaNs in-place; cached .npy arrays are read-only memmaps.
        values = np.array(field.transpose(*dims).values, dtype=np.float32, copy=True)
        log.debug("GLOENS: %s materialized in %.1fs (%.1f MiB)", name, time.monotonic() - started, values.nbytes / 2**20)
        return values, vertical[0] if vertical else None

    u, u_zdim = materialize_component(u_name, "U")
    v, v_zdim = materialize_component(v_name, "V")
    if u.shape[-2:] != ulon.shape or v.shape[-2:] != vlon.shape:
        raise ValueError("GLOENS velocity arrays and NEMO U/V meshmask shapes differ")
    if bool(u_zdim) != bool(v_zdim):
        raise ValueError("U/V must both be 3D or both be 2D")
    times = np.asarray((pd.to_datetime(velocities.time.values) - pd.Timestamp(day)).total_seconds(), dtype=np.float64)
    u_coords = {"lon": ulon, "lat": ulat, "time": times}
    v_coords = {"lon": vlon, "lat": vlat, "time": times}
    if u_zdim:
        u_coords["depth"] = np.asarray(velocities[u_zdim].values, dtype=np.float32)
        v_coords["depth"] = np.asarray(velocities[v_zdim].values, dtype=np.float32)
    started = time.monotonic()
    log.debug("GLOENS: constructing Parcels FieldSet from U/V arrays")
    fieldset = FieldSet.from_data(
        {"U": u, "V": v},
        {"U": u_coords, "V": v_coords},
        mesh="spherical", allow_time_extrapolation=True,
    )
    log.debug("GLOENS: FieldSet constructed in %.1fs", time.monotonic() - started)
    if scheme not in ("AdvectionRK4", "AdvectionEE", "K_smag"):
        raise ValueError(f"Unknown GLOENS Parcels scheme: {scheme}")
    if scheme == "K_smag":
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
    log.debug("GLOENS: initializing %d particles", len(lon0))
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                       lon=np.asarray(lon0, dtype=np.float64),
                       lat=np.asarray(lat0, dtype=np.float64),
                       depth=np.full(len(lon0), float(depth), dtype=np.float32),
                       time=np.zeros(len(lon0), dtype=np.float64))
    log.debug("GLOENS: particles initialized in %.1fs", time.monotonic() - started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pfile = pset.ParticleFile(name=str(output_path), outputdt=timedelta(hours=sampling_hours))
    kernel = AdvectionEE if scheme == "AdvectionEE" else AdvectionRK4
    if scheme == "K_smag":
        kernel = kernel + pset.Kernel(smagdiff)
    log.info("GLOENS scheme=%s, dt=%s minutes, sampling=%s hours", scheme, dt_minutes, sampling_hours)
    log.debug("GLOENS: starting Parcels execution for %d days", drift_days)
    started = time.monotonic()
    pset.execute(kernel, runtime=timedelta(days=int(drift_days)),
                 dt=timedelta(minutes=dt_minutes), output_file=pfile)
    log.debug("GLOENS: Parcels execution finished in %.1fs", time.monotonic() - started)
    return str(output_path)
