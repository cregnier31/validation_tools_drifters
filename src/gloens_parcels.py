"""Parcels 3.x advection on GLOENS NEMO curvilinear C-grid forecast velocities."""
from datetime import timedelta
import math
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr



def run_gloens_parcels(velocities, meshmask, day, lon0, lat0, output_path, drift_days=28, u_name="uo", v_name="vo", depth=0.0, scheme="AdvectionRK4", dt_minutes=30, sampling_hours=1, log=None, cache_dir=None):
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, AdvectionEE, Field, ParcelsRandom

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
    fieldset = FieldSet.from_xarray_dataset(
        ds, {"U": u_name, "V": v_name},
        {"lon": "glamf", "lat": "gphif", "time": "time"},
        mesh="spherical", allow_time_extrapolation=True,
    )
    log.debug("GLOENS: 2D FieldSet constructed in %.1fs", time.monotonic() - started)
    if fieldset.U.grid.zdim != 1 or fieldset.V.grid.zdim != 1:
        raise ValueError("Expected a 2D (single-level) GLOENS FieldSet")
    # Parcels 3.1.4 CurvilinearGrid does not support periodic halos.
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
                       depth=np.zeros(len(lon0), dtype=np.float32),
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
