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
    # Build directly from the disk-backed ensemble-mean DataArrays, following
    # the working xarray/F-point workflow. No additional daily NetCDF copies.
    u = velocities[u_name]
    v = velocities[v_name]
    u_z = [d for d in u.dims if d not in ("time", "y", "x")]
    v_z = [d for d in v.dims if d not in ("time", "y", "x")]
    if len(u_z) != 1 or len(v_z) != 1:
        raise ValueError(f"Expected one depth dimension for U/V: {u_z}, {v_z}")
    if u.sizes[u_z[0]] != 75 or v.sizes[v_z[0]] != 75:
        raise ValueError("GLOENS requires all 75 depth levels")
    u_depth = np.asarray(u[u_z[0]].values)
    v_depth = np.asarray(v[v_z[0]].values)
    if not np.allclose(u_depth, v_depth, rtol=1e-6, atol=1e-4):
        raise ValueError("U and V depth coordinates differ; cannot share a 3D grid")
    u = u.rename({u_z[0]: "depth"})
    v = v.rename({v_z[0]: "depth"}).assign_coords(depth=u_depth)
    # Avoid xarray aligning distinct coordinates or materializing full velocity arrays.
    ds = xr.Dataset({u_name: u, v_name: v})
    with xr.open_dataset(meshmask, engine="netcdf4") as grid:
        for name in ("glamf", "gphif"):
            coord = grid[name]
            if "time_counter" in coord.dims:
                coord = coord.isel(time_counter=0, drop=True)
            ds[name] = coord.load()
    log.debug("GLOENS: constructing 3D xarray FieldSet at all 75 depths")
    started = time.monotonic()
    fieldset = FieldSet.from_xarray_dataset(
        ds, {"U": u_name, "V": v_name},
        {"lon": "glamf", "lat": "gphif", "depth": "depth", "time": "time"},
        mesh="spherical", allow_time_extrapolation=True,
    )
    log.debug("GLOENS: 3D xarray FieldSet constructed in %.1fs", time.monotonic() - started)
    for component in ("U", "V"):
        nz = getattr(fieldset, component).grid.zdim
        if nz != 75:
            raise ValueError(f"GLOENS {component} grid has {nz} depth levels; expected 75")
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
