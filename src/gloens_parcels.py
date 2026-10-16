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
    # The legacy irregular-grid workflow uses FieldSet.from_nemo with meshmask
    # coordinates and NEMO-specific interpolation. Reuse that path here.
    # Write one lead day per file so Parcels can defer reading time slices.
    from netCDF4 import Dataset

    cache_root = Path(cache_dir) if cache_dir else Path(output_path).parent / "gloens_nemo_daily"
    run_key = pd.Timestamp(day).strftime("%Y%m%d")
    daily_root = cache_root / f"nemo_daily_R{run_key}"
    daily_root.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(velocities.time.values)
    if len(times) != int(drift_days):
        raise ValueError(f"Expected {drift_days} daily velocity samples, got {len(times)}")

    def write_daily(var, label):
        field = velocities[var]
        vertical = [dim for dim in field.dims if dim not in ("time", "y", "x")]
        if len(vertical) != 1:
            raise ValueError(f"{label} requires one vertical dimension, got {vertical}")
        zdim = vertical[0]
        if field.sizes[zdim] != 75:
            log.warning("GLOENS %s: expected 75 depths, found %d", label, field.sizes[zdim])
        paths = []
        for i, timestamp in enumerate(times):
            path = daily_root / f"{label}_{timestamp.strftime('%Y%m%d')}.nc"
            if path.exists():
                with Dataset(path) as check:
                    expected_shape = (1, field.sizes[zdim], field.sizes["y"], field.sizes["x"])
                    if var in check.variables and check.variables[var].shape == expected_shape:
                        log.debug("GLOENS: reuse daily %s %d/%d: %s", label, i + 1, len(times), path)
                        paths.append(str(path))
                        continue
                raise ValueError(
                    f"Existing daily NetCDF cache is incompatible: {path}. "
                    "Refusing to overwrite it; remove it manually to regenerate."
                )
            started = time.monotonic()
            slab = field.isel(time=slice(i, i + 1)).transpose("time", zdim, "y", "x")
            # A single day is loaded at a time, retaining all vertical levels.
            temp = path.with_suffix(".partial.nc")
            if temp.exists():
                temp.unlink()
            slab.to_dataset(name=var).to_netcdf(temp, engine="netcdf4",
                                                encoding={var: {"dtype": "float32"}})
            temp.replace(path)
            log.debug("GLOENS: wrote daily %s %d/%d in %.1fs: %s",
                      label, i + 1, len(times), time.monotonic() - started, path)
            paths.append(str(path))
        return paths

    u_files = write_daily(u_name, "U")
    v_files = write_daily(v_name, "V")
    filenames = {
        "U": {"lon": str(meshmask), "lat": str(meshmask),
              "depth": u_files, "data": u_files},
        "V": {"lon": str(meshmask), "lat": str(meshmask),
              "depth": v_files, "data": v_files},
    }
    variables = {"U": u_name, "V": v_name}
    dimensions = {
        "U": {"lon": "nav_lon", "lat": "nav_lat", "depth": "depthu", "time": "time"},
        "V": {"lon": "nav_lon", "lat": "nav_lat", "depth": "depthv", "time": "time"},
    }
    started = time.monotonic()
    log.debug("GLOENS: constructing NEMO FieldSet with deferred time loading")
    fieldset = FieldSet.from_nemo(
        filenames, variables, dimensions,
        allow_time_extrapolation=True, time_periodic=False,
        deferred_load=True,
    )
    log.debug("GLOENS: NEMO FieldSet constructed in %.1fs", time.monotonic() - started)
    for component in ("U", "V"):
        nz = getattr(fieldset, component).grid.zdim
        if nz != 75:
            raise ValueError(f"GLOENS {component} Parcels grid has {nz} depth levels; expected 75")
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
