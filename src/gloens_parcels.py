"""Parcels 3.x advection on GLOENS NEMO curvilinear C-grid forecast velocities."""
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def run_gloens_parcels(velocities, meshmask, day, lon0, lat0, output_path, drift_days=28, u_name="uo", v_name="vo", depth=0.0):
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4

    meshmask = Path(meshmask)
    if not meshmask.is_file():
        raise FileNotFoundError(f"GLOENS NEMO meshmask missing: {meshmask}")
    with xr.open_dataset(meshmask, decode_times=False) as mesh:
        def coords(name):
            da = mesh[name]
            if "time_counter" in da.dims:
                da = da.isel(time_counter=0)
            return np.asarray(da.transpose("y", "x").values, dtype=np.float32)
        ulon, ulat = coords("glamu"), coords("gphiu")
        vlon, vlat = coords("glamv"), coords("gphiv")

    import logging
    log = logging.getLogger(__name__)

    def materialize_component(var, name):
        field = velocities[var]
        vertical = [dim for dim in field.dims if dim not in ("time", "y", "x")]
        if len(vertical) > 1:
            raise ValueError(f"{name} has unsupported extra dimensions: {vertical}")
        dims = ("time", *vertical, "y", "x")
        log.info("Computing %s velocities separately: dims=%s, shape=%s", name, dims, field.shape)
        values = np.asarray(field.transpose(*dims).values, dtype=np.float32)
        log.info("%s velocity materialized: %.1f MiB", name, values.nbytes / 2**20)
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
    fieldset = FieldSet.from_data(
        {"U": u, "V": v},
        {"U": u_coords, "V": v_coords},
        mesh="spherical", allow_time_extrapolation=True,
    )
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                       lon=np.asarray(lon0, dtype=np.float64),
                       lat=np.asarray(lat0, dtype=np.float64),
                       depth=np.full(len(lon0), float(depth), dtype=np.float32),
                       time=np.zeros(len(lon0), dtype=np.float64))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pfile = pset.ParticleFile(name=str(output_path), outputdt=timedelta(hours=1))
    pset.execute(AdvectionRK4, runtime=timedelta(days=int(drift_days)),
                 dt=timedelta(minutes=30), output_file=pfile)
    return str(output_path)
