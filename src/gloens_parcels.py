"""Parcels 3.x advection on GLOENS NEMO curvilinear C-grid forecast velocities."""
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def run_gloens_parcels(velocities, meshmask, day, lon0, lat0, output_path, drift_days=28, u_name="uo", v_name="vo"):
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

    u = np.asarray(velocities[u_name].transpose("time", "y", "x").values, dtype=np.float32)
    v = np.asarray(velocities[v_name].transpose("time", "y", "x").values, dtype=np.float32)
    if u.shape[1:] != ulon.shape or v.shape[1:] != vlon.shape:
        raise ValueError("GLOENS velocity arrays and NEMO U/V meshmask shapes differ")
    times = np.asarray((pd.to_datetime(velocities.time.values) - pd.Timestamp(day)).total_seconds(), dtype=np.float64)
    fieldset = FieldSet.from_data(
        {"U": u, "V": v},
        {"U": {"lon": ulon, "lat": ulat, "time": times},
         "V": {"lon": vlon, "lat": vlat, "time": times}},
        mesh="spherical", allow_time_extrapolation=True,
    )
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                       lon=np.asarray(lon0, dtype=np.float64),
                       lat=np.asarray(lat0, dtype=np.float64),
                       time=np.zeros(len(lon0), dtype=np.float64))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pfile = pset.ParticleFile(name=str(output_path), outputdt=timedelta(hours=1))
    pset.execute(AdvectionRK4, runtime=timedelta(days=int(drift_days)),
                 dt=timedelta(minutes=30), output_file=pfile)
    return str(output_path)
