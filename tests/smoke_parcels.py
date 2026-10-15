#!/usr/bin/env python3
"""Parcels 3.x execution/output smoke test (synthetic velocities, not GLOENS)."""
from datetime import timedelta
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr
from parcels import AdvectionRK4, FieldSet, JITParticle, ParticleSet

def main():
    lon = np.linspace(-10, 10, 21, dtype=np.float32)
    lat = np.linspace(30, 50, 21, dtype=np.float32)
    time = np.array([0, 3600, 7200], dtype=np.float64)
    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {"uo": (("time", "lat", "lon"), np.full(shape, 0.1, dtype=np.float32)),
         "vo": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32))},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    fieldset = FieldSet.from_xarray_dataset(
        ds, {"U": "uo", "V": "vo"},
        {"lon": "lon", "lat": "lat", "time": "time"},
        mesh="spherical", allow_time_extrapolation=False,
    )
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=[0.0], lat=[40.0], time=[0.0])
    with tempfile.TemporaryDirectory() as tmp:
        output = str(Path(tmp) / "parcels_smoke.zarr")
        pfile = pset.ParticleFile(name=output, outputdt=timedelta(hours=1))
        pset.execute(AdvectionRK4, runtime=timedelta(hours=2),
                     dt=timedelta(minutes=10), output_file=pfile)
        pfile.close()
        with xr.open_zarr(output) as traj:
            if "lon" not in traj or "lat" not in traj or "time" not in traj:
                raise AssertionError(f"Missing trajectory variables: {list(traj)}")
            if not np.isfinite(traj.lon.values).any():
                raise AssertionError("No finite simulated longitude")
            print("PASS: Parcels executed and wrote valid Zarr trajectory:", dict(traj.sizes))

if __name__ == "__main__":
    main()
