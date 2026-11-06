"""Parcels 4 advection from daily NEMO NetCDF U/V files.

Uses the same F-point SGRID construction as the GLOENS runner.  This is
an opt-in replacement for the legacy Parcels 3 NetCDF path.
"""
from datetime import timedelta
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr


def _paths(files):
    if isinstance(files, (str, Path)):
        return [str(files)]
    paths = sorted(str(path) for path in files)
    if not paths:
        raise ValueError("No NetCDF velocity files provided")
    return paths


def _open_surface(files, variable, depth, depth_axis):
    """Open only the requested NEMO depth, retaining lazy time slices."""
    def preprocess(ds):
        if variable not in ds:
            raise KeyError(f"Missing velocity variable {variable!r}")
        arr = ds[variable]
        if depth_axis not in arr.dims:
            raise ValueError(f"{variable} lacks {depth_axis}: {arr.dims}")
        arr = arr.sel({depth_axis: float(depth)}, method="nearest")
        arr = arr.reset_coords(drop=True)
        return arr.to_dataset(name=variable)

    return xr.open_mfdataset(
        _paths(files), combine="by_coords", preprocess=preprocess,
        chunks={"time_counter": 1}, compat="no_conflicts",
    )


def run_netcdf_parcels4(u_files, v_files, u_variable, v_variable, meshmask,
                        lon0, lat0, drifter_ids, drift_days, output_path, *,
                        depth=0.0, scheme="AdvectionRK4", dt_minutes=5,
                        sampling_hours=6, log=None):
    """Advect observed seeds on a 2D NEMO C-grid and write Parcels 4 Parquet.

    The original seed order is retained: Parquet particle_id is matched to
    observed drifter IDs by Retrieve_traj_parquet downstream.
    """
    import parcels

    log = log or logging.getLogger(__name__)
    if int(parcels.__version__.split(".")[0]) < 4:
        raise RuntimeError("NetCDF SGRID runner requires Parcels >= 4")
    if scheme not in ("AdvectionRK4", "AdvectionEE"):
        raise NotImplementedError(f"Parcels 4 NetCDF scheme {scheme!r} not supported")
    if dt_minutes <= 0 or sampling_hours <= 0 or drift_days <= 0:
        raise ValueError("Drift, integration step and sampling interval must be positive")
    if 24 % sampling_hours:
        raise ValueError("Sampling interval must divide 24 hours")
    if (drift_days * 24 * 60) % dt_minutes:
        raise ValueError("Drift duration must be divisible by integration step")

    lon0 = np.asarray(lon0, dtype=float).ravel()
    lat0 = np.asarray(lat0, dtype=float).ravel()
    if len(lon0) == 0 or len(lon0) != len(lat0) or len(lon0) != len(drifter_ids):
        raise ValueError("Seed positions and observed drifter IDs must match and be nonempty")
    if not np.isfinite(lon0).all() or not np.isfinite(lat0).all():
        raise ValueError("Nonfinite drifter seed coordinates")

    # Retrieve_data_local returns meshmask as a list (one path per input).
    if isinstance(meshmask, (list, tuple)):
        unique_meshes = {str(item) for item in meshmask}
        if len(unique_meshes) != 1:
            raise ValueError(f"Expected one shared NEMO meshmask, got: {sorted(unique_meshes)}")
        meshmask = unique_meshes.pop()
    # Legacy Retrieve_data_local wraps variable names in lists.
    def _single_variable(value, component):
        if isinstance(value, (list, tuple)):
            names = {str(item) for item in value}
            if len(names) != 1:
                raise ValueError(
                    f"Expected one {component} variable name, got {sorted(names)}"
                )
            return names.pop()
        if not isinstance(value, str):
            raise TypeError(f"{component} variable name must be a string: {value!r}")
        return value

    u_variable = _single_variable(u_variable, "U")
    v_variable = _single_variable(v_variable, "V")
    grid_path = Path(meshmask)
    if not grid_path.is_file():
        raise FileNotFoundError(f"NEMO meshmask not found: {grid_path}")
    with _open_surface(u_files, u_variable, depth, "depthu") as uds, \
         _open_surface(v_files, v_variable, depth, "depthv") as vds:
        u = uds[u_variable].rename({"time_counter": "time"})
        v = vds[v_variable].rename({"time_counter": "time"})
        if u.sizes != v.sizes:
            raise ValueError(f"U/V dimensions differ: {u.sizes} / {v.sizes}")
        times = pd.to_datetime(u.time.values).to_numpy(dtype="datetime64[ns]")
        vtimes = pd.to_datetime(v.time.values).to_numpy(dtype="datetime64[ns]")
        if len(times) < 2 or not np.array_equal(times, vtimes):
            raise ValueError("U/V times must match and contain at least two samples")
        if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
            raise ValueError("Velocity timestamps must be strictly increasing")
        # The final RK4 step samples velocities no later than the requested
        # endpoint. A velocity record at that endpoint is sufficient.
        needed = timedelta(days=float(drift_days))
        coverage = pd.Timestamp(times[-1]) - pd.Timestamp(times[0])
        if coverage < needed:
            raise ValueError(
                f"Velocity coverage {coverage} insufficient for drift {drift_days}d. "
                "Supply NetCDF records through the requested integration endpoint."
            )
        with xr.open_dataset(grid_path) as geometry:
            coords = geometry[["glamf", "gphif"]]
            if "t" in coords.dims:
                coords = coords.isel(t=0, drop=True)
            if "time_counter" in coords.dims:
                coords = coords.isel(time_counter=0, drop=True)
            coords = coords.load()
        if coords["glamf"].shape != u.shape[-2:] or coords["gphif"].shape != u.shape[-2:]:
            raise ValueError("F-grid geometry shape does not match NetCDF velocity grid")
        log.info("GLO12 NetCDF SGRID: %d time samples from %s to %s; depth %.2fm",
                 len(times), times[0], times[-1], depth)
        sgrid = parcels.convert.nemo_to_sgrid(
            fields={"U": u, "V": v}, coords=coords
        )
        fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid)
        pset = parcels.ParticleSet(
            fieldset=fieldset, pclass=parcels.Particle,
            x=lon0, y=lat0, t=pd.Timestamp(times[0]).to_pydatetime()
        )
        destination = Path(output_path).with_suffix(".parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite trajectory: {destination}")
        output = parcels.ParticleFile(
            str(destination), outputdt=timedelta(hours=sampling_hours)
        )
        kernel = (parcels.kernels.AdvectionRK4 if scheme == "AdvectionRK4"
                  else parcels.kernels.AdvectionEE)
        log.info("Running Parcels 4 NetCDF %s: %d seeds, %s days, dt=%s min",
                 scheme, len(lon0), drift_days, dt_minutes)
        pset.execute(
            kernel, runtime=timedelta(days=float(drift_days)),
            dt=timedelta(minutes=dt_minutes), output_file=output
        )
        log.info("Saved NetCDF Parcels 4 trajectories: %s", destination)
        return str(destination)
