"""HPC diagnostic for GLOENS daily NetCDF / NEMO meshmask compatibility.

Run: python -m pytest -s tests/test_gloens_nemo_grid.py
Optional: GLOENS_CACHE_DIR=/scratch/.../nemo_daily_R20240104
          GLOENS_MESHMASK=/scratch/.../iORCA025_meshmask_v3.1.nc

Reads original NetCDF files without changing them. The 3D test uses one day only.
"""
import os
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

CACHE = Path(os.environ.get(
    "GLOENS_CACHE_DIR",
    "/scratch/cregnier/WORK_lagrangian/zarr_ensemble_mean_cache/nemo_daily_R20240104",
))
MESH = Path(os.environ.get(
    "GLOENS_MESHMASK", "/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc"
))


def _daily_file(prefix):
    files = sorted(CACHE.glob(f"{prefix}_*.nc"))
    if not files:
        pytest.skip(f"No {prefix} daily NetCDF files in {CACHE}")
    return files[0]


def _metadata(prefix, variable):
    path = _daily_file(prefix)
    with netCDF4.Dataset(path) as ds:
        assert variable in ds.variables, f"{path}: missing {variable}"
        data = ds.variables[variable]
        dims, shape = data.dimensions, data.shape
        print(f"{prefix}: {path.name} {variable} dims={dims} shape={shape}")
        assert len(shape) == 4, f"{prefix}: expected time/depth/y/x"
        assert shape[0] == 1
        assert shape[1] == 75, f"{prefix}: must retain all 75 depths"
        assert dims[-2:] == ("y", "x")
        return dims, shape


def test_daily_velocity_dimensions():
    u_dims, u_shape = _metadata("U", "uo")
    v_dims, v_shape = _metadata("V", "vo")
    assert u_shape == v_shape, f"U/V shape mismatch: {u_shape} vs {v_shape}"
    assert u_dims[1] == "depthu"
    assert v_dims[1] == "depthv"


def test_meshmask_horizontal_coordinates():
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    _, shape = _metadata("U", "uo")
    with netCDF4.Dataset(MESH) as ds:
        for name in ("glamu", "gphiu", "glamv", "gphiv", "nav_lon", "nav_lat"):
            if name not in ds.variables:
                print(f"Meshmask does not contain {name}")
                continue
            var = ds.variables[name]
            print(f"meshmask {name}: dims={var.dimensions}, shape={var.shape}")
            assert var.shape[-2:] == shape[-2:], (
                f"{name}: meshmask horizontal shape {var.shape[-2:]} != velocity {shape[-2:]}"
            )
            # Inspect just a few grid coordinates, without reading whole 2D arrays.
            print(f"  corners: {var[..., 0, 0]}, {var[..., -1, -1]}")



def test_meshmask_interior_coordinates():
    """Print interior geographic coordinates, not just masked boundary corners."""
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    with netCDF4.Dataset(MESH) as ds:
        for name in ("glamu", "gphiu", "glamv", "gphiv", "nav_lon", "nav_lat"):
            if name not in ds.variables:
                pytest.fail(f"Meshmask missing coordinate {name}")
            var = ds.variables[name]
            value = var[..., 500, 700]
            print(f"meshmask {name} center [500,700]: {value}")
            assert np.all(np.isfinite(value)), f"{name}: nonfinite interior coordinate"




def test_meshmask_vertical_coordinate_metadata():
    """Report NEMO vertical variables without reading large arrays or changing cache."""
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    names = ("depthu", "depthv", "gdept_0", "gdepu_0", "gdepv_0")
    with netCDF4.Dataset(MESH) as ds:
        for name in names:
            if name in ds.variables:
                var = ds.variables[name]
                print(f"meshmask {name}: dims={var.dimensions}, shape={var.shape}")
            else:
                print(f"meshmask {name}: MISSING")
    for prefix, varname in (("U", "uo"), ("V", "vo")):
        path = _daily_file(prefix)
        with netCDF4.Dataset(path) as ds:
            depth_dim = ds.variables[varname].dimensions[1]
            print(f"daily {prefix}: depth dimension={depth_dim}, "
                  f"coordinate variable present={depth_dim in ds.variables}")
            if depth_dim in ds.variables:
                coord = ds.variables[depth_dim]
                print(f"daily {prefix} {depth_dim}: dims={coord.dimensions}, shape={coord.shape}")



def test_fpoint_3d_xarray_one_day():
    """Build the production-style 3D F-point FieldSet from just one cached day."""
    parcels = pytest.importorskip("parcels")
    xr = pytest.importorskip("xarray")
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    with xr.open_dataset(_daily_file("U")) as u_ds, xr.open_dataset(_daily_file("V")) as v_ds:
        u_depth = u_ds["depthu"].values
        v_depth = v_ds["depthv"].values
        assert len(u_depth) == len(v_depth) == 75
        np.testing.assert_allclose(u_depth, v_depth, rtol=1e-6, atol=1e-4)
        u = u_ds["uo"].rename({"depthu": "depth"})
        v = v_ds["vo"].rename({"depthv": "depth"}).assign_coords(depth=u_depth)
        ds = xr.Dataset({"uo": u, "vo": v})
        with xr.open_dataset(MESH) as mesh:
            for name in ("glamf", "gphif"):
                coord = mesh[name]
                if "time_counter" in coord.dims:
                    coord = coord.isel(time_counter=0, drop=True)
                ds = ds.assign_coords({name: coord.load()})
        fieldset = parcels.FieldSet.from_xarray_dataset(
            ds, {"U": "uo", "V": "vo"},
            {"lon": "glamf", "lat": "gphif", "depth": "depth", "time": "time"},
            mesh="spherical", allow_time_extrapolation=True,
        )
        for name in ("U", "V"):
            grid = getattr(fieldset, name).grid
            print(f"{name}: xdim={grid.xdim} ydim={grid.ydim} zdim={grid.zdim} tdim={grid.tdim}")
            assert grid.zdim == 75


def test_fpoint_3d_one_day_advection():
    """Advect one particle for one day on a single 75-level GLOENS forecast slice."""
    from datetime import timedelta
    parcels = pytest.importorskip("parcels")
    xr = pytest.importorskip("xarray")
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    with xr.open_dataset(_daily_file("U")) as u_ds, xr.open_dataset(_daily_file("V")) as v_ds:
        u_depth = u_ds["depthu"].values
        v_depth = v_ds["depthv"].values
        assert len(u_depth) == len(v_depth) == 75
        np.testing.assert_allclose(u_depth, v_depth, rtol=1e-6, atol=1e-4)
        ds = xr.Dataset({
            "uo": u_ds["uo"].rename({"depthu": "depth"}),
            "vo": v_ds["vo"].rename({"depthv": "depth"}).assign_coords(depth=u_depth),
        })
        with xr.open_dataset(MESH) as mesh:
            for name in ("glamf", "gphif"):
                coord = mesh[name]
                if "time_counter" in coord.dims:
                    coord = coord.isel(time_counter=0, drop=True)
                ds = ds.assign_coords({name: coord.load()})
        fieldset = parcels.FieldSet.from_xarray_dataset(
            ds, {"U": "uo", "V": "vo"},
            {"lon": "glamf", "lat": "gphif", "depth": "depth", "time": "time"},
            mesh="spherical", allow_time_extrapolation=True,
        )
        assert fieldset.U.grid.zdim == fieldset.V.grid.zdim == 75
        # Interior ocean position previously used in GLOENS trajectory diagnostics.
        pset = parcels.ParticleSet(
            fieldset=fieldset, pclass=parcels.JITParticle,
            lon=[-58.001], lat=[23.021], depth=[15.0], time=[0.0],
        )
        print("Starting one-day RK4 advection at lon=-58.001, lat=23.021, depth=15 m")
        pset.execute(parcels.AdvectionRK4, runtime=timedelta(days=1),
                     dt=timedelta(minutes=5))
        assert len(pset) == 1
        particle = next(iter(pset))
        print(f"End position: lon={particle.lon}, lat={particle.lat}, depth={particle.depth}")
        assert np.isfinite(particle.lon) and np.isfinite(particle.lat)
        assert np.isclose(particle.depth, 15.0)


def test_fpoint_particle_location_diagnostic():
    """Inspect F-point coordinate validity near the failed particle, without advection."""
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    with netCDF4.Dataset(MESH) as ds:
        lon_var, lat_var = ds.variables["glamf"], ds.variables["gphif"]
        lon = np.ma.filled(lon_var[0] if lon_var.ndim == 3 else lon_var[:], np.nan)
        lat = np.ma.filled(lat_var[0] if lat_var.ndim == 3 else lat_var[:], np.nan)
    valid = np.isfinite(lon) & np.isfinite(lat)
    valid &= (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    target_lon, target_lat = -58.001, 23.021
    dlon = ((lon - target_lon + 180) % 360) - 180
    distance = dlon**2 * np.cos(np.deg2rad(target_lat))**2 + (lat - target_lat)**2
    distance[~valid] = np.inf
    j, i = np.unravel_index(np.argmin(distance), distance.shape)
    print(f"F-point valid coordinates: {np.count_nonzero(valid)}/{valid.size}")
    print(f"F-point [0,0]: lon={lon[0,0]}, lat={lat[0,0]}")
    print(f"Closest F-point to particle: (j={j}, i={i}), lon={lon[j,i]}, lat={lat[j,i]}, distance_deg={np.sqrt(distance[j,i])}")
    print(f"Neighboring 3x3 F-point longitudes:\\n{lon[max(j-1,0):j+2,max(i-1,0):i+2]}")
    print(f"Neighboring 3x3 F-point latitudes:\\n{lat[max(j-1,0):j+2,max(i-1,0):i+2]}")
    assert np.isfinite(distance[j,i])
    assert np.sqrt(distance[j,i]) < 1, "Particle is not near a valid F-point cell"


def test_fpoint_2d_one_day_advection():
    """Exercise the new 2D nearest-15m F-point setup for one day."""
    from datetime import timedelta
    parcels = pytest.importorskip("parcels")
    xr = pytest.importorskip("xarray")
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    with xr.open_dataset(_daily_file("U")) as u_ds, xr.open_dataset(_daily_file("V")) as v_ds:
        u = u_ds["uo"].sel(depthu=15.0, method="nearest").reset_coords(drop=True)
        v = v_ds["vo"].sel(depthv=15.0, method="nearest").reset_coords(drop=True)
        ds = xr.Dataset({"uo": u, "vo": v})
        with xr.open_dataset(MESH) as mesh:
            for name in ("glamf", "gphif"):
                coord = mesh[name]
                if "time_counter" in coord.dims:
                    coord = coord.isel(time_counter=0, drop=True)
                ds = ds.assign_coords({name: coord.load()})
        fieldset = parcels.FieldSet.from_xarray_dataset(
            ds, {"U": "uo", "V": "vo"},
            {"lon": "glamf", "lat": "gphif", "time": "time"},
            mesh="spherical", allow_time_extrapolation=True,
        )
        assert fieldset.U.grid.zdim == fieldset.V.grid.zdim == 1
        pset = parcels.ParticleSet(fieldset=fieldset, pclass=parcels.JITParticle,
                                    lon=[-58.001], lat=[23.021], time=[0.0])
        pset.execute(parcels.AdvectionRK4, runtime=timedelta(days=1),
                     dt=timedelta(minutes=5))
        assert len(pset) == 1
        particle = next(iter(pset))
        print(f"2D one-day end position: lon={particle.lon}, lat={particle.lat}")
        assert np.isfinite(particle.lon) and np.isfinite(particle.lat)


def test_fpoint_2d_wet_cell_and_search_diagnostic():
    """One-day read-only comparison: observed start versus nearby wet F point."""
    from datetime import timedelta
    xr = pytest.importorskip("xarray")
    parcels = pytest.importorskip("parcels")
    if not MESH.is_file():
        pytest.skip(f"Missing meshmask: {MESH}")
    j0, i0 = 621, 915
    with xr.open_dataset(_daily_file("U")) as ud, xr.open_dataset(_daily_file("V")) as vd:
        u = ud["uo"].sel(depthu=15., method="nearest").reset_coords(drop=True)
        v = vd["vo"].sel(depthv=15., method="nearest").reset_coords(drop=True)
        with xr.open_dataset(MESH) as mesh:
            lon = mesh["glamf"].isel(time_counter=0, drop=True).load()
            lat = mesh["gphif"].isel(time_counter=0, drop=True).load()
            print("Meshmask masks available:", [n for n in ("tmask", "umask", "vmask", "fmask") if n in mesh])
            for name in ("tmask", "umask", "vmask", "fmask"):
                if name in mesh:
                    mask = mesh[name]
                    sel = {d: (0 if d == "time_counter" else 0) for d in mask.dims if d not in ("y", "x")}
                    print(name, "near start:", mask.isel(y=slice(j0-1,j0+2), x=slice(i0-1,i0+2), **sel).values)
            print("U near start:", u.isel(y=slice(j0-1,j0+2), x=slice(i0-1,i0+2)).values)
            print("V near start:", v.isel(y=slice(j0-1,j0+2), x=slice(i0-1,i0+2)).values)
            ds = xr.Dataset({"uo": u, "vo": v}).assign_coords(glamf=lon, gphif=lat)
        fs = parcels.FieldSet.from_xarray_dataset(
            ds, {"U":"uo","V":"vo"},
            {"lon":"glamf","lat":"gphif","time":"time"},
            mesh="spherical", allow_time_extrapolation=True)
        print("Parcels grid dimensions:", fs.U.grid.xdim, fs.U.grid.ydim)
        for label, x, y in (("drifter", -58.001, 23.021),
                            ("nearest_F_point", float(lon[j0,i0]), float(lat[j0,i0]))):
            pset = parcels.ParticleSet(fieldset=fs, pclass=parcels.JITParticle,
                                       lon=[x], lat=[y], time=[0.])
            try:
                pset.execute(parcels.AdvectionRK4, runtime=timedelta(minutes=5),
                             dt=timedelta(minutes=5))
            except Exception as exc:
                print(f"{label}: FAILED {type(exc).__name__}: {exc}")
            else:
                p = next(iter(pset))
                print(f"{label}: OK lon={p.lon} lat={p.lat}")


def test_fpoint_2d_seeded_grid_indices():
    """Check whether seeding the initial curvilinear cell bypasses (0,0)."""
    from datetime import timedelta
    xr = pytest.importorskip("xarray")
    parcels = pytest.importorskip("parcels")
    if not MESH.is_file():
        pytest.skip(f"Missing meshmask: {MESH}")
    with xr.open_dataset(_daily_file("U")) as ud, xr.open_dataset(_daily_file("V")) as vd:
        u = ud["uo"].sel(depthu=15., method="nearest").reset_coords(drop=True)
        v = vd["vo"].sel(depthv=15., method="nearest").reset_coords(drop=True)
        with xr.open_dataset(MESH) as mesh:
            coords = {}
            for name in ("glamf", "gphif"):
                coord = mesh[name]
                if "time_counter" in coord.dims:
                    coord = coord.isel(time_counter=0, drop=True)
                coords[name] = coord.load()
        ds = xr.Dataset({"uo": u, "vo": v}).assign_coords(coords)
        fs = parcels.FieldSet.from_xarray_dataset(
            ds, {"U":"uo","V":"vo"},
            {"lon":"glamf","lat":"gphif","time":"time"},
            mesh="spherical", allow_time_extrapolation=True)
        pset = parcels.ParticleSet(fieldset=fs, pclass=parcels.JITParticle,
                                   lon=[-58.001], lat=[23.021], time=[0.])
        p = next(iter(pset))
        print("Particle index variables:", [name for name in ("xi", "yi") if hasattr(p, name)])
        print("Grid count:", len(fs.gridset.grids))
        print("Initial indices:", p.xi, p.yi)
        # In Parcels 3.x indices are arrays indexed by the grid ID.
        for grid in fs.gridset.grids:
            gid = grid._gtype if False else fs.gridset.grids.index(grid)
            p.xi[gid] = 915
            p.yi[gid] = 621
        print("Seeded indices:", p.xi, p.yi)
        for label, field in (("U", fs.U), ("V", fs.V)):
            try:
                value = field[0., 0., float(p.lat), float(p.lon)]
                print(f"Direct {label} sample:", value)
            except Exception as exc:
                print(f"Direct {label} sample FAILED: {type(exc).__name__}: {exc}")
        print("Before execution: time=", p.time, "state=", p.state)
        pset.execute(parcels.AdvectionRK4, runtime=timedelta(minutes=5),
                     dt=timedelta(minutes=5))
        print("Seeded particle after 5 minutes:", p.lon, p.lat)
        print("Particle time:", p.time, "state:", p.state)
        print("Field time axis:", fs.U.grid.time, "time_origin:", fs.U.grid.time_origin)
        print("Particle initial time was explicitly 0.0; inspect whether it lies in field time range")
        print("Particle state meanings:", {name: getattr(parcels.StatusCode, name, None) for name in ("Success", "Evaluate", "Repeat", "Delete", "StopExecution", "Error") } if hasattr(parcels, "StatusCode") else "StatusCode not exported")
        print("Particle grid indices after step:", p.xi, p.yi)
        assert np.isfinite(p.lon) and np.isfinite(p.lat)
        assert abs(float(p.lon) + 58.001) + abs(float(p.lat) - 23.021) > 1e-6, (
            "Particle did not advect despite finite local U/V; inspect time and kernel execution"
        )


def test_fpoint_2d_seeded_two_timestamps():
    """Check whether a genuine time interval enables one RK4 step."""
    from datetime import timedelta
    xr = pytest.importorskip("xarray")
    parcels = pytest.importorskip("parcels")
    if not MESH.is_file():
        pytest.skip(f"Missing meshmask: {MESH}")
    with xr.open_dataset(_daily_file("U")) as ud, xr.open_dataset(_daily_file("V")) as vd:
        u = ud["uo"].sel(depthu=15., method="nearest").reset_coords(drop=True)
        v = vd["vo"].sel(depthv=15., method="nearest").reset_coords(drop=True)
        with xr.open_dataset(MESH) as mesh:
            coords = {}
            for name in ("glamf", "gphif"):
                c = mesh[name]
                if "time_counter" in c.dims:
                    c = c.isel(time_counter=0, drop=True)
                coords[name] = c.load()
        ds = xr.Dataset({"uo": u, "vo": v}).assign_coords(coords)
        # Duplicate the single day's field in memory only; do not modify NetCDF.
        t0 = ds.time.values[0]
        t1 = t0 + np.timedelta64(1, "D")
        ds = xr.concat([ds, ds], dim="time").assign_coords(time=[t0, t1])
        fs = parcels.FieldSet.from_xarray_dataset(
            ds, {"U":"uo","V":"vo"},
            {"lon":"glamf","lat":"gphif","time":"time"},
            mesh="spherical", allow_time_extrapolation=False)
        pset = parcels.ParticleSet(fieldset=fs, pclass=parcels.JITParticle,
                                   lon=[-58.001], lat=[23.021], time=[0.])
        p = next(iter(pset))
        p.xi[0], p.yi[0] = 915, 621
        print("Two-timestamp field time axis:", fs.U.grid.time)
        print("Before:", p.lon, p.lat, p.time, p.state)
        pset.execute(parcels.AdvectionRK4, runtime=timedelta(minutes=5),
                     dt=timedelta(minutes=5))
        p_after = next(iter(pset))
        print("After (fresh ParticleSet lookup):", p_after.lon, p_after.lat, p_after.time, p_after.state)
        print("Before-reference after execute:", p.lon, p.lat, p.time, p.state)
        assert float(p_after.time) >= 300., "Particle did not advance in time"
        assert abs(float(p_after.lon) + 58.001) + abs(float(p_after.lat) - 23.021) > 1e-6


def test_parcels_minimal_execution_control():
    """Independent Parcels 3.1.4 control with a tiny regular grid."""
    from datetime import timedelta
    parcels = pytest.importorskip("parcels")
    fieldset = parcels.FieldSet.from_data(
        {"U": np.full((2, 3, 3), 0.2, dtype=np.float32),
         "V": np.full((2, 3, 3), 0.1, dtype=np.float32)},
        {"lon": np.array([-59., -58., -57.], dtype=np.float32),
         "lat": np.array([22., 23., 24.], dtype=np.float32),
         "time": np.array([0., 86400.], dtype=np.float64)},
        mesh="spherical", allow_time_extrapolation=False,
    )
    pset = parcels.ParticleSet(fieldset=fieldset, pclass=parcels.JITParticle,
                               lon=[-58.001], lat=[23.021], time=[0.])
    before = next(iter(pset))
    print("Control before:", before.lon, before.lat, before.time, before.state)
    pset.execute(parcels.AdvectionRK4, runtime=timedelta(minutes=5),
                 dt=timedelta(minutes=5))
    after = next(iter(pset))
    print("Control after:", after.lon, after.lat, after.time, after.state)
    assert float(after.time) >= 300., "Even the independent regular-grid control did not execute"
    assert abs(float(after.lon) + 58.001) + abs(float(after.lat) - 23.021) > 1e-6


@pytest.mark.parametrize("particle_type", ["JITParticle", "ScipyParticle"])
def test_parcels_execution_backend_control(particle_type):
    """Check both Parcels execution backends without GLOENS dependencies."""
    from datetime import timedelta
    parcels = pytest.importorskip("parcels")
    fieldset = parcels.FieldSet.from_data(
        {"U": np.full((2, 3, 3), 0.2, dtype=np.float32),
         "V": np.full((2, 3, 3), 0.1, dtype=np.float32)},
        {"lon": np.array([-59., -58., -57.], dtype=np.float32),
         "lat": np.array([22., 23., 24.], dtype=np.float32),
         "time": np.array([0., 86400.], dtype=np.float64)},
        mesh="spherical", allow_time_extrapolation=False,
    )
    cls = getattr(parcels, particle_type)
    pset = parcels.ParticleSet(fieldset=fieldset, pclass=cls,
                               lon=[-58.001], lat=[23.021], time=[0.])
    pset.execute(parcels.AdvectionRK4, runtime=timedelta(minutes=10),
                 dt=timedelta(minutes=5))
    p = next(iter(pset))
    print(f"{particle_type} after 10min:", p.lon, p.lat, p.time, p.state)
    assert float(p.time) >= 600., f"{particle_type} did not advance"


def test_fpoint_parcels4_sgrid_one_day():
    """Parcels 4 NEMO SGRID smoke test: one cached day, nearest 15m only."""
    from datetime import timedelta
    parcels = pytest.importorskip("parcels")
    if int(parcels.__version__.split(".")[0]) < 4:
        pytest.skip("Parcels 4 test")
    xr = pytest.importorskip("xarray")
    if not MESH.is_file():
        pytest.skip(f"Missing meshmask: {MESH}")
    with xr.open_dataset(_daily_file("U")) as ud, xr.open_dataset(_daily_file("V")) as vd:
        u = ud["uo"].sel(depthu=15., method="nearest").reset_coords(drop=True)
        v = vd["vo"].sel(depthv=15., method="nearest").reset_coords(drop=True)
        with xr.open_dataset(MESH) as mesh:
            sgrid = parcels.convert.nemo_to_sgrid(
                fields={"U": u, "V": v}, coords=mesh
            )
            fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid)
        fieldset.describe()
        pset = parcels.ParticleSet(
            fieldset, x=[-58.001], y=[23.021], t=[u.time.values[0]]
        )
        print("Parcels 4 before:", pset.x, pset.y, pset.t)
        pset.execute([parcels.kernels.AdvectionRK4],
                     runtime=timedelta(minutes=10), dt=timedelta(minutes=5))
        print("Parcels 4 after:", pset.x, pset.y, pset.t)
        assert np.isfinite(pset.x).all() and np.isfinite(pset.y).all()
        assert abs(float(pset.x[0]) + 58.001) + abs(float(pset.y[0]) - 23.021) > 1e-6
