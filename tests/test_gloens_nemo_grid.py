"""HPC diagnostic for GLOENS daily NetCDF / NEMO meshmask compatibility.

Run: python -m pytest -s tests/test_gloens_nemo_grid.py
Optional: GLOENS_CACHE_DIR=/scratch/.../nemo_daily_R20240104
          GLOENS_MESHMASK=/scratch/.../iORCA025_meshmask_v3.1.nc

Reads original NetCDF files without changing them. FieldSet test creates/reuses\na separate shared-depth NetCDF cache (requires additional disk space).
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



def test_parcels_nemo_fieldset_construction():
    """Reproduce the reported assertion without running trajectories or writing files."""
    parcels = pytest.importorskip("parcels")
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    u_files = sorted(CACHE.glob("U_*.nc"))
    v_files = sorted(CACHE.glob("V_*.nc"))
    if not u_files or not v_files:
        pytest.skip(f"No complete daily files in {CACHE}")
    assert len(u_files) == len(v_files), "U/V day counts differ"
    from src.gloens_parcels import prepare_shared_depth_files
    shared_root = CACHE.parent / CACHE.name.replace("nemo_daily_", "nemo_shared_depth_")
    shared_u, shared_v = prepare_shared_depth_files(
        [str(p) for p in u_files], [str(p) for p in v_files], shared_root
    )
    filenames = {
        "U": {"lon": str(MESH), "lat": str(MESH), "depth": shared_u,
              "data": shared_u},
        "V": {"lon": str(MESH), "lat": str(MESH), "depth": shared_v,
              "data": shared_v},
    }
    dims = {
        "U": {"lon": "nav_lon", "lat": "nav_lat", "depth": "depth", "time": "time"},
        "V": {"lon": "nav_lon", "lat": "nav_lat", "depth": "depth", "time": "time"},
    }
    fieldset = parcels.FieldSet.from_nemo(
        filenames, {"U": "uo", "V": "vo"}, dims,
        allow_time_extrapolation=True, time_periodic=False, deferred_load=True,
    )
    for name in ("U", "V"):
        grid = getattr(fieldset, name).grid
        print(f"{name} grid: xdim={grid.xdim}, ydim={grid.ydim}, "
              f"zdim={grid.zdim}, tdim={grid.tdim}, "
              f"lon_shape={np.shape(grid.lon)}, lat_shape={np.shape(grid.lat)}")
    assert fieldset.U.grid.xdim == fieldset.V.grid.xdim
    assert fieldset.U.grid.ydim == fieldset.V.grid.ydim
    assert fieldset.U.grid.zdim == 75
    assert fieldset.V.grid.zdim == 75


def test_fpoint_2d_fieldset_at_15m():
    """Diagnostic: one daily slice, nearest 15 m, using NEMO F-point coordinates."""
    parcels = pytest.importorskip("parcels")
    xr = pytest.importorskip("xarray")
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    arrays = []
    for label, var, zdim in (("U", "uo", "depthu"), ("V", "vo", "depthv")):
        with xr.open_dataset(_daily_file(label)) as source:
            chosen = source[zdim].sel({zdim: 15.0}, method="nearest").item()
            print(f"{label}: nearest model depth to 15 m = {chosen}")
            arrays.append(source[var].sel({zdim: 15.0}, method="nearest").load())
    with xr.open_dataset(MESH) as mesh:
        lon = mesh["glamf"].isel(time_counter=0).load() if "time_counter" in mesh["glamf"].dims else mesh["glamf"].load()
        lat = mesh["gphif"].isel(time_counter=0).load() if "time_counter" in mesh["gphif"].dims else mesh["gphif"].load()
    ds = xr.Dataset({"uo": arrays[0], "vo": arrays[1], "glamf": lon, "gphif": lat})
    fieldset = parcels.FieldSet.from_xarray_dataset(
        ds, {"U": "uo", "V": "vo"},
        {"lon": "glamf", "lat": "gphif", "time": "time"},
        mesh="spherical", allow_time_extrapolation=True,
    )
    print(f"2D F-point: U zdim={fieldset.U.grid.zdim}, V zdim={fieldset.V.grid.zdim}")
    assert fieldset.U.grid.xdim == fieldset.V.grid.xdim == 1440
    assert fieldset.U.grid.ydim == fieldset.V.grid.ydim == 1049


def test_fpoint_3d_nemo_fieldset():
    """Diagnostic: 75-level deferred NEMO FieldSet using F-point coordinates."""
    parcels = pytest.importorskip("parcels")
    if not MESH.is_file():
        pytest.skip(f"Meshmask unavailable: {MESH}")
    from src.gloens_parcels import prepare_shared_depth_files
    u_files = sorted(CACHE.glob("U_*.nc"))[:1]
    v_files = sorted(CACHE.glob("V_*.nc"))[:1]
    if not u_files or not v_files:
        pytest.skip(f"No daily NetCDF files in {CACHE}")
    assert len(u_files) == len(v_files)
    print(f"3D F-point diagnostic: testing only {u_files[0].name} and {v_files[0].name}")
    shared_root = CACHE.parent / CACHE.name.replace("nemo_daily_", "nemo_shared_depth_")
    shared_u, shared_v = prepare_shared_depth_files(
        [str(p) for p in u_files], [str(p) for p in v_files], shared_root
    )
    filenames = {
        "U": {"lon": str(MESH), "lat": str(MESH), "depth": shared_u, "data": shared_u},
        "V": {"lon": str(MESH), "lat": str(MESH), "depth": shared_v, "data": shared_v},
    }
    dims = {"lon": "glamf", "lat": "gphif", "depth": "depth", "time": "time"}
    fieldset = parcels.FieldSet.from_nemo(
        filenames, {"U": "uo", "V": "vo"}, dims,
        allow_time_extrapolation=True, time_periodic=False, deferred_load=True,
    )
    for name in ("U", "V"):
        grid = getattr(fieldset, name).grid
        print(f"3D F-point {name}: xdim={grid.xdim}, ydim={grid.ydim}, "
              f"zdim={grid.zdim}, tdim={grid.tdim}")
        assert grid.zdim == 75
    assert fieldset.U.grid.xdim == fieldset.V.grid.xdim
    assert fieldset.U.grid.ydim == fieldset.V.grid.ydim
