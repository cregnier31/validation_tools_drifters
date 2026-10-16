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
