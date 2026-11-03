"""Conservative spatial filtering of GLO12 velocities before Parcels advection.

Generate weights with xESMF in a separate environment; apply with SciPy in Parcels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr


def _bounds(centers, *, latitude=False):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or not np.all(np.diff(centers) > 0):
        raise ValueError("Conservative remapping requires strictly increasing 1-D coordinates")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers[1:] + centers[:-1]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    if latitude:
        edges = np.clip(edges, -90., 90.)
    return edges


def grids_and_path(velocities, cache_dir):
    """Create grids and deterministic weight path from actual Zarr coordinates."""
    u = velocities
    if not {"latitude", "longitude"}.issubset(u.dims):
        raise ValueError("Expected regular latitude/longitude dimensions")
    lon = np.sort(((u.longitude.values.astype(float) + 180) % 360) - 180)
    lat = np.sort(u.latitude.values.astype(float))
    if np.unique(lon).size != len(lon):
        raise ValueError("Longitude coordinates are not unique")
    lon_b, lat_b = _bounds(lon), _bounds(lat, latitude=True)
    tlon = np.arange(-179.875, 180, .25)
    tlat = np.arange(-89.875, 90, .25)
    tlon = tlon[(tlon - .125 >= lon_b[0] - 1e-8) &
                (tlon + .125 <= lon_b[-1] + 1e-8)]
    tlat = tlat[(tlat - .125 >= lat_b[0] - 1e-8) &
                (tlat + .125 <= lat_b[-1] + 1e-8)]
    if min(len(tlon), len(tlat)) < 2:
        raise ValueError("Insufficient 1/4-degree target coverage")
    src = xr.Dataset(coords={"lon": ("lon", lon), "lat": ("lat", lat),
                             "lon_b": ("lon_b", lon_b), "lat_b": ("lat_b", lat_b)})
    dst = xr.Dataset(coords={"lon": ("lon", tlon), "lat": ("lat", tlat),
                             "lon_b": ("lon_b", _bounds(tlon)),
                             "lat_b": ("lat_b", _bounds(tlat, latitude=True))})
    digest = hashlib.sha256()
    for values in (lon, lat, tlon, tlat):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    path = Path(cache_dir) / ("glo12_025_conservative_" + digest.hexdigest()[:20] + ".nc")
    return src, dst, path


def generate_weights(source, cache_dir):
    """Run ONLY in the dedicated xESMF/ESMF preprocessing environment."""
    import xesmf as xe
    src, dst, path = grids_and_path(source, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    xe.Regridder(src, dst, "conservative", filename=str(path),
                 reuse_weights=False, periodic=False)
    return path


def _remap_block(block, *, weights, output_shape):
    """Map one time/depth block using a sparse source-to-destination matrix."""
    original_shape = block.shape[:-2]
    flat = block.reshape((-1, weights.shape[1]))
    values = np.where(np.isfinite(flat), flat, 0.0)
    valid = np.isfinite(flat).astype(np.float64)
    numerator = (weights @ values.T).T
    coverage = (weights @ valid.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(coverage >= 0.5, numerator / coverage, np.nan)
    return result.reshape((*original_shape, *output_shape))


def conservative_glo12_to_025(velocities, *, u_name="uo", v_name="vo",
                              cache_dir=None, log=None):
    """Apply precomputed ESMF conservative weights with SciPy, without xESMF.

    Missing velocities are excluded from averages; cells with less than 50%
    valid source area are masked. Weights must be generated separately first.
    """
    from scipy import sparse
    if not cache_dir:
        raise ValueError("A persistent remapping weight cache directory is required")
    u, v = velocities[u_name], velocities[v_name]
    if not u.latitude.equals(v.latitude) or not u.longitude.equals(v.longitude):
        raise ValueError("GLO12 U and V coordinates must match")
    src, dst, path = grids_and_path(u, cache_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing conservative weights: {path}. Generate them with "
            "scripts/generate_glo12_025_weights.py in the glo12_regrid environment.")
    with xr.open_dataset(path) as ds:
        row = ds["row"].values.astype(np.int64) - 1
        col = ds["col"].values.astype(np.int64) - 1
        data = ds["S"].values.astype(np.float64)
    shape = (len(dst.lat) * len(dst.lon), len(src.lat) * len(src.lon))
    weights = sparse.csr_matrix((data, (row, col)), shape=shape)
    source = velocities.assign_coords(
        longitude=((velocities.longitude + 180) % 360) - 180
    ).sortby("longitude").sortby("latitude")
    output = xr.Dataset()
    for name in (u_name, v_name):
        field = source[name]
        result = xr.apply_ufunc(
            _remap_block, field,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[["latitude_025", "longitude_025"]],
            kwargs={"weights": weights, "output_shape": (len(dst.lat), len(dst.lon))},
            dask="parallelized", output_dtypes=[np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {"latitude_025": len(dst.lat),
                                 "longitude_025": len(dst.lon)},
                "allow_rechunk": True,
            },
        ).rename({"latitude_025": "latitude", "longitude_025": "longitude"})
        result = result.assign_coords(latitude=dst.lat.values, longitude=dst.lon.values)
        output[name] = result
        output[name].attrs = dict(field.attrs)
    if log:
        log.info("GLO12_025 SciPy sparse conservative remapping %s -> %s using %s",
                 (len(src.lat), len(src.lon)), (len(dst.lat), len(dst.lon)), path)
    return output
