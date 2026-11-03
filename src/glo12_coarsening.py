"""Conservative spatial filtering of GLO12 velocities before Parcels advection.

Requires xesmf and an ESMF backend in the Parcels environment.
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


def conservative_glo12_to_025(velocities, *, u_name="uo", v_name="vo",
                              cache_dir=None, log=None):
    """Area-overlap average onto a regular quarter-degree grid.

    Regrids only u/v (not all 3-D forecast variables), preserving forecast
    times and vertical levels. NaN source cells are excluded; output requires
    at least 50% valid contributing area. Uses a reusable sparse ESMF weight file.
    """
    try:
        import xesmf as xe
    except ImportError as exc:
        raise RuntimeError(
            "GLO12_025 requires xesmf with ESMF/esmpy installed in the "
            "Parcels environment (for example conda-forge::xesmf and esmpy)"
        ) from exc

    u, v = velocities[u_name], velocities[v_name]
    if not {"latitude", "longitude"}.issubset(u.dims):
        raise ValueError("GLO12_025 requires regular latitude/longitude velocity dimensions")
    if not u.latitude.equals(v.latitude) or not u.longitude.equals(v.longitude):
        raise ValueError("GLO12 U and V must have matching horizontal coordinates")

    # Sort and normalize to a common -180..180 domain before computing bounds.
    # Avoid interpolating at this stage: the transformation only reorders cells.
    source = xr.Dataset({u_name: u, v_name: v})
    source = source.assign_coords(
        longitude=((source.longitude + 180) % 360) - 180
    ).sortby("longitude").sortby("latitude")
    lon = source.longitude.values
    lat = source.latitude.values
    if np.unique(lon).size != lon.size:
        raise ValueError("Duplicate source longitudes after normalization")
    lon_bounds = _bounds(lon)
    lat_bounds = _bounds(lat, latitude=True)

    # Fixed quarter-degree cell centers, restricted to native grid coverage.
    # Keeping cells wholly inside the source domain avoids extrapolation.
    target_lon_all = np.arange(-179.875, 180., 0.25)
    target_lat_all = np.arange(-89.875, 90., 0.25)
    target_lon = target_lon_all[
        (target_lon_all - .125 >= lon_bounds[0] - 1e-8)
        & (target_lon_all + .125 <= lon_bounds[-1] + 1e-8)
    ]
    target_lat = target_lat_all[
        (target_lat_all - .125 >= lat_bounds[0] - 1e-8)
        & (target_lat_all + .125 <= lat_bounds[-1] + 1e-8)
    ]
    if len(target_lon) < 2 or len(target_lat) < 2:
        raise ValueError("Native GLO12 extent cannot support a quarter-degree target grid")

    src_grid = xr.Dataset(coords={
        "lon": ("lon", lon), "lat": ("lat", lat),
        "lon_b": ("lon_b", lon_bounds), "lat_b": ("lat_b", lat_bounds),
    })
    dst_grid = xr.Dataset(coords={
        "lon": ("lon", target_lon), "lat": ("lat", target_lat),
        "lon_b": ("lon_b", _bounds(target_lon)),
        "lat_b": ("lat_b", _bounds(target_lat, latitude=True)),
    })
    # xESMF expects lon/lat dimension names for rectilinear input.
    digest = hashlib.sha256()
    for values in (lon, lat, target_lon, target_lat):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    weights = None
    if cache_dir:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / f"glo12_025_conservative_normed_{digest.hexdigest()[:20]}.nc"
    regridder = xe.Regridder(
        src_grid, dst_grid, "conservative_normed",
        filename=str(weights) if weights else None,
        reuse_weights=bool(weights and weights.is_file()),
        periodic=False,
    )
    result = xr.Dataset()
    for name in (u_name, v_name):
        # A 50% valid-ocean coverage threshold prevents poorly represented
        # coastal cells from receiving a velocity based on tiny ocean fragments.
        field = source[name].rename({"latitude": "lat", "longitude": "lon"})
        mapped = regridder(field, skipna=True, na_thres=0.5)
        result[name] = mapped.rename({"lat": "latitude", "lon": "longitude"})
        result[name].attrs = dict(source[name].attrs)
    if log:
        log.info("GLO12_025 conservative remap: %s -> %s; weights=%s; "
                 "valid-area threshold=50%%", (len(lat), len(lon)),
                 (len(target_lat), len(target_lon)), weights)
    return result
