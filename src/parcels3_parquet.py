"""Convert legacy Parcels 3 Zarr trajectories to the Parcels 4 Parquet schema.

Parquet particle_id is the zero-based observation seeding index, not a
Parcels-internal ID. Times are stored as absolute datetime64[ns].
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import xarray as xr


def convert_parcels3_zarr(zarr_path, parquet_path, observed_ids, log=None):
    """Write particle_id,t,x,y, preserving the original observation seed order."""
    destination = Path(parquet_path)
    if destination.exists():
        # Do not silently reuse a truncated or incompatible output.
        import pyarrow.parquet as pq
        pq.read_metadata(destination)
        return str(destination)

    with xr.open_zarr(zarr_path) as ds:
        for name in ("lon", "lat", "time"):
            if name not in ds:
                raise ValueError(f"Parcels 3 output lacks {name}: {zarr_path}")
        lon = np.asarray(ds["lon"].values)
        lat = np.asarray(ds["lat"].values)
        times = np.asarray(ds["time"].values)
        if lon.ndim != 2 or lat.shape != lon.shape or times.shape != lon.shape:
            raise ValueError("Expected matching (trajectory, observation) lon/lat/time arrays")
        expected = np.asarray(observed_ids)
        if lon.shape[0] != len(expected):
            raise ValueError(
                f"Zarr has {lon.shape[0]} trajectories, but {len(expected)} seeds; "
                "cannot safely infer original seed indices"
            )
        if len(np.unique(expected)) != len(expected):
            raise ValueError("Duplicate observation seed IDs")
        if "label" in ds:
            labels = np.asarray(ds["label"].values)
            # Parcels 3 labels are usually stored as character arrays.
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = labels[:, 0]
            def normalized(v):
                if isinstance(v, np.ndarray):
                    v = b"".join(v.tolist()) if v.dtype.kind == "S" else "".join(v.tolist())
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                return str(v).strip().strip("\\x00")
            if labels.ndim == 1:
                actual = [normalized(v) for v in labels]
                wanted = [normalized(v) for v in expected]
                if actual != wanted:
                    raise ValueError("Zarr particle labels differ from observation seed order")

        clock = pd.to_datetime(times.reshape(-1), errors="coerce")
        if clock.isna().all():
            raise ValueError("Parcels 3 trajectory times cannot be decoded")
        frame = pd.DataFrame({
            "particle_id": np.repeat(np.arange(lon.shape[0], dtype=np.int64), lon.shape[1]),
            "t": clock,
            "x": lon.reshape(-1),
            "y": lat.reshape(-1),
        })
        frame = frame.dropna(subset=["t"])
        frame = frame.loc[np.isfinite(frame["x"]) & np.isfinite(frame["y"])]
        if frame.empty:
            raise ValueError("No valid Parcels 3 trajectory samples")
        if frame.duplicated(["particle_id", "t"]).any():
            raise ValueError("Duplicate particle/time samples in Parcels 3 output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    if log:
        log.info("Converted Parcels 3 Zarr to Parquet: %s (%d records)", destination, len(frame))
    return str(destination)
