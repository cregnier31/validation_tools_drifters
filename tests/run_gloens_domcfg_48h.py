#!/usr/bin/env python3
"""Read-only-input 48h four-seed trajectory test with domcfg F geometry."""
import argparse
import json
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import parcels

SEEDS = {
    "3201769": (-72.087997, 30.035000),
    "4102665": (-66.238998, 36.525002),
    "6203615": (-78.365997, 30.718000),
    "6203642": (-77.361000, 32.047001),
}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle", default="20240229")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--geometry-source",
                    default="/scratch/cregnier/static/iORCA025_domcfg_v3.1.nc")
    ap.add_argument("--output",
                    default="/scratch/cregnier/static/gloens_domcfg_48h.parquet")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--dt-minutes", type=int, default=30)
    args = ap.parse_args()
    if args.hours <= 0 or args.dt_minutes <= 0:
        ap.error("hours and dt-minutes must be positive")
    if args.hours * 60 % args.dt_minutes:
        ap.error("duration must be divisible by timestep")
    if Path(args.output).exists():
        ap.error(f"Output already exists; choose a new path: {args.output}")
    cache = Path(args.cache_dir)
    fields = {}
    for comp, var in (("U", "uo"), ("V", "vo")):
        matches = sorted(cache.glob(f"FCST_R{args.cycle}_*_{var}_ensmean_28days.json"))
        if len(matches) != 1:
            ap.error(f"Expected one {var} metadata file, got {len(matches)}")
        meta = json.loads(matches[0].read_text())
        array = np.load(matches[0].with_suffix(".npy"), mmap_mode="r")
        if tuple(meta["shape"]) != array.shape:
            raise ValueError(f"{var}: cache shape mismatch")
        fields[comp] = xr.DataArray(array[:, 0, :, :],
             dims=("time", "y", "x"),
             coords={"time": pd.to_datetime(meta["times"])})
    ds = xr.Dataset(fields)
    times = ds.time.values.astype("datetime64[ns]")
    expected = np.datetime64(pd.Timestamp(args.cycle).normalize()
                             + pd.Timedelta(hours=12), "ns")
    if times[0] != expected:
        raise ValueError(f"Forecast begins {times[0]}, expected {expected}")
    if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Forecast time must be strictly increasing")
    needed = timedelta(hours=args.hours, minutes=args.dt_minutes)
    available = pd.Timestamp(times[-1]) - pd.Timestamp(times[0])
    if available < needed:
        raise ValueError(f"Insufficient velocity coverage: {available} < {needed}")
    with xr.open_dataset(args.geometry_source, engine="netcdf4",
                         decode_cf=False) as nc:
        coord = {}
        for name in ("glamf", "gphif"):
            da = nc[name]
            for dim in da.dims:
                if dim not in ("y", "x"):
                    da = da.isel({dim: 0})
            coord[name] = (("y", "x"), np.array(da.values, copy=True))
    fcoords = xr.Dataset(coord)
    zeros = ((fcoords.glamf.values == 0) & (fcoords.gphif.values == 0))
    print(f"F paired-zero={int(zeros.sum())}", flush=True)
    if zeros.any() or not np.isfinite(fcoords.glamf.values).all() or not np.isfinite(fcoords.gphif.values).all():
        raise ValueError("Geometry has zero/invalid coordinates; refusing run")
    sgrid = parcels.convert.nemo_to_sgrid(
        fields={"U": ds.U, "V": ds.V}, coords=fcoords)
    fieldset = parcels.FieldSet.from_sgrid_conventions(sgrid)
    ids = list(SEEDS)
    lon = np.array([SEEDS[k][0] for k in ids], dtype=float)
    lat = np.array([SEEDS[k][1] for k in ids], dtype=float)
    pset = parcels.ParticleSet(
        fieldset=fieldset, pclass=parcels.Particle,
        x=lon, y=lat,
        t=pd.Timestamp(times[0]).to_pydatetime())
    from src.gloens_periodic_kernel import AdvectionRK4_periodicBC
    print(f"Parcels={parcels.__version__}, seeds={ids}, "
          f"start={times[0]}, hours={args.hours}, dt_minutes={args.dt_minutes}",
          flush=True)
    pfile = parcels.ParticleFile(args.output, outputdt=timedelta(hours=1))
    pset.execute(AdvectionRK4_periodicBC,
                 runtime=timedelta(hours=args.hours),
                 dt=timedelta(minutes=args.dt_minutes),
                 output_file=pfile)
    print(f"Final particle count={len(pset.x)}", flush=True)
    for i, ident in enumerate(ids):
        if i >= len(pset.x):
            print(f"{ident}: missing from final particle array", flush=True)
            continue
        lo, la = float(pset.x[i]), float(pset.y[i])
        dlon = np.deg2rad(((lo-lon[i]+180)%360)-180)
        dlat = np.deg2rad(la-lat[i])
        a = np.sin(dlat/2)**2 + np.cos(np.deg2rad(lat[i]))*np.cos(np.deg2rad(la))*np.sin(dlon/2)**2
        km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))
        print(f"{ident}: initial=({lon[i]:.6f},{lat[i]:.6f}), "
              f"final=({lo:.6f},{la:.6f}), displacement_km={km:.3f}, "
              f"state={np.asarray(pset.state).ravel()[i]}", flush=True)
    print(f"Trajectory output: {args.output}", flush=True)

if __name__ == "__main__":
    main()
