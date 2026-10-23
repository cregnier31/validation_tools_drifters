#!/usr/bin/env python3
"""One calendar-time Gulf Stream animation combining all weekly GLOENS cycles.

Reads existing Parcels Parquet and observed CSV files only. No advection, no
deletion. Each cycle contributes tracks from its initialization until drift_days
later; tracks accumulate on a common map as calendar time advances.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

CYCLES = [
    "20240104", "20240111", "20240118", "20240125",
    "20240201", "20240208", "20240215", "20240222", "20240229",
    "20240307", "20240314", "20240321", "20240328",
]
REGION = (-80, -40, 25, 50)


def load_cycle(root, cycle, days, max_tracks):
    dirs = sorted(root.glob(
        f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift{days}_dt5_{cycle}-{cycle}*/ZONE_*/{cycle}"
    ))
    if len(dirs) != 1:
        raise ValueError(f"{cycle}: expected one trajectory directory; found {len(dirs)}")
    folder = dirs[0]
    parquet = sorted(folder.glob("Trajectories_GLOENS_*at_SVPloc*.parquet"))
    csvs = sorted((folder.parent / "DRIFTERS").glob("Drifters_*.csv"))
    if len(parquet) != 1 or len(csvs) != 1:
        raise ValueError(f"{cycle}: expected one Parquet and observation CSV")
    obs = pd.read_csv(csvs[0], usecols=["id", "time", "lon", "lat"])
    obs["id"] = obs["id"].astype(str)
    obs["time"] = pd.to_datetime(obs["time"], utc=True).dt.tz_localize(None)
    obs.loc[(obs.lon == 0) & (obs.lat == 0), ["lon", "lat"]] = np.nan
    ids = sorted(obs.id.unique(), key=lambda x: (0, int(x)) if x.lstrip("-").isdigit() else (1, x))
    import json
    mapping = np.arange(len(ids))
    mapfile = parquet[0].with_suffix(".seed_indices.json")
    if mapfile.exists():
        meta = json.loads(mapfile.read_text())
        if meta["original_seed_count"] != len(ids):
            raise ValueError(f"{cycle}: seed map does not match observed IDs")
        mapping = np.asarray(meta["original_indices"], dtype=int)
    model = pd.read_parquet(parquet[0], columns=["particle_id", "t", "x", "y"])
    model["particle_id"] = model.particle_id.astype(int)
    if model.particle_id.min() < 0 or model.particle_id.max() >= len(mapping):
        raise ValueError(f"{cycle}: particle ID outside seed map")
    model["id"] = np.asarray(ids, dtype=object)[mapping[model.particle_id.to_numpy()]]
    start = pd.Timestamp(cycle) + pd.Timedelta(hours=12)
    if pd.api.types.is_numeric_dtype(model.t):
        model["time"] = start + pd.to_timedelta(model.t, unit="s")
    else:
        model["time"] = pd.to_datetime(model.t, utc=True).dt.tz_localize(None)
    end = start + pd.Timedelta(days=days)
    model = model.loc[model.time.between(start, end), ["id", "time", "x", "y"]]
    if max_tracks:
        chosen = set(sorted(model.id.unique())[:max_tracks])
        model = model[model.id.isin(chosen)]
        obs = obs[obs.id.isin(chosen)]
    data = model.merge(obs, on=["id", "time"], how="left", validate="one_to_one")
    data = data.rename(columns={"x": "model_lon", "y": "model_lat",
                                "lon": "obs_lon", "lat": "obs_lat"})
    data["cycle"] = cycle
    return data


def make_segments(df, prefix):
    segments, times = [], []
    for _, track in df.groupby(["cycle", "id"], sort=False):
        track = track.sort_values("time")
        xy = track[[prefix + "_lon", prefix + "_lat"]].to_numpy(dtype=float)
        ts = track.time.to_numpy(dtype="datetime64[ns]")
        good = np.isfinite(xy).all(axis=1)
        good &= (xy[:, 0] >= REGION[0]) & (xy[:, 0] <= REGION[1])
        good &= (xy[:, 1] >= REGION[2]) & (xy[:, 1] <= REGION[3])
        delta_h = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600
        adjacent = (good[:-1] & good[1:] & (delta_h > 0) &
                    (delta_h <= 1.01) & (np.abs(np.diff(xy[:, 0])) < 180))
        idx = np.flatnonzero(adjacent)
        segments.extend(xy[j:j + 2] for j in idx)
        times.extend(ts[idx + 1])
    if not segments:
        return np.empty((0, 2, 2)), np.empty(0, dtype="datetime64[ns]")
    return np.asarray(segments), np.asarray(times, dtype="datetime64[ns]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    ap.add_argument("--drift-days", type=int, default=15)
    ap.add_argument("--frame-hours", type=int, default=12)
    ap.add_argument("--max-tracks-per-cycle", type=int, default=120,
                    help="Limit displayed tracks per cycle for manageable rendering; 0=all")
    ap.add_argument("--output", default="outputs/gulf_stream_all_cycles_2024.gif")
    ap.add_argument("--cycles", nargs="+", default=CYCLES)
    args = ap.parse_args()
    if args.drift_days < 1 or args.frame_hours < 1 or args.max_tracks_per_cycle < 0:
        ap.error("drift-days and frame-hours must be positive; max-tracks must be nonnegative")

    parts = []
    for cycle in args.cycles:
        print(f"Reading {cycle}", flush=True)
        parts.append(load_cycle(Path(args.root), cycle, args.drift_days, args.max_tracks_per_cycle))
    df = pd.concat(parts, ignore_index=True)
    for prefix in ("model", "obs"):
        df[prefix + "_lon"] = ((df[prefix + "_lon"] + 180) % 360) - 180

    cartopy_dir = Path("/home/cregnier/code/lib_pyt/lib_pyt/statics/cartopy_data")
    if not cartopy_dir.is_dir():
        raise SystemExit(f"Cartopy local data directory missing: {cartopy_dir}")
    import cartopy
    cartopy.config["data_dir"] = str(cartopy_dir)
    cartopy.config["pre_existing_data_dir"] = str(cartopy_dir)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    model_segs, model_times = make_segments(df, "model")
    obs_segs, obs_times = make_segments(df, "obs")
    first = pd.Timestamp(min(args.cycles)) + pd.Timedelta(hours=12)
    last = pd.Timestamp(max(args.cycles)) + pd.Timedelta(hours=12, days=args.drift_days)
    frames = pd.date_range(first, last, freq=f"{args.frame_hours}h")
    if frames[-1] != last:
        frames = frames.append(pd.DatetimeIndex([last]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(REGION, crs=ccrs.PlateCarree())
    ax.set_facecolor("#dbeef7")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#dedbd1", edgecolor="none")
    ax.coastlines(resolution="50m", linewidth=.65)
    grid = ax.gridlines(draw_labels=True, linewidth=.4, alpha=.4)
    grid.top_labels = False
    grid.right_labels = False
    ax.set_title("GLOENS vs observed drifters — Gulf Stream, Jan–Apr 2024")
    ml = LineCollection([], colors="tab:blue", linewidths=.6, alpha=.4,
                        transform=ccrs.PlateCarree(), label="GLOENS")
    ol = LineCollection([], colors="tab:orange", linewidths=.6, alpha=.5,
                        transform=ccrs.PlateCarree(), label="Observed")
    ax.add_collection(ml)
    ax.add_collection(ol)
    ax.legend(loc="lower right")
    stamp = ax.text(.02, .97, "", transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))

    def update(frame):
        now = np.datetime64(frame.to_datetime64(), "ns")
        ml.set_segments(model_segs[model_times <= now])
        ol.set_segments(obs_segs[obs_times <= now])
        active = sum(pd.Timestamp(c) + pd.Timedelta(hours=12) <= frame for c in args.cycles)
        stamp.set_text(f"{frame:%Y-%m-%d %H:%M} UTC  |  cycles started: {active}/{len(args.cycles)}")
        return ml, ol, stamp

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=frames, interval=150, blit=False)
    animation.save(output, writer=PillowWriter(fps=6), dpi=100)
    plt.close(fig)
    print(f"Saved combined animation: {output} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
