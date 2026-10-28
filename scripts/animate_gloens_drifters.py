#!/usr/bin/env python3
"""Animate matched observed drifters and GLOENS Parcels tracks.

Example:
 python scripts/animate_gloens_drifters.py --model /path/Trajectories_GLOENS_....parquet \
   --observations /path/Drifters_....csv --output comparison.mp4
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def load_tracks(model_path, observations_path, step_hours):
    model = pd.read_parquet(model_path, columns=["particle_id", "t", "x", "y"])
    obs = pd.read_csv(observations_path, usecols=["id", "time", "lon", "lat"],
                      parse_dates=["time"])
    # Seeding uses drifter IDs sorted ascending, not the numeric drifter ID.
    ids = np.sort(obs["id"].unique())
    obs["particle_id"] = pd.Categorical(obs["id"], categories=ids, ordered=True).codes
    obs = obs.drop_duplicates(["particle_id", "time"])
    start = obs["time"].min()
    obs["hour"] = (obs["time"] - start).dt.total_seconds() / 3600
    model["hour"] = model["t"] / 3600
    for frame in (model, obs):
        frame["hour"] = np.rint(frame["hour"]).astype(int)
    model = model[model["hour"] % step_hours == 0]
    obs = obs[obs["hour"] % step_hours == 0]
    model = model.drop_duplicates(["particle_id", "hour"])
    obs = obs.drop_duplicates(["particle_id", "hour"])
    matched = model.merge(obs[["particle_id", "hour", "lon", "lat"]],
                          on=["particle_id", "hour"], how="inner")
    matched = matched.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["x", "y", "lon", "lat"])
    matched = matched[(matched["y"].abs() <= 90) & (matched["lat"].abs() <= 90)]
    if matched.empty:
        raise ValueError("No matched model/observation samples; check seeding IDs and times")
    return matched, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Parcels 4 trajectory Parquet")
    parser.add_argument("--observations", type=Path, required=True, help="Seeded drifter CSV")
    parser.add_argument("--output", type=Path, required=True, help="MP4 or GIF output")
    parser.add_argument("--step-hours", type=int, default=6, help="Hours per animation frame")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-particles", type=int, default=300,
                        help="Stable subset for readable display (0 means all)")
    args = parser.parse_args()
    if args.step_hours < 1 or args.fps < 1:
        parser.error("--step-hours and --fps must be positive")
    tracks, start = load_tracks(args.model, args.observations, args.step_hours)
    if args.max_particles > 0:
        keep = np.sort(tracks.particle_id.unique())[:args.max_particles]
        tracks = tracks[tracks.particle_id.isin(keep)]
    times = sorted(tracks.hour.unique())
    # Longitude unwrapping relative to the observed location avoids long dateline links.
    tracks["model_x"] = tracks["lon"] + (tracks["x"] - tracks["lon"] + 180) % 360 - 180
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.set_facecolor("#edf5fa")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.grid(alpha=0.2)
    obs_sc = ax.scatter([], [], s=11, c="#1877ba", label="Observed drifters", zorder=3)
    mod_sc = ax.scatter([], [], s=11, c="#f28e2b", label="GLOENS / Parcels", zorder=3)
    from matplotlib.collections import LineCollection
    links = LineCollection([], colors="#777777", alpha=0.22, linewidths=0.65)
    ax.add_collection(links)
    ax.legend(loc="lower left")
    all_x = np.r_[tracks.lon.to_numpy(), tracks.model_x.to_numpy()]
    all_y = np.r_[tracks.lat.to_numpy(), tracks.y.to_numpy()]
    lo, hi = np.nanpercentile(all_x, [1, 99])
    la, lb = np.nanpercentile(all_y, [1, 99])
    ax.set_xlim(lo - max(3, (hi-lo)*0.1), hi + max(3, (hi-lo)*0.1))
    ax.set_ylim(max(-90, la-5), min(90, lb+5))
    title = ax.set_title("")
    grouped = {h: df for h, df in tracks.groupby("hour")}
    def draw(k):
        hour = times[k]
        frame = grouped[hour]
        obs_sc.set_offsets(frame[["lon", "lat"]].to_numpy())
        mod_sc.set_offsets(frame[["model_x", "y"]].to_numpy())
        a = frame[["lon", "lat"]].to_numpy()
        b = frame[["model_x", "y"]].to_numpy()
        links.set_segments(np.stack([a, b], axis=1))
        dlat = np.radians(frame.y.to_numpy() - frame.lat.to_numpy())
        dlon = np.radians(frame.model_x.to_numpy() - frame.lon.to_numpy())
        aa = np.sin(dlat/2)**2 + np.cos(np.radians(frame.lat.to_numpy())) * np.cos(
            np.radians(frame.y.to_numpy())) * np.sin(dlon/2)**2
        km = 6371 * 2 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
        date = start + pd.Timedelta(hours=int(hour))
        title.set_text(f"GLOENS vs observed drifters  |  {date:%Y-%m-%d %H:%M} UTC  "
                       f"|  lead {hour/24:.1f} d  |  matched {len(frame)}  "
                       f"|  mean separation {np.mean(km):.1f} km")
        return obs_sc, mod_sc, links, title
    anim = FuncAnimation(fig, draw, frames=len(times), interval=1000/args.fps, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    elif args.output.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=args.fps, bitrate=2200)
    else:
        parser.error("--output must end in .mp4 or .gif")
    anim.save(str(args.output), writer=writer, dpi=130)
    plt.close(fig)
    print(f"Saved {args.output} ({len(times)} frames, {tracks.particle_id.nunique()} drifters)")


if __name__ == "__main__":
    main()
