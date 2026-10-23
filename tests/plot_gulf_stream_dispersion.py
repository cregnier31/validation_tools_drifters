#!/usr/bin/env python3
"""GLOENS 20240229 Gulf Stream extension: all-particle maps and GIF.

Reads existing Parcels Parquet and observation CSV. No simulation/cache writes.
Default region: 80W–35W, 25N–50N. GIF needs Pillow.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", default="20240229")
    p.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    p.add_argument("--bounds", type=float, nargs=4, default=[-80, -35, 25, 50],
                   metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--frame-hours", type=int, default=3)
    p.add_argument("--tail-hours", type=int, default=48)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    west, east, south, north = args.bounds
    if not west < east or not south < north or args.frame_hours < 1:
        p.error("Invalid bounds or frame-hours")
    dirs = sorted(Path(args.root).glob(
        f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift10_dt5_{args.cycle}-{args.cycle}/ZONE_*/{args.cycle}"))
    if len(dirs) != 1:
        raise SystemExit(f"Expected one cycle directory, found {dirs}")
    directory = dirs[0]
    pq = list(directory.glob("Trajectories_GLOENS_*at_SVPloc*.parquet"))
    observations = list((directory.parent / "DRIFTERS").glob("Drifters_*.csv"))
    if len(pq) != 1 or len(observations) != 1:
        raise SystemExit(f"Need one trajectory Parquet and one observed drifter CSV; got {pq}, {observations}")
    out = Path(args.output) if args.output else directory / "gulf_stream_extension"
    out.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(observations[0], usecols=["id", "time", "lon", "lat"])
    obs["id"] = obs["id"].astype(str)
    obs["time"] = pd.to_datetime(obs["time"])
    obs.loc[(obs.lon == 0) & (obs.lat == 0), ["lon", "lat"]] = np.nan
    ids = np.sort(obs.id.unique())  # matches Retrieve_drifters_traj's sorted seed order
    model = pd.read_parquet(pq[0], columns=["particle_id", "t", "x", "y"])
    seed_file = pq[0].with_suffix(".seed_indices.json")
    mapping = np.arange(len(ids))
    if seed_file.exists():
        seed = json.loads(seed_file.read_text())
        if seed["original_seed_count"] != len(ids):
            raise SystemExit("Seed map does not match observations")
        mapping = np.asarray(seed["original_indices"], dtype=int)
    model["particle_id"] = model.particle_id.astype(int)
    if model.particle_id.min() < 0 or model.particle_id.max() >= len(mapping):
        raise SystemExit("Invalid particle_id")
    model["id"] = ids[mapping[model.particle_id.to_numpy()]]
    start = pd.Timestamp(args.cycle) + pd.Timedelta(hours=12)
    model["time"] = (start + pd.to_timedelta(model.t, unit="s")
                     if pd.api.types.is_numeric_dtype(model.t)
                     else pd.to_datetime(model.t))
    model = model.rename(columns={"x": "lon", "y": "lat"})
    # Select seeds within the geographical window at initialization.
    seeds = model.loc[(model.time - start).abs() <= pd.Timedelta(seconds=1),
                      ["id", "lon", "lat"]]
    seeds = seeds[(seeds.lon.between(west, east)) & (seeds.lat.between(south, north))]
    chosen = set(seeds.id)
    if not chosen:
        raise SystemExit("No model particles seeded inside the chosen Gulf Stream region")
    model = model[model.id.isin(chosen)].copy()
    obs = obs[obs.id.isin(chosen)].copy()
    model = model[model.time.between(start, start + pd.Timedelta(days=10))]
    obs = obs[obs.time.between(start, start + pd.Timedelta(days=10))]
    # Only draw trajectories within the displayed region; never wrap/connect
    # geographically distant points across the plotting boundary.
    def clipped(df):
        d = df.copy()
        outside = ~d.lon.between(west, east) | ~d.lat.between(south, north)
        d.loc[outside, ["lon", "lat"]] = np.nan
        return d

    model, obs = clipped(model), clipped(obs)
    print(f"Selected {len(chosen)} particles seeded inside {args.bounds}")
    print(f"Model timestamps: {model.time.min()} to {model.time.max()}")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, df, name, color in zip(
            axes, [obs, model], ["Observed drifters", "GLOENS Parcels"], ["tab:blue", "tab:orange"]):
        for _, g in df.groupby("id", sort=False):
            g = g.sort_values("time")
            ax.plot(g.lon.to_numpy(), g.lat.to_numpy(), color=color, alpha=.27, lw=.7)
        ax.scatter(seeds.lon, seeds.lat, s=4, c="black", alpha=.45, label="Seed locations")
        ax.set(xlim=(west, east), ylim=(south, north), xlabel="Longitude (°E)",
               ylabel="Latitude (°N)", title=name)
        ax.grid(alpha=.25)
    fig.suptitle(f"Gulf Stream extension — all {len(chosen)} drifters — {args.cycle} +10 days")
    fig.tight_layout()
    fig.savefig(out / "all_tracks_observed_vs_model.png", dpi=180)
    plt.close(fig)

    # Common hourly clock: model positions at exact timestamps; no interpolation.
    clock = pd.date_range(start, start + pd.Timedelta(days=10), freq="h")
    def matrix(df):
        d = df[df.time.isin(clock)]
        if d.duplicated(["id", "time"]).any():
            raise SystemExit("Duplicate particle/time positions")
        lo = d.pivot(index="id", columns="time", values="lon").reindex(index=sorted(chosen), columns=clock)
        la = d.pivot(index="id", columns="time", values="lat").reindex(index=sorted(chosen), columns=clock)
        return lo.to_numpy(dtype=float), la.to_numpy(dtype=float)
    ox, oy = matrix(obs)
    mx, my = matrix(model)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    scatters = []
    trails = []
    for ax, label, color in zip(axes, ["Observed", "GLOENS model"], ["tab:blue", "tab:orange"]):
        ax.set(xlim=(west, east), ylim=(south, north), xlabel="Longitude (°E)",
               ylabel="Latitude (°N)", title=label)
        ax.grid(alpha=.2)
        lines = [ax.plot([], [], lw=.7, alpha=.25, color=color)[0] for _ in chosen]
        trails.append(lines)
        scatters.append(ax.scatter([], [], s=10, color=color, alpha=.85))
    title = fig.suptitle("")
    fig.tight_layout()
    frames = list(range(0, len(clock), args.frame_hours))
    if frames[-1] != len(clock) - 1:
        frames.append(len(clock) - 1)
    def update(i):
        idx = frames[i]
        lo_idx = max(0, idx - args.tail_hours)
        for panel, (xs, ys) in enumerate([(ox, oy), (mx, my)]):
            coords = np.column_stack([xs[:, idx], ys[:, idx]])
            scatters[panel].set_offsets(coords)
            for row, line in enumerate(trails[panel]):
                line.set_data(xs[row, lo_idx:idx+1], ys[row, lo_idx:idx+1])
        title.set_text(f"Gulf Stream extension — {clock[idx]:%Y-%m-%d %H:%M} UTC — +{idx/24:.2f} days")
        return scatters + [line for group in trails for line in group] + [title]
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    anim.save(out / "dispersion_observed_vs_model.gif", writer=PillowWriter(fps=args.fps), dpi=95)
    plt.close(fig)
    print("Wrote:", out / "all_tracks_observed_vs_model.png")
    print("Wrote:", out / "dispersion_observed_vs_model.gif")


if __name__ == "__main__":
    main()
