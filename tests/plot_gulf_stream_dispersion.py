#!/usr/bin/env python3
"""GLOENS 20240229 Gulf Stream extension: all-particle maps and GIF.

Reads existing Parcels Parquet and observation CSV. No simulation/cache writes.
Default region: 80W–35W, 25N–50N. GIF needs Pillow.
"""
import argparse
import os
import json
from pathlib import Path

# Use the existing HPC Cartopy/Natural Earth data; do not download at runtime.
CARTOPY_DATA = "/home/cregnier/code/lib_pyt/lib_pyt/statics/cartopy_data"
os.environ.setdefault("CARTOPY_DATA_DIR", CARTOPY_DATA)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
import cartopy
cartopy.config["data_dir"] = CARTOPY_DATA
cartopy.config["pre_existing_data_dir"] = CARTOPY_DATA
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", default="20240229")
    p.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    p.add_argument("--bounds", type=float, nargs=4, default=[-80, -35, 25, 50],
                   metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--frame-hours", type=int, default=3)
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
    ids = sorted(obs.id.unique(), key=lambda x: (0, int(x)) if x.lstrip("-").isdigit() else (1, x))
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

    # Diagnose missing model tracks before clipping positions to map bounds.
    coverage = []
    for identifier in sorted(chosen):
        m = model[model.id == identifier].sort_values("time")
        o = obs[obs.id == identifier].sort_values("time")
        m = m[np.isfinite(m.lon) & np.isfinite(m.lat)]
        o = o[np.isfinite(o.lon) & np.isfinite(o.lat)]
        coverage.append(dict(id=identifier, model_samples=len(m), observed_samples=len(o),
                             model_last_time=m.time.max() if len(m) else pd.NaT,
                             observed_last_time=o.time.max() if len(o) else pd.NaT,
                             model_last_lon=m.lon.iloc[-1] if len(m) else np.nan,
                             model_last_lat=m.lat.iloc[-1] if len(m) else np.nan))
    pd.DataFrame(coverage).to_csv(out / "particle_coverage.csv", index=False)
    print("Particle coverage:", out / "particle_coverage.csv")
    model, obs = clipped(model), clipped(obs)
    print(f"Selected {len(chosen)} particles seeded inside {args.bounds}")
    print(f"Model timestamps: {model.time.min()} to {model.time.max()}")
    def setup_map(ax):
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#deded8", edgecolor="black",
                       linewidth=.6, zorder=5)
        ax.coastlines(resolution="110m", linewidth=.6, zorder=6)
        grid = ax.gridlines(draw_labels={"bottom": "x", "left": "y"},
                            linewidth=.5, alpha=.35)
        grid.top_labels = False
        grid.right_labels = False

    legend = [
        Line2D([0], [0], color="tab:blue", lw=1.5, label="Observed track"),
        Line2D([0], [0], color="tab:orange", lw=1.5, label="Model track"),
        Line2D([0], [0], marker="o", color="tab:blue", linestyle="None",
               markersize=5, label="Observed position"),
        Line2D([0], [0], marker="o", color="tab:orange", linestyle="None",
               markersize=5, label="Model position"),
    ]
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    setup_map(ax)
    for df, color in [(obs, "tab:blue"), (model, "tab:orange")]:
        for _, g in df.groupby("id", sort=False):
            g = g.sort_values("time")
            ax.plot(g.lon.to_numpy(), g.lat.to_numpy(), color=color,
                    alpha=.35, lw=.75, transform=ccrs.PlateCarree(), zorder=2)
    ax.scatter(seeds.lon, seeds.lat, s=6, c="black", alpha=.6,
               transform=ccrs.PlateCarree(), zorder=4)
    ax.legend(handles=legend[:2], loc="lower left")
    ax.set_title(f"Gulf Stream extension — all {len(chosen)} drifters — {args.cycle} +10 days")
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
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    setup_map(ax)
    trails = []
    dots = []
    for color in ["tab:blue", "tab:orange"]:
        lines = [ax.plot([], [], color=color, lw=.8, alpha=.38,
                         transform=ccrs.PlateCarree(), zorder=2)[0]
                 for _ in sorted(chosen)]
        trails.append(lines)
        dots.append(ax.scatter([], [], s=12, color=color, alpha=.9,
                               transform=ccrs.PlateCarree(), zorder=4))
    ax.legend(handles=legend, loc="lower left", ncol=2)
    title = ax.set_title("")
    fig.tight_layout()
    frames = list(range(0, len(clock), args.frame_hours))
    if frames[-1] != len(clock) - 1:
        frames.append(len(clock) - 1)

    def update(i):
        idx = frames[i]
        # Full trajectory history from launch, not just the last 48 hours.
        # NaNs naturally break tracks where observations are missing.
        for panel, (xs, ys) in enumerate([(ox, oy), (mx, my)]):
            coords = np.column_stack([xs[:, idx], ys[:, idx]])
            dots[panel].set_offsets(coords)
            for row, line in enumerate(trails[panel]):
                line.set_data(xs[row, :idx + 1], ys[row, :idx + 1])
        title.set_text(f"Gulf Stream extension — {clock[idx]:%Y-%m-%d %H:%M} UTC"
                       f" — +{idx / 24:.2f} days")
        return dots + [line for group in trails for line in group] + [title]
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    anim.save(out / "dispersion_observed_vs_model.gif", writer=PillowWriter(fps=args.fps), dpi=95)
    plt.close(fig)
    print("Wrote:", out / "all_tracks_observed_vs_model.png")
    print("Wrote:", out / "dispersion_observed_vs_model.gif")


if __name__ == "__main__":
    main()
