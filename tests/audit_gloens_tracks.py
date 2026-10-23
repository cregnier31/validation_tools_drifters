#!/usr/bin/env python3
"""Audit one GLOENS cycle from existing Parcels Parquet and observed drifter CSV.

No velocity data, simulation, or cache modification. Requires pandas, pyarrow,
numpy, matplotlib. Example: python tests/audit_gloens_tracks.py --cycle 20240229
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EARTH_KM = 6371.0


def distance_km(lon1, lat1, lon2, lat2):
    dlon = np.deg2rad((np.asarray(lon2) - np.asarray(lon1) + 180) % 360 - 180)
    dlat = np.deg2rad(np.asarray(lat2) - np.asarray(lat1))
    a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle", default="20240229")
    ap.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-tracks", type=int, default=12)
    ap.add_argument("--gulf-stream-animation", action="store_true", help="Create Gulf Stream zoom GIF from existing model/observations")
    ap.add_argument("--frame-hours", type=int, default=6, help="Hours between animation frames")
    args = ap.parse_args()
    base = Path(args.root)
    candidates = sorted(base.glob(f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift10_dt5_{args.cycle}-{args.cycle}*/ZONE_*/{args.cycle}"))
    if len(candidates) != 1:
        raise SystemExit(f"Expected one cycle directory; found {len(candidates)}: {candidates}")
    cycle_dir = candidates[0]
    pq = sorted(cycle_dir.glob("Trajectories_GLOENS_*at_SVPloc*.parquet"))
    obsfiles = list((cycle_dir.parent / "DRIFTERS").glob("Drifters_*.csv"))
    if len(pq) != 1 or len(obsfiles) != 1:
        raise SystemExit(f"Expected one Parquet and one observation CSV: parquet={pq}, observations={obsfiles}")
    out = Path(args.output) if args.output else cycle_dir / "trajectory_audit"
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(pq[0], columns=["particle_id", "t", "x", "y"])
    obs = pd.read_csv(obsfiles[0], usecols=["id", "time", "lon", "lat"])
    obs["time"] = pd.to_datetime(obs["time"])
    obs.loc[(obs["lon"] == 0) & (obs["lat"] == 0), ["lon", "lat"]] = np.nan
    obs["id"] = obs["id"].astype(str)
    ids = sorted(obs["id"].unique(), key=lambda x: int(x) if x.lstrip("-").isdigit() else x)
    # Parcels particle_id is the original sorted seed index unless a seed map exists.
    import json
    mapping_file = pq[0].with_suffix(".seed_indices.json")
    mapping = np.arange(len(ids))
    if mapping_file.exists():
        data = json.loads(mapping_file.read_text())
        if data["original_seed_count"] != len(ids):
            raise SystemExit("Seed map does not match observed IDs")
        mapping = np.asarray(data["original_indices"], dtype=int)
    raw["particle_id"] = raw["particle_id"].astype(int)
    if raw["particle_id"].min() < 0 or raw["particle_id"].max() >= len(mapping):
        raise SystemExit("Parcels particle IDs outside seed mapping")
    raw["id"] = [ids[mapping[i]] for i in raw["particle_id"]]
    start = pd.Timestamp(args.cycle) + pd.Timedelta(hours=12)
    if pd.api.types.is_numeric_dtype(raw["t"]):
        raw["time"] = start + pd.to_timedelta(raw["t"], unit="s")
    else:
        raw["time"] = pd.to_datetime(raw["t"])
    if raw.duplicated(["id", "time"]).any() or obs.duplicated(["id", "time"]).any():
        raise SystemExit("Duplicate particle/time or observation ID/time")
    raw = raw.rename(columns={"x": "model_lon", "y": "model_lat"})
    obs = obs.rename(columns={"lon": "obs_lon", "lat": "obs_lat"})
    merged = raw[["id", "time", "model_lon", "model_lat"]].merge(
        obs, on=["id", "time"], how="left", validate="one_to_one")
    merged["hour"] = (merged["time"] - start).dt.total_seconds() / 3600
    merged["sep_km"] = distance_km(merged.obs_lon, merged.obs_lat, merged.model_lon, merged.model_lat)
    merged.loc[~np.isfinite(merged[["model_lon", "model_lat", "obs_lon", "obs_lat"]]).all(axis=1), "sep_km"] = np.nan
    merged = merged[(merged.hour >= 0) & (merged.hour <= 240)].copy()
    if merged.empty:
        raise SystemExit("No trajectory samples in the requested 10-day window")
    per_hour = merged.groupby("hour").agg(
        model_positions=("model_lon", "count"), valid_pairs=("sep_km", "count"),
        median_sep_km=("sep_km", "median"),
        p90_sep_km=("sep_km", lambda x: x.quantile(.9)),
        mean_sep_km=("sep_km", "mean")).reset_index()
    per_hour.to_csv(out / "hourly_separation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(per_hour.hour / 24, per_hour.median_sep_km, label="Median")
    ax.plot(per_hour.hour / 24, per_hour.p90_sep_km, label="90th percentile")
    ax.plot(per_hour.hour / 24, per_hour.mean_sep_km, label="Mean", alpha=.7)
    ax.set(xlabel="Days after noon launch", ylabel="Model-observation separation (km)",
           title=f"GLOENS {args.cycle}: separation distance", xlim=(0, 10))
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "separation_distance.png", dpi=160); plt.close(fig)

    # Cumulative Liu skill by particle, with missing fixes excluded and
    # no displacement calculated across observation gaps.
    scored = merged.sort_values(["id", "time"]).copy()
    for c in ("obs_lon", "obs_lat"):
        scored["prev_" + c] = scored.groupby("id")[c].shift()
    scored["obs_step_km"] = distance_km(
        scored.prev_obs_lon, scored.prev_obs_lat, scored.obs_lon, scored.obs_lat)
    good_step = (scored.groupby("id").hour.diff() == 1) & scored.sep_km.notna() & scored.obs_step_km.notna()
    scored["step_path"] = scored.obs_step_km.where(good_step, 0.0)
    scored["step_error"] = scored.sep_km.where(good_step, 0.0)
    scored["cum_path"] = scored.groupby("id").step_path.cumsum()
    scored["cum_error"] = scored.groupby("id").step_error.cumsum()
    # NCLS denominator: sum of cumulative observed trajectory lengths.
    scored["path_for_normalizer"] = scored.cum_path.where(good_step, 0.0)
    scored["ncls_denominator"] = scored.groupby("id").path_for_normalizer.cumsum()
    scored["liu_skill"] = (1 - scored.cum_error / scored.ncls_denominator).clip(lower=0, upper=1)
    scored.loc[~good_step | (scored.cum_path <= 0), "liu_skill"] = np.nan
    liu_hourly = scored.groupby("hour").liu_skill.agg(
        median="median", mean="mean", count="count").reset_index()
    liu_hourly.to_csv(out / "hourly_liu_skill.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(liu_hourly.hour / 24, liu_hourly["median"], label="Median")
    ax.plot(liu_hourly.hour / 24, liu_hourly["mean"], label="Mean")
    ax.set(xlabel="Days after noon launch", ylabel="Liu skill (0–1)",
           title=f"GLOENS {args.cycle}: bounded Liu trajectory skill",
           xlim=(0, 10), ylim=(0, 1))
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "liu_skill.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_hour.hour / 24, per_hour.valid_pairs, label="Valid observation-model pairs")
    ax.plot(per_hour.hour / 24, per_hour.model_positions, label="Model positions")
    ax.axhline(len(ids), ls=":", color="black", label="Seeded drifters")
    ax.set(xlabel="Days after noon launch", ylabel="Number of particles",
           title=f"GLOENS {args.cycle}: sample retention", xlim=(0, 10))
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=160); plt.close(fig)

    # Track maps: show selected drifters with high day-1 separation and
    # selected low-error controls. Longitudes unwrapped independently to
    # avoid false jumps at the dateline.
    at24 = merged[(merged.hour >= 23) & (merged.hour <= 25)].groupby("id").sep_km.median().dropna()
    worst = at24.sort_values(ascending=False).head(max(1, args.max_tracks // 2)).index.tolist()
    best = at24.sort_values().head(max(1, args.max_tracks - len(worst))).index.tolist()
    selected = list(dict.fromkeys(worst + best))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, names, title in zip(axes, [worst, best], ["Largest 24h separation", "Smallest 24h separation"]):
        for name in names:
            sub = merged[merged.id == name].sort_values("time")
            for x, y, style in [("obs_lon", "obs_lat", "-"), ("model_lon", "model_lat", "--")]:
                good = sub[[x, y]].dropna()
                if good.empty:
                    continue
                unwrapped = np.rad2deg(np.unwrap(np.deg2rad(good[x].to_numpy())))
                ax.plot(unwrapped, good[y], style, lw=1.2, label=f"{name} {'obs' if style == '-' else 'model'}")
        ax.set(xlabel="Longitude (unwrapped degrees)", ylabel="Latitude (degrees)", title=title)
        ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"GLOENS {args.cycle}: solid=observed, dashed=model")
    fig.tight_layout(); fig.savefig(out / "tracks_comparison.png", dpi=170); plt.close(fig)

    # Global overview of every available trajectory, not only audit selections.
    # Split at longitude discontinuities so dateline crossings do not draw
    # artificial lines across the entire map.
    fig, ax = plt.subplots(figsize=(16, 8))
    for prefix, color, label in (("obs", "tab:orange", "Observed"),
                                 ("model", "tab:blue", "GLOENS model")):
        for _, sub in merged.groupby("id", sort=False):
            sub = sub.sort_values("time")
            x = ((sub[prefix + "_lon"].to_numpy(dtype=float) + 180) % 360) - 180
            y = sub[prefix + "_lat"].to_numpy(dtype=float)
            breaks = np.flatnonzero(np.abs(np.diff(x)) > 180) + 1
            x[breaks] = np.nan
            ax.plot(x, y, color=color, alpha=.25, lw=.65)
        ax.plot([], [], color=color, lw=1.5, label=label)
    ax.set(xlim=(-180, 180), ylim=(-90, 90),
           xlabel="Longitude (degrees east)", ylabel="Latitude (degrees north)",
           title=f"GLOENS {args.cycle}: all observed and simulated trajectories")
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(alpha=.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "global_all_trajectories.png", dpi=180)
    plt.close(fig)

    # Flag implausibly fast model segments. Hourly outputs may have gaps,
    # so normalize each segment by its actual elapsed time.
    segments = raw.sort_values(["id", "time"]).copy()
    segments["prev_lon"] = segments.groupby("id").model_lon.shift()
    segments["prev_lat"] = segments.groupby("id").model_lat.shift()
    segments["dt_h"] = segments.groupby("id").time.diff().dt.total_seconds() / 3600
    segments["speed_kmh"] = distance_km(segments.prev_lon, segments.prev_lat, segments.model_lon, segments.model_lat) / segments.dt_h
    segments.loc[segments.dt_h <= 0, "speed_kmh"] = np.nan
    segments.sort_values("speed_kmh", ascending=False).head(30)[
        ["id", "time", "model_lon", "model_lat", "speed_kmh"]].to_csv(out / "fastest_segments.csv", index=False)

    if args.gulf_stream_animation:
        from matplotlib.animation import FuncAnimation, PillowWriter
        if args.frame_hours < 1:
            raise SystemExit("--frame-hours must be positive")
        # Animate only existing paired records; never interpolate across gaps.
        region = (-80, -40, 25, 50)
        anim_data = merged.copy()
        for prefix in ("model", "obs"):
            anim_data[prefix + "_lon"] = ((anim_data[prefix + "_lon"] + 180) % 360) - 180
        frames = np.arange(0, 241, args.frame_hours)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.set(xlim=region[:2], ylim=region[2:], xlabel="Longitude (degrees east)",
               ylabel="Latitude (degrees north)",
               title=f"GLOENS {args.cycle}: Gulf Stream drifters")
        ax.grid(alpha=.25)
        model_scatter = ax.scatter([], [], s=11, c="tab:blue", label="GLOENS model", alpha=.7)
        obs_scatter = ax.scatter([], [], s=11, c="tab:orange", label="Observed", alpha=.7)
        ax.legend(loc="lower right")
        stamp_text = ax.text(.02, .97, "", transform=ax.transAxes, va="top")
        def update(hour):
            at = anim_data[np.isclose(anim_data.hour.to_numpy(), hour)]
            for scatter, prefix in ((model_scatter, "model"), (obs_scatter, "obs")):
                xy = at[[prefix + "_lon", prefix + "_lat"]].to_numpy(dtype=float)
                xy = xy[np.isfinite(xy).all(axis=1)]
                xy = xy[(xy[:, 0] >= region[0]) & (xy[:, 0] <= region[1]) &
                        (xy[:, 1] >= region[2]) & (xy[:, 1] <= region[3])]
                scatter.set_offsets(xy.reshape(-1, 2))
            stamp_text.set_text(f"Lead: {hour / 24:.2f} days")
            return model_scatter, obs_scatter, stamp_text
        animation = FuncAnimation(fig, update, frames=frames, interval=180, blit=False)
        gif_path = out / "gulf_stream_zoom.gif"
        animation.save(gif_path, writer=PillowWriter(fps=5), dpi=110)
        plt.close(fig)
        print("Saved Gulf Stream animation:", gif_path)

    print("Cycle:", args.cycle, "start:", start, "model end:", raw.time.max())
    print("Seeded:", len(ids), "model IDs:", raw.id.nunique())
    print("Initial maximum separation (km):", merged.loc[merged.hour == 0, "sep_km"].max())
    print("Model records at hour 240:", int((merged.hour == 240).sum()))
    print("Paired records at hour 240:", int(merged.loc[merged.hour == 240, "sep_km"].notna().sum()))
    print("Largest hourly segment speed (km/h):", segments.speed_kmh.max())
    print("Saved audit to:", out)


if __name__ == "__main__":
    main()
