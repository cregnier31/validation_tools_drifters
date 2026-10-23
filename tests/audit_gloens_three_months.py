#!/usr/bin/env python3
"""Audit all weekly GLOENS cycles using existing Parquet trajectories and observations.

Runs the single-cycle audit without animations and aggregates per-cycle quality
metrics. Read-only with respect to trajectories, observations, and caches.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CYCLES = ("20240104 20240111 20240118 20240125 20240201 20240208 "
          "20240215 20240222 20240229 20240307 20240314 20240321 20240328").split()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drift-days", type=int, default=15)
    ap.add_argument("--root", default="/scratch/cregnier/WORK_lagrangian/AdvectionRK4/DRIFT_GLOENS")
    ap.add_argument("--output", default=None)
    ap.add_argument("--cycles", nargs="+", default=CYCLES)
    args = ap.parse_args()
    if args.drift_days < 1:
        ap.error("--drift-days must be positive")
    root = Path(args.root)
    out = Path(args.output) if args.output else root / f"three_month_audit_drift{args.drift_days}d"
    out.mkdir(parents=True, exist_ok=True)
    single = Path(__file__).with_name("audit_gloens_tracks.py")
    records = []
    for cycle in args.cycles:
        print(f"Auditing {cycle}...", flush=True)
        matches = sorted(root.glob(
            f"Parcels_Drift-GLOENS_FCST_AdvectionRK4_drift{args.drift_days}_dt5_{cycle}-{cycle}*/ZONE_*/{cycle}"))
        row = {"cycle": cycle, "status": "FAIL", "reason": ""}
        try:
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one cycle directory; found {len(matches)}")
            directory = matches[0]
            pq = sorted(directory.glob("Trajectories_GLOENS_*at_SVPloc*.parquet"))
            if len(pq) != 1:
                raise ValueError(f"Expected one trajectory Parquet; found {len(pq)}")
            df = pd.read_parquet(pq[0], columns=["particle_id", "t", "x", "y"])
            if df.empty:
                raise ValueError("Empty trajectory Parquet")
            t = df.t
            if pd.api.types.is_numeric_dtype(t):
                duration_h = (t.max() - t.min()) / 3600
            else:
                duration_h = (pd.to_datetime(t).max() - pd.to_datetime(t).min()).total_seconds() / 3600
            row.update(model_particles=int(df.particle_id.nunique()),
                       records=len(df), timestamps=int(t.nunique()),
                       duration_hours=float(duration_h),
                       invalid_positions=int((~np.isfinite(df[["x", "y"]].to_numpy(dtype=float))).any(axis=1).sum()))
            if duration_h < args.drift_days * 24 - 1:
                raise ValueError(f"Short trajectory: {duration_h:.1f} of {args.drift_days * 24} hours")
            result = subprocess.run(
                [sys.executable, str(single), "--cycle", cycle, "--drift-days",
                 str(args.drift_days), "--root", str(root)],
                capture_output=True, text=True)
            (out / f"{cycle}.log").write_text(result.stdout + "\n" + result.stderr)
            if result.returncode:
                raise ValueError(f"Single-cycle audit failed; see {cycle}.log")
            audit = directory / "trajectory_audit"
            sep = pd.read_csv(audit / "hourly_separation.csv")
            skill = pd.read_csv(audit / "hourly_liu_skill.csv")
            row.update(
                paired_initial=int(sep.iloc[0].valid_pairs),
                paired_final=int(sep.iloc[-1].valid_pairs),
                final_hour=float(sep.iloc[-1].hour),
                final_median_sep_km=float(sep.iloc[-1].median_sep_km),
                final_p90_sep_km=float(sep.iloc[-1].p90_sep_km),
                final_mean_liu=float(skill.iloc[-1]["mean"]),
                final_median_liu=float(skill.iloc[-1]["median"]),
                final_liu_count=int(skill.iloc[-1]["count"]))
            if row["final_hour"] < args.drift_days * 24 - 1:
                raise ValueError("Validation does not reach the final forecast day")
            if row["paired_initial"] <= 0:
                raise ValueError("No initial observation-model matches")
            row["status"] = "PASS"
        except Exception as exc:
            row["reason"] = str(exc)
        records.append(row)
        print(f"  {row['status']}: {row['reason'] or 'duration and validation checked'}", flush=True)
    summary = pd.DataFrame(records)
    summary.to_csv(out / "cycle_summary.csv", index=False)
    print("\n", summary.to_string(index=False))
    print(f"\nSummary: {out / 'cycle_summary.csv'}")
    print("Per-cycle audit plots: each cycle's trajectory_audit directory")
    if (summary.status != "PASS").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
