#!/usr/bin/env python3
"""Batch all validated drifters for each available GLOENS 28-day cycle in 2025.

Default: discover actual initialization dates from forecast Zarr filenames.
Each cycle needs matching CloudFerro U/V stores and local observations.
"""
import argparse
import csv
import re
import s3fs
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--step-days", type=int, default=None, help="Optional subsampling of discovered cycles")
    p.add_argument("--start", default=None, help="Optional YYYYMMDD first cycle")
    p.add_argument("--end", default=None, help="Optional YYYYMMDD last cycle")
    p.add_argument("--output", default=None)
    p.add_argument("--continue-on-error", action="store_true", help="Continue after failed cycle")
    args = p.parse_args()
    if args.step_days is not None and args.step_days < 1:
        p.error("--step-days must be positive")
    # Enumerate actual weekly forecast initialization dates from CloudFerro S3.
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
    from forecast_inputs import GLOENS_BUCKET, GLOENS_ENDPOINT_URL
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": GLOENS_ENDPOINT_URL})
    pattern = re.compile(r"glo4-ens50_ng_1d-m_(\d{8})-\d{8}_3DU-uo_fcst_R(\d{8})\.zarr$")
    dates = sorted({
        date.fromisoformat(m.group(1)[:4] + "-" + m.group(1)[4:6] + "-" + m.group(1)[6:])
        for item in fs.ls(GLOENS_BUCKET)
        if (m := pattern.search(str(item))) and m.group(1) == m.group(2)
    })
    first = date.fromisoformat(args.start[:4] + "-" + args.start[4:6] + "-" + args.start[6:]) if args.start else date(args.year, 1, 1)
    last = date.fromisoformat(args.end[:4] + "-" + args.end[4:6] + "-" + args.end[6:]) if args.end else date(args.year, 12, 31)
    dates = [d for d in dates if first <= d <= last and d.year == args.year]
    if args.step_days is not None:
        selected = []
        next_date = first
        for d in dates:
            if d >= next_date:
                selected.append(d)
                next_date = d + timedelta(days=args.step_days)
        dates = selected
    if not dates:
        raise SystemExit("No matching GLOENS forecast initialization cycles found")
    print(f"Discovered {len(dates)} GLOENS cycles for {args.year}", flush=True)
    output = Path(args.output or f"outputs/gloens_annual_{args.year}")
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.csv"
    script = Path(__file__).resolve().parents[1] / "Lagrangian_tool.py"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "status", "returncode", "log", "plot"])
        for current in dates:
            stamp = current.strftime("%Y%m%d")
            cycle_output = output / stamp
            cycle_output.mkdir(parents=True, exist_ok=True)
            log_path = cycle_output / "run.log"
            command = [sys.executable, str(script), "--date1", stamp,
                       "--date2", stamp, "--lead", "FCST", "--sys", "GLOENS",
                       "--scheme", "AdvectionRK4", "--engine", "parcels",
                       "--typeobs", "COPERNICUS_DAILY"]
            print(f"Running {stamp} (all valid particles, 28 days)", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            plot = None  # Lagrangian_tool.py manages its own figures and output paths.
            ok = result.returncode == 0
            writer.writerow([stamp, "PASS" if ok else "FAIL", result.returncode,
                             str(log_path), ""])
            f.flush()
            print(f"{stamp}: {'PASS' if ok else 'FAIL'}; log: {log_path}", flush=True)
            if not ok and not args.continue_on_error:
                raise SystemExit(f"Stopped on failed cycle {stamp}; see {log_path}. Use --continue-on-error to proceed.")
    print(f"Annual summary: {summary}")


if __name__ == "__main__":
    main()
