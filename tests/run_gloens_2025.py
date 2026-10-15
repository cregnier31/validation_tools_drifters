#!/usr/bin/env python3
"""Batch the experimental 10-drifter/28-day GLOENS comparison over 2025.

Default: daily initializations with complete 28-day observation windows inside
2025 (Jan 1 to Dec 3). Use --step-days 7 or 30 for cheaper initial tests.
Each cycle needs a matching CloudFerro forecast store and local observations.
"""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--step-days", type=int, default=1)
    p.add_argument("--start", default=None, help="Optional YYYYMMDD first cycle")
    p.add_argument("--end", default=None, help="Optional YYYYMMDD last cycle")
    p.add_argument("--output", default=None)
    p.add_argument("--meshmask", default="/scratch/cregnier/static/iORCA025_meshmask_v3.1.nc")
    p.add_argument("--continue-on-error", action="store_true", help="Continue after failed cycle")
    args = p.parse_args()
    if args.step_days < 1:
        p.error("--step-days must be positive")
    first = date.fromisoformat(args.start[:4] + "-" + args.start[4:6] + "-" + args.start[6:]) if args.start else date(args.year, 1, 1)
    last = date.fromisoformat(args.end[:4] + "-" + args.end[4:6] + "-" + args.end[6:]) if args.end else date(args.year, 12, 31) - timedelta(days=28)
    if first.year != args.year or last.year != args.year or last < first:
        p.error("Requested range must be ordered and within --year")
    output = Path(args.output or f"outputs/gloens_annual_{args.year}")
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.csv"
    script = Path(__file__).resolve().with_name("compare_gloens_10_drifters.py")
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "status", "returncode", "log", "plot"])
        current = first
        while current <= last:
            stamp = current.strftime("%Y%m%d")
            cycle_output = output / stamp
            cycle_output.mkdir(parents=True, exist_ok=True)
            log_path = cycle_output / "run.log"
            command = [sys.executable, str(script), "--date", stamp,
                       "--output", str(cycle_output), "--meshmask", args.meshmask]
            print(f"Running {stamp} (10 particles, 28 days)", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            plot = cycle_output / "comparison_2d.png"
            ok = result.returncode == 0 and plot.is_file()
            writer.writerow([stamp, "PASS" if ok else "FAIL", result.returncode,
                             str(log_path), str(plot) if plot.is_file() else ""])
            f.flush()
            print(f"{stamp}: {'PASS' if ok else 'FAIL'}; log: {log_path}", flush=True)
            if not ok and not args.continue_on_error:
                raise SystemExit(f"Stopped on failed cycle {stamp}; see {log_path}. Use --continue-on-error to proceed.")
            current += timedelta(days=args.step_days)
    print(f"Annual summary: {summary}")


if __name__ == "__main__":
    main()
