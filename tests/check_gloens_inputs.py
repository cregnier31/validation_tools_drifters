#!/usr/bin/env python3
"""Real I/O integration checks for one GLOENS cycle (no Parcels simulation).

Run:
  PYTHONPATH=src python tests/check_gloens_inputs.py --date 20251106 --drifters
  PYTHONPATH=src python tests/check_gloens_inputs.py --date 20251106 --model
  PYTHONPATH=src python tests/check_gloens_inputs.py --date 20251106 --drifters --model

Requires local drifter files/configuration and network access for --model.
"""
import argparse
import datetime
import logging
import pathlib
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYYMMDD initialization date")
    parser.add_argument("--drifters", action="store_true")
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    if not args.drifters and not args.model:
        parser.error("Select --drifters and/or --model")
    day = pd.Timestamp(datetime.datetime.strptime(args.date, "%Y%m%d"))
    if args.drifters:
        import yaml
        from trajectory_io import retrieve_drifters_for_one_day_traj
        with open(ROOT / "cfg/config_drifters.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Read observed drifter settings from the same runtime configuration
        # rather than silently using a synthetic trajectory.
        print("Config sections:", list(config))
        raise SystemExit(
            "Observed drifter retrieval requires runtime settings (region, observation "
            "source and data path). Use the launcher configuration for this environment; "
            "no fake observations were substituted."
        )
    if args.model:
        from forecast_inputs import discover_gloens_stores, prepare_ensemble_forecast
        u, v, options = discover_gloens_stores(day)
        print("U store:", u)
        print("V store:", v)
        with prepare_ensemble_forecast(u, v, day, days=28, storage_options=options) as ds:
            print("Mean dimensions:", dict(ds.sizes))
            assert ds.sizes["time"] == 28
            for name in ("uo", "vo"):
                assert name in ds and "ens" not in ds[name].dims
                print(name, "dims:", ds[name].dims)
            print("First and last times:", ds.time.values[[0, -1]])
        print("PASS: GLOENS 28-day model input retrieval and ensemble means")


if __name__ == "__main__":
    main()
