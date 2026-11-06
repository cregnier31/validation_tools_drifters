"""One-day GLOENS launcher-path test with 28-day dispersion (mocked I/O).

Run: python -m unittest discover -s tests -p 'test_gloens_launcher.py' -v

This checks the launcher -> process_one_day -> GLOENS Zarr -> Parcels handoff,
without downloading global data or launching a computationally expensive run.
"""
import contextlib
import datetime as dt
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


class GloensLauncherTest(unittest.TestCase):
    def test_single_day_28_day_parcels_dispatch(self):
        from lagrangian_workflow import process_one_day
        day = pd.Timestamp("2025-11-06")
        with tempfile.TemporaryDirectory() as root:
            config = {
                "config_info": {
                    "DIR_OUT": root + "/", "lon_min": -180, "lon_max": 180,
                    "lat_min": -90, "lat_max": 90, "SYSTEM": "GLOENS",
                    "leadtime": "FCST", "dir_in": root, "dir_work": root,
                    "path_data": root, "depth": 0, "advection_diffusion_type": "AdvectionRK4",
                    "nb_repeats": 0, "STR_REGION": "global", "drift_time": 28,
                    "obs_type": "COPERNICUS_DAILY", "loglevel": 0,
                    "engine": "parcels", "input_format": "zarr",
                    "zarr_bucket": "MOISICEEF",
                    "zarr_endpoint_url": "https://s3.waw3-1.cloudferro.com",
                }
            }
            trajs = [{"id": 123, "lon": np.array([1.]), "lat": np.array([2.])}]
            t = pd.date_range(day, periods=28, freq="D")
            ds = xr.Dataset(
                {"uo": (("time", "lat", "lon"), np.zeros((28, 2, 2))),
                 "vo": (("time", "lat", "lon"), np.zeros((28, 2, 2)))},
                coords={"time": t, "lat": [1., 2.], "lon": [1., 2.]},
            )
            calls = []

            @contextlib.contextmanager
            def fake_forecast(u, v, start, **kwargs):
                calls.append(("forecast", u, v, pd.Timestamp(start), kwargs))
                yield ds

            def fake_parcels(*args, **kwargs):
                calls.append(("parcels", args, kwargs))

            import forecast_inputs
            import parcels_tools
            import lagrangian_workflow

            with patch.object(lagrangian_workflow, "retrieve_drifters_for_one_day_traj",
                              return_value=(trajs, root + "/drifters.csv")), \
                 patch.object(forecast_inputs, "discover_gloens_stores",
                              return_value=("s3://MOISICEEF/U.zarr", "s3://MOISICEEF/V.zarr", {})), \
                 patch.object(forecast_inputs, "prepare_ensemble_forecast", fake_forecast), \
                 patch.object(parcels_tools, "run_parcels_Drift_remote", fake_parcels):
                result = process_one_day(day, config)
            self.assertEqual(result, root + "/drifters.csv")
            self.assertEqual([call[0] for call in calls], ["forecast", "parcels"])
            self.assertEqual(calls[0][4]["days"], 28)
            self.assertEqual(calls[1][1][7], 28)  # T_DRIFT
            self.assertEqual(calls[1][1][0], "GLOENS")
            self.assertEqual(calls[1][1][1].sizes["time"], 28)


if __name__ == "__main__":
    unittest.main()
