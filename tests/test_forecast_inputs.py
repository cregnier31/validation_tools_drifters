"""Synthetic tests for 28-day ensemble velocity preparation.

Run: PYTHONPATH=src python -m unittest discover -s tests
"""
import unittest
import numpy as np
import xarray as xr

from forecast_inputs import ensemble_mean_velocity, select_forecast_window


class ForecastInputsTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(np.datetime64("2026-01-01"),
                         np.datetime64("2026-01-30"),
                         np.timedelta64(1, "D"))
        self.ds = xr.Dataset(
            {"uo": (("ens", "time", "depthu", "lat", "lon"),
                    np.stack([np.ones((29, 1, 2, 2)),
                              3 * np.ones((29, 1, 2, 2))])),
             "vo": (("ens", "time", "depthv", "lat", "lon"),
                    np.stack([2 * np.ones((29, 1, 2, 2)),
                              4 * np.ones((29, 1, 2, 2))]))},
            coords={"ens": [0, 1], "time": time,
                    "depthu": [0.0], "depthv": [0.0],
                    "lat": [0.0, 1.0], "lon": [0.0, 1.0]})

    def test_ensemble_mean(self):
        mean = ensemble_mean_velocity(self.ds)
        self.assertNotIn("ens", mean.dims)
        self.assertTrue(np.allclose(mean.uo.values, 2))
        self.assertTrue(np.allclose(mean.vo.values, 3))

    def test_28_day_window(self):
        result = select_forecast_window(self.ds, "2026-01-01")
        self.assertEqual(result.sizes["time"], 29)

    def test_reject_incomplete_window(self):
        with self.assertRaises(ValueError):
            select_forecast_window(self.ds.isel(time=slice(0, 20)),
                                   "2026-01-01")


if __name__ == "__main__":
    unittest.main()
