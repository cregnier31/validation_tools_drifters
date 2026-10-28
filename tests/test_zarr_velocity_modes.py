"""Pure in-memory tests for velocity mode selection."""
import unittest
import numpy as np
import xarray as xr
from forecast_inputs import select_velocity


class TestVelocitySelection(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {"uo": (("ens", "time", "y", "x"), np.array([[[[1.0]]], [[[3.0]]]])),
             "vo": (("ens", "time", "y", "x"), np.array([[[[2.0]]], [[[4.0]]]]))},
            coords={"ens": [10, 20], "time": np.array(["2024-01-04"], dtype="datetime64[D]")},
        )

    def test_mean(self):
        selected = select_velocity(self.ds, mode="mean")
        self.assertEqual(float(selected.uo.values.item()), 2.0)
        self.assertEqual(float(selected.vo.values.item()), 3.0)

    def test_member_by_label(self):
        selected = select_velocity(self.ds, mode="member", member=20)
        self.assertEqual(float(selected.uo.values.item()), 3.0)
        self.assertNotIn("ens", selected.uo.dims)

    def test_deterministic(self):
        ds = self.ds.sel(ens=10, drop=True)
        selected = select_velocity(ds, mode="deterministic")
        self.assertEqual(float(selected.uo.values.item()), 1.0)

    def test_deterministic_rejects_ensemble(self):
        with self.assertRaises(ValueError):
            select_velocity(self.ds, mode="deterministic")

    def test_member_required(self):
        with self.assertRaises(ValueError):
            select_velocity(self.ds, mode="member")


if __name__ == "__main__":
    unittest.main()
