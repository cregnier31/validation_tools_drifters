"""Tests for single-file, multi-file NetCDF and Zarr velocity inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from forecast_inputs import open_velocity_dataset


class TestOpenVelocityDataset(unittest.TestCase):
    def test_single_netcdf(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "velocity.nc"
            xr.Dataset({"uo": ("time", [1.0])}, coords={"time": [0]}).to_netcdf(path)
            with open_velocity_dataset(path) as result:
                self.assertEqual(float(result.uo.values[0]), 1.0)

    def test_multifile_netcdf_sequence_and_glob(self):
        with TemporaryDirectory() as directory:
            for day in range(2):
                path = Path(directory) / f"velocity_{day}.nc"
                xr.Dataset(
                    {"uo": ("time", np.array([day + 1.0]))},
                    coords={"time": [day]},
                ).to_netcdf(path)
            sources = sorted(Path(directory).glob("*.nc"))
            for selection in (sources, str(Path(directory) / "*.nc")):
                with self.subTest(selection=str(selection)):
                    with open_velocity_dataset(selection) as result:
                        np.testing.assert_array_equal(result.uo.values, [1.0, 2.0])

    def test_empty_sequence(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            open_velocity_dataset([])

    def test_missing_glob(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                open_velocity_dataset(str(Path(directory) / "*.nc"))

    def test_reject_mixed_zarr_sequence(self):
        with self.assertRaisesRegex(ValueError, "single Zarr"):
            open_velocity_dataset(["a.nc", "b.zarr"])

    def test_reject_storage_options_for_netcdf(self):
        with self.assertRaisesRegex(ValueError, "only applies to Zarr"):
            open_velocity_dataset(["a.nc"], storage_options={"anon": True})

    def test_zarr_dispatch(self):
        with patch("xarray.open_zarr") as opened:
            open_velocity_dataset("example.zarr/")
            opened.assert_called_once_with("example.zarr/", consolidated=False, chunks={})


if __name__ == "__main__":
    unittest.main()
