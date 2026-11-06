"""Regression tests for Liu trajectory diagnostics."""
import unittest
import numpy as np
from src.lagrangian_metrics import LiuScore2, liu_score_diagnostics


class TestLiuDiagnostics(unittest.TestCase):
    def test_perfect_track(self):
        lon = np.array([[0., 1., 2.]])
        lat = np.zeros_like(lon)
        result = liu_score_diagnostics(lon, lat, lon, lat)
        np.testing.assert_allclose(result["skill"], 1.)
        np.testing.assert_allclose(result["normalized_error"], 0.)
        np.testing.assert_array_equal(result["valid_count"], [1, 1])
        np.testing.assert_array_equal(result["clipped_count"], [0, 0])

    def test_agrees_with_existing_skill(self):
        lon = np.array([[0., 1., 2.]])
        lat = np.zeros_like(lon)
        model_lat = np.array([[0., 0.1, 0.2]])
        result = liu_score_diagnostics(lon, lat, lon, model_lat)
        expected = LiuScore2(lon, lat, lon, model_lat)
        np.testing.assert_allclose(result["skill"], expected, equal_nan=True)

    def test_missing_sample_invalidates_following_leads(self):
        lon = np.array([[0., np.nan, 2.], [0., 1., 2.]])
        lat = np.zeros_like(lon)
        result = liu_score_diagnostics(lon, lat, lon, lat)
        np.testing.assert_array_equal(result["valid_count"], [1, 1])
        self.assertTrue(np.isnan(result["skill"][:, 0]).all())

    def test_zero_displacement_has_no_skill(self):
        lon = np.zeros((1, 3))
        result = liu_score_diagnostics(lon, lon, lon, lon)
        self.assertTrue(np.isnan(result["skill"]).all())
        np.testing.assert_array_equal(result["valid_count"], [0, 0])

    def test_invalid_tolerance(self):
        lon = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            liu_score_diagnostics(lon, lon, lon, lon, n=0)


if __name__ == "__main__":
    unittest.main()
