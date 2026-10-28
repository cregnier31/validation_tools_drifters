"""Unit tests for experimental Parcels 4 periodic RK4 (no velocity downloads)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "src" / "gloens_periodic_kernel.py"
spec = importlib.util.spec_from_file_location("gloens_periodic_kernel", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Particles:
    def __init__(self, x, y, dt=1.0):
        self.x = np.array([x], dtype=float)
        self.y = np.array([y], dtype=float)
        self.z = np.zeros(1)
        self.t = np.zeros(1)
        self.dt = np.array([dt])
        self.dx = np.zeros(1)
        self.dy = np.zeros(1)


class Velocity:
    def __init__(self, u, v=0):
        self.u = u
        self.v = v
        self.sampled = []

    def __getitem__(self, key):
        if isinstance(key, tuple):
            x = np.asarray(key[3])
        else:
            x = key.x
        self.sampled.extend(x.tolist())
        return np.full_like(x, self.u), np.full_like(x, self.v)


class TestPeriodicRK4(unittest.TestCase):
    def test_eastward_dateline(self):
        p = Particles(179.8, 20.0)
        uv = Velocity(0.5)
        module.AdvectionRK4_periodicBC(p, type("F", (), {"UV": uv})())
        self.assertAlmostEqual(float(p.x[0] + p.dx[0]), -179.7)
        self.assertTrue(all(-180 <= x < 180 for x in uv.sampled[1:]))
        self.assertAlmostEqual(float(p.dy[0]), 0)

    def test_westward_dateline(self):
        p = Particles(-179.8, -10.0)
        uv = Velocity(-0.5)
        module.AdvectionRK4_periodicBC(p, type("F", (), {"UV": uv})())
        self.assertAlmostEqual(float(p.x[0] + p.dx[0]), 179.7)
        self.assertTrue(all(-180 <= x < 180 for x in uv.sampled[1:]))

    def test_non_boundary_matches_standard_rk4_constant_flow(self):
        p = Particles(15, 35, dt=10)
        uv = Velocity(0.02, -0.01)
        module.AdvectionRK4_periodicBC(p, type("F", (), {"UV": uv})())
        self.assertAlmostEqual(float(p.dx[0]), 0.2)
        self.assertAlmostEqual(float(p.dy[0]), -0.1)


if __name__ == "__main__":
    unittest.main()
