import math
import unittest

import numpy as np

from dewarp_geometry import canvas_from_aspect, estimate_aspect_from_grid


def plane_grid(width=2.0, height=1.0, rows=21, columns=31):
    x = np.linspace(0, width, columns)
    y = np.linspace(0, height, rows)
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx, yy, np.zeros_like(xx)), axis=-1)


class AspectEstimatorTest(unittest.TestCase):
    def test_rectangle(self):
        result = estimate_aspect_from_grid(plane_grid(2.0, 1.0), layout="hwc")
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.aspect_ratio, 2.0, places=7)

    def test_rigid_transform_and_uniform_scale_invariance(self):
        grid = plane_grid(1.3, 2.4)
        angle = 0.72
        rotation = np.array([
            [math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ])
        transformed = 7.0 * (grid @ rotation.T) + np.array([4.0, -2.0, 9.0])
        result = estimate_aspect_from_grid(transformed, layout="hwc")
        self.assertAlmostEqual(result.aspect_ratio, 1.3 / 2.4, places=7)

    def test_cylindrical_surface_uses_arc_length(self):
        width, height, radius = 2.4, 1.2, 1.0
        u = np.linspace(-width / 2, width / 2, 81)
        v = np.linspace(0, height, 31)
        uu, vv = np.meshgrid(u, v)
        grid = np.stack((radius * np.sin(uu / radius), vv, radius * np.cos(uu / radius)), axis=-1)
        result = estimate_aspect_from_grid(grid, layout="hwc")
        self.assertAlmostEqual(result.aspect_ratio, width / height, delta=2e-4)

    def test_chw_and_batch(self):
        grid = np.moveaxis(plane_grid(), -1, 0)[None]
        self.assertAlmostEqual(estimate_aspect_from_grid(grid, layout="chw").aspect_ratio, 2.0)

    def test_invalid_grid(self):
        grid = plane_grid()
        grid[:, 10:20] = np.nan
        result = estimate_aspect_from_grid(grid, layout="hwc")
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_canvas_has_requested_ratio(self):
        width, height = canvas_from_aspect(0.7, long_edge=1400)
        self.assertEqual((width, height), (980, 1400))


if __name__ == "__main__":
    unittest.main()
