"""Verify flat-endpoint design against actual mesh geometry and free folding."""

import math
import unittest
from dataclasses import replace

import numpy as np

from trussbuckling.model import (
    Parameters, choose_creases, endpoint_angle, endpoint_strain, endpoint_twist,
    mesh, state, sweep,
)


class DesignTests(unittest.TestCase):
    def design(self, radius=100.0, length=71.2, creases=17):
        return Parameters(radius=radius, length=length, creases=creases,
                          gamma=endpoint_angle(radius, length, creases))

    def test_extended_and_flat_meshes_have_identical_edge_lengths(self):
        cases = ((10, 3, 3), (100, 71.2, 17), (250, 402, 32),
                 (100, 160.8, 256), (100, 2, 256))
        for radius, length, creases in cases:
            with self.subTest(radius=radius, length=length, creases=creases):
                p = self.design(radius, length, creases)
                twist = endpoint_twist(radius, length, creases)
                extended, edges, _ = mesh(p, 0, length)
                folded, folded_edges, _ = mesh(p, twist, 0)
                np.testing.assert_array_equal(edges, folded_edges)
                lengths = np.linalg.norm(extended[edges[:, 1]] - extended[edges[:, 0]], axis=1)
                flat_lengths = np.linalg.norm(folded[edges[:, 1]] - folded[edges[:, 0]], axis=1)
                np.testing.assert_allclose(flat_lengths, lengths, rtol=2e-13, atol=radius * 1e-13)
                for angle, height in ((0, length), (twist, 0)):
                    np.testing.assert_allclose(state(p, angle, height=height).strain,
                                               0, atol=2e-13)
                self.assertLess(state(p, twist).height, radius * 1e-7)

    def test_reported_peak_matches_dense_free_folding_path(self):
        for radius, length, creases in ((100, 22.9, 3), (100, 71.2, 17),
                                       (50, 80.4, 64)):
            with self.subTest(length=length, creases=creases):
                p = self.design(radius, length, creases)
                twists = np.linspace(0, endpoint_twist(radius, length, creases), 301)
                path = sweep(p, twists)
                measured = np.max(np.abs(path["strain"]), axis=1)
                peak = endpoint_strain(radius, length, creases)
                self.assertAlmostEqual(peak, measured.max(), places=13)
                self.assertEqual(int(measured.argmax()), len(twists) // 2)
                self.assertTrue(np.all(np.diff(path["height"]) <= radius * 1e-9))
                self.assertTrue(np.all(path["height"][:-1] > 0))

    def test_small_strain_design_still_has_an_energy_barrier(self):
        p = self.design()
        twist = endpoint_twist(p.radius, p.length, p.creases)
        before, middle, after = (state(p, fraction * twist) for fraction in (0.4, 0.5, 0.6))
        self.assertGreater(middle.energy, 0)
        self.assertGreater(middle.energy, before.energy)
        self.assertGreater(middle.energy, after.energy)
        self.assertGreater(before.torque, 0)
        self.assertLess(after.torque, 0)
        self.assertLess(middle.stiffness, 0)
        self.assertAlmostEqual(middle.torque, 0, delta=1e-6)

    def test_default_design_numeric_reference(self):
        self.assertAlmostEqual(math.degrees(endpoint_angle(100, 71.2, 17)), 29.534187175790123)
        self.assertAlmostEqual(math.degrees(endpoint_twist(100, 71.2, 17)), 14.815681257411033)
        self.assertAlmostEqual(endpoint_strain(100, 71.2, 17), 0.0007267025847781376, places=13)
        self.assertAlmostEqual(endpoint_strain(100, 71.2, 13), 0.0009837029217120272, places=13)

    def test_automatic_count_is_the_smallest_meeting_the_allowance(self):
        for radius, length, allowance, expected in (
            (100, 71.2, 0.001, 13), (100, 71.2, 0.01, 4),
            (100, 113.8, 0.001, 66), (100, 160.8, 0.001, 228),
        ):
            with self.subTest(length=length, allowance=allowance):
                count = choose_creases(radius, length, allowance)
                self.assertEqual(count, expected)
                self.assertLessEqual(endpoint_strain(radius, length, count), allowance)
                for smaller in range(3, count):
                    try:
                        peak = endpoint_strain(radius, length, smaller)
                    except ValueError:
                        continue  # This count cannot have compatible end states.
                    self.assertGreater(peak, allowance)

    def test_count_search_respects_geometry_before_strain(self):
        # A generous strain allowance must still reject infeasible polygons.
        self.assertEqual(choose_creases(100, 160.8, 1), 5)
        with self.assertRaisesRegex(ValueError, "No flat-endpoint design.*n <= 4"):
            choose_creases(100, 160.8, 1, limit=4)
        with self.assertRaisesRegex(ValueError, "No flat-endpoint design"):
            choose_creases(100, 200, 1)

    def test_unmet_allowance_reports_the_best_achievable_strain(self):
        best = endpoint_strain(100, 113.8, 64)
        with self.assertRaises(ValueError) as caught:
            choose_creases(100, 113.8, 0.001, limit=64)
        message = str(caught.exception)
        self.assertIn("No n <= 64", message)
        self.assertIn("0.1%", message)
        self.assertIn(f"best is {100 * best:.4g}%", message)

    def test_maximum_height_includes_vertical_reference_bars(self):
        for creases in (3, 17, 256):
            with self.subTest(creases=creases):
                radius = 100
                length = 2 * radius * math.cos(math.pi / creases)
                p = self.design(radius, length, creases)
                self.assertAlmostEqual(p.gamma, math.pi / 2, places=12)
                twist = endpoint_twist(radius, length, creases)
                np.testing.assert_allclose(state(p, twist, height=0).strain, 0, atol=2e-13)
                self.assertGreater(state(p, twist / 2).height, 0)
                self.assertTrue(math.isfinite(endpoint_strain(radius, length, creases)))
                with self.assertRaisesRegex(ValueError, "L <= 2 R"):
                    endpoint_angle(radius, length * (1 + 1e-7), creases)

    def test_short_large_count_design_retains_its_small_positive_twist(self):
        p = self.design(100, 2, 256)
        twist = endpoint_twist(p.radius, p.length, p.creases)
        self.assertGreater(twist, 0.0002)
        self.assertLess(twist, 0.000201)
        middle = state(p, twist / 2)
        self.assertGreater(middle.height, 1)
        self.assertLess(middle.height, p.length)
        self.assertGreater(endpoint_strain(p.radius, p.length, p.creases), 0)
        self.assertLess(endpoint_strain(p.radius, p.length, p.creases), 1e-9)
        np.testing.assert_allclose(state(p, twist, height=0).strain, 0, atol=2e-13)

    def test_uniform_scaling_preserves_design_and_strain(self):
        angle = endpoint_angle(100, 71.2, 17)
        twist = endpoint_twist(100, 71.2, 17)
        peak = endpoint_strain(100, 71.2, 17)
        for scale in (0.01, 3, 100):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(endpoint_angle(100 * scale, 71.2 * scale, 17), angle)
                self.assertAlmostEqual(endpoint_twist(100 * scale, 71.2 * scale, 17), twist)
                self.assertAlmostEqual(endpoint_strain(100 * scale, 71.2 * scale, 17), peak, places=13)
                self.assertEqual(choose_creases(100 * scale, 71.2 * scale, 0.001), 13)

    def test_stiffness_values_do_not_change_the_free_folding_geometry(self):
        p = self.design()
        twist = endpoint_twist(p.radius, p.length, p.creases) / 2
        original = state(p, twist)
        for changed in (replace(p, modulus=0.1), replace(p, area=1000)):
            result = state(changed, twist)
            self.assertEqual(result.height, original.height)
            np.testing.assert_array_equal(result.strain, original.strain)
            self.assertNotEqual(result.energy, original.energy)

    def test_endpoint_design_requires_the_height_to_change(self):
        p = self.design()
        twist = endpoint_twist(p.radius, p.length, p.creases)
        held = state(p, twist, height=p.length)
        self.assertTrue(np.all(held.strain > 0))
        self.assertGreater(held.energy, 0)
        shifted = state(p, 0, height=1.01 * p.length)
        self.assertTrue(np.all(shifted.strain > 0))

    def test_design_rejects_invalid_inputs(self):
        for radius, length, creases in ((0, 1, 17), (100, 0, 17),
                                        (100, math.inf, 17), (math.nan, 1, 17),
                                        (100, 1, 2), (100, 1, 4.5), (100, 1, True)):
            with self.subTest(radius=radius, length=length, creases=creases):
                for function in (endpoint_angle, endpoint_twist, endpoint_strain):
                    with self.assertRaises(ValueError):
                        function(radius, length, creases)
        for allowance in (0, -1, math.nan, math.inf):
            with self.subTest(allowance=allowance), self.assertRaises(ValueError):
                choose_creases(100, 71.2, allowance)
        for limit in (2, 3.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                choose_creases(100, 71.2, 0.001, limit=limit)


if __name__ == "__main__":
    unittest.main()
