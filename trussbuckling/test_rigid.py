"""Geometric checks of the exact slit-prism reconstruction."""

import unittest
from math import pi, sin

import numpy as np

from trussbuckling.research.rigid_prism import (
    cell_height, fold_limit, ideal_aspect, mesh, panel_lengths, reference_fraction,
)


class RigidTests(unittest.TestCase):
    def test_every_triangle_remains_congruent_throughout_folding(self):
        for sides in (4, 5, 6, 17, 32):
            side = 200 * sin(pi / sides)
            expected = np.array([side/np.sqrt(2), side/np.sqrt(2), side])
            for fraction in np.linspace(0, 1, 101):
                vertices, faces, _, _ = mesh(100, sides, float(fraction))
                np.testing.assert_allclose(panel_lengths(vertices, faces),
                                           np.tile(expected, (4*sides, 1)), rtol=3e-14)

    def test_entire_vertex_set_stays_in_fixed_cylinder_and_between_rings(self):
        for sides in (3, 4, 6, 17, 64):
            for fraction in np.linspace(0, 1, 101):
                vertices, _, _, _ = mesh(100, sides, float(fraction))
                radius = np.linalg.norm(vertices[:, :2], axis=1)
                np.testing.assert_allclose(radius[:2*sides], 100, atol=1e-12)
                self.assertLessEqual(radius.max(), 100 + 1e-12)
                self.assertGreaterEqual(vertices[:, 2].min(), -1e-12)
                self.assertLessEqual(vertices[:, 2].max(), cell_height(100, sides)*fraction + 1e-12)

    def test_flat_state_is_full_collapse_with_rigid_nondegenerate_panels(self):
        for sides in (4, 6, 17):
            vertices, faces, _, twist = mesh(100, sides, 0)
            np.testing.assert_array_equal(vertices[:, 2], 0)
            np.testing.assert_allclose(vertices[sides:2*sides], np.roll(vertices[:sides], -1, axis=0), atol=1e-12)
            self.assertAlmostEqual(twist, 2*pi/sides)
            triangles = vertices[faces]
            areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                            triangles[:, 2]-triangles[:, 0]), axis=1)/2
            np.testing.assert_allclose(areas, (200*sin(pi/sides))**2/4, rtol=1e-13)

    def test_slit_lips_are_distinct_material_points_that_open(self):
        for sides in (3, 4):
            for fraction in (1., .7, .3):
                vertices, _, cuts, _ = mesh(100, sides, fraction)
                self.assertEqual(cuts.shape, (4*sides, 2))
                gap = np.linalg.norm(vertices[2*sides:3*sides] - vertices[3*sides:], axis=1)
                if fraction == 1:
                    np.testing.assert_allclose(gap, 0, atol=1e-12)
                else:
                    self.assertTrue(np.all(gap > 0))

    def test_triangular_rectangles_fold_with_congruent_panels_and_fixed_radius(self):
        # Bottom/top triangles: sides R,R,sqrt(3)R. Vertical triangles: R,R,R.
        expected = 100 * np.tile([[1, 1, np.sqrt(3)], [1, 1, 1]], (6, 1))
        for fraction in np.linspace(0, 1, 201):
            vertices, faces, _, theta = mesh(100, 3, float(fraction))
            np.testing.assert_allclose(panel_lengths(vertices, faces), expected, rtol=2e-14)
            self.assertAlmostEqual(fraction**2, 2*np.cos(theta)-1, places=14)
            np.testing.assert_allclose(vertices[3:6, 2], 100*fraction)
        flat, _, _, theta = mesh(100, 3, 0.)
        np.testing.assert_array_equal(flat[:, 2], 0.)
        np.testing.assert_allclose(flat[6:], 0., atol=3e-14)
        self.assertAlmostEqual(theta, pi/3)
        self.assertAlmostEqual(reference_fraction(100, 3, 71.2), .712)

    def test_triangular_model_matches_independent_original_constraint_solution(self):
        from trussbuckling.research.verification.triangle import mesh as independent_mesh
        for fraction in np.linspace(.01, .99, 41):
            vertices, _, _, _ = mesh(1., 3, float(fraction))
            original, _ = independent_mesh(float(fraction))
            np.testing.assert_allclose(vertices, original, atol=2e-13)

    def test_square_height_rotation_relation_and_reference_size(self):
        for fraction in (.01, .3, .7, 1.):
            vertices, _, _, theta = mesh(100, 4, fraction)
            self.assertAlmostEqual(vertices[4, 2]**2, 20000*np.cos(theta), places=10)
        f = reference_fraction(100, 17, 71.2, cells=2)
        self.assertAlmostEqual(2*mesh(100, 17, f)[0][17, 2], 71.2)

    def test_invalid_geometry_is_rejected(self):
        for radius, sides, fraction in ((0, 4, .5), (100, 2, .5), (100, 4.5, .5),
                                         (100, 4, -1), (100, 4, 1.1)):
            with self.assertRaises(ValueError):
                mesh(radius, sides, fraction)

    def test_rectangular_aspects_preserve_every_panel_and_the_envelope(self):
        for sides in (3, 4, 6, 24):
            width = 200*sin(pi/sides)
            for aspect in (.1, .4, ideal_aspect(sides), 1., 1.5, 2.):
                height = width*aspect
                leg = np.hypot(width, height)/2
                expected = np.tile(np.sort([[leg, leg, width], [leg, leg, height]], axis=1),
                                   (2*sides, 1))
                minimum, _ = fold_limit(sides, aspect)
                for fraction in np.linspace(minimum, 1, 31):
                    vertices, faces, _, _ = mesh(100, sides, float(fraction), aspect)
                    np.testing.assert_allclose(panel_lengths(vertices, faces), expected, rtol=2e-13)
                    self.assertGreaterEqual(vertices[:, 2].min(), -1e-10)
                    self.assertLessEqual(vertices[:, 2].max(), height*fraction + 1e-10)
                    self.assertLessEqual(np.linalg.norm(vertices[:, :2], axis=1).max(), 100+1e-10)

    def test_aspect_limits_match_the_actual_contact_geometry(self):
        self.assertAlmostEqual(ideal_aspect(3), 1/np.sqrt(3))
        for sides in (3, 4, 6, 24):
            self.assertEqual(fold_limit(sides, ideal_aspect(sides)), (0., "flat"))
            minimum, reason = fold_limit(sides, .4)
            self.assertEqual(reason, "end_plane")
            vertices, _, _, _ = mesh(100, sides, minimum, .4)
            np.testing.assert_allclose(vertices[2*sides:3*sides, 2], 0, atol=1e-10)
            minimum, reason = fold_limit(sides, 1.5)
            self.assertEqual(reason, "panel_contact")
            vertices, _, _, _ = mesh(100, sides, minimum, 1.5)
            lower, upper = vertices[2*sides:3*sides], vertices[3*sides:]
            if sides == 3:
                np.testing.assert_allclose(lower[:, :2], 0, atol=1e-10)
                np.testing.assert_allclose(lower, upper, atol=1e-10)
            else:
                np.testing.assert_allclose(lower, np.roll(upper, 1, axis=0), atol=1e-10)
        self.assertAlmostEqual(fold_limit(3, 1.)[0], np.sqrt(2/3))

    def test_custom_aspect_cannot_move_past_its_geometric_limit(self):
        for aspect in (.1, .4, 1., 2.):
            with self.assertRaises(ValueError):
                mesh(100, 3, 0., aspect)
            with self.assertRaises(ValueError):
                reference_fraction(100, 3, 1., aspect=aspect)
        for aspect in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mesh(100, 3, 1., aspect)


if __name__ == "__main__":
    unittest.main()
