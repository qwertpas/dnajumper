"""Geometry, force balance and independent energy checks for the cord viewer."""

from dataclasses import replace
from math import radians, sqrt
import unittest

import numpy as np

from trussbuckling.cable_model import (
    Parameters, deform, evaluate, response, response_warnings, twist_limit,
)
from trussbuckling.research.offset_links import crossing_angle
from trussbuckling.research.verify_alignment import full_model


class CableTests(unittest.TestCase):
    def test_geometry_rank_and_plate_equilibrium_for_six_through_sixteen_cords(self):
        for family in (3, 4, 8):
            for ratio in (.3, .65, 1.):
                p = Parameters(per_family=family, ratio=ratio, cells=3)
                limit, _ = twist_limit(p)
                for twist in (0., limit*.5, limit):
                    p = replace(p, twist=twist)
                    report = evaluate(p)
                    links = np.diff(report.points, axis=0)
                    np.testing.assert_allclose(np.linalg.norm(links, axis=2), report.length, rtol=1e-12)
                    self.assertEqual(report.rank, 3 if ratio == 1 else 5)
                    self.assertGreater(report.height, 0)
                    self.assertLess(radians(twist), crossing_angle(1, ratio, family))
                    forces = report.tension*links/report.length
                    np.testing.assert_allclose(forces.sum(axis=1), np.tile([0, 0, p.force], (3, 1)), atol=1e-10)
                    # Each free intermediate plate has zero resultant force and moment.
                    for plate in (1, 2):
                        force = forces[plate] - forces[plate-1]
                        arms = report.points[plate] - report.centers[plate]
                        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-10)
                        np.testing.assert_allclose(np.cross(arms, force).sum(axis=0), 0, atol=1e-8)
                    arms = report.points[-1] - report.centers[-1]
                    torque = np.cross(arms, -forces[-1]).sum(axis=0)
                    np.testing.assert_allclose(torque, [0, 0, -report.torque], atol=1e-8)

    def test_normalized_hessian_matches_independent_existing_derivation(self):
        for cells, ratio, twist in ((3, .5, 30), (5, .65, 65), (12, 1., 45)):
            p = Parameters(cells=cells, ratio=ratio, twist=twist, per_family=4, height=sqrt(2))
            report = evaluate(p)
            other, _, _ = full_model(cells, radians(twist), report.length/p.radius,
                inner=ratio, rigidity=p.rigidity/p.force)
            np.testing.assert_allclose(report._matrix[np.ix_(report._free, report._free)], other,
                                       atol=2e-10)

    def test_six_cord_hessian_matches_exact_nonlinear_energy(self):
        rng = np.random.default_rng(191)
        for p in (Parameters(cells=3), Parameters(cells=5, ratio=1., height=sqrt(2))):
            report = evaluate(p)
            length = report.length/p.radius
            rest = length*(1-report.extension/100)

            def energy(values):
                motion = np.zeros((p.cells+1, 6))
                motion[1:] = values.reshape(p.cells, 6)
                motion[:, :3] *= p.radius
                _, _, points = deform(report, motion)
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=2)/p.radius
                return p.rigidity/(2*p.force*length)*np.sum(np.maximum(lengths-rest, 0)**2)-values[-4]

            for _ in range(4):
                direction = rng.normal(size=p.cells*6)
                direction /= np.linalg.norm(direction)
                step = 1e-5
                measured = (energy(step*direction)+energy(-step*direction)
                            - 2*energy(np.zeros_like(direction)))/step**2
                expected = direction @ report._matrix @ direction
                self.assertAlmostEqual(measured/expected, 1., places=5)

    def test_six_and_eight_cords_match_at_equal_total_axial_rigidity(self):
        p = Parameters(cells=4)
        six = evaluate(p)
        eight = evaluate(replace(p, per_family=4, rigidity=p.rigidity*6/8))
        np.testing.assert_allclose(six._matrix, eight._matrix, atol=2e-11)

    def test_default_stability_equal_radius_instability_and_mode_normalization(self):
        stable = evaluate(Parameters())
        unstable = evaluate(Parameters(ratio=1., height=sqrt(2)))
        self.assertTrue(stable.stable)
        self.assertFalse(unstable.tilt_stable)
        for report in (stable, unstable, evaluate(Parameters(ratio=1., twist=0))):
            displacement = report.mode[:, None, :3] + np.cross(
                report.mode[:, None, 3:], report.points-report.centers[:, None, :]
            )
            self.assertAlmostEqual(np.linalg.norm(displacement, axis=2).max(), report.radius)
            np.testing.assert_array_equal(report.mode[[0, -1]], 0)
        with self.assertRaisesRegex(ValueError, "No stable linear response"):
            response(unstable, .01, .01, 0)

    def test_end_error_response_obeys_boundary_and_free_plate_equilibrium(self):
        report = evaluate(Parameters())
        motion = response(report, .01, .01, 35)
        values = motion[1:].copy()
        values[:, :3] /= report.radius
        residual = report._matrix @ values.ravel()
        np.testing.assert_allclose(residual[report._free], 0, atol=1e-12)
        np.testing.assert_array_equal(motion[0], 0)
        self.assertAlmostEqual(np.linalg.norm(motion[-1, :2]), .01)
        self.assertAlmostEqual(np.linalg.norm(motion[-1, 3:5]), radians(.01))
        self.assertFalse(response_warnings(report, motion))
        single = evaluate(Parameters(), cells=1)
        warnings = response_warnings(single, response(single, .1, .1, 0))
        self.assertIn("slacken", warnings[0])

    def test_equal_radius_zero_twist_and_invalid_inputs(self):
        report = evaluate(Parameters(ratio=1., twist=0))
        self.assertTrue(report.stable)
        self.assertTrue(np.isfinite(report.eigenvalues).all())
        for p in (Parameters(cells=2.0), Parameters(ratio=0), Parameters(force=0),
                  Parameters(per_family=2), Parameters(rigidity=1, force=1000)):
            with self.assertRaises(ValueError):
                evaluate(p)


if __name__ == "__main__":
    unittest.main()
