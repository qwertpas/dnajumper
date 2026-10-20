"""Verify five-cord optimization candidates against geometry and exact energy."""

import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from trussbuckling.research.five_cords import (
    contact_angles, geometry, make_design, mirror_design, slack_margins, stack_matrix, stiffness,
)


class FiveCordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads((Path(__file__).parent/"research/five_designs.json").read_text())
        cls.saved = data["designs"]

    def test_all_designs_preserve_lengths_balance_load_and_have_rank_five(self):
        for item in self.saved:
            d = make_design(np.radians(item["angles_degrees"]), item["bottom_radii"])
            self.assertGreater(d.shares.min(), 0)
            self.assertGreaterEqual(min(d.bottom.min(), d.top.min()), .6-1e-12)
            self.assertLessEqual(max(d.bottom.max(), d.top.max()), 1+1e-12)
            lengths = np.sqrt(1.2+(d.bottom-d.top)**2)
            for angle in np.radians([0, 19, 45, 75]):
                centers, points = geometry(d, angle)
                links = np.diff(points, axis=0)
                np.testing.assert_allclose(np.linalg.norm(links, axis=2), np.tile(lengths, (12, 1)), atol=1e-13)
                h = centers[1, 2]
                tension = d.shares*lengths/h
                forces = links/lengths[None, :, None]*tension[None, :, None]
                np.testing.assert_allclose(forces.sum(axis=1), np.tile([0, 0, 1], (12, 1)), atol=1e-13)
                for j in range(1, 12):
                    net = forces[j]-forces[j-1]
                    np.testing.assert_allclose(net.sum(axis=0), 0, atol=1e-13)
                    np.testing.assert_allclose(np.cross(points[j]-centers[j], net).sum(axis=0), 0, atol=1e-13)
                top_arms = points[1]-centers[1]
                jacobian = np.column_stack((links[0], np.cross(top_arms, links[0])))/lengths[:, None]
                self.assertEqual(np.linalg.matrix_rank(jacobian), 5)

    def test_body_frame_hessian_matches_exact_finite_rotation_energy(self):
        rng = np.random.default_rng(52)
        for item in (self.saved[0], self.saved[-1]):
            d = make_design(np.radians(item["angles_degrees"]), item["bottom_radii"])
            angle, cells = np.radians(49), 4
            matrix = stack_matrix(d, angle, cells=cells)
            centers, points = geometry(d, angle, cells=cells)
            lengths = np.sqrt(1.2+(d.bottom-d.top)**2)
            tension = d.shares*lengths/centers[1, 2]
            spring = 1200/lengths
            rest = lengths-tension/spring
            turns = Rotation.from_rotvec(np.column_stack((np.zeros(cells+1),
                np.zeros(cells+1), np.arange(cells+1)*angle))).as_matrix()

            def energy(values):
                pose = np.vstack((np.zeros(6), values.reshape(cells, 6)))
                translation = np.einsum("pij,pj->pi", turns, pose[:, :3])
                rotation = np.einsum("pij,pj->pi", turns, pose[:, 3:])
                arms = points-centers[:, None, :]
                changed = np.einsum("pij,pcj->pci", Rotation.from_rotvec(rotation).as_matrix(), arms)
                changed += (centers+translation)[:, None, :]
                extension = np.maximum(np.linalg.norm(np.diff(changed, axis=0), axis=2)-rest, 0)
                return .5*np.sum(spring*extension**2)-values[-4]

            free = np.r_[np.arange(6*(cells-1)), 6*(cells-1)+2]
            for _ in range(4):
                direction = np.zeros(6*cells)
                direction[free] = rng.normal(size=len(free))
                direction /= np.linalg.norm(direction)
                step = 1e-5
                measured = (energy(step*direction)+energy(-step*direction)
                            - 2*energy(np.zeros_like(direction)))/step**2
                self.assertAlmostEqual(measured/(direction@matrix@direction), 1., places=5)

    def test_full_stability_and_stroke_scores(self):
        for item in self.saved:
            d = make_design(np.radians(item["angles_degrees"]), item["bottom_radii"])
            for angle in np.radians(np.linspace(0, 75, 31)):
                full = stack_matrix(d, angle)
                free = np.r_[np.arange(66), 68]
                np.linalg.cholesky(full[np.ix_(free, free)])
                self.assertGreater(stiffness(d, angle), item["worst_alignment_stiffness"]-1e-5)
            at_worst = stiffness(d, np.radians(item["worst_twist_degrees"]))
            self.assertAlmostEqual(at_worst, item["worst_alignment_stiffness"], places=6)
            self.assertGreater(np.degrees(contact_angles(d).min()), 79.99999)

    def test_mirror_construction_and_contact_angle_are_exact(self):
        alpha = np.radians(31.50333703822236)
        beta = alpha+2*np.arctan(.25)
        d = mirror_design(alpha, beta)
        self.assertAlmostEqual(np.degrees(contact_angles(d).min()), 90.)
        # Mirror about alpha, then match paired anchors 0<->1,2<->4,3<->3.
        order = [1, 0, 4, 3, 2]
        np.testing.assert_allclose(d.bottom[order], d.top)
        difference = (2*alpha-d.angles-d.angles[order]+np.pi)%(2*np.pi)-np.pi
        np.testing.assert_allclose(difference, 0, atol=1e-14)
        # At the predicted pair contact, one alternating parity intersects
        # at a common height fraction. Use positive arbitrary h=1 here.
        i, j = 0, 4
        theta = np.pi/2
        t = d.bottom[i]*d.bottom[j]/(d.bottom[i]*d.bottom[j]+d.product)
        residuals = []
        for a, b in ((d.bottom, d.top), (d.top, d.bottom)):
            z = np.exp(1j*d.angles)*((1-t)*a+t*b*np.exp(1j*theta))
            residuals.append(abs(z[i]-z[j]))
        self.assertLess(min(residuals), 1e-13)

    def test_linear_slack_margins_are_positive_and_finite(self):
        for item in self.saved:
            d = make_design(np.radians(item["angles_degrees"]), item["bottom_radii"])
            for angle in np.radians([0, 45, 75]):
                margins = np.asarray(slack_margins(d, angle))
                self.assertTrue(np.isfinite(margins).all())
                self.assertTrue((margins>0).all())


if __name__ == "__main__":
    unittest.main()
