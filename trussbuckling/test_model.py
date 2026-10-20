"""Check the paper geometry, spatial mesh, and energy derivatives end to end.

Run from the repository root: python3.11 -m unittest trussbuckling.test_model
"""

import math
import unittest
from dataclasses import replace

import numpy as np

from trussbuckling.model import Parameters, collapse_twist, folding_angles, mesh, rigid_endpoint, state, sweep


class ModelTests(unittest.TestCase):
    def setUp(self):
        _, gamma = folding_angles(100.0, 71.2, 17)
        self.p = Parameters(radius=100.0, length=71.2, creases=17, gamma=gamma,
                            modulus=5550.0, area=1.0)

    def close(self, actual, expected, relative=1e-5, absolute=1e-7):
        np.testing.assert_allclose(actual, expected, rtol=relative, atol=absolute)

    def test_hunt_table_3_folding_angles(self):
        # Table 3 reports whole degrees. These are source values, not formula copies.
        rows = ((22.9, 23, 16, 24), (35.9, 21, 21, 30),
                (51.0, 19, 26, 35), (71.2, 17, 32, 42),
                (113.8, 14, 43, 56), (160.8, 12, 57, 72))
        for length, creases, beta, gamma in rows:
            with self.subTest(length=length):
                found = np.degrees(folding_angles(100.0, length, creases))
                self.close(found, (beta, gamma), relative=0, absolute=0.5)

    def test_paper_triangle_geometry(self):
        beta, gamma = folding_angles(100.0, 71.2, 17)
        alpha = gamma - beta
        self.close(alpha, math.pi / 17)
        short_fold = 71.2 / math.sin(gamma)
        self.close(short_fold / math.sin(beta),
                   (2 * math.pi * 100 / 17) / math.sin(alpha))

    def test_maximum_folding_length_produces_valid_parameters(self):
        for creases in (3, 12, 17, 64):
            length = 2 * math.pi * 100 / creases / math.tan(math.pi / creases)
            beta, gamma = folding_angles(100, length, creases)
            self.close(gamma, math.pi / 2)
            p = Parameters(radius=100, length=length, creases=creases, gamma=gamma)
            self.close(state(p, 0).height, length)

    def test_reference_state_is_unloaded(self):
        for height in (None, self.p.length):
            result = state(self.p, 0.0, height=height)
            self.close(result.height, self.p.length)
            self.close(result.lengths, self.p.rest_lengths)
            self.close(result.strain, 0)
            self.close(result.energy, 0)
            self.close(result.torque, 0, absolute=1e-6)
            self.close(result.force, 0)

    def test_closed_rims_and_complete_triangulation(self):
        count = self.p.creases
        vertices, edges, faces = mesh(self.p, 0.23, 54.0)
        self.assertEqual(vertices.shape, (2 * count, 3))
        self.assertEqual(edges.shape, (4 * count, 2))
        self.assertEqual(faces.shape, (2 * count, 3))
        self.assertEqual(len({tuple(sorted(edge)) for edge in edges}), 4 * count)
        self.close(np.linalg.norm(vertices[:, :2], axis=1), self.p.radius)
        for rim in (vertices[:count], vertices[count:]):
            lengths = np.linalg.norm(np.roll(rim, -1, axis=0) - rim, axis=1)
            self.close(lengths, 2 * self.p.radius * math.sin(math.pi / count))
        # A cylinder has Euler characteristic zero; every rim edge has one face,
        # and every side diagonal has two.
        incidence = {tuple(sorted(edge)): 0 for edge in edges}
        for face in faces:
            for a, b in zip(face, np.roll(face, -1)):
                incidence[tuple(sorted((a, b)))] += 1
        self.assertEqual(sorted(incidence.values()), [1] * (2 * count) + [2] * (2 * count))
        self.assertEqual(len(vertices) - len(edges) + len(faces), 0)

    def test_spatial_member_lengths_match_reported_lengths(self):
        count = self.p.creases
        for twist, height in ((0.0, 71.2), (0.25, 50.0), (1.0, 0.0)):
            with self.subTest(twist=twist, height=height):
                vertices, edges, _ = mesh(self.p, twist, height)
                lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
                result = state(self.p, twist, height=height)
                self.close(lengths[:2 * count], 2 * self.p.radius * math.sin(math.pi / count))
                self.close(np.sort(lengths[2 * count:]),
                           np.sort(np.repeat(result.lengths, count)))

    def test_fixed_height_torque_is_energy_derivative(self):
        angle, height, step = 0.22, 56.0, 1e-5
        before = state(self.p, angle - step, height=height)
        after = state(self.p, angle + step, height=height)
        result = state(self.p, angle, height=height)
        self.close(result.torque, (after.energy - before.energy) / (2 * step))

    def test_axial_force_is_energy_derivative(self):
        angle, height, step = 0.22, 56.0, 1e-3
        before = state(self.p, angle, height=height - step)
        after = state(self.p, angle, height=height + step)
        result = state(self.p, angle, height=height)
        self.close(result.force, (after.energy - before.energy) / (2 * step))

    def test_fixed_height_stiffness_is_torque_derivative(self):
        angle, height, step = 0.22, 56.0, 1e-5
        before = state(self.p, angle - step, height=height)
        after = state(self.p, angle + step, height=height)
        result = state(self.p, angle, height=height)
        self.close(result.stiffness, (after.torque - before.torque) / (2 * step))

    def test_axial_stiffness_is_force_derivative(self):
        angle, height, step = 0.22, 56.0, 1e-3
        before = state(self.p, angle, height=height - step)
        after = state(self.p, angle, height=height + step)
        result = state(self.p, angle, height=height)
        self.close(result.axial_stiffness, (after.force - before.force) / (2 * step))

    def test_free_height_minimizes_actual_energy(self):
        for angle in (0.0, 0.2, 0.6, 1.4):
            with self.subTest(twist=angle):
                result = state(self.p, angle)
                self.assertGreaterEqual(result.height, 0)
                self.close(result.force, 0, absolute=1e-6)
                heights = np.linspace(0, max(self.p.length * 1.5, result.height * 1.2), 151)
                energies = [state(self.p, angle, height=height).energy for height in heights]
                self.assertLessEqual(result.energy, min(energies) + 1e-7)
                for height in (max(0, result.height - 0.01), result.height + 0.01):
                    self.assertLessEqual(result.energy,
                                         state(self.p, angle, height=height).energy + 1e-7)

    def test_relaxed_torque_and_stiffness_are_energy_derivatives(self):
        # Away from the contact transition, the relaxed path must obey the same
        # energy balance as prescribed-height states.
        angle, step = 0.12, 1e-5
        before, result, after = (state(self.p, value) for value in
                                 (angle - step, angle, angle + step))
        self.assertGreater(result.height, 0)
        self.close(result.torque, (after.energy - before.energy) / (2 * step))
        self.close(result.stiffness, (after.torque - before.torque) / (2 * step))

    def test_modulus_and_area_scale_loads_without_changing_geometry(self):
        reference = state(self.p, 0.2)
        for changed in (replace(self.p, modulus=self.p.modulus * 3),
                        replace(self.p, area=self.p.area * 3)):
            result = state(changed, 0.2)
            self.close(result.height, reference.height)
            self.close(result.lengths, reference.lengths)
            self.close(result.strain, reference.strain)
            for field in ("energy", "torque", "stiffness", "axial_stiffness"):
                self.close(getattr(result, field), 3 * getattr(reference, field))

    def test_paper_folding_angles_do_not_guarantee_strain_free_collapse(self):
        # Searching every relative orientation still leaves strain at zero height
        # for this two-ring model with the source's developed-sheet angles.
        angles = np.linspace(1e-4 - self.p.reference_twist,
                             math.pi - self.p.reference_twist, 361)
        best = min(state(self.p, angle, height=0).energy for angle in angles)
        self.assertGreater(best, 1e-3)

    def test_strain_free_endpoints_still_require_intermediate_strain(self):
        alpha = math.pi / self.p.creases
        theta = math.acos(self.p.length ** 2 /
                          (4 * self.p.radius ** 2 * math.cos(alpha))) - alpha
        gamma = math.atan2(self.p.length, self.p.radius * theta)
        p = replace(self.p, gamma=gamma)
        folded_twist = math.pi - 2 * theta - 2 * alpha
        self.assertGreater(folded_twist, 0)
        flat = state(p, folded_twist, height=0)
        self.close(flat.lengths, p.rest_lengths)
        self.close(flat.strain, 0)
        self.close(flat.energy, 0)
        middle = state(p, folded_twist / 2)
        self.assertGreater(middle.energy, 1e-3)
        self.assertGreater(np.max(np.abs(middle.strain)), 1e-6)

    def test_analytical_rigid_endpoint_is_an_actual_second_unloaded_state(self):
        self.assertIsNone(rigid_endpoint(self.p))
        p = replace(self.p, gamma=math.radians(25))
        twist, height = rigid_endpoint(p)
        self.assertGreater(height, 0)
        self.assertGreater(abs(twist), 1e-3)
        result = state(p, twist, height=height)
        self.close(result.strain, 0)
        self.close(result.energy, 0)
        self.close(result.torque, 0)
        self.close(result.force, 0)

    def test_bistable_path_contains_negative_torque_stiffness(self):
        p = replace(self.p, gamma=math.radians(25))
        endpoint = rigid_endpoint(p)
        results = sweep(p, np.linspace(0, endpoint[0], 61))
        self.assertGreater(max(results["energy"]), 1e-3)
        self.assertLess(min(results["stiffness"]), 0)
        self.assertGreater(results["stiffness"][0], 0)
        self.assertGreater(results["stiffness"][-1], 0)

    def test_sweep_matches_individual_states(self):
        angles = np.linspace(0, 0.5, 13)
        for height in (None, 45.0):
            result = sweep(self.p, angles, height=height)
            states = [state(self.p, angle, height=height) for angle in angles]
            for field in ("height", "torque", "strain", "stiffness", "energy"):
                self.close(result[field], [getattr(item, field) for item in states])

    def test_collapse_limit_for_all_paper_folding_geometries(self):
        for length, creases in ((22.9, 23), (35.9, 21), (51, 19),
                                (71.2, 17), (113.8, 14), (160.8, 12)):
            gamma = folding_angles(100, length, creases)[1]
            p = replace(self.p, length=length, creases=creases, gamma=gamma)
            twist = collapse_twist(p)
            self.assertIsNotNone(twist)
            self.assertGreater(twist, 0)
            self.assertGreater(state(p, twist - 1e-5).height, 0)
            self.assertLess(state(p, twist).height, 1e-5 * length)
            self.assertEqual(state(p, twist + 1e-5).height, 0)

    def test_relaxed_stiffness_remains_accurate_near_flat_limit(self):
        twist = collapse_twist(self.p) - 1e-8
        before = state(self.p, twist - 1e-9)
        after = state(self.p, twist + 1e-9)
        self.close(state(self.p, twist).stiffness,
                   (after.torque - before.torque) / 2e-9, relative=1e-3)

    def test_uniform_length_scaling_preserves_strain(self):
        reference = state(self.p, 0.2)
        p = replace(self.p, radius=2 * self.p.radius, length=2 * self.p.length)
        result = state(p, 0.2)
        self.close(result.height, 2 * reference.height)
        self.close(result.lengths, 2 * reference.lengths)
        self.close(result.strain, reference.strain)
        self.close(result.energy, 2 * reference.energy)
        self.close(result.torque, 2 * reference.torque)

    def test_rejects_invalid_or_nonfinite_parameters(self):
        for field, value in (("radius", 0), ("radius", float("nan")),
                             ("length", -1), ("length", float("inf")),
                             ("creases", 2), ("creases", 5.5),
                             ("gamma", 0), ("gamma", math.pi),
                             ("modulus", 0), ("area", -1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(self.p, **{field: value})
        with self.assertRaises(ValueError):
            folding_angles(100, 250, 17)
        with self.assertRaises(ValueError):
            state(self.p, 0, height=-1)


if __name__ == "__main__":
    unittest.main()
