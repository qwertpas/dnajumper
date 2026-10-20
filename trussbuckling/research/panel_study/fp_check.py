"""Unconstrained plate/tip audit of the existing FP-ori mesh.

Run with python3.11 -m trussbuckling.research.panel_study.fp_check.
Fix the lower plate; allow all six upper-plate motions and all slit-tip
coordinates. Triangle rigidity means its three edge lengths are fixed.
This tests local constraints and load paths, not fabric constitutive behavior.
"""

from math import sin

import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation

from ..rigid_prism import cell_height, mesh


def cross(point):
    x, y, z = point
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def model(sides, fraction):
    vertices, faces, _, angle = mesh(radius=1., sides=sides, fraction=fraction)
    edges = sorted({tuple(sorted((face[j], face[(j + 1) % 3])))
                    for face in faces for j in range(3)})
    # Distances within each rigid plate are already enforced by its pose.
    edges = [edge for edge in edges
             if not (edge[1] < sides or sides <= edge[0] < edge[1] < 2*sides)]
    edges = np.asarray(edges)
    height = cell_height(1., sides) * fraction
    center = np.array([0., 0., height])
    size = 6 + 6*sides
    maps = np.zeros((4*sides, 3, size))
    for i in range(sides, 2*sides):
        maps[i, :, :6] = np.column_stack((np.eye(3), -cross(vertices[i] - center)))
    for i in range(2*sides, 4*sides):
        start = 6 + 3*(i - 2*sides)
        maps[i, :, start:start + 3] = np.eye(3)
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    jacobian = np.einsum('ij,ijk->ik', vectors / lengths[:, None],
                         maps[edges[:, 1]] - maps[edges[:, 0]])

    def distances(values):
        moved = vertices.copy()
        moved[sides:2*sides] = ((vertices[sides:2*sides] - center)
                               @ Rotation.from_rotvec(values[3:6]).as_matrix().T
                               + center + values[:3])
        moved[2*sides:] += values[6:].reshape(-1, 3)
        return np.linalg.norm(moved[edges[:, 1]] - moved[edges[:, 0]], axis=1)

    return jacobian, edges, lengths, height, angle, distances


def verify():
    print('n  fraction  variables  constraints  rank  nullity  tip rank  reduced rank')
    worst_force = 0.
    worst_derivative = 0.
    for sides in (3, 4, 6):
        for fraction in (.1, .25, .5, .75, .9):
            jacobian, edges, lengths, height, angle, distances = model(sides, fraction)
            rank = np.linalg.matrix_rank(jacobian, tol=1e-10)
            tip_rank = np.linalg.matrix_rank(jacobian[:, 6:], tol=1e-10)
            reduced = null_space(jacobian[:, 6:].T).T @ jacobian[:, :6]
            reduced_rank = np.linalg.matrix_rank(reduced, tol=1e-10)
            assert len(edges) == 7*sides
            assert rank == 6*sides + 3
            assert tip_rank == 6*sides
            assert reduced_rank == 3

            # A physical tensile equilibrium carries force only on corner edges.
            tensions = np.zeros(len(edges))
            corners = edges[:, 1] - edges[:, 0] == sides
            corners &= edges[:, 0] < sides
            assert np.count_nonzero(corners) == sides
            tensions[corners] = cell_height(1., sides) / (sides * height)
            load = np.zeros(jacobian.shape[1])
            load[2], load[5] = 1., sin(angle) / height
            force_error = np.max(np.abs(jacobian.T @ tensions - load))
            assert force_error < 1e-12
            worst_force = max(worst_force, force_error)

            # Independently finite-difference all pose and tip derivatives.
            step = 1e-6
            basis = np.eye(jacobian.shape[1]) * step
            measured = np.column_stack([(distances(v) - distances(-v)) / (2*step)
                                        for v in basis])
            derivative_error = np.max(np.abs(measured - jacobian))
            assert derivative_error < 1e-8
            worst_derivative = max(worst_derivative, derivative_error)
            print(f'{sides}  {fraction:8.2f}  {jacobian.shape[1]:9d}  '
                  f'{len(edges):11d}  {rank:4d}  {jacobian.shape[1]-rank:7d}  '
                  f'{tip_rank:8d}  {reduced_rank:12d}')
    print(f'Maximum force residual: {worst_force:.3g}')
    print(f'Maximum finite-difference derivative error: {worst_derivative:.3g}')
    print('Every case: 3 tangent motions, and zero force in every slit-tip edge.')


if __name__ == '__main__':
    verify()
