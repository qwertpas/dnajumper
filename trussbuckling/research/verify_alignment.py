"""Independent 3D force/stiffness checks for alignment.py and offset_links.py.

Run python3.11 -m trussbuckling.research.verify_alignment from the repo root.
This builds the full plate translation/rotation Hessian directly from cable
endpoints, then checks it against the much smaller analytical tilt model.
"""

from math import cos, pi, sin, sqrt

import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation

from .alignment import lowest_mode, rotation, tilt_matrix
from . import offset_links


def cross_matrix(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])


def full_model(cells, angle, length=sqrt(2), outer=1.0, inner=None,
               rigidity=1000.0, force=1.0):
    """Return full energy Hessian, cable Jacobian, and exact energy function.

    Each internal plate has six variables: translation, then rotation vector
    about its center in the global frame. The final variable is top height.
    End plates cannot tilt, translate sideways, or change their applied twist.
    rigidity is E*A per cable; stiffness is rigidity/length.
    """
    sides = 4 if inner is None else 8
    other = outer if inner is None else inner
    height = sqrt(length**2 - outer**2 - other**2 + 2*outer*other*cos(angle))
    size = 6 * (cells - 1) + 1
    angles = np.arange(sides) * 2*pi/sides
    arms, maps = [], []
    for j in range(cells + 1):
        radii = np.full(sides, outer) if inner is None else np.where(
            np.arange(sides) % 2 == j % 2, outer, inner
        )
        arm = np.column_stack((radii*np.cos(angles + j*angle),
                               radii*np.sin(angles + j*angle), np.zeros(sides)))
        arms.append(arm)
        maps.append([])
        for point in arm:
            change = np.zeros((3, size))
            if 0 < j < cells:
                change[:, 6*(j-1):6*j] = np.column_stack((np.eye(3), -cross_matrix(point)))
            elif j == cells:
                change[2, -1] = 1
            maps[j].append(change)

    geometric = np.zeros((size, size))
    rows = []
    for j in range(cells):
        for i in range(sides):
            bottom, top = arms[j][i], arms[j+1][i]
            link = top - bottom + [0, 0, height]
            change = maps[j+1][i] - maps[j][i]
            rows.append(link @ change / length)
            geometric += change.T @ change
            # Exact second derivative of exp(rotation_vector) @ point.
            for plate, point, sign in ((j, bottom, -1), (j+1, top, 1)):
                if 0 < plate < cells:
                    part = slice(6*(plate-1)+3, 6*plate)
                    geometric[part, part] += sign * (
                        (np.outer(link, point) + np.outer(point, link))/2
                        - np.dot(link, point)*np.eye(3)
                    )
    jacobian = np.asarray(rows)
    tension = force * length / (sides * height)
    stiffness = rigidity / length
    hessian = (tension / length) * geometric + (
        stiffness - tension / length
    ) * jacobian.T @ jacobian
    rest = length - tension / stiffness
    if rest <= 0:
        raise ValueError("Rigidity must be high enough for positive rest length.")

    def energy(values):
        points = []
        for j, arm in enumerate(arms):
            center = np.array([0., 0., j*height])
            if 0 < j < cells:
                center += values[6*(j-1):6*(j-1)+3]
                arm = arm @ Rotation.from_rotvec(values[6*(j-1)+3:6*j]).as_matrix().T
            elif j == cells:
                center[2] += values[-1]
            points.append(arm + center)
        lengths = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=2)
        extension = np.maximum(lengths - rest, 0)
        return stiffness/2 * np.sum(extension**2) - force*values[-1]

    return hessian, jacobian, energy


def verify():
    rng = np.random.default_rng(914)
    comparisons = 0
    for cells, degrees in ((4, 35), (5, 65), (8, 55), (11, 45)):
        angle = np.deg2rad(degrees)
        matrix, jacobian, energy = full_model(cells, angle)
        value, tilts, centers = lowest_mode(cells, angle)
        direction = np.zeros(matrix.shape[0])
        for j in range(1, cells):
            direction[6*(j-1):6*(j-1)+2] = centers[j]
            direction[6*(j-1)+3:6*(j-1)+5] = tilts[j]
        assert np.max(np.abs(jacobian @ direction)) < 1e-12
        assert np.linalg.norm(centers[-1]) < 1e-12
        predicted = value * np.sum(tilts**2)
        np.testing.assert_allclose(direction @ matrix @ direction, predicted, atol=1e-9)

        # Check all 3D Hessian terms, not just a chosen transverse mode.
        for _ in range(3):
            direction = rng.normal(size=matrix.shape[0])
            direction /= np.linalg.norm(direction)
            step = 1e-5
            measured = (energy(step*direction) + energy(-step*direction)
                        - 2*energy(np.zeros_like(direction))) / step**2
            np.testing.assert_allclose(measured, direction @ matrix @ direction, rtol=3e-6)
            comparisons += 1
        # The reference geometry is in static equilibrium in every free DOF.
        for column in range(matrix.shape[0]):
            direction = np.eye(matrix.shape[0])[column] * 1e-7
            assert abs((energy(direction) - energy(-direction))/2e-7) < 1e-6

    for ratio in (.5, .65, .7):
        length = sqrt(1 + ratio**2)
        for angle in (0., .4, 1.2, pi/2 - 1e-3):
            for swap in (False, True):
                bottom, top = offset_links.points(1, ratio, length, angle, swap)
                np.testing.assert_allclose(np.linalg.norm(top-bottom, axis=1), length)
                jacobian = offset_links.jacobian(bottom, top)
                assert np.linalg.matrix_rank(jacobian) == 5
                # Independent material Hessian and elimination of horizontal translation.
                material = jacobian.T @ jacobian / length
                tilt = material[3:5, 3:5] - material[3:5, :2] @ np.linalg.solve(
                    material[:2, :2], material[:2, 3:5]
                )
                expected = offset_links.tilt_stiffness(1, ratio, length, angle, 1)
                np.testing.assert_allclose(tilt, expected*np.eye(2), atol=1e-12)

    print(f"Passed: reduced formula vs full 3D Hessian; {comparisons} exact-energy finite differences;")
    print("static equilibrium; 24 swapped-radius states with rank five and analytical material stiffness.")
    print("Example full elastic stack, theta=45 degrees, E*A/F=1000 per link:")
    for cells in (5, 11, 20):
        original = np.linalg.eigvalsh(full_model(cells, pi/4)[0])[0]
        changed = np.linalg.eigvalsh(full_model(cells, pi/4, sqrt(1+.65**2), inner=.65)[0])[0]
        print(f"  N={cells:2d}: minimum Hessian eigenvalue, original={original:.6f}, changed={changed:.6f}")
    print("Eigenvalues above use R=F=1 and the stated pose coordinates; their signs are the result.")


if __name__ == "__main__":
    verify()
