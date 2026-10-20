"""Exact open, four-sector fabric fan with three concurrent fold axes.

The corner is a swiveling load attachment. The fan is an unloaded connection
between two planar pieces; this model makes no collision or strength claim.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


RAYS = np.deg2rad([0.0, 45.0, 135.0, 225.0, 270.0])
FLAT = np.vstack((np.zeros(3), np.column_stack((
    np.cos(RAYS), np.sin(RAYS), np.zeros(5)
))))
FACES = np.array([[0, i, i + 1] for i in range(1, 5)])
BASIS = np.column_stack((FLAT[2], FLAT[3], [0.0, 0.0, 1.0]))


def fan_mesh(center, first, last, radius=1.0, end_angle=45.0, branch=0):
    """Join frames ``first`` and ``last`` at ``center`` by an isometric fan.

    Frames are 3x3 proper rotation matrices in world coordinates. Their columns
    map the same flat xy pattern into each endpoint facet. ``radius`` is the
    distance from the common corner to every outer vertex, in caller units.
    ``end_angle`` is each end sector's angle in degrees; the two middle sectors
    are 90 degrees. ``branch=1`` selects the other equivalent Euler solution.
    The returned triangles can fold through each other; check contact separately.
    """
    center = np.asarray(center, dtype=float)
    first, last = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
    if radius <= 0 or center.shape != (3,):
        raise ValueError("Use a positive radius and a three-coordinate center")
    for frame in (first, last):
        if (frame.shape != (3, 3)
                or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10, rtol=0)
                or abs(np.linalg.det(frame) - 1.0) > 1e-10):
            raise ValueError("Endpoint frames must be proper rotation matrices")
    rays = np.deg2rad([0, end_angle, end_angle+90, end_angle+180, 2*end_angle+180])
    pattern = np.vstack((np.zeros(3), np.column_stack((np.cos(rays), np.sin(rays), np.zeros(5)))))
    basis = np.column_stack((pattern[2], pattern[3], [0, 0, 1]))
    target = basis.T @ first.T @ last @ basis
    # At an Euler singularity scipy chooses one of the equivalent valid folds.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        alpha, beta, gamma = Rotation.from_matrix(target).as_euler("XYX")
    if branch:
        alpha, beta, gamma = alpha+np.pi, -beta, gamma+np.pi
    one = Rotation.from_rotvec(alpha * pattern[2]).as_matrix()
    two = one @ Rotation.from_rotvec(beta * pattern[3]).as_matrix()
    three = two @ Rotation.from_rotvec(-gamma * pattern[4]).as_matrix()
    rotations = np.array([first, first @ one, first @ two, first @ three])
    flat = radius * pattern
    # Each shared ray is unchanged by the rotation about that ray.
    vertices = np.array([
        center,
        center + rotations[0] @ flat[1],
        center + rotations[0] @ flat[2],
        center + rotations[1] @ flat[3],
        center + rotations[2] @ flat[4],
        center + rotations[3] @ flat[5],
    ])
    return {
        "vertices": vertices,
        "faces": FACES.copy(),
        "flat": flat,
        "folds": np.array([alpha, beta, -gamma]),
        "rotations": rotations,
    }


def verify(samples=2000):
    """Check arbitrary endpoint frames, every triangle metric and every seam."""
    rng = np.random.default_rng(1492)
    pairs = list(zip(Rotation.random(samples, rng=rng).as_matrix(),
                     Rotation.random(samples, rng=rng).as_matrix()))
    # Include exact Euler singularities, which still have valid folded meshes.
    for beta in (0.0, np.pi):
        for alpha, gamma in ((0.0, 0.0), (0.37, -0.91), (np.pi, np.pi)):
            last = BASIS @ Rotation.from_euler("XYX", [alpha, beta, gamma]).as_matrix() @ BASIS.T
            pairs.append((np.eye(3), last))
    errors = dict(metric=0.0, seam=0.0, endpoint=0.0, edge=0.0)
    for first, last in pairs:
        center = rng.uniform(-3, 3, 3)
        mesh = fan_mesh(center, first, last, 2.3)
        for triangle, frame in zip(mesh["faces"], mesh["rotations"]):
            flat = mesh["flat"][triangle]
            folded = mesh["vertices"][triangle]
            errors["seam"] = max(errors["seam"], float(np.max(np.abs(
                folded - (flat @ frame.T + center)))))
            a, b = flat[1:] - flat[0], folded[1:] - folded[0]
            errors["metric"] = max(errors["metric"], float(np.max(np.abs(a @ a.T - b @ b.T))))
            for i, j in ((0, 1), (1, 2), (2, 0)):
                errors["edge"] = max(errors["edge"], abs(float(
                    np.linalg.norm(flat[i] - flat[j]) - np.linalg.norm(folded[i] - folded[j]))))
        errors["endpoint"] = max(errors["endpoint"], float(np.max(np.abs(
            mesh["rotations"][-1] - last))))
    assert max(errors.values()) < 1e-10, errors
    return {"poses": len(pairs), "max_errors": errors,
            "contact_checked": False,
            "note": "Exact zero-thickness metric and orientation checks; no strength or collision claim."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
