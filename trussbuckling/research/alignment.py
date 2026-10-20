"""Tilt stability of a stack of rigid plates joined by tensile corner links.

Assumptions: identical cells, equal corresponding attachment radii, uniform
twist, positive axial tension, straight taut links, and fixed end centers and
end normals. The end distance may respond to the axial load. Fabric bending,
friction, slack, contact, and unequal link tensions are not represented.

This is the tangent stability limit of links with large but finite axial
stiffness. Perfectly inextensible links at this singular configuration require
additional second-order compatibility; a negative tangent eigenvalue alone
does not prove an exact, all-taut, strain-free bending path.

Let w[j] be the small horizontal rotation vector of plate j, and write
w[j] = rotation(j*angle) @ u[j]. End tilts u[0] = u[N] = 0. First-order link
length preservation determines horizontal center displacements by

  p[j+1]-p[j] = h/(2*sin(angle/2)) *
      (rotation(-angle/2)@w[j+1] - rotation(angle/2)@w[j]).

The fixed end centers impose sum(w[1:N]) = 0. On these motions the quadratic
potential energy, including initial cable tension, is

  energy = force/(4*h) * (b*sum(|u[j+1]-u[j]|**2)
                          - 2*radius**2*d*sum(|u[j]|**2)),
  d = 1-cos(angle), b = h**2/d + radius**2,
  h**2 = length**2 - 2*radius**2*d.

The positive term resists changes in tilt between plates. The negative term
comes from transmitted torque. Long stacks admit slowly varying tilted modes
that make the positive term small. Positive eigenvalues mean local stability
within this model; zero means marginal; negative means instability.
"""

from math import cos, sin, sqrt

import numpy as np
from scipy.linalg import null_space


def rotation(angle):
    """Two-dimensional rotation; angles throughout are radians."""
    return np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


def tilt_matrix(cells, angle, length=sqrt(2), radius=1.0, force=1.0):
    """Return the tilt Hessian, its admissible basis, and the cell height.

    The basis maps reduced coordinates to u[1], ..., u[cells-1]. Multiplying
    the Hessian by force changes stiffness, but not the stability boundary.
    At least three cells are needed for a length-preserving transverse mode
    with both end plates fixed in direction and horizontal position.
    """
    if cells < 3 or int(cells) != cells:
        raise ValueError("Use an integer number of cells >= 3.")
    if min(length, radius, force) <= 0 or not 0 < angle < np.pi:
        raise ValueError("Use positive lengths/load and 0 < angle < pi.")
    d = 2 * sin(angle / 2)**2
    height_squared = length**2 - 2 * radius**2 * d
    if height_squared <= 0:
        raise ValueError("The chosen twist has no positive cell height.")
    height = sqrt(height_squared)
    count = cells - 1
    laplace = 2 * np.eye(count) - np.eye(count, k=1) - np.eye(count, k=-1)
    end_constraint = np.hstack([rotation(j * angle) for j in range(1, cells)])
    basis = null_space(end_constraint)
    b = height_squared / d + radius**2
    full = force / (2 * height) * (
        b * np.kron(laplace, np.eye(2)) - 2 * radius**2 * d * np.eye(2 * count)
    )
    return basis.T @ full @ basis, basis, height


def lowest_mode(cells, angle, length=sqrt(2), radius=1.0, force=1.0):
    """Return smallest tilt stiffness, plate tilts, and horizontal offsets.

    The returned mode is an infinitesimal direction, NOT a finite deformed
    configuration satisfying all cable lengths. Its largest plate tilt is 1.
    """
    matrix, basis, height = tilt_matrix(cells, angle, length, radius, force)
    values, vectors = np.linalg.eigh(matrix)
    local = (basis @ vectors[:, 0]).reshape(cells - 1, 2)
    tilts = np.zeros((cells + 1, 2))
    for j in range(1, cells):
        tilts[j] = rotation(j * angle) @ local[j - 1]
    tilts /= np.linalg.norm(tilts, axis=1).max()
    centers = np.zeros_like(tilts)
    for j in range(cells):
        centers[j + 1] = centers[j] + height / (2 * sin(angle / 2)) * (
            rotation(-angle / 2) @ tilts[j + 1]
            - rotation(angle / 2) @ tilts[j]
        )
    return values[0], tilts, centers


def minimum_length(cells, angle, radius=1.0):
    """Minimum link length for positive tilt stiffness at a prescribed twist.

    Also enforce the separate physical condition length > 2*R*sin(angle/2).
    Equality here is a marginal mode, so a design needs a positive margin.
    """
    count = cells - 1
    laplace = 2 * np.eye(count) - np.eye(count, k=1) - np.eye(count, k=-1)
    end_constraint = np.hstack([rotation(j * angle) for j in range(1, cells)])
    basis = null_space(end_constraint)
    value = np.linalg.eigvalsh(basis.T @ np.kron(laplace, np.eye(2)) @ basis)[0]
    d = 2 * sin(angle / 2)**2
    return radius * sqrt(d + 2 * d*d / value)


if __name__ == "__main__":
    from scipy.optimize import brentq

    print("Square cells: L/R = sqrt(2), fixed end centers and normals")
    print("cells  critical twist/cell (degrees)")
    for cells in (3, 4, 5, 6, 8, 10, 12, 16, 20):
        def value(angle):
            return np.linalg.eigvalsh(tilt_matrix(cells, angle)[0])[0]
        if value(np.pi/2 - 1e-5) > 0:
            print(f"{cells:5d}  none before 90 degrees")
        else:
            angle = brentq(value, 0.01, np.pi/2 - 1e-5)
            print(f"{cells:5d}  {np.rad2deg(angle):8.3f}")
