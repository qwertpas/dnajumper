"""Exact rigid FP-ori prism with diagonal slits and fixed outer radius.

Based on Liu et al. (2024), doi:10.1038/s41598-023-50353-4.
Ideal faces are square for n>=4 and height R, width sqrt(3)R for n=3.
Other aspect ratios stop at panel contact or at the end-plane boundary.
The explicit hinge solutions are derived in README.md. Thickness is zero.
"""

from math import asin, cos, isfinite, pi, sin, sqrt, tan
from numbers import Integral

import numpy as np


def ideal_aspect(sides):
    """Initial height / face width for complete folding without crossings."""
    if isinstance(sides, bool) or not isinstance(sides, Integral) or sides < 3:
        raise ValueError("Use an integer number of sides >= 3.")
    return 1 / sqrt(3) if sides == 3 else 1.0


def cell_height(radius, sides, aspect=None):
    """Height of the straight prism; aspect means initial height / face width."""
    ideal = ideal_aspect(sides)
    aspect = ideal if aspect is None else aspect
    if not isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be positive and finite.")
    if not isfinite(aspect) or aspect <= 0:
        raise ValueError("Aspect ratio must be positive and finite.")
    return 2 * radius * sin(pi / sides) * aspect


def fold_limit(sides, aspect=None):
    """Minimum height fraction while panels stay between the two end planes.

    Tall faces stop at panel contact. Short faces reach an end plane, which
    is contact only if a cap is present. The ideal ratio reaches a flat stack.
    These are analytical geometric limits, not an elastic calculation.
    """
    ideal = ideal_aspect(sides)
    aspect = ideal if aspect is None else aspect
    if not isfinite(aspect) or aspect <= 0:
        raise ValueError("Aspect ratio must be positive and finite.")
    if aspect == ideal:
        return 0.0, "flat"
    if aspect > ideal:
        return sqrt((aspect - ideal) * (aspect + ideal)) / aspect, "panel_contact"
    # p=0 in the hinge equation gives this quadratic in u=tan(theta/2).
    a = 1 / tan(pi / sides)
    root = sqrt((a - aspect)**2 + 4*aspect**2*(1 + a*aspect - aspect**2))
    u = 2*aspect**2 / (a - aspect + root)
    chord = sqrt(a*a + 1) * u / (aspect * sqrt(1 + u*u))
    return sqrt(max(0.0, (1 - chord) * (1 + chord))), "end_plane"


def mesh(radius=100.0, sides=6, fraction=1.0, aspect=None):
    """Return vertices, triangle indices, slit edges, and top rotation (radians).

    fraction is height / cell_height, from 1 (straight prism) to 0 (flat).
    Vertex groups are bottom B, top T, and the two free slit centers J and I.
    Rigid end polygons constrain B and T; their cap panels are not drawn.
    """
    ideal = ideal_aspect(sides)
    aspect = ideal if aspect is None else aspect
    maximum = cell_height(radius, sides, aspect)
    if not isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Height fraction must be between 0 and 1.")
    minimum, _ = fold_limit(sides, aspect)
    if fraction < minimum:
        raise ValueError(f"Height fraction must be at least {minimum:.8g} for this aspect ratio.")
    alpha = pi / sides
    side = 2 * radius * sin(alpha)
    height = maximum * fraction
    a, b = radius * cos(alpha), side / 2
    half_height = maximum / 2
    theta = 2 * asin(maximum / (2 * radius) * sqrt(1 - fraction**2))
    # t=tan(phi/2) gives the inward-folding branch without a constraint solver.
    if aspect != ideal:
        # hp-Xq=D enforces the last edge of each triangle; p^2+q^2=1.
        # Work at unit radius to avoid scaling the discriminant with dimensions.
        h, c = height/radius, half_height/radius
        x = (a/radius)*(cos(theta)-1) + (b/radius)*sin(theta)
        y = (a/radius)*sin(theta) + (b/radius)*(1-cos(theta))
        d = 2*c - (b/radius)*y/c
        t = (d+x) / (h + sqrt(max(0.0, h*h+x*x-d*d)))
    elif sides == 3:
        # Rectangular faces: h^2/R^2 = 2*cos(theta)-1, flat at theta=60 degrees.
        t = fraction / (1 + sqrt(1 - cos(theta) + sqrt(3) * sin(theta)))
    else:
        d, u = 1 / tan(alpha), tan(theta / 2)
        # Replacing sqrt(1-d*u) by a multiple of fraction is stable near flat.
        t = fraction * sqrt((1 + u*u) / (1 + d*u)) * (1 + u) / (
            sqrt((1 + d * u) * (1 + u*u)) + sqrt(2 * u * (d + u))
        )
    q, p = (1 - t*t) / (1 + t*t), 2*t / (1 + t*t)
    angles = np.arange(sides) * 2 * alpha
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    tangents = np.column_stack((-np.sin(angles), np.cos(angles)))
    rotation = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    bottom = np.column_stack((a * normals - b * tangents, np.zeros(sides)))
    top = np.column_stack((bottom[:, :2] @ rotation.T, np.full(sides, height)))
    lower = np.column_stack(((a - half_height*q) * normals, np.full(sides, half_height*p)))
    upper = np.column_stack((lower[:, :2] @ rotation.T, np.full(sides, height - half_height*p)))
    vertices = np.vstack((bottom, top, lower, upper))
    faces, cuts = [], []
    for k in range(sides):
        j = (k + 1) % sides
        faces += [(k, j, 2*sides+k), (k, sides+k, 2*sides+k),
                  (sides+k, sides+j, 3*sides+k), (j, sides+j, 3*sides+k)]
        cuts += [(j, 2*sides+k), (2*sides+k, sides+k),
                 (j, 3*sides+k), (3*sides+k, sides+k)]
    return vertices, np.asarray(faces), np.asarray(cuts), theta


def panel_lengths(vertices, faces):
    """All three side lengths of every triangle, independent of indexing order."""
    triangles = vertices[faces]
    return np.sort(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2), axis=1)


def reference_fraction(radius, sides, length, cells=1, aspect=None):
    """Starting fraction for cells stacked axially; each cell has the same fold."""
    maximum = cells * cell_height(radius, sides, aspect)
    if not 0 < length <= maximum:
        raise ValueError(f"Reference length must be in (0, {maximum:g}] for {cells} cells.")
    if length / maximum < fold_limit(sides, aspect)[0]:
        raise ValueError("Reference length is below this pattern's folding limit.")
    return length / maximum
