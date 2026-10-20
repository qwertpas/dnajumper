"""Exact slit-cube FP-ori, Liu et al. (2024), DOI 10.1038/s41598-023-50353-4.

This reconstruction uses n=4 and alpha=45 degrees. `height` is the end-ring
separation divided by the square side length. All side panels are rigid right
isosceles triangles. A slit along one diagonal of each original side square
separates its two center copies J and I. Zero-thickness panels touch/overlap in
the fully folded state. A fixed-radius outer envelope is not a closed skin.
"""

import json
from pathlib import Path

import numpy as np
import mpmath as mp


def mesh(height, radius=1.0, caps=True):
    """Return vertices, triangles, and cut-edge pairs for height in [0, 1].

    Vertex groups B,T,J,I have four vertices each. For sector k, the bottom
    square edge is B[k]--B[k+1]. Its slit has two free copies:
    B[k+1]--J[k]--T[k] and B[k+1]--I[k]--T[k].
    """
    if not 0 <= height <= 1 or radius <= 0:
        raise ValueError("Require 0 <= height <= 1 and radius > 0")
    side = radius * np.sqrt(2)
    c = height * height
    s = np.sqrt(max(0.0, 1 - c * c))
    t = height / (1 + np.sqrt(s))
    q, p = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
    rotation = np.array([[c, -s], [s, c]])
    sectors = np.array(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)))
    bottom = side / 2 * np.column_stack((sectors[:, 0] + sectors[:, 1],
                                       sectors[:, 1] - sectors[:, 0],
                                       np.zeros(4)))
    top = bottom.copy()
    top[:, :2] = bottom[:, :2] @ rotation.T
    top[:, 2] = side * height
    lower = np.column_stack((side * (1 - q) / 2 * sectors,
                             np.full(4, side * p / 2)))
    upper = lower.copy()
    upper[:, :2] = lower[:, :2] @ rotation.T
    upper[:, 2] = side * height - lower[:, 2]
    vertices = np.concatenate((bottom, top, lower, upper))
    faces, cuts = [], []
    for k in range(4):
        j = (k + 1) % 4
        faces += [(k, j, 8 + k), (k, 4 + k, 8 + k),
                  (4 + k, 4 + j, 12 + k), (j, 4 + j, 12 + k)]
        cuts += [(j, 8 + k), (8 + k, 4 + k),
                 (j, 12 + k), (12 + k, 4 + k)]
    if caps:
        faces += [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]
    return vertices, np.array(faces), np.array(cuts)


def intersections(vertices, faces, tolerance=1e-10):
    """Find transverse edge/interior-face intersections, excluding contact.

    This diagnostic is numerical, not a continuous collision certificate.
    Coplanar contacts require separate inspection (the endpoint flat stack is
    intentionally coplanar). For interior heights this checks every directed
    triangle-pair edge/triangle test, including pairs sharing vertices.
    """
    triangles = vertices[faces]
    pairs = np.array([(i, j) for i in range(len(faces))
                      for j in range(len(faces)) if i != j])
    a = triangles[pairs[:, 1], 0]
    e = triangles[pairs[:, 1], 1] - a
    f = triangles[pairs[:, 1], 2] - a
    hits = set()
    for edge in range(3):
        start = triangles[pairs[:, 0], edge]
        direction = triangles[pairs[:, 0], (edge + 1) % 3] - start
        cross = np.cross(direction, f)
        det = np.einsum("ij,ij->i", e, cross)
        valid = np.abs(det) > tolerance
        inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
        delta = start - a
        u = np.einsum("ij,ij->i", delta, cross) * inv
        cross2 = np.cross(delta, e)
        v = np.einsum("ij,ij->i", direction, cross2) * inv
        distance = np.einsum("ij,ij->i", f, cross2) * inv
        valid &= ((u > tolerance) & (v > tolerance)
                  & (u + v < 1 - tolerance) & (distance > tolerance)
                  & (distance < 1 - tolerance))
        hits.update(tuple(sorted(pair)) for pair in pairs[valid])
    return sorted(hits)


def precise_intersections(height, pairs):
    """Resolve ill-conditioned near-contact candidates at 80-digit precision."""
    with mp.workdps(80):
        z = mp.mpf(str(height))
        c, s = z * z, mp.sqrt(1 - z**4)
        t = z / (1 + mp.sqrt(s))
        q, p = (1 - t*t) / (1 + t*t), 2*t / (1 + t*t)
        side = mp.sqrt(2)
        sectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
        bottom = [mp.matrix([side*(x+y)/2, side*(y-x)/2, 0]) for x, y in sectors]
        top = [mp.matrix([c*b[0]-s*b[1], s*b[0]+c*b[1], side*z]) for b in bottom]
        lower = [mp.matrix([side*(1-q)*x/2, side*(1-q)*y/2, side*p/2]) for x, y in sectors]
        upper = [mp.matrix([c*b[0]-s*b[1], s*b[0]+c*b[1], side*z-b[2]]) for b in lower]
        vertices, faces = bottom + top + lower + upper, mesh(height)[1]
        def cross(a, b):
            return mp.matrix([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
        found = []
        eps = mp.mpf("1e-60")
        for i, j in pairs:
            hit = False
            for source, target in ((i, j), (j, i)):
                a, b, cc = (vertices[k] for k in faces[target])
                e, f = b-a, cc-a
                for edge in range(3):
                    start = vertices[faces[source, edge]]
                    direction = vertices[faces[source, (edge+1)%3]] - start
                    d = start-a
                    v1 = cross(direction, f)
                    det = mp.fdot(e, v1)
                    if abs(det) <= eps:
                        continue
                    v2 = cross(d, e)
                    u, v, w = mp.fdot(d, v1)/det, mp.fdot(direction, v2)/det, mp.fdot(f, v2)/det
                    if min(u, v, w) > eps and max(u+v, w) < 1-eps:
                        hit = True
            if hit:
                found.append((int(i), int(j)))
        return found


def verify(samples=1001):
    base, faces, cuts = mesh(1.0)
    edge_ids = np.unique(np.sort(np.concatenate((faces[:, [0, 1]],
                                                faces[:, [1, 2]],
                                                faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(base[edge_ids[:, 0]] - base[edge_ids[:, 1]], axis=1)
    max_error, max_radius_error, max_z_outside = 0., 0., 0.
    crossings, precise_checks = [], 0
    for height in np.linspace(0, 1, samples):
        vertices, _, _ = mesh(float(height))
        current = np.linalg.norm(vertices[edge_ids[:, 0]] - vertices[edge_ids[:, 1]], axis=1)
        max_error = max(max_error, float(np.max(np.abs(current / lengths - 1))))
        radii = np.linalg.norm(vertices[:, :2], axis=1)
        max_radius_error = max(max_radius_error, float(abs(np.max(radii) - 1)))
        max_z_outside = max(max_z_outside, float(-np.min(vertices[:, 2])),
                            float(np.max(vertices[:, 2]) - height * np.sqrt(2)))
        if 0 < height < 1:
            hits = intersections(vertices, faces)
            if hits:
                precise_checks += len(hits)
                hits = precise_intersections(float(height), hits)
                if hits:
                    crossings.append({"height": float(height), "pairs": hits})
    result = {"samples": samples, "vertices": len(base), "triangles_with_caps": len(faces),
              "unique_edges": len(edge_ids), "slit_free_edges": len(cuts),
              "maximum_relative_edge_error": max_error,
              "maximum_outer_radius_error": max_radius_error,
              "maximum_axial_envelope_violation": max_z_outside,
              "transverse_intersections": crossings,
              "near_contact_pairs_checked_at_80_digits": precise_checks,
              "endpoint_contacts": "Flat lamination at height=0; slit lips coincide at height=1",
              "collision_check": "Numerical transverse-intersection sampling; not a continuous certificate"}
    assert max_error < 1e-12 and max_radius_error < 1e-12 and max_z_outside < 1e-12
    assert not crossings
    return result


if __name__ == "__main__":
    result = verify()
    Path(__file__).with_name("square.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
