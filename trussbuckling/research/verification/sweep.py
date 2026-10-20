"""Independent finite-path checks for the 45-degree slit prism family."""

import json
from pathlib import Path

import numpy as np

from .square import intersections, precise_intersections


def mesh(n, height):
    """Unit outer radius, normalized height in [0,1]."""
    gamma = np.pi / n
    b, a = np.sin(gamma), np.cos(gamma)
    A = a / b
    c = np.cos(2 * gamma) + (1 - np.cos(2 * gamma)) * height**2
    s = np.sqrt(max(0., 1 - c*c))
    u = s / (1 + c)
    # This form avoids subtracting 1-A*u close to the flat endpoint.
    tau = (height * (1 + u) * np.sqrt((1 + u*u) / (1 + A*u))
           / (np.sqrt((1 + A*u) * (1 + u*u)) + np.sqrt(2*u*(A+u))))
    q, p = (1 - tau*tau) / (1 + tau*tau), 2*tau / (1 + tau*tau)
    angles = 2 * gamma * np.arange(n)
    axes = np.column_stack((np.cos(angles), np.sin(angles)))
    rotation = np.array([[c, -s], [s, c]])
    bottom = np.column_stack((a*axes[:, 0]+b*axes[:, 1],
                              a*axes[:, 1]-b*axes[:, 0], np.zeros(n)))
    top = bottom.copy()
    top[:, :2] = bottom[:, :2] @ rotation.T
    top[:, 2] = 2*b*height
    lower = np.column_stack(((a-b*q)*axes, np.full(n, b*p)))
    upper = lower.copy()
    upper[:, :2] = lower[:, :2] @ rotation.T
    upper[:, 2] = 2*b*height - lower[:, 2]
    faces = []
    for k in range(n):
        j = (k + 1) % n
        faces += [(k, j, 2*n+k), (k, n+k, 2*n+k),
                  (n+k, n+j, 3*n+k), (j, n+j, 3*n+k)]
    return np.concatenate((bottom, top, lower, upper)), np.array(faces)


def coplanar_overlap(vertices, faces, tolerance=1e-10):
    """Positive-area overlap of coplanar triangles, excluding edge contact."""
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0],
                       triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    found = []
    for i in range(len(faces)):
        parallel = np.linalg.norm(np.cross(normals[i], normals), axis=1) < tolerance
        distances = np.abs((triangles[:, 0]-triangles[i, 0]) @ normals[i])
        for j in np.flatnonzero(parallel & (distances < tolerance)):
            if j <= i:
                continue
            projection = np.delete(triangles[[i, j]], np.argmax(np.abs(normals[i])), axis=2)
            a, b = projection
            edges = np.concatenate((np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b))
            axes = np.column_stack((-edges[:, 1], edges[:, 0]))
            axes /= np.linalg.norm(axes, axis=1)[:, None]
            pa, pb = a @ axes.T, b @ axes.T
            overlap = np.minimum(pa.max(0), pb.max(0))-np.maximum(pa.min(0), pb.min(0))
            if np.all(overlap > tolerance):
                found.append((i, int(j)))
    return found


def verify(n, samples=401):
    original, faces = mesh(n, 1.)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(original[edges[:, 1]]-original[edges[:, 0]], axis=1)
    error, radius_error, axial_error = 0., 0., 0.
    crossings, overlaps, precise_checks = [], [], 0
    # Interior samples stop clear of the flat-contact singularity. The two
    # endpoints are checked for lengths/envelope only.
    heights = np.r_[0., np.linspace(.0025, .9975, samples), 1.]
    for height in heights:
        vertices, _ = mesh(n, height)
        current = np.linalg.norm(vertices[edges[:, 1]]-vertices[edges[:, 0]], axis=1)
        error = max(error, float(np.max(np.abs(current / lengths - 1))))
        radius_error = max(radius_error, float(abs(np.max(np.linalg.norm(vertices[:, :2], axis=1))-1)))
        axial_error = max(axial_error, float(-vertices[:, 2].min()),
                          float(vertices[:, 2].max()-2*np.sin(np.pi/n)*height))
        if 0 < height < 1:
            hit = intersections(vertices, faces)
            if hit and n == 4:
                precise_checks += len(hit)
                hit = precise_intersections(float(height), hit)
            if hit:
                crossings.append({"height": float(height), "pairs": [[int(i), int(j)] for i, j in hit]})
            overlap = coplanar_overlap(vertices, faces)
            if overlap:
                overlaps.append({"height": float(height), "pairs": overlap})
    return dict(n=n, states=len(heights), maximum_relative_edge_error=error,
                maximum_radius_error=radius_error, maximum_axial_error=axial_error,
                transverse_crossings=crossings, coplanar_overlaps=overlaps,
                near_contact_pairs_checked_at_80_digits=precise_checks)


if __name__ == "__main__":
    results = [verify(n) for n in (4, 5, 6, 8, 12, 17, 24, 32)]
    Path(__file__).with_name("general.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))
