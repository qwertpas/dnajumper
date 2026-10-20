"""Independent audit of the triangular slit prism with height R and side sqrt(3) R.

The coordinate solution here solves the unsimplified hinge equation directly,
independently of rigid_prism.mesh. Caps are included in every collision check.
Run: python3.11 -m trussbuckling.research.verification.triangle
"""

import json
from pathlib import Path

import mpmath as mp
import numpy as np

from .square import intersections
from .sweep import coplanar_overlap


def mesh(height):
    """Unit radius and height in [0, 1]; solve h*p-X*q=D, p*p+q*q=1."""
    a, b, c = .5, np.sqrt(3)/2, .5
    cosine = (1+height*height)/2
    sine = np.sqrt(max(0., 1-cosine*cosine))
    x = a*(cosine-1)+b*sine
    y = a*sine+b*(1-cosine)
    d = 2*c-(b/c)*y
    scale = height*height+x*x
    root = np.sqrt(max(0., scale-d*d))
    q = (-d*x+height*root)/scale
    p = (d*height+x*root)/scale
    if height == 0:
        q, p = 1., 0.
    angles = np.arange(3)*2*np.pi/3
    axes = np.column_stack((np.cos(angles), np.sin(angles)))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    bottom = np.column_stack((a*axes[:, 0]+b*axes[:, 1],
                              a*axes[:, 1]-b*axes[:, 0], np.zeros(3)))
    top = bottom.copy()
    top[:, :2] = bottom[:, :2] @ rotation.T
    top[:, 2] = height
    lower = np.column_stack(((a-c*q)*axes, np.full(3, c*p)))
    upper = lower.copy()
    upper[:, :2] = lower[:, :2] @ rotation.T
    upper[:, 2] = height-lower[:, 2]
    faces = []
    for k in range(3):
        j = (k+1) % 3
        faces += [(k, j, 6+k), (k, 3+k, 6+k),
                  (3+k, 3+j, 9+k), (j, 3+j, 9+k)]
    faces += [(0, 1, 2), (3, 4, 5)]
    return np.concatenate((bottom, top, lower, upper)), np.array(faces)


def precise_intersections(height, pairs):
    """Recheck near-contact intersection candidates using 80-digit arithmetic."""
    with mp.workdps(80):
        h = mp.mpf(str(height))
        a, b = mp.mpf('.5'), mp.sqrt(3)/2
        cosine, sine = (1+h*h)/2, mp.sqrt(1-((1+h*h)/2)**2)
        x, y = a*(cosine-1)+b*sine, a*sine+b*(1-cosine)
        d, scale = 1-2*b*y, h*h+x*x
        root = mp.sqrt(scale-d*d)
        q, p = (-d*x+h*root)/scale, (d*h+x*root)/scale
        axes = [(mp.cos(2*k*mp.pi/3), mp.sin(2*k*mp.pi/3)) for k in range(3)]
        bottom = [mp.matrix([a*x+b*y, a*y-b*x, 0]) for x, y in axes]
        top = [mp.matrix([cosine*v[0]-sine*v[1], sine*v[0]+cosine*v[1], h])
               for v in bottom]
        lower = [mp.matrix([a*(1-q)*x, a*(1-q)*y, a*p]) for x, y in axes]
        upper = [mp.matrix([cosine*v[0]-sine*v[1], sine*v[0]+cosine*v[1], h-v[2]])
                 for v in lower]
        vertices, faces = bottom+top+lower+upper, mesh(height)[1]

        def cross(u, v):
            return mp.matrix([u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
                              u[0]*v[1]-u[1]*v[0]])

        found, eps = [], mp.mpf('1e-60')
        for i, j in pairs:
            hit = False
            for source, target in ((i, j), (j, i)):
                a, b, c = (vertices[k] for k in faces[target])
                e, f = b-a, c-a
                for edge in range(3):
                    start = vertices[faces[source, edge]]
                    direction = vertices[faces[source, (edge+1) % 3]]-start
                    delta, v1 = start-a, cross(direction, f)
                    det = mp.fdot(e, v1)
                    if abs(det) <= eps:
                        continue
                    v2 = cross(delta, e)
                    u, v, t = mp.fdot(delta, v1)/det, mp.fdot(direction, v2)/det, mp.fdot(f, v2)/det
                    if min(u, v, t) > eps and max(u+v, t) < 1-eps:
                        hit = True
            if hit:
                found.append((int(i), int(j)))
        return found


def verify(samples=1001):
    original, faces = mesh(1.)
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(original[edges[:, 0]]-original[edges[:, 1]], axis=1)
    errors = np.zeros(3)
    crossings, overlaps, precise_checks = [], [], 0
    heights = np.unique(np.r_[0., np.linspace(.001, .999, samples),
                              np.logspace(-6, -3, 100), 1-np.logspace(-6, -3, 100), 1.])
    for height in heights:
        vertices, _ = mesh(height)
        current = np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)
        errors = np.maximum(errors, [np.max(np.abs(current/lengths-1)),
            abs(np.max(np.linalg.norm(vertices[:, :2], axis=1))-1),
            max(-vertices[:, 2].min(), vertices[:, 2].max()-height)])
        if 0 < height < 1:
            hit = intersections(vertices, faces)
            if hit:
                precise_checks += len(hit)
                hit = precise_intersections(float(height), hit)
                if hit:
                    crossings.append({'height': float(height), 'pairs': hit})
            overlap = coplanar_overlap(vertices, faces)
            if overlap:
                overlaps.append({'height': float(height), 'pairs': overlap})
    result = dict(sides=3, initial_height_over_radius=1, final_rotation_degrees=60,
                  states=len(heights), vertices=len(original), triangles_with_caps=len(faces),
                  maximum_relative_edge_error=errors[0], maximum_radius_error=errors[1],
                  maximum_axial_error=errors[2], transverse_crossings=crossings,
                  coplanar_overlaps=overlaps, near_contact_pairs_checked_at_80_digits=precise_checks,
                  collision_check='Numerical sampling; continuous nonintersection is proved separately.',
                  endpoint_contact='The flat endpoint permits face-to-face stacking.')
    assert np.max(errors) < 1e-12
    assert not crossings and not overlaps
    return result


if __name__ == '__main__':
    result = verify()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
