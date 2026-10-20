"""Continuous rectangular woven straps: exact yarn map and separation checks.

The fabric is an ideal freely shearing orthogonal yarn net. Its warp and
weft lengths are fixed; its crossing angles are allowed to change. This is
not an isometry of a rigid sheet and is not a calibrated fabric stress model.

Each free rectangle has one flexible diagonal fold. A finite-width hairpin
band at every plate joins successive cells with a full-width clamped area.
No cuts, disconnected tips, or zero-width ligaments are required by this map.
"""

import json
from itertools import combinations
from math import acos, asin, atan2, cos, pi, sin, sqrt
from pathlib import Path

import numpy as np


def slice_plane(triangle, distances, direction, tolerance=1e-10):
    values = []
    for i in range(3):
        j = (i+1)%3
        if abs(distances[i]) <= tolerance:
            values.append(float(triangle[i]@direction))
        if distances[i]*distances[j] < 0:
            point = triangle[i]+distances[i]/(distances[i]-distances[j])*(triangle[j]-triangle[i])
            values.append(float(point@direction))
    return (min(values), max(values)) if values else None


def crossing(first, second, tolerance=1e-9):
    normal_a = np.cross(first[1]-first[0], first[2]-first[0])
    normal_b = np.cross(second[1]-second[0], second[2]-second[0])
    normal_a /= np.linalg.norm(normal_a)
    normal_b /= np.linalg.norm(normal_b)
    da, db = (first-second[0])@normal_b, (second-first[0])@normal_a
    if min(da) > tolerance or max(da) < -tolerance or min(db) > tolerance or max(db) < -tolerance:
        return False
    direction = np.cross(normal_a, normal_b)
    norm = np.linalg.norm(direction)
    if norm < tolerance:
        return False  # Coplanar contacts need a separate endpoint/thickness check.
    direction /= norm
    sa, sb = slice_plane(first, da, direction), slice_plane(second, db, direction)
    return sa is not None and sb is not None and min(sa[1], sb[1])-max(sa[0], sb[0]) > tolerance

FACES = np.array([[0, 1, 3], [0, 2, 3]])


def dimensions(ratio):
    return sqrt(1+ratio*ratio-ratio), sqrt(1+ratio*ratio)


def plate(ratio, theta, index):
    height = 0. if theta == pi/2 else sqrt(2*ratio*cos(theta))
    radii = (1., ratio) if index%2 == 0 else (ratio, 1.)
    angles = (index*theta, index*theta+pi/3)
    return np.array([[r*cos(a), r*sin(a), index*height] for r, a in zip(radii, angles)])


def cell(ratio, theta, index=0):
    """A,B on the lower plate and D,C on the upper; material diagonal A-C."""
    width, length = dimensions(ratio)
    vertices = np.vstack((plate(ratio, theta, index), plate(ratio, theta, index+1)))
    a, b, d, c = vertices
    gradients = np.array([np.column_stack(((b-a)/width, (c-b)/length)),
                          np.column_stack(((c-d)/width, (d-a)/length))])
    return vertices, gradients


def point(vertices, gradients, width, length, x, y):
    face = 0 if y/length <= x/width else 1
    return vertices[0]+gradients[face]@np.array([x, y])


def bridge(ratio, theta, index, band, x, u):
    """Two equal layers of a finite material band lie flat under the clamp.

    u=0 and u=band meet the same plate attachment seam. The mid-band fold
    turns the fabric back by 180 degrees. All lengths are normalized by R.
    """
    width, _ = dimensions(ratio)
    a, b = plate(ratio, theta, index)
    tangent = (b-a)/width
    inward = np.array([-tangent[1], tangent[0], 0.])
    if inward@(a+b) > 0:
        inward *= -1
    return a+x*tangent+min(u, band-u)*inward


def collisions(vertices):
    triangles = []
    for strap in range(3):
        angle = 2*pi*strap/3
        rotation = np.array([[cos(angle), -sin(angle), 0.],
                             [sin(angle), cos(angle), 0.], [0., 0., 1.]])
        triangles.extend(vertices[FACES]@rotation.T)
    return [[i, j] for i, j in combinations(range(6), 2)
            if i//2 != j//2 and crossing(triangles[i], triangles[j])]


def maximum_shear(ratio):
    """Exact extrema of the two cell-parity shear functions on 0..90 deg."""
    width, length = dimensions(ratio)
    rows = []
    for parity in (0, 1):
        a, b = (1., ratio) if parity == 0 else (ratio, 1.)
        candidates = [0., pi/2]
        critical = atan2(a*a*sin(pi/3), a*b-a*a*cos(pi/3))
        if 0 < critical < pi/2:
            candidates.append(critical)
        values = []
        for theta in candidates:
            value = (a*b*(cos(theta)+cos(pi/3))-b*b-a*a*cos(theta+pi/3))/(width*length)
            values.append((abs(asin(value)), theta, value))
        shear, theta, value = max(values)
        rows.append(dict(parity=parity, shear_degrees=shear*180/pi,
                         twist_degrees=theta*180/pi, metric_off_diagonal=value,
                         minimum_yarn_angle_degrees=acos(abs(value))*180/pi,
                         area_ratio_at_maximum_shear=sqrt(1-value*value)))
    return rows


def verify(ratio=.8, band=.1, cells=12, samples=2001):
    width, length = dimensions(ratio)
    assert 3/8 < ratio < 1
    error = dict(yarn_metric=0., equal_triangle_metric=0., crease=0.,
                 yarn_length=0., cell_seam=0., bridge_join=0., bridge_yarn_length=0.)
    maximum_angle = 0.
    minimum_triangle_area = float('inf')
    for theta in np.linspace(0., pi/2, samples):
        for parity in (0, 1):
            vertices, gradients = cell(ratio, theta, parity)
            metric = np.einsum('vji,vjk->vik', gradients, gradients)
            error['yarn_metric'] = max(error['yarn_metric'], float(np.max(np.abs(np.diagonal(metric, axis1=1, axis2=2)-1))))
            error['equal_triangle_metric'] = max(error['equal_triangle_metric'], float(np.max(np.abs(metric[0]-metric[1]))))
            maximum_angle = max(maximum_angle, float(np.max(np.abs(np.arcsin(metric[:, 0, 1])))))
            minimum_triangle_area = min(minimum_triangle_area, float(np.min(np.sqrt(np.linalg.det(metric)))))
            for fraction in np.linspace(0, 1, 9):
                xy = np.array([fraction*width, fraction*length])
                error['crease'] = max(error['crease'], float(np.linalg.norm((gradients[0]-gradients[1])@xy)))
                warp = [point(vertices, gradients, width, length, fraction*width, y)
                        for y in (0., fraction*length, length)]
                weft = [point(vertices, gradients, width, length, x, fraction*length)
                        for x in (0., fraction*width, width)]
                error['yarn_length'] = max(error['yarn_length'],
                    abs(float(np.linalg.norm(np.diff(warp, axis=0), axis=1).sum())-length),
                    abs(float(np.linalg.norm(np.diff(weft, axis=0), axis=1).sum())-width))
            if theta < pi/2:
                assert not collisions(vertices), (parity, theta)
            # Cell faces are convex combinations of their four anchors: their
            # height and radius bounds follow exactly, not only from sampling.
            assert np.max(np.linalg.norm(vertices[:, :2], axis=1)) < 1+1e-12
        # Verify actual alternating plate seams over the complete chain.
        for index in range(cells):
            vertices, _ = cell(ratio, theta, index)
            following, _ = cell(ratio, theta, index+1)
            error['cell_seam'] = max(error['cell_seam'], float(np.max(np.abs(vertices[2:]-following[:2]))))
    # The bridge rule is independent of twist up to an in-plane rotation.
    bridge_angles = []
    for parity in (0, 1):
        a, b = plate(ratio, 0., parity)
        outline = []
        for x in (0., width):
            for u in (0., band/2, band):
                p = bridge(ratio, 0., parity, band, x, u)
                assert np.linalg.norm(p[:2]) <= 1+1e-12
                if u in (0., band):
                    expected = a+x*(b-a)/width
                    error['bridge_join'] = max(error['bridge_join'], float(np.linalg.norm(p-expected)))
                outline.append(p)
        angles = np.unwrap(np.arctan2(np.array(outline)[:, 1], np.array(outline)[:, 0]))
        span = float(np.ptp(angles))
        assert span < 2*pi/3
        bridge_angles.append(span*180/pi)
        for fraction in np.linspace(0, 1, 9):
            warp = [bridge(ratio, 0., parity, band, fraction*width, u)
                    for u in (0., band/2, band)]
            weft = [bridge(ratio, 0., parity, band, x, fraction*band) for x in (0., width)]
            error['bridge_yarn_length'] = max(error['bridge_yarn_length'],
                abs(float(np.linalg.norm(np.diff(warp, axis=0), axis=1).sum())-band),
                abs(float(np.linalg.norm(weft[1]-weft[0]))-width))
    assert max(error.values()) < 2e-12
    maximum = maximum_shear(ratio)
    assert abs(maximum_angle-max(row['shear_degrees'] for row in maximum)*pi/180) < 1e-7
    return dict(ratio=ratio, width_over_radius=width, free_length_over_radius=length,
                straight_height_over_radius=sqrt(2*ratio), cells=cells,
                clamp_band_over_radius=band, clamp_footprint_depth_over_radius=band/2,
                raw_length_over_radius=cells*length+(cells+1)*band,
                samples_per_parity=samples, maximum_errors=error,
                exact_shear_extrema=maximum, minimum_sampled_area_ratio=minimum_triangle_area,
                bridge_angular_spans_degrees=bridge_angles,
                continuous_collision_proof=dict(condition='3/8 < r/R < 1; 0 <= theta < 90 degrees',
                    explanation='At each height each strap is two segments L-E-R. All three endpoints '
                    'lie in an angular sector strictly narrower than 120 degrees. The other straps '
                    'are rotations by 120 and 240 degrees. Adjacent cells occupy disjoint open height slabs.',
                    even_E_dot_endpoint_lower_bound='u^2-u*v*sin(phi)+r*v^2*cos(phi), and the reversed u/v form',
                    positive_discriminant_condition='4*r*cos(phi)>sin(phi)^2; phi=60 degrees gives r>3/8',
                    even_edge_angle_bound='phi+2*atan((1-r)/(1+r)*tan(theta/2)) < 120 degrees',
                    odd_E_dot_endpoint_lower_bound='r*u^2-r*u*v*sin(phi)+v^2*cos(phi), and the reversed u/v form',
                    odd_positive_condition='4*cos(phi)>r*sin(phi)^2; holds for 0<r<1',
                    odd_edge_angle_bound='0 < angle(R)-angle(L) <= phi = 60 degrees when r>3/8'),
                limitations=['Ideal zero-thickness geometry and freely rotating yarn crossings.',
                             'Finite shear stiffness, weave locking, yarn slip, creep, clamp pressure and wear are not modeled.',
                             'A fixed-length diagonal stitch or tape would obstruct this motion.',
                             'The prescribed crease branch is geometrically possible; passive shape selection is not established.',
                             'Exactly flat is unloaded storage; finite fabric/plate thickness requires a nonzero folded gap.'])


if __name__ == '__main__':
    result = verify()
    result['millimeters_at_R100'] = {key: result[source]*100 for key, source in
        (('width', 'width_over_radius'), ('free_length', 'free_length_over_radius'),
         ('clamp_band', 'clamp_band_over_radius'), ('raw_length', 'raw_length_over_radius'))}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
