"""Three continuous routed strips with flat load diamonds and open fan joints."""

import json
from math import cos, sin, sqrt, pi
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .fan import fan_mesh, FLAT, FACES
from ..manufacturable_panels.woven_check import crossing

RADIUS = 100.0
RATIO = 0.65
OFFSET = np.deg2rad(28.9)
CELLS = 6
WIDTH = 10.0
FAN_RADIUS = 4.0
LENGTH = RADIUS * sqrt(1 + RATIO**2)
BRIDGE = RADIUS * sqrt(1 + RATIO**2 - 2*RATIO*cos(OFFSET))
LENGTHS = np.array([LENGTH]*CELLS + [BRIDGE] + [LENGTH]*CELLS)


def turn(angle):
    return np.array([[cos(angle), -sin(angle), 0.0],
                     [sin(angle), cos(angle), 0.0], [0.0, 0.0, 1.0]])


def flat_net():
    angles = 2*np.arctan(WIDTH/LENGTHS)
    headings = [0.0]
    starts, signs, ends = [], [], []
    for j in range(1, len(LENGTHS)):
        sign = 1 if j % 2 else -1
        starts.append(headings[-1]+pi+sign*angles[j-1]/2)
        signs.append(sign)
        ends.append(45-np.rad2deg(angles[j-1]+angles[j])/4)
        headings.append(headings[-1]+sign*pi/2)
    directions = np.array([[cos(a), sin(a), 0] for a in headings])
    nodes = np.vstack((np.zeros(3), np.cumsum(LENGTHS[:, None]*directions, axis=0)))
    diamonds = []
    for a, b, u in zip(nodes[:-1], nodes[1:], directions):
        v = np.cross([0, 0, 1], u)
        diamonds.append(np.array([a, (a+b)/2-WIDTH*v/2, b, (a+b)/2+WIDTH*v/2]))
    frames = [turn(start) @ np.diag([1.0, sign, sign]) for start, sign in zip(starts, signs)]
    patterns = [fan_mesh([0, 0, 0], np.eye(3), np.eye(3), FAN_RADIUS, end_angle=end)['flat'] for end in ends]
    fans = [pattern @ frame.T + node for pattern, frame, node in zip(patterns, frames, nodes[1:-1])]
    return dict(nodes=nodes, diamonds=np.array(diamonds), fans=np.array(fans),
                frames=np.array(frames), signs=np.array(signs), directions=directions,
                ends=ends, patterns=patterns)


def world_nodes(theta):
    """Radians per cell; return the 14-node U route for one strip, in mm."""
    height = RADIUS*sqrt(2*RATIO*cos(theta))
    a, b = [], []
    for i in range(CELLS+1):
        ra, rb = (RADIUS, RATIO*RADIUS) if i % 2 == 0 else (RATIO*RADIUS, RADIUS)
        angle = i*theta
        a.append([ra*cos(angle), ra*sin(angle), i*height])
        b.append([rb*cos(angle+OFFSET), rb*sin(angle+OFFSET), i*height])
    return np.array(a+b[::-1])


def strip_mesh(theta, strip=0):
    """Triangle records contain world/flat arrays, kind and route index."""
    net = flat_net()
    nodes = world_nodes(theta) @ turn(strip*2*pi/3).T
    records, rotations = [], []
    for j, (a, b, flat_u, diamond) in enumerate(zip(nodes[:-1], nodes[1:], net['directions'], net['diamonds'])):
        u = (b-a)/np.linalg.norm(b-a)
        v = np.cross([0., 0., 1.], u)
        v /= np.linalg.norm(v)
        world_frame = np.column_stack((u, v, np.cross(u, v)))
        flat_frame = np.column_stack((flat_u, np.cross([0, 0, 1], flat_u), [0, 0, 1]))
        rotation = world_frame @ flat_frame.T
        rotations.append(rotation)
        folded = (diamond-net['nodes'][j]) @ rotation.T + a
        for face in ([0, 1, 2], [0, 2, 3]):
            records.append(dict(world=folded[face], flat=diamond[face], kind='bridge' if j == CELLS else 'load', index=j, strip=strip))
    for j, (frame, sign) in enumerate(zip(net['frames'], net['signs']), 1):
        fan = fan_mesh(nodes[j], rotations[j-1] @ frame, rotations[j] @ frame, FAN_RADIUS, net['ends'][j-1])
        for face in FACES:
            face = face if sign > 0 else face[::-1]
            records.append(dict(world=fan['vertices'][face], flat=net['fans'][j-1][face], kind='fan', index=j, strip=strip))
    return dict(triangles=records, nodes=nodes, net=net, rotations=np.array(rotations))


def flat_check(net):
    polygons = [Polygon(d[:, :2]) for d in net['diamonds']]
    polygons += [Polygon(f[face, :2]) for f in net['fans'] for face in FACES]
    area = sum(p.area for p in polygons)
    union = unary_union(polygons)
    overlap = area-union.area
    # Floating-point rotations can leave microscopic cracks between seams.
    connected = union.buffer(1e-8).geom_type == 'Polygon'
    return dict(overlap_area_mm2=float(overlap), connected=connected,
                bounds_mm=list(union.bounds), material_area_mm2=float(area))


def intersections(records):
    hits = []
    bounds = np.array([r['world'] for r in records])
    lower, upper = bounds.min(axis=1), bounds.max(axis=1)
    candidates = np.all(upper[:, None, :] >= lower[None, :, :]-1e-8, axis=2)
    candidates &= candidates.T
    for i, j in zip(*np.where(np.triu(candidates, 1))):
        a, b = records[i], records[j]
        if a['kind'] == b['kind'] and a['index'] == b['index'] and a['strip'] == b['strip'] and a['kind'] != 'fan':
            continue
        p, q = a['world'], b['world']
        if np.any(p.max(axis=0) < q.min(axis=0)-1e-8) or np.any(q.max(axis=0) < p.min(axis=0)-1e-8):
            continue
        shared = np.any(np.linalg.norm(p[:, None]-q[None, :], axis=-1) < 1e-8, axis=1).sum()
        # Two noncoplanar triangles sharing an edge can only meet on that edge.
        # Near a fully folded hinge, plane-intersection arithmetic is ill-conditioned.
        # Coplanar positive-area contact is still checked even for adjacent facets.
        p = p.mean(axis=0)+(p-p.mean(axis=0))*(1-1e-7)
        q = q.mean(axis=0)+(q-q.mean(axis=0))*(1-1e-7)
        kind = None
        if shared < 2 and crossing(p, q, tolerance=1e-8):
            kind = 'crossing'
        else:
            n = np.cross(p[1]-p[0], p[2]-p[0])
            n /= np.linalg.norm(n)
            if np.max(np.abs((q-p[0])@n)) < 1e-8:
                keep = [k for k in range(3) if k != int(np.argmax(np.abs(n)))]
                if Polygon(p[:, keep]).intersection(Polygon(q[:, keep])).area > 1e-8:
                    kind = 'coplanar'
        if kind:
            hits.append([int(i), int(j), a['kind'], a['index'], b['kind'], b['index'], kind, a['strip'], b['strip']])
    return hits


def write_svg(path):
    net = flat_net()
    polygons = [Polygon(d[:, :2]) for d in net['diamonds']]
    polygons += [Polygon(f[face, :2]) for f in net['fans'] for face in FACES]
    # Snap microscopic numerical seam gaps; this is far below fabrication scale.
    shape = unary_union(polygons).buffer(1e-8).buffer(-1e-8)
    xmin, ymin, xmax, ymax = shape.bounds
    margin = 15.0
    def points(rows):
        return ' '.join(f'{x-xmin+margin:.6f},{y-ymin+margin:.6f}' for x, y, *_ in rows)
    w, h = xmax-xmin+2*margin, ymax-ymin+2*margin
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.6f}mm" height="{h:.6f}mm" viewBox="0 0 {w:.6f} {h:.6f}">',
             '<text x="15" y="6" font-family="sans-serif" font-size="3">Kinematic prototype in mm. Cut black; fold dashed brown; blue marks are pivot centers. Align warp/weft with horizontal/vertical load diagonals.</text>',
             '<text x="15" y="10" font-family="sans-serif" font-size="3">Swivel hardware, finite thickness and clamp allowances require a fabrication coupon. Cut three identical pieces.</text>',
             '<g fill="none" stroke="black" stroke-width="0.15">']
    parts = [shape] if shape.geom_type == 'Polygon' else list(shape.geoms)
    for part in parts:
        lines.append(f'<polygon points="{points(part.exterior.coords)}"/>')
        for hole in part.interiors:
            lines.append(f'<polygon points="{points(hole.coords)}"/>')
    lines.append('</g><g fill="none" stroke="#a04e2d" stroke-width="0.12" stroke-dasharray="1,0.7">')
    for fan in net['fans']:
        for ray in (2, 3, 4):
            lines.append(f'<polyline points="{points(fan[[0, ray]])}"/>')
    lines.append('</g><g fill="#146c91" font-family="sans-serif" font-size="2.5">')
    for i, node in enumerate(net['nodes']):
        x, y, _ = node
        lines.append(f'<circle cx="{x-xmin+margin:.6f}" cy="{y-ymin+margin:.6f}" r="0.5"/>')
        lines.append(f'<text x="{x-xmin+margin+1:.6f}" y="{y-ymin+margin-1:.6f}">{i}</text>')
    lines.append('</g></svg>')
    Path(path).write_text('\n'.join(lines))


def verify(samples=541, minimum=11.0, maximum=65.0):
    metric = seam = lengths = 0.0
    collision_rows = []
    previous = None
    movement = 0.0
    for degrees in np.linspace(minimum, maximum, samples):
        model = strip_mesh(np.deg2rad(degrees))
        local = np.array([row['world']-model['nodes'][row['index']]
                          for row in model['triangles'] if row['kind'] == 'fan'])
        if previous is not None:
            movement = max(movement, float(np.max(np.linalg.norm(local-previous, axis=-1))))
        previous = local
        for record in model['triangles']:
            p, q = record['flat'], record['world']
            p, q = p[1:]-p[0], q[1:]-q[0]
            metric = max(metric, float(np.max(np.abs(p@p.T-q@q.T))))
        lengths = max(lengths, float(np.max(np.abs(np.linalg.norm(np.diff(model['nodes'], axis=0), axis=1)-LENGTHS))))
        # Compare both fan boundary rays with their incident rigid load facet maps.
        for j, frame in enumerate(model['net']['frames'], 1):
            fan = fan_mesh(model['nodes'][j], model['rotations'][j-1]@frame, model['rotations'][j]@frame, FAN_RADIUS, model['net']['ends'][j-1])
            for k, link in ((1, j-1), (5, j)):
                expected = model['nodes'][j] + model['rotations'][link] @ (frame@model['net']['patterns'][j-1][k])
                seam = max(seam, float(np.linalg.norm(fan['vertices'][k]-expected)))
        all_records = model['triangles']
        for strip in (1, 2):
            all_records += strip_mesh(np.deg2rad(degrees), strip)['triangles']
        hits = intersections(all_records)
        if hits:
            collision_rows.append(dict(degrees=float(degrees), count=len(hits), examples=hits[:4]))
    net = flat_check(flat_net())
    assert metric < 1e-7 and seam < 1e-9 and lengths < 1e-9
    assert abs(net['overlap_area_mm2']) < 1e-6 and net['connected'], net
    return dict(samples=samples, angle_range_degrees=[minimum, maximum], strips=3,
                max_metric_error_mm2=metric, max_seam_error_mm=seam,
                max_link_error_mm=lengths, flat_net=net, intersections=collision_rows,
                fan_continuity=dict(sample_step_degrees=(maximum-minimum)/(samples-1),
                                    max_vertex_move_mm=movement),
                load_length_mm=LENGTH, bridge_length_mm=BRIDGE,
                limits='Triangle metrics verified. Sampled triangle crossings and coplanar overlap checked; finite thickness and plate contact untested.')


if __name__ == '__main__':
    output = Path(__file__).parent
    write_svg(output/'routed_strip.svg')
    results = verify()
    (output/'routed_check.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
