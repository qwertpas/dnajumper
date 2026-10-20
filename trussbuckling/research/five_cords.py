"""Exact five-cord geometry and alignment stiffness for design optimization.

R=F=1. All bottom/top radius products equal p; corresponding anchors have the
same azimuth in the untwisted state. Therefore all five fixed cord lengths
permit the same h(theta), even though their lengths need not be equal.

Positive axial-load shares c solve two weighted-centroid equations plus sum(c)=1.
Then T_i = c_i L_i/h and motor torque = p*sin(theta)/h. Swapping the radius
arrays on alternating plates gives a stack with the same force balance.

The alignment metric is the least eigenvalue of the transverse stiffness after
minimizing energy over all free axial translations and internal twists. Both
translations/R and rotations in radians enter the transverse norm. This is a
local, prestressed elastic metric, not a nonlinear buckling simulation.
"""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np
from scipy.linalg import eigh, solve


@dataclass
class Design:
    angles: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    shares: np.ndarray
    product: float


def make_design(angles, bottom, product=.6):
    """Angles in radians, radii in units of the maximum outer radius."""
    angles, bottom = np.asarray(angles, float), np.asarray(bottom, float)
    top = product/bottom
    ux, uy = np.cos(angles), np.sin(angles)
    balance = np.array([bottom*ux, bottom*uy, top*ux, top*uy, np.ones(len(bottom))])
    shares = np.linalg.solve(balance, [0., 0., 0., 0., 1.])
    return Design(angles, bottom, top, shares, product)


def pentagon():
    """Balanced closed-form seed attaining min/max radius = 0.6."""
    angles = np.arange(5)*2*pi/5
    wave = .25/cos(pi/10)*np.cos(2*angles+pi/10)
    bottom = np.sqrt(.6*(1+wave)/(1-wave))
    return make_design(angles, bottom)


def mirror_design(alpha, beta, outer_pair=1., product=.6):
    """Interpretable reflection-and-radius-swap family; angles are radians.

    Relative to the mirror axis, the five anchors lie at -alpha,+alpha,+beta,
    pi,-beta. A global rotation sets the first angle to zero. The top pattern
    is the mirror image of the bottom pattern, so alternate plates can flip.
    outer_pair=1 uses only the three radii 1,product,sqrt(product).
    """
    angles = np.array([0., 2*alpha, alpha+beta, pi+alpha, 2*pi+alpha-beta])
    bottom = np.array([1., product, outer_pair, sqrt(product), product/outer_pair])
    return make_design(angles, bottom, product)


def contact_angles(design):
    """Exact first contact for each cord pair over both alternating parities.

    At a common height fraction, intersection requires
    (1-s)*(a_i*u_i-a_j*u_j)+s*exp(i*theta)*(b_i*u_i-b_j*u_j)=0.
    Eliminating s yields this half-angle expression. Angles are radians;
    pi means no positive-height contact before a half turn.
    """
    values = []
    for i in range(5):
        for j in range(i+1, 5):
            delta = abs((design.angles[i]-design.angles[j]+pi) % (2*pi)-pi)
            a, b = design.bottom[i], design.bottom[j]
            values.append(2*np.arctan2((a+b)*sin(delta/2), abs(a-b)*cos(delta/2)))
    return np.asarray(values)


def _cross(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def cell_matrix(design, angle, height0=sqrt(1.2), total_rigidity=6000., swap=False):
    """12x12 two-plate energy Hessian in each plate's own rotated coordinates.

    total_rigidity is sum(EA_i)/F, divided equally among the five cords. The
    dimensionless energy Hessian includes initial tension and is normalized
    by F*R. Coordinates are dx/R,dy/R,dz/R,rx,ry,rz for each plate.
    """
    a, b = (design.top, design.bottom) if swap else (design.bottom, design.top)
    h = sqrt(height0**2 - 4*design.product*sin(angle/2)**2)
    lengths = np.sqrt(height0**2+(a-b)**2)
    turn = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1.]])
    unit = np.column_stack((np.cos(design.angles), np.sin(design.angles), np.zeros(len(a))))
    matrix = np.zeros((12, 12))
    for i in range(len(a)):
        lower, upper_local = a[i]*unit[i], b[i]*unit[i]
        upper = turn @ upper_local
        link = upper-lower+[0, 0, h]
        derivative = np.column_stack((-np.eye(3), _cross(lower), turn, -turn@_cross(upper_local)))
        row = link @ derivative/lengths[i]
        prestress = design.shares[i]/h
        material = total_rigidity/(len(a)*lengths[i])
        matrix += prestress*derivative.T@derivative+(material-prestress)*np.outer(row, row)
        for where, point, sign, frame in ((3, lower, -1, np.eye(3)), (9, upper, 1, turn)):
            correction = (np.outer(link, point)+np.outer(point, link))/2-np.dot(link, point)*np.eye(3)
            matrix[where:where+3, where:where+3] += sign*prestress*frame.T@correction@frame
    return matrix


def stack_matrix(design, angle, cells=12, height0=sqrt(1.2), total_rigidity=6000.):
    """All plate coordinates except the fixed bottom; final end constraints follow."""
    blocks = [cell_matrix(design, angle, height0, total_rigidity, bool(parity)) for parity in (0, 1)]
    matrix = np.zeros((6*cells, 6*cells))
    for j in range(cells):
        block = blocks[j % 2]
        if j == 0:
            matrix[:6, :6] += block[6:, 6:]
        else:
            where = slice(6*(j-1), 6*(j+1))
            matrix[where, where] += block
    return matrix


def alignment_matrix(design, angle, cells=12, height0=sqrt(1.2), total_rigidity=6000.):
    """Condense axial/twist motion; fixed end centers/normals and imposed end twist."""
    matrix = stack_matrix(design, angle, cells, height0, total_rigidity)
    transverse = np.array([6*j+k for j in range(cells-1) for k in (0, 1, 3, 4)])
    axial = np.r_[[6*j+k for j in range(cells-1) for k in (2, 5)], 6*(cells-1)+2]
    aa = matrix[np.ix_(axial, axial)]
    ta = matrix[np.ix_(transverse, axial)]
    # Positive aa is required: these coordinates must minimize energy.
    relaxed = matrix[np.ix_(transverse, transverse)]-ta@solve(aa, ta.T, assume_a="pos")
    return relaxed


def stiffness(design, angle, cells=12, height0=sqrt(1.2), total_rigidity=6000.):
    matrix = alignment_matrix(design, angle, cells, height0, total_rigidity)
    return float(eigh(matrix, subset_by_index=[0, 0], eigvals_only=True)[0])


def geometry(design, angle, cells=12, height0=sqrt(1.2)):
    """Return nominal plate centers and cord attachment points, R=1."""
    h = sqrt(height0**2-4*design.product*sin(angle/2)**2)
    centers = np.column_stack((np.zeros(cells+1), np.zeros(cells+1), np.arange(cells+1)*h))
    points = []
    for j in range(cells+1):
        radii = design.top if j % 2 else design.bottom
        points.append(np.column_stack((radii*np.cos(design.angles+j*angle),
            radii*np.sin(design.angles+j*angle), np.full(len(radii), j*h))))
    return centers, np.asarray(points)


def clearance(design, angle, height0=sqrt(1.2)):
    """Smallest cord-centerline distance within a cell, checking both parities."""
    _, points = geometry(design, angle, cells=2, height0=height0)
    best = np.inf
    for layer in range(2):
        for i in range(len(design.bottom)):
            for j in range(i+1, len(design.bottom)):
                p, q = points[layer, i], points[layer, j]
                d, e = points[layer+1, i]-p, points[layer+1, j]-q
                delta = p-q
                dd, ee, de = d@d, e@e, d@e
                candidates = []
                for s in (0., 1.):
                    candidates.append((s, np.clip(e@(delta+s*d)/ee, 0, 1)))
                for t in (0., 1.):
                    candidates.append((np.clip(d@(t*e-delta)/dd, 0, 1), t))
                if dd*ee-de**2 > 1e-14*dd*ee:
                    s, t = np.linalg.solve([[dd, -de], [-de, ee]], [-d@delta, e@delta])
                    if 0 <= s <= 1 and 0 <= t <= 1:
                        candidates.append((s, t))
                best = min(best, *(np.linalg.norm(delta+s*d-t*e) for s, t in candidates))
    return float(best)


def slack_margins(design, angle, cells=12, height0=sqrt(1.2), total_rigidity=6000.):
    """Worst-direction end tilt/offset that first unloads a cord, in linear theory.

    Returns (tilt in radians, lateral end offset / R). Each is varied alone,
    with the other zero. These are local predictions, not nonlinear limits.
    """
    matrix = stack_matrix(design, angle, cells, height0, total_rigidity)
    free = np.r_[np.arange(6*(cells-1)), 6*(cells-1)+2]
    boundary = 6*(cells-1)+np.array([0, 1, 3, 4])
    change = np.zeros((6*cells, 4))
    change[boundary] = np.eye(4)
    change[free] = -solve(matrix[np.ix_(free, free)], matrix[np.ix_(free, boundary)], assume_a="pos")
    h = sqrt(height0**2-4*design.product*sin(angle/2)**2)
    lengths = np.sqrt(height0**2+(design.bottom-design.top)**2)
    turn = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1.]])
    unit = np.column_stack((np.cos(design.angles), np.sin(design.angles), np.zeros(len(lengths))))
    tension_changes = []
    for j in range(cells):
        a, b = (design.top, design.bottom) if j % 2 else (design.bottom, design.top)
        local_change = np.vstack((np.zeros((6, 4)), change[:6])) if j == 0 else change[6*(j-1):6*(j+1)]
        for i in range(len(lengths)):
            lower, upper = a[i]*unit[i], b[i]*unit[i]
            link = turn@upper-lower+[0, 0, h]
            derivative = np.column_stack((-np.eye(3), _cross(lower), turn, -turn@_cross(upper)))
            row = link@derivative/lengths[i]
            tension_changes.append(total_rigidity/(len(lengths)*lengths[i])*row@local_change)
    tension_changes = np.asarray(tension_changes)
    tensions = np.tile(design.shares*lengths/h, cells)
    margins = []
    for columns in (slice(2, 4), slice(0, 2)):
        norm = np.linalg.norm(tension_changes[:, columns], axis=1)
        margins.append(np.divide(tensions, norm, out=np.full_like(tensions, np.inf), where=norm>1e-12).min())
    return tuple(margins)


if __name__ == "__main__":
    design = pentagon()
    print("angle, bottom radius, top radius, axial-load share")
    print(np.column_stack((np.degrees(design.angles), design.bottom, design.top, design.shares)))
    for degrees in (0, 15, 30, 45, 60, 75):
        print(degrees, stiffness(design, np.radians(degrees)))
