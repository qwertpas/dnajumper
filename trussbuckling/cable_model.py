"""Twisting plates with two families of tensile cords at exchanged radii.

Geometry is analytical and inextensible:
    L/R = sqrt((H0/R)**2 + (1-r/R)**2)
    h/R = sqrt((H0/R)**2 - 4*(r/R)*sin(theta/2)**2).

Stability is a local elastic calculation at that geometry and axial preload.
Each cord has tangent stiffness EA/L. Its local zero-force length is chosen
as L - T/(EA/L); thus the displayed ideal folding path is not a nonlinear
elastic continuation with one fixed natural length. Extension is shown in the
GUI, so this approximation is visible when the cords are too compliant.

All matrix coordinates are translations/R and rotation vectors in radians.
The Hessian is divided by F*R. This avoids mixing numerical millimeters with
radians when comparing modes. A negative eigenvalue means local instability.
The 3D Hessian includes the geometric stiffness from the existing tension.
"""

from dataclasses import dataclass, field
from math import asin, atan, cos, degrees, isfinite, pi, radians, sin, sqrt, tan

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Parameters:
    radius: float = 100.0           # mm
    ratio: float = .65              # inner / outer radius
    height: float = sqrt(1.3)       # untwisted geometric height / outer radius
    twist: float = 45.0             # degrees per cell
    per_family: int = 3             # total cords = 2 * per_family
    cells: int = 12
    force: float = 10.0             # N, tensile load resisting contraction
    rigidity: float = 10000.0       # N, EA per cord


@dataclass
class Report:
    cells: int
    radius: float
    height: float
    length: float
    tension: float
    torque: float
    extension: float
    rank: int
    stable: bool
    tilt_stable: bool
    smallest: float
    tilt_smallest: float
    eigenvalues: np.ndarray
    centers: np.ndarray
    rotations: np.ndarray
    points: np.ndarray
    mode: np.ndarray
    warnings: tuple[str, ...]
    _matrix: np.ndarray = field(repr=False)
    _free: np.ndarray = field(repr=False)
    _rigidity: float = field(repr=False)


def twist_limit(p):
    """Maximum GUI twist in degrees, before contact or h/R = .02.

    Cord crossings are exact centerline intersections (zero cord thickness).
    A half-degree margin is kept before the first crossing. Zero cell height
    is excluded because carrying a nonzero axial load there is singular.
    """
    q = p.ratio
    flat = 2*asin(sqrt(max(0., p.height**2 - .02**2)/(4*q))) if (
        p.height**2 - .02**2 < 4*q
    ) else pi
    crossing = pi if q == 1 else 2*atan(tan(pi/(2*p.per_family))*(1+q)/(1-q))
    if flat < crossing - radians(.5):
        return degrees(flat), "minimum cell height 0.02 R"
    return degrees(min(pi - radians(.5), crossing - radians(.5))), "first cord crossing"


def _cross(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _check(p, cells):
    if not all(isfinite(x) for x in (p.radius, p.ratio, p.height, p.twist, p.force, p.rigidity)):
        raise ValueError("Parameters must be finite.")
    if min(p.radius, p.height, p.force, p.rigidity) <= 0 or not 0 < p.ratio <= 1:
        raise ValueError("Use positive dimensions/load/stiffness and 0 < r/R <= 1.")
    if (not isinstance(cells, (int, np.integer)) or cells < 1
            or not isinstance(p.per_family, (int, np.integer)) or p.per_family < 3):
        raise ValueError("Use at least one cell and at least three cords per family.")
    if not 0 <= p.twist < 180:
        raise ValueError("Twist must lie between zero and 180 degrees.")


def evaluate(p, cells=None):
    """Analytical geometry, full local Hessian, and weakest transverse mode.

    A single cell has a top plate free to translate and tilt, with prescribed
    twist. A stack has fixed end centers/normals/twist; its top can move axially.
    Mode rows are translations (mm), then rotations (rad), normalized so the
    largest first-order anchor displacement equals R. They are directions,
    not finite folding configurations or post-buckled equilibria.
    """
    cells = p.cells if cells is None else cells
    _check(p, cells)
    count = 2 * p.per_family
    theta = radians(p.twist)
    length = sqrt(p.height**2 + (1-p.ratio)**2)
    square = p.height**2 - 4*p.ratio*sin(theta/2)**2
    if square <= 0:
        raise ValueError("This twist reaches zero height; reduce twist or increase initial height.")
    height = sqrt(square)
    tension = p.force * length / (count * height)
    extension = tension / p.rigidity
    if extension >= 1:
        raise ValueError(f"Cord tension ({tension:.3g} N) exceeds EA. Increase rigidity or reduce load/twist.")
    size = 6 * cells
    matrix = np.zeros((size, size))
    angles = np.arange(count)*2*pi/count
    centers = np.zeros((cells+1, 3))
    centers[:, 2] = np.arange(cells+1)*height
    rotations = Rotation.from_rotvec(np.column_stack((np.zeros(cells+1),
        np.zeros(cells+1), np.arange(cells+1)*theta))).as_matrix()
    arms = []
    for j in range(cells+1):
        radii = np.where(np.arange(count) % 2 == j % 2, 1., p.ratio)
        arms.append(np.column_stack((radii*np.cos(angles+j*theta),
            radii*np.sin(angles+j*theta), np.zeros(count))))
    arms = np.asarray(arms)
    jacobian_first = []
    prestress = tension / (p.force * length)
    material = p.rigidity / (p.force * length)
    for j in range(cells):
        for i in range(count):
            bottom, top = arms[j, i], arms[j+1, i]
            link = top - bottom + [0, 0, height]
            # Only the two adjacent plates enter a cord's derivatives.
            derivative = np.column_stack((-np.eye(3), _cross(bottom), np.eye(3), -_cross(top)))
            row = link @ derivative / length
            block = prestress*(derivative.T @ derivative) + (material-prestress)*np.outer(row, row)
            for offset, point, sign in ((3, bottom, -1), (9, top, 1)):
                block[offset:offset+3, offset:offset+3] += sign*prestress*(
                    (np.outer(link, point)+np.outer(point, link))/2
                    - np.dot(link, point)*np.eye(3)
                )
            if j == 0:
                matrix[:6, :6] += block[6:, 6:]
                jacobian_first.append(row[6:])
            else:
                where = slice(6*(j-1), 6*(j+1))
                matrix[where, where] += block
    free = np.array([0, 1, 2, 3, 4]) if cells == 1 else np.r_[
        np.arange(6*(cells-1)), 6*(cells-1)+2
    ]
    transverse = np.array([0, 1, 3, 4]) if cells == 1 else np.array([
        6*j+k for j in range(cells-1) for k in (0, 1, 3, 4)
    ])
    values, vectors = eigh(matrix[np.ix_(transverse, transverse)])
    smallest = float(eigh(matrix[np.ix_(free, free)], subset_by_index=[0, 0], eigvals_only=True)[0])
    motion = np.zeros((cells+1, 6))
    full_mode = np.zeros(size)
    full_mode[transverse] = vectors[:, 0]
    motion[1:] = full_mode.reshape(cells, 6)
    # Choose a repeatable phase within the rotationally symmetric mode pair.
    horizontal = motion[:, :2]
    reference = horizontal[np.argmax(np.linalg.norm(horizontal, axis=1))]
    phase = -np.arctan2(reference[1], reference[0])
    turn = np.array([[cos(phase), -sin(phase)], [sin(phase), cos(phase)]])
    motion[:, :2] = motion[:, :2] @ turn.T
    motion[:, 3:5] = motion[:, 3:5] @ turn.T
    displacement = motion[:, None, :3] + np.cross(motion[:, None, 3:], arms)
    motion /= np.linalg.norm(displacement, axis=2).max()
    motion[:, :3] *= p.radius
    warnings = []
    if extension > .01:
        warnings.append("Extension exceeds 1%: the ideal folding geometry is only a local approximation.")
    allowed, reason = twist_limit(p)
    if p.twist > allowed + 1e-7:
        warnings.append(f"Beyond the viewer limit for {reason}; contact is not modeled.")
    return Report(cells, p.radius, height*p.radius, length*p.radius, tension,
        p.force*p.radius*p.ratio*sin(theta)/height, 100*extension,
        int(np.linalg.matrix_rank(jacobian_first)), smallest > 1e-9, values[0] > 1e-9,
        smallest, float(values[0]), values, centers*p.radius, rotations,
        (arms+centers[:, None, :])*p.radius, motion, tuple(warnings), matrix, free, p.rigidity)


def deform(report, motion, scale=1.):
    """Draw a small-motion direction using exact rotations of each rigid plate.

    The plates are rigid in the drawing, but the displaced cords do not keep
    exact length. An exaggerated eigenmode must not be read as a finite fold.
    """
    moved = np.asarray(motion)*scale
    turns = Rotation.from_rotvec(moved[:, 3:]).as_matrix()
    centers = report.centers + moved[:, :3]
    rotations = turns @ report.rotations
    arms = report.points - report.centers[:, None, :]
    points = np.einsum("pij,pcj->pci", turns, arms) + centers[:, None, :]
    return centers, rotations, points


def response(report, offset_mm, tilt_deg, direction_deg):
    """Linear equilibrium response to a prescribed top-end offset and tilt.

    The end centers/normals are now prescribed with this small error. The top
    remains free axially, and its twist is held. No pseudoinverse or artificial
    stabilization is used when the nominal equilibrium is unstable.
    """
    if not report.stable:
        raise ValueError("No stable linear response at this state. Select the weakest-mode view.")
    if not all(isfinite(x) for x in (offset_mm, tilt_deg, direction_deg)):
        raise ValueError("End misalignment must be finite.")
    values = np.zeros(report._matrix.shape[0])
    angle = radians(direction_deg)
    direction = np.array([cos(angle), sin(angle)])
    start = 6*(report.cells-1)
    values[start:start+2] = offset_mm/report.radius*direction
    values[start+3:start+5] = radians(tilt_deg)*direction
    free = np.r_[np.arange(start), start+2]
    matrix = report._matrix
    values[free] = np.linalg.solve(matrix[np.ix_(free, free)], -(matrix @ values)[free])
    motion = np.zeros((report.cells+1, 6))
    motion[1:] = values.reshape(report.cells, 6)
    motion[:, :3] *= report.radius
    return motion


def response_warnings(report, motion):
    """Check whether a linear end-error prediction would unload any cord."""
    arms = report.points - report.centers[:, None, :]
    displacement = motion[:, None, :3] + np.cross(motion[:, None, 3:], arms)
    links = np.diff(report.points, axis=0)
    change = np.sum(links*np.diff(displacement, axis=0), axis=2) / report.length
    tensions = report.tension + report._rigidity/report.length*change
    messages = []
    slack = int(np.count_nonzero(tensions <= 0))
    if slack:
        messages.append(f"{slack} cords would slacken: the taut-cord linear prediction is invalid.")
    if np.linalg.norm(displacement, axis=2).max() > .05*report.radius:
        messages.append("Displacement exceeds 5% R; the small-motion approximation may be inaccurate.")
    return tuple(messages)
