"""Two-ring Hunt truss. Units: mm, N, MPa, radians; energy/torque: N mm.

The circular end rings stay rigid. There are n identical bars in each of
two inclined families. See README.md for the derivation and assumptions.
"""

from dataclasses import dataclass
from math import acos, asin, atan2, cos, isfinite, pi, sin, sqrt, tan
from numbers import Integral

import numpy as np
from scipy.optimize import brentq


def folding_angles(radius, length, creases):
    """Return Hunt's developed-sheet (beta, gamma), in radians, Eqs. (1-2).

    Limit to non-overhanging creases: 0 < beta < gamma <= pi/2.
    The paper uses the arc pitch 2*pi*R/n, not the polygon chord.
    """
    if not all(isfinite(v) and v > 0 for v in (radius, length)):
        raise ValueError("Radius and length must be positive and finite.")
    if isinstance(creases, bool) or not isinstance(creases, Integral) or creases < 3:
        raise ValueError("Creases must be an integer of at least 3.")
    alpha = pi / creases
    pitch = 2 * pi * radius / creases
    if length > pitch / tan(alpha) * (1 + 1e-12):
        raise ValueError("This length exceeds the non-overhanging Hunt geometry.")
    value = cos(alpha) - 2 * length / pitch * sin(alpha)
    gamma = min(pi / 2, (alpha + acos(np.clip(value, -1, 1))) / 2)
    return gamma - alpha, gamma


@dataclass(frozen=True)
class Parameters:
    radius: float = 100.0
    length: float = 71.2
    creases: int = 17
    gamma: float = folding_angles(100.0, 71.2, 17)[1]
    modulus: float = 5550.0
    area: float = 1.0

    def __post_init__(self):
        values = (self.radius, self.length, self.modulus, self.area)
        if not all(isfinite(v) and v > 0 for v in values):
            raise ValueError("Radius, length, modulus, and area must be positive and finite.")
        if isinstance(self.creases, bool) or not isinstance(self.creases, Integral) or self.creases < 3:
            raise ValueError("Creases must be an integer of at least 3.")
        if not isfinite(self.gamma) or not 0 < self.gamma <= pi / 2:
            raise ValueError("Crease angle gamma must be between 0 and 90 degrees.")

    @property
    def pitch(self):
        """Angular spacing between vertices, radians."""
        return 2 * pi / self.creases

    @property
    def reference_twist(self):
        """Embed the developed crease run L*cot(gamma) as an arc on radius R."""
        return self.length / (self.radius * tan(self.gamma))

    @property
    def rest_lengths(self):
        angles = self.reference_twist + np.array([0.0, self.pitch])
        return np.hypot(self.length, 2 * self.radius * np.sin(angles / 2))


@dataclass(frozen=True)
class State:
    height: float
    lengths: np.ndarray
    strain: np.ndarray
    energy: float
    torque: float
    force: float
    stiffness: float
    axial_stiffness: float


def state(p, twist, height=None):
    """Evaluate the symmetric truss at ADDED twist (radians).

    If height is None, minimize energy over h >= 0 (no applied axial force).
    Otherwise both twist and height are prescribed. Torque = dU/d(twist);
    force = dU/dh. Stiffness follows the selected height condition.
    """
    if not isfinite(twist):
        raise ValueError("Twist must be finite.")
    if height is not None and (not isfinite(height) or height < 0):
        raise ValueError("Height must be nonnegative and finite.")
    angles = p.reference_twist + twist + np.array([0.0, p.pitch])
    chords = 2 * p.radius * np.abs(np.sin(angles / 2))
    rest = p.rest_lengths
    free = height is None

    if free:
        # U_h/(n*EA*h) = sum(1/l0 - 1/l). Its derivative in h is positive.
        # Therefore there is either one positive minimum, or the minimum h=0.
        def balance(h):
            lengths = np.hypot(h, chords)
            if np.any(lengths == 0):
                return -np.inf
            return float(np.sum(1 / rest - 1 / lengths))

        height = 0.0 if balance(0.0) >= 0 else brentq(
            balance, 0.0, float(max(rest)), xtol=1e-11, rtol=1e-13
        )

    lengths = np.hypot(height, chords)
    if np.any(lengths == 0):
        raise ValueError("A bar has zero length; move away from this degenerate geometry.")
    change = lengths - rest
    strain = change / rest
    scale = p.creases * p.modulus * p.area / rest

    # Analytical first and second length derivatives for q = (h, theta).
    h_first = height / lengths
    t_first = p.radius**2 * np.sin(angles) / lengths
    hh = chords**2 / lengths**3
    tt = p.radius**2 * np.cos(angles) / lengths - t_first**2 / lengths

    energy = float(np.sum(scale * change**2) / 2)
    force = float(np.sum(scale * change * h_first))
    torque = float(np.sum(scale * change * t_first))
    u_hh = float(np.sum(scale * (h_first**2 + change * hh)))
    u_tt = float(np.sum(scale * (t_first**2 + change * tt)))
    stiffness = u_tt
    if free and height > 0:
        # Along U_h=0, simplify the Schur complement analytically. Cancelling
        # h before division keeps the flat-limit stiffness accurate as h -> 0.
        inverse_cube = 1 / lengths**3
        weight = p.creases * p.modulus * p.area
        u_hh = weight * height**2 * float(np.sum(inverse_cube))
        stiffness -= weight * p.radius**4 * float(np.sum(np.sin(angles) * inverse_cube))**2 / float(np.sum(inverse_cube))
    return State(height, lengths, strain, energy, torque, force, stiffness, u_hh)


def mesh(p, twist, height):
    """Return vertices, 4n edges, and 2n triangular faces for plotting.

    Edge order: bottom rim, top rim, short bars, long bars (n of each).
    Faces visualize the truss only; they contribute no bending stiffness.
    """
    count = p.creases
    angle = np.arange(count) * p.pitch
    bottom = np.column_stack((p.radius * np.cos(angle), p.radius * np.sin(angle), np.zeros(count)))
    angle = angle + p.reference_twist + twist
    top = np.column_stack((p.radius * np.cos(angle), p.radius * np.sin(angle), np.full(count, height)))
    vertices = np.vstack((bottom, top))
    i = np.arange(count)
    j = (i + 1) % count
    edges = np.vstack((
        np.column_stack((i, j)), np.column_stack((i + count, j + count)),
        np.column_stack((i, i + count)), np.column_stack((i, j + count)),
    ))
    faces = np.vstack((np.column_stack((i, j, j + count)), np.column_stack((i, j + count, i + count))))
    return vertices, edges, faces


def sweep(p, twists, height=None):
    """Sample states independently; no stepping history or numerical derivatives."""
    states = [state(p, float(twist), height) for twist in twists]
    names = ("height", "torque", "strain", "stiffness", "energy")
    return {name: np.array([getattr(s, name) for s in states]) for name in names}


def collapse_twist(p):
    """First axially free h=0 state, or None on the chosen non-wrapping branch.

    Restrict total twist to [theta0, pi-pitch], where both chords increase.
    Stop the GUI here: further twist would push through the flat configuration.
    """
    start = p.reference_twist
    end = pi - p.pitch
    if start >= end:
        return None
    target = float(np.sum(1 / p.rest_lengths))

    def balance(theta):
        chords = 2 * p.radius * np.sin((theta + np.array([0.0, p.pitch])) / 2)
        return target - float(np.sum(1 / chords))

    if balance(end) < 0:
        return None
    return brentq(balance, start, end, xtol=1e-13) - start


def rigid_endpoint(p):
    """The other zero-strain state, as (added twist, height), if it exists.

    Two isolated equal-length states do not imply a rigid path between them.
    """
    alpha = pi / p.creases
    theta = p.reference_twist
    square = p.length**2 - 4 * p.radius**2 * cos(alpha) * cos(theta + alpha)
    if square < -1e-10 * p.radius**2:
        return None
    return pi - 2 * alpha - 2 * theta, sqrt(max(0.0, square))


def endpoint_angle(radius, length, creases):
    """Gamma for stress-free extended AND flat states of the fixed-ring truss.

    Set h2=0 in rigid_endpoint: L²=4R² cos(alpha) cos(theta0+alpha).
    This fixes the reference rotation for each chosen integer n. It does not
    remove the small strains between the two end states.
    """
    Parameters(radius=radius, length=length, creases=creases, gamma=pi / 2)
    alpha = pi / creases
    if length > 2 * radius * cos(alpha) * (1 + 1e-12):
        raise ValueError("Flat end states require L <= 2 R cos(pi/n).")
    half_twist = asin(length**2 / (4 * radius**2 * cos(alpha)))
    theta = max(0.0, pi / 2 - alpha - half_twist)
    return atan2(length, radius * theta)


def endpoint_twist(radius, length, creases):
    """Exact added rotation from the designed reference state to its flat state."""
    endpoint_angle(radius, length, creases)  # Validate the design domain.
    return 2 * asin(length**2 / (4 * radius**2 * cos(pi / creases)))


def endpoint_strain(radius, length, creases):
    """Exact peak absolute bar strain along the axially free designed path.

    Free equilibrium fixes 1/a+1/b. The difference b²-a² reaches its maximum
    at theta=pi/2-alpha, halfway between the two end rotations. Both bar strain
    magnitudes therefore peak there. Only one height solve is needed.
    """
    p = Parameters(radius=radius, length=length, creases=creases,
                   gamma=endpoint_angle(radius, length, creases))
    middle = state(p, endpoint_twist(radius, length, creases) / 2)
    return float(np.max(np.abs(middle.strain)))


def choose_creases(radius, length, max_strain, limit=256):
    """Smallest feasible n whose free-path peak strain meets the given limit.

    Strain is a fraction, e.g. 0.001 for 0.1%. Integer candidates are checked
    directly; there is no assumption that strain decreases monotonically in n.
    """
    Parameters(radius=radius, length=length, gamma=pi / 2)
    if not isfinite(max_strain) or max_strain <= 0:
        raise ValueError("Allowed peak strain must be positive and finite.")
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 3:
        raise ValueError("The crease-count limit must be an integer of at least 3.")
    if length > 2 * radius * cos(pi / limit):
        raise ValueError(f"No flat-endpoint design for L/R={length / radius:.4g} with n <= {limit}.")
    best = np.inf
    for creases in range(3, limit + 1):
        if length > 2 * radius * cos(pi / creases):
            continue
        peak = endpoint_strain(radius, length, creases)
        if peak <= max_strain:
            return creases
        best = min(best, peak)
    raise ValueError(
        f"No n <= {limit} meets {100 * max_strain:.4g}% peak strain; "
        f"best is {100 * best:.4g}%. Increase the allowance or reduce L/R."
    )
