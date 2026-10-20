"""Two families of tensile links with exchanged attachment radii.

With m links per family, one family runs from radius a to b and the other
from b to a. Their azimuths alternate, separated by pi/m. Use m=3 for six
links or m=4 for eight. All have the same length L. Their coaxial motion
is h**2 = L**2 - a**2 - b**2 + 2*a*b*cos(theta). Choosing
L**2 = a**2 + b**2 gives h**2 = 2*a*b*cos(theta), flat at 90 degrees.

Why this removes the original infinitesimal tilt mode: let t be a lateral
translation, w a horizontal rotation, Q a planar rotation through theta,
and J a planar rotation through 90 degrees. For m >= 3 the length
constraints in each group resolve constant, sine and cosine parts. Their
horizontal parts are

    (b*Q.T - a*I)*t + h*b*Q.T*J*w = 0,
    (a*Q.T - b*I)*t + h*a*Q.T*J*w = 0.

Multiply the first by a and subtract b times the second. This gives
(b**2-a**2)*t = 0, hence t = 0 and w = 0 when a != b and h > 0.
The remaining constraint is h*dh + a*b*sin(theta)*dtheta = 0. Thus the
length Jacobian has rank five and only the intended folding freedom.
This is a local kinematic result for taut, inextensible links. Finite
elasticity, preload and the complete stack still need a stability analysis.

Alternating plates exchange their radii: a,b,a,b,... then b,a,b,a,... .
Use swap=True for the next cell; its bottom anchors match the previous
cell's top anchors after applying the accumulated rigid transformation.

Nonintersection for positive height: two links from different groups can
meet only at the same height fraction s. Their squared radii differ by
(a**2-b**2)*(1-2*s), so s must equal 1/2. Their midpoint phase difference
is +/-2*atan((a-b)/(a+b)*tan(theta/2)). The first crossing angle is
2*atan(tan(pi/(2*m))*(a+b)/(a-b)). Six links avoid crossings before
90 degrees for b/a >= 2-sqrt(3); eight require b/a >= sqrt(2)-1.
Links within either group cannot meet before pi. Equal radii recover the
original weak alignment mode; all links meet on the axis at theta=pi.
At h=0 this nonintersection argument no longer applies.
"""

from math import atan2, isfinite, pi, sin, sqrt, tan

import numpy as np


def height(outer, inner, length, twist):
    """Axial separation; twist is in radians and all lengths use one unit."""
    if not all(isfinite(x) for x in (outer, inner, length, twist)):
        raise ValueError("Parameters must be finite.")
    if not 0 < inner <= outer or length <= 0 or not 0 <= twist <= pi:
        raise ValueError("Use 0 < inner <= outer, length > 0 and 0 <= twist <= pi.")
    square = length**2 - (outer-inner)**2 - 4*outer*inner*sin(twist/2)**2
    if square < -1e-12 * length**2:
        raise ValueError("This twist exceeds the flat configuration.")
    return sqrt(max(0.0, square))


def points(outer, inner, length, twist, swap=False, per_family=4):
    """Return bottom and top cable endpoints, each shape (2*per_family, 3)."""
    if not isinstance(per_family, (int, np.integer)) or per_family < 3:
        raise ValueError("Use an integer per_family >= 3.")
    h = height(outer, inner, length, twist)
    count = 2 * per_family
    angles = np.arange(count) * pi / per_family
    lower = np.where(np.arange(count) % 2 == int(swap), outer, inner)
    upper = outer + inner - lower
    bottom = np.column_stack((lower*np.cos(angles), lower*np.sin(angles), np.zeros(count)))
    top = np.column_stack((upper*np.cos(angles + twist), upper*np.sin(angles + twist),
                           np.full(count, h)))
    return bottom, top


def crossing_angle(outer, inner, per_family=4):
    """First cord crossing in radians, assuming positive plate separation."""
    if not isinstance(per_family, (int, np.integer)) or per_family < 3:
        raise ValueError("Use an integer per_family >= 3.")
    if not all(isfinite(x) for x in (outer, inner)) or not 0 < inner <= outer:
        raise ValueError("Use finite radii with 0 < inner <= outer.")
    return 2 * atan2(tan(pi / (2*per_family)) * (outer + inner), outer - inner)


def jacobian(bottom, top):
    """Cable length derivatives for top-center translation then rotation xyz."""
    links = top - bottom
    arms = top - top.mean(axis=0)
    return np.column_stack((links, np.cross(arms, links))) / np.linalg.norm(
        links, axis=1
    )[:, None]


def tilt_stiffness(outer, inner, length, twist, rigidity, per_family=4):
    """Material stiffness per tilt axis after allowing lateral translation.

    Each cable has axial rigidity E*A = rigidity, so spring stiffness E*A/L.
    The minimized quadratic elastic energy is stiffness*|horizontal tilt|**2/2.
    This excludes the additional geometric stiffness from existing tension.
    """
    if not isinstance(per_family, (int, np.integer)) or per_family < 3:
        raise ValueError("Use an integer per_family >= 3.")
    h = height(outer, inner, length, twist)
    if outer == inner and twist == 0:
        raise ValueError("Lateral stiffness is singular here; use the full pose Hessian.")
    horizontal = (outer-inner)**2 + 4*outer*inner*sin(twist/2)**2
    return (per_family / 4 * rigidity * h**2 * (outer**2 - inner**2)**2
            / (length**3 * horizontal))
