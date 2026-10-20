"""Exact symbolic checks for the square FP-ori proof in square_proof.txt.

Run with python3.11. Requires the already available SymPy package.
Coordinates use side length 1; scaling preserves every identity.
"""

import sympy as sp


u, c, h = sp.symbols("u c h", real=True)
den = (1 + u) ** 2 + c
r = c / den
p = 2 * h * (1 + u) / den
q = ((1 + u) ** 2 - c) / den
half = sp.Rational(1, 2)


def reduced(value):
    numerator = sp.factor(sp.together(value)).as_numer_denom()[0]
    numerator = sp.rem(numerator, h**2 - c, h)
    return sp.factor(sp.rem(numerator, c**2 + u**4 - 1, c))


def equal(left, right):
    assert reduced(left - right) == 0, sp.factor(left - right)


def turn(point):
    return sp.Matrix([-point[1], point[0], point[2]])


bottom = [sp.Matrix([half, -half, 0])]
top = [sp.Matrix([(c + u**2) / 2, (u**2 - c) / 2, h])]
lower = [sp.Matrix([r, 0, p / 2])]
upper = [sp.Matrix([r * c, r * u**2, h - p / 2])]
for _ in range(3):
    for points in (bottom, top, lower, upper):
        points.append(turn(points[-1]))

faces = []
for k in range(4):
    j = (k + 1) % 4
    faces.extend(
        [
            (bottom[k], bottom[j], lower[k]),
            (bottom[k], top[k], lower[k]),
            (top[k], top[j], upper[k]),
            (bottom[j], top[j], upper[k]),
        ]
    )

equal(p**2 + q**2, 1)
equal(2 * h * p - (c + u**2 - 1) * q, c - u**2 + 1)
equal(h - p, h * (c + u**2 - 1) / den)
for points in faces:
    a, b, center = points
    equal((a - b).dot(a - b), 1)
    equal((a - center).dot(a - center), half)
    equal((b - center).dot(b - center), half)

normal = sp.Matrix([p, 0, q])
equal(normal.dot(top[1] - bottom[1]), h * u * (c + 1 - u**2) / den)
equal(normal.dot(upper[0] - bottom[1]), 2 * h * u**3 * (1 - u**2) / den**2)
normal = (top[0] - bottom[0]).cross(lower[0] - bottom[0])
equal(normal.dot(bottom[1] - bottom[0]), -h * u / 2)
equal(normal.dot(upper[3] - bottom[0]), h * (1 + u) * (c + u - u**2) / den**2)

gap = upper[0] - lower[0]
equal(gap.dot(gap), 2 * r**2 * (1 - c) + (h - p) ** 2)
print("PASS: all 48 triangle distances and all symbolic proof identities.")
