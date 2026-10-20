"""Exact panel and separation checks for triangle_proof.txt (R=2)."""

import sympy as s


d, v, h = s.symbols("d v h", real=True)
a, b = s.Integer(1), s.sqrt(3)
C, S = 1-d, (v*v-d)/b
E = 2-2*d+2*v+v*v
p, q = h*(1+v)/E, (2*d+2*v+v*v)/E
r, X, Y = 1-q, C-1+b*S, S+b*(1-C)


def reduced(value):
    numerator = s.together(value).as_numer_denom()[0]
    numerator = s.rem(numerator, h*h-4*(1-2*d), h)
    return s.factor(s.rem(numerator, v**4-2*d*v*v+4*d*d-6*d, v))


def equal(left, right):
    assert reduced(left-right) == 0, s.factor(left-right)


def turn(point):
    return s.Matrix([(-point[0]-b*point[1])/2,
                     (b*point[0]-point[1])/2, point[2]])


bottom = [s.Matrix([a, -b, 0])]
top = [s.Matrix([a*C+b*S, a*S-b*C, h])]
lower = [s.Matrix([r, 0, p])]
upper = [s.Matrix([r*C, r*S, h-p])]
for _ in range(2):
    for points in (bottom, top, lower, upper):
        points.append(turn(points[-1]))

equal(C*C+S*S, 1)
equal(p*p+q*q, 1)
equal(h*h+8*(1-C), 4)
equal(h*p-X*q, h*h/2-X)
equal(2-b*Y, h*h/2-X)
equal(E-2*(1+v), X)
equal(r, h*h/(2*E))

faces = []
for k in range(3):
    j = (k+1) % 3
    faces.extend([
        (bottom[k], bottom[j], lower[k], 12),
        (bottom[k], top[k], lower[k], 4),
        (top[k], top[j], upper[k], 12),
        (bottom[j], top[j], upper[k], 4),
    ])
for first, second, center, base in faces:
    equal((first-second).dot(first-second), base)
    equal((first-center).dot(first-center), 4)
    equal((second-center).dot(second-center), 4)

normal = s.Matrix([p, 0, q])
G = 2*v*(b*S-2*d*C)+2*d*(2-d+b*S)
P = 9-12*d+(12-4*d)*v+3*v*v-2*v**3
equal(normal.dot(top[1]-bottom[1]), h*(2*d+2*v-v**3)/E)
equal(normal.dot(upper[0]-bottom[1]), h*G/E**2)
normal = (top[0]-bottom[0]).cross(lower[0]-bottom[0])
equal(normal.dot(bottom[1]-bottom[0]), -2*b*(h*q+X*p))
equal(normal.dot(upper[2]-bottom[0]), 2*b*h*P/(3*E**2))
print("PASS: all 36 triangle distances and all triangular separation identities.")
