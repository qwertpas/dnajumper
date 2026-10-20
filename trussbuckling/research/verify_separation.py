"""Exact symbolic checks for the general-n separation proof."""

import sympy as s

A, b, u, q, h = s.symbols("A b u q h", positive=True)
a = A*b
c, z = (1-u*u)/(1+u*u), 2*u/(1+u*u)
cd, sd = (A*A-1)/(A*A+1), 2*A/(A*A+1)
D = 2*b*(1-A*u)/(1+u*u)
X = u*D
# This identity follows directly from the rigid-edge equation and h^2.
p = h*(1+u*q)/(2*b*(1+A*u))
r = a-b*q
B0, B1 = s.Matrix([a, -b, 0]), s.Matrix([a, b, 0])
T0, T1 = s.Matrix([a*c+b*z, a*z-b*c, h]), s.Matrix([a*c-b*z, a*z+b*c, h])
J = s.Matrix([r, 0, b*p])
I = s.Matrix([r*c, r*z, h-b*p])
Ip = s.Matrix([r*(c*cd+z*sd), r*(z*cd-c*sd), h-b*p])
N = s.Matrix([p, 0, q])
m = (T0-B0).cross(J-B0)

identities = {
    "bottom plane / next top vertex": N.dot(T1-B1)-h*(q-u)/(1+u*u),
    "bottom plane / upper slit vertex": N.dot(I-B1)-h*u*(A-q)*(q-u)/((1+A*u)*(1+u*u)),
    "vertical plane / next bottom vertex": m.dot(B1-B0)+2*b*b*(h*q+X*p),
    "vertical plane / previous upper slit vertex": m.dot(Ip-B0)-2*b*b*h*(A-q)*(A*(q-u)+q*u+1)/((A*A+1)*(A*u+1)),
}
K2 = (1-A*u)/((1+A*u)*(1+u*u))
identities["q greater than u"] = ((1+K2*u*u)*u*u+2*K2*u*u+K2-1
                                   -2*u*(u-A)/(1+A*u))
for name, expression in identities.items():
    assert s.factor(expression) == 0, name
    print("Verified:", name)
print("All five general-n separation identities hold exactly.")
