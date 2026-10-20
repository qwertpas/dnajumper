"""Local stiffness of continuous reciprocal-radius warp panels.

Three panels connect alternating radii r and C/r, with C=a*b. Every warp
has L(r)^2=r^2+C^2/r^2 and the common motion h^2=2*C*cos(twist).
The textile carrier is NOT modeled: this is a quadrature of tensile fibers,
including their geometric stiffness, not an isometric fabric simulation.

Rest lengths remain fixed at L(r). At each twist, solve the extended cell
height from axial force balance using Hookean warp fibers. Thus the same
physical springs are used throughout each force-controlled deployment.
Lengths use R=b=1; each strip has EA=1. Rotations use world exponential
coordinates. End centers and orientations are fixed except top axial motion.
"""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def curvature(d, arm):
    return (np.outer(d, arm) + np.outer(arm, d)) / 2 - np.dot(d, arm) * np.eye(3)


def build(angle, force, cells=12, points=24, edge_only=False, a=.65, b=1.):
    """Return full pose Hessian, equilibrium gradient, and fiber data."""
    if edge_only:
        radii, weights = np.array([a, b]), np.array([.5, .5])
    else:
        x, g = leggauss(points)
        radii, weights = (a+b)/2 + (b-a)/2*x, g/2
    count, product = len(radii), a*b
    upper = product/radii
    rest = np.sqrt(radii*radii+upper*upper)
    height0 = np.sqrt(max(0., 2*product*np.cos(angle)))
    def balance(h):
        length = np.sqrt(rest*rest+h*h-2*product*np.cos(angle))
        return 3*h*np.sum(weights*(length/rest-1)/length)-force
    height = brentq(balance, height0, height0+10*b, xtol=1e-14) if force else height0
    lengths = np.sqrt(rest*rest+height*height-2*product*np.cos(angle))
    strains = lengths/rest-1
    centers = np.column_stack((np.zeros(cells+1), np.zeros(cells+1),
                               np.arange(cells+1)*height))
    arms = np.empty((cells+1, 3*count, 3))
    for j in range(cells+1):
        r = radii if j % 2 == 0 else upper
        phase = np.repeat(np.arange(3)*2*np.pi/3+j*angle, count)
        rr = np.tile(r, 3)
        arms[j] = np.column_stack((rr*np.cos(phase), rr*np.sin(phase), np.zeros(3*count)))
    ea, rest = np.tile(weights, 3), np.tile(rest, 3)
    size = 6*(cells+1)
    matrix, gradient = np.zeros((size, size)), np.zeros(size)
    for j in range(cells):
        index = np.arange(6*j, 6*j+12)
        for i in range(3*count):
            d = centers[j+1]+arms[j+1, i]-centers[j]-arms[j, i]
            length = np.linalg.norm(d)
            unit = d/length
            force_i = ea[i]*(length/rest[i]-1)
            spring = ea[i]/rest[i]
            first = np.column_stack((-np.eye(3), skew(arms[j, i]),
                                      np.eye(3), -skew(arms[j+1, i])))
            g = first.T@unit
            block = spring*np.outer(g, g) + force_i/length * first.T@(np.eye(3)-np.outer(unit, unit))@first
            block[3:6, 3:6] -= force_i/length*curvature(d, arms[j, i])
            block[9:12, 9:12] += force_i/length*curvature(d, arms[j+1, i])
            matrix[np.ix_(index, index)] += block
            gradient[index] += force_i*g
    gradient[6*cells+2] -= force
    free = np.r_[np.arange(6, 6*cells), 6*cells+2]
    tilt = np.array([6*j+k for j in range(1, cells) for k in (0, 1, 3, 4)])
    axial = np.array([6*j+k for j in range(1, cells) for k in (2, 5)]+[6*cells+2])
    return dict(matrix=matrix, gradient=gradient, free=free, tilt=tilt, axial=axial,
                centers=centers, arms=arms, ea=ea, rest=rest, force=force,
                height=height, height0=height0, strains=strains, radii=radii, weights=weights)


def energy(model, pose):
    centers = model['centers']+pose[:, :3]
    rotation = Rotation.from_rotvec(pose[:, 3:]).as_matrix()
    anchors = centers[:, None, :]+np.einsum('jab,jib->jia', rotation, model['arms'])
    lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=2)
    return np.sum(model['ea']/(2*model['rest'])*(lengths-model['rest'])**2) - model['force']*centers[-1, 2]


def metrics(model):
    h = model['matrix']
    values = {name: np.linalg.eigvalsh(h[np.ix_(model[name], model[name])])
              for name in ('free', 'tilt', 'axial')}
    return dict(minimum_strain=np.min(model['strains']), maximum_strain=np.max(model['strains']),
                height=model['height'], unloaded_height=model['height0'],
                minimum_full=values['free'][0], minimum_tilt=values['tilt'][0],
                minimum_axial=values['axial'][0],
                equilibrium_error=np.max(np.abs(model['gradient'][model['free']])),
                negative_eigenvalues=int(np.sum(values['free'] < -1e-10)))


def verify(model):
    """Directional second differences of the physical spring potential."""
    rng = np.random.default_rng(1729)
    result = []
    shape = model['centers'].shape[0], 6
    zero = np.zeros(shape)
    start = energy(model, zero)
    for _ in range(4):
        direction = np.zeros(zero.size)
        direction[model['free']] = rng.normal(size=len(model['free']))
        direction /= np.linalg.norm(direction)
        exact = direction@model['matrix']@direction
        errors = []
        for step in (1e-3, 3e-4, 1e-4):
            pose = step*direction.reshape(shape)
            numeric = (energy(model, pose)+energy(model, -pose)-2*start)/step**2
            errors.append(abs(numeric-exact)/max(1e-12, abs(exact)))
        result.append(dict(analytic=exact, relative_errors=errors))
    return result


def verify_unstable(model, step=1e-3):
    """Check that the weakest mode lowers energy while every fiber stays taut."""
    free = model['free']
    values, vectors = np.linalg.eigh(model['matrix'][np.ix_(free, free)])
    pose = np.zeros((len(model['centers']), 6))
    pose.ravel()[free] = step*vectors[:, 0]
    start = energy(model, np.zeros_like(pose))
    changes, minimum_strains = [], []
    for sign in (-1, 1):
        trial = sign*pose
        centers = model['centers']+trial[:, :3]
        rotation = Rotation.from_rotvec(trial[:, 3:]).as_matrix()
        anchors = centers[:, None, :]+np.einsum('jab,jib->jia', rotation, model['arms'])
        lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=2)
        changes.append(energy(model, trial)-start)
        minimum_strains.append(np.min(lengths/model['rest']-1))
    return dict(eigenvalue=values[0], amplitude=step, energy_changes=changes,
                quadratic_prediction=.5*values[0]*step**2,
                minimum_strains=minimum_strains)


def local_tilt(angle, a=.65, b=1.):
    """Analytic material tilt stiffness / (EA_strip*R), before prestrain."""
    product = a*b
    w = lambda r: 1/(b-a)/(r*r+product*product/r**2)**1.5
    r2 = quad(lambda r: w(r)*r*r, a, b)[0]
    q2 = quad(lambda r: w(r)*product*product/r**2, a, b)[0]
    rq = product*quad(w, a, b)[0]
    height2 = 2*product*np.cos(angle)
    return 1.5*height2*(r2*q2-rq*rq)/(r2+q2-2*rq*np.cos(angle))


def verify_local(degree):
    angle = np.deg2rad(degree)
    model = build(angle, force=0, cells=1)
    h = model['matrix'][6:, 6:]
    index = [0, 1, 3, 4]
    block = h[np.ix_(index, index)]
    reduced = block[2:, 2:]-block[2:, :2]@np.linalg.solve(block[:2, :2], block[:2, 2:])
    exact = local_tilt(angle)
    return dict(angle=degree, rank=int(np.linalg.matrix_rank(h, tol=1e-10)),
                stiffness=exact, numeric_stiffness=np.linalg.eigvalsh(reduced).tolist(),
                maximum_error=np.max(np.abs(reduced-exact*np.eye(2))))


def textile_metrics(a=.65, b=1.):
    product, width = a*b, b-a
    density = lambda r: max(1., product/r**2)
    allowance = quad(density, a, b, points=[np.sqrt(product)])[0]
    rows = []
    for degree in (0, 15, 30, 45, 60, 75, 90):
        angle = np.deg2rad(degree)
        for t in (0, .25, .5, .75, 1):
            path = quad(lambda r: np.sqrt((1-t)**2+(t*product/r**2)**2
                           -2*t*(1-t)*product/r**2*np.cos(angle)), a, b)[0]
            rows.append(dict(angle=degree, fraction=t, crosswise_path=path,
                             slack_vs_local_allowance=1-path/allowance))
    return dict(seam_width=width, local_allowance=allowance,
                exact_allowance=2*(b-np.sqrt(product)),
                allowance_over_seam=allowance/width-1,
                longest_over_shortest_warp=np.sqrt((a*a+b*b)/(2*product))-1,
                rows=rows)


def main():
    rows = []
    for ratio in (.001, 1/300, .01, .05):
        for degree in (15, 30, 45, 60, 75, 85, 90):
            for edges in (False, True):
                model = build(np.deg2rad(degree), force=3*ratio, edge_only=edges)
                rows.append(dict(force_over_total_ea=ratio, angle=degree,
                                 material='edges' if edges else 'panel', **metrics(model)))
    example = build(np.deg2rad(45), force=.01)
    convergence = []
    for count in (8, 16, 24, 40):
        convergence.append(dict(points=count, **metrics(build(np.deg2rad(75), .01, points=count))))
    critical = []
    for ratio in (.001, 1/300, .01, .05):
        for edges in (False, True):
            def minimum(degree):
                return metrics(build(np.deg2rad(degree), 3*ratio, points=12,
                                     edge_only=edges))['minimum_tilt']
            degree = brentq(minimum, 15, 90, xtol=1e-7)
            critical.append(dict(force_over_total_ea=ratio,
                                 material='edges' if edges else 'panel',
                                 critical_angle=degree))
    report = dict(model='12 cells, 3 strips, a=.65R, b=R, EA_strip=1, R=1',
                  limitations=['Independent elastic warp fibers; carrier, clamps, contact and yarn slip omitted.',
                               'Positive tangent eigenvalues are local stability, not global stability.',
                               'Translation scaled by R; rotations in radians; Hessian divided by EA_strip*R.',
                               'Under a positive tensile load the chain has positive height even at 90 degrees.'],
                  representative='force=.01 corresponds to F=10 N and EA_strip=1000 N',
                  rows=rows, critical_angles=critical, finite_difference=verify(example),
                  unstable_mode_check=verify_unstable(build(np.deg2rad(85), .01)),
                  convergence=convergence, local_tilt=[verify_local(d) for d in (15,45,75,85)],
                  textile=textile_metrics())
    path = Path(__file__).with_name('reciprocal_results.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(path)
    print('Representative F=10 N, EA_strip=1000 N:')
    for row in rows:
        if row['force_over_total_ea'] == 1/300:
            print(row)
    print('Worst finite difference relative error:', max(max(r['relative_errors']) for r in report['finite_difference']))
    print('Critical angles:', critical)


if __name__ == '__main__':
    main()
