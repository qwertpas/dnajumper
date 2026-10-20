"""Prestress stability of three taut paths, not a verified fabric surface.

One path per 120-degree sector; all three use the same radius on each plate.
Plate radii alternate R and r. The natural path length sqrt(R^2+r^2) permits
an unloaded flat state at 90 degrees. Loaded height is solved with fixed rest
lengths. End positions and orientations are fixed except top axial motion.

R=1, EA=1 per path and F=.04 represent R=100 mm, EA=1000 N and F=40 N.
The elastic Hessian includes prestress. Rank three alone does not imply
instability: tension stabilizes the tested short chains below about 61 deg.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from ..panel_study.reciprocal import skew, curvature


def build(angle, ratio=1., cells=6, force=.04):
    length = np.sqrt(1+ratio*ratio)
    height0 = np.sqrt(2*ratio*np.cos(angle))
    def balance(h):
        current = np.sqrt(length*length+h*h-2*ratio*np.cos(angle))
        return 3*(current/length-1)*h/current-force
    height = brentq(balance, height0, height0+10.)
    centers = np.column_stack((np.zeros(cells+1), np.zeros(cells+1), np.arange(cells+1)*height))
    arms = np.empty((cells+1, 3, 3))
    for j in range(cells+1):
        radius = 1. if j % 2 == 0 else ratio
        phi = np.arange(3)*2*np.pi/3+j*angle
        arms[j] = np.column_stack((radius*np.cos(phi), radius*np.sin(phi), np.zeros(3)))
    size = 6*(cells+1)
    matrix, gradient = np.zeros((size, size)), np.zeros(size)
    strains = []
    for j in range(cells):
        index = np.arange(6*j, 6*j+12)
        for i in range(3):
            d = centers[j+1]+arms[j+1, i]-centers[j]-arms[j, i]
            current = np.linalg.norm(d)
            unit = d/current
            tension = current/length-1
            strains.append(tension)
            first = np.column_stack((-np.eye(3), skew(arms[j, i]), np.eye(3), -skew(arms[j+1, i])))
            row = first.T@unit
            block = np.outer(row, row)/length+tension/current*first.T@(np.eye(3)-np.outer(unit, unit))@first
            block[3:6, 3:6] -= tension/current*curvature(d, arms[j, i])
            block[9:12, 9:12] += tension/current*curvature(d, arms[j+1, i])
            matrix[np.ix_(index, index)] += block
            gradient[index] += tension*row
    gradient[6*cells+2] -= force
    free = np.r_[np.arange(6, 6*cells), 6*cells+2]
    tilt = np.array([6*j+k for j in range(1, cells) for k in (0, 1, 3, 4)])
    return dict(matrix=matrix, gradient=gradient, free=free, tilt=tilt,
                centers=centers, arms=arms, rest=length, height=height,
                force=force, strains=np.array(strains))


def metrics(model):
    matrix = model['matrix']
    return dict(minimum_full=float(np.linalg.eigvalsh(matrix[np.ix_(model['free'], model['free'])])[0]),
                minimum_tilt=float(np.linalg.eigvalsh(matrix[np.ix_(model['tilt'], model['tilt'])])[0]),
                height=model['height'], minimum_strain=float(model['strains'].min()),
                equilibrium_residual=float(max(abs(model['gradient'][model['free']]))))


def independent_check(model):
    """Autodiff of the spring potential, independent of the analytic Hessian."""
    import torch
    from ..manufacturable_panels.cloth_mechanics import skew as tensor_skew
    dtype = torch.float64
    centers = torch.tensor(model['centers'], dtype=dtype)
    arms = torch.tensor(model['arms'], dtype=dtype)
    free = model['free']
    def energy(q):
        pose = torch.zeros(model['matrix'].shape[0], dtype=dtype)
        pose[free] = q
        pose = pose.reshape(-1, 6)
        rotation = torch.matrix_exp(tensor_skew(pose[:, 3:]))
        points = centers[:, None]+pose[:, None, :3]+torch.einsum('jab,jib->jia', rotation, arms)
        lengths = torch.linalg.vector_norm(points[1:]-points[:-1], dim=2)
        return torch.sum((lengths-model['rest'])**2)/(2*model['rest'])-model['force']*(centers[-1, 2]+pose[-1, 2])
    zero = torch.zeros(len(free), dtype=dtype)
    actual = torch.autograd.functional.hessian(energy, zero).numpy()
    expected = model['matrix'][np.ix_(free, free)]
    values, vectors = np.linalg.eigh(actual)
    step = 1e-4
    start = float(energy(zero))
    changes = [float(energy(torch.tensor(s*step*vectors[:, 0], dtype=dtype)))-start for s in (-1, 1)]
    return dict(maximum_hessian_difference=float(abs(actual-expected).max()),
                weakest_eigenvalue=float(values[0]), mode_step=step, energy_changes=changes,
                quadratic_prediction=float(.5*values[0]*step**2))


def main():
    rows, critical = [], []
    for ratio in (1., .8, .65):
        for degree in (0, 15, 30, 45, 55, 60, 70, 80):
            rows.append(dict(ratio=ratio, angle=degree, **metrics(build(np.deg2rad(degree), ratio))))
        threshold = brentq(lambda degree: metrics(build(np.deg2rad(degree), ratio))['minimum_tilt'], .001, 89)
        critical.append(dict(ratio=ratio, angle=threshold))
    checks = [dict(angle=angle, **independent_check(build(np.deg2rad(angle), .8))) for angle in (45, 60, 70)]
    report = dict(cells=6, force_N=40, EA_per_path_N=1000, radius_mm=100,
                  model='Three straight elastic tensile paths; no fabric continuity or attachment geometry is asserted.',
                  coordinates='Translations/R, rotations/radians; energy divided by EA_per_path*R.',
                  rows=rows, critical_angles=critical, independent_checks=checks)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Critical angles:', critical)
    print('Independent checks:', checks)


if __name__ == '__main__':
    main()
