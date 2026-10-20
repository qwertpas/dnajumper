"""Finite shear in a continuous, two-facet woven-fabric strap chain.

Each strap has two reinforced edges (effective EA=.5 each), and two affine cloth
triangles joined along the same material diagonal in every cell. Cloth energy
per strap is kappa/4*(gamma1**2+gamma2**2). Here gamma is yarn-angle shear,
and kappa=G2D*W*L. Uniform body warp rigidity can be included in total EA:
each affine triangle has the axial strain of its adjoining long edge and
half the material area, so half the body's rigidity belongs to each edge.
Independent interior deformation, bending, contact, friction, locking and
hysteresis are omitted; this is a finite-shear two-facet model.

Lengths use outer radius R=1; forces use total EA per strap=1. F=.01 thus
represents F=10 N with EA=1000 N per strap. Tensor rotations are exponential
coordinates. The full Hessian includes geometric stiffness automatically.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root
import torch


DTYPE = torch.float64


def pair_energy(x, average, ratio, force, kappa, offset=np.pi/3):
    length = np.sqrt(1+ratio**2)
    width = np.sqrt(1+ratio**2-2*ratio*np.cos(offset))
    total = -force*(x[0]+x[1])
    for j, (a, b) in enumerate(((1., ratio), (ratio, 1.))):
        angle = average+(1 if j == 0 else -1)*x[2]
        current = torch.sqrt(length**2+x[j]**2-2*ratio*torch.cos(angle))
        cosine = (b*(a*np.cos(offset)-b)+a*(b-a*np.cos(offset))*torch.cos(angle)
                  +a*a*np.sin(offset)*torch.sin(angle))/(width*current)
        shear = torch.asin(cosine)
        total = total+3/(2*length)*(current-length)**2+1.5*kappa*shear**2
    return total


def equilibrium(average, ratio, force, kappa, offset=np.pi/3):
    length = np.sqrt(1+ratio**2)
    height0 = np.sqrt(2*ratio*np.cos(average))
    def balance(h):
        current = np.sqrt(length**2+h*h-2*ratio*np.cos(average))
        return 3*h*(current/length-1)/current-force
    height = brentq(balance, height0, height0+1.)
    def gradient(x):
        tensor = torch.tensor(x, dtype=DTYPE, requires_grad=True)
        value = pair_energy(tensor, average, ratio, force, kappa, offset)
        return torch.autograd.grad(value, tensor)[0].detach().numpy()
    def hessian(x):
        tensor = torch.tensor(x, dtype=DTYPE)
        return torch.autograd.functional.hessian(
            lambda v: pair_energy(v, average, ratio, force, kappa, offset), tensor).numpy()
    result = root(gradient, [height, height, 0.], jac=hessian, tol=1e-11)
    residual = max(abs(gradient(result.x)))
    if residual > 1e-9:
        raise ValueError(f'Pair equilibrium residual {residual}')
    return result.x, residual


def skew(v):
    zero = torch.zeros_like(v[:, 0])
    return torch.stack((zero, -v[:, 2], v[:, 1], v[:, 2], zero, -v[:, 0],
                        -v[:, 1], v[:, 0], zero), dim=1).reshape(-1, 3, 3)


def chain_energy(average, ratio, force, kappa, pair, cells=12, offset=np.pi/3,
                 top_shift=(0., 0.), top_tilt=(0., 0., 0.)):
    length = np.sqrt(1+ratio**2)
    heights = np.array([pair[j % 2] for j in range(cells)])
    angles = np.array([average+(1 if j % 2 == 0 else -1)*pair[2] for j in range(cells)])
    centers = np.zeros((cells+1, 3))
    centers[:, 2] = np.r_[0, np.cumsum(heights)]
    centers[-1, :2] = top_shift
    phase = np.r_[0, np.cumsum(angles)]
    arms = np.empty((cells+1, 6, 3))
    for j in range(cells+1):
        radii = np.where((np.arange(6)+j) % 2 == 0, 1., ratio)
        azimuth = np.repeat(np.arange(3)*2*np.pi/3, 2)+np.tile([0, offset], 3)+phase[j]
        arms[j] = np.column_stack((radii*np.cos(azimuth), radii*np.sin(azimuth), np.zeros(6)))
    centers = torch.tensor(centers, dtype=DTYPE)
    arms = torch.tensor(arms, dtype=DTYPE)
    free = np.r_[np.arange(6, 6*cells), 6*cells+2]
    tilt_full = [6*j+k for j in range(1, cells) for k in (0, 1, 3, 4)]
    tilt = np.array([list(free).index(i) for i in tilt_full])
    def energy(vector):
        pose = torch.zeros(6*(cells+1), dtype=DTYPE)
        pose[free] = vector
        pose = pose.reshape(cells+1, 6)
        fixed_rotation = torch.zeros((cells+1, 3), dtype=DTYPE)
        fixed_rotation[-1] = torch.tensor(top_tilt, dtype=DTYPE)
        rotation = torch.matrix_exp(skew(pose[:, 3:]+fixed_rotation))
        points = centers[:, None, :]+pose[:, None, :3]+torch.einsum('jab,jib->jia', rotation, arms)
        lengths = torch.linalg.vector_norm(points[1:]-points[:-1], dim=2)
        total = torch.sum((lengths-length)**2)/(4*length)-force*(centers[-1, 2]+pose[-1, 2])
        a, b = points[:-1, ::2], points[:-1, 1::2]
        c, d = points[1:, 1::2], points[1:, ::2]
        for u, v in ((b-a, c-b), (c-d, d-a)):
            cosine = torch.sum(u*v, dim=2)/(torch.linalg.vector_norm(u, dim=2)*torch.linalg.vector_norm(v, dim=2))
            total = total+kappa/4*torch.sum(torch.asin(cosine)**2)
        return total
    return energy, len(free), tilt, heights, angles


def analyze(average_degrees, ratio, shear_ratio, force=.01, cells=12, offset=np.pi/3):
    average = np.deg2rad(average_degrees)
    kappa = shear_ratio*force
    pair, pair_error = equilibrium(average, ratio, force, kappa, offset)
    energy, size, tilt, heights, angles = chain_energy(average, ratio, force, kappa, pair, cells, offset)
    zero = torch.zeros(size, dtype=DTYPE, requires_grad=True)
    gradient = torch.autograd.grad(energy(zero), zero)[0].detach().numpy()
    matrix = torch.autograd.functional.hessian(energy, zero).detach().numpy()
    length = np.sqrt(1+ratio**2)
    strains = np.sqrt(length**2+heights**2-2*ratio*np.cos(angles))/length-1
    return dict(ratio=ratio, average_angle=average_degrees, kappa_over_force_radius=shear_ratio,
                angles=np.rad2deg(angles[:2]).tolist(), heights=heights[:2].tolist(),
                edge_strains=strains[:2].tolist(), minimum_full=float(np.linalg.eigvalsh(matrix)[0]),
                minimum_tilt=float(np.linalg.eigvalsh(matrix[np.ix_(tilt, tilt)])[0]),
                pair_residual=float(pair_error), full_gradient_residual=float(max(abs(gradient)))), energy, matrix


def verify(energy, matrix):
    rng = np.random.default_rng(1729)
    direction = rng.normal(size=len(matrix))
    direction /= np.linalg.norm(direction)
    exact = direction@matrix@direction
    zero = torch.zeros(len(matrix), dtype=DTYPE)
    start = float(energy(zero))
    rows = []
    for step in (1e-3, 3e-4, 1e-4):
        q = torch.tensor(step*direction, dtype=DTYPE)
        numeric = float(energy(q)+energy(-q)-2*start)/step**2
        rows.append(dict(step=step, numeric=numeric, analytic=exact,
                         relative_error=abs(numeric-exact)/abs(exact)))
    return rows


def main():
    rows = []
    for angle in (45, 60, 65):
        for shear in (0., .001, .01, .05):
            row, energy, matrix = analyze(angle, .8, shear)
            rows.append(row)
            print(row, flush=True)
    report = dict(model='12 cells, q=.8, F=10 N, EA=1000 N per strap; finite shear two-facet cloth model',
                  energy='U_s per strap = kappa/4*(gamma_1^2+gamma_2^2), kappa=G2D*W*L',
                  assumptions='Total EA may include uniform body warp rigidity and equal hems, lumped exactly within the affine-facet ansatz. Independent interior deformation, bending, locking, friction, hysteresis, thickness and contact omitted. End centers and rotations fixed, except axial displacement.',
                  rows=rows, finite_difference=verify(energy, matrix))
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
