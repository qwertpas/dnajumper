"""Reproduce the six-cell 28.9-degree local stability audit."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from .cloth_mechanics import analyze, verify


def check(angle, shear):
    return analyze(angle, .65, shear, cells=6, offset=np.deg2rad(28.9))


def main():
    rows=[]
    for angle in (0,15,30,45,60,65,75,80,85,86,89):
        for shear in (0,.01,.05):
            row,energy,matrix=check(angle,shear)
            rows.append(row)
    critical=[]
    for shear in (0,.01,.05):
        angle=brentq(lambda x:check(x,shear)[0]['minimum_full'],80,89,xtol=1e-7)
        critical.append(dict(shear_ratio=shear,angle=angle))
    row,energy,matrix=check(45,.05)
    cholesky=np.linalg.cholesky(matrix)
    stable=dict(row=row,finite_difference=verify(energy,matrix),
                cholesky_residual=float(np.max(abs(cholesky@cholesky.T-matrix))))
    row,energy,matrix=check(86,.05)
    values,vectors=np.linalg.eigh(matrix)
    base=float(energy(torch.zeros(len(matrix),dtype=torch.float64)))
    changes=[]
    for step in (1e-3,3e-4,1e-4):
        delta=torch.tensor(step*vectors[:,0],dtype=torch.float64)
        changes.append(dict(step=step,plus=float(energy(delta))-base,
                            minus=float(energy(-delta))-base,
                            prediction=.5*float(values[0])*step**2))
    report=dict(assumptions='Six cells, r/R=.65, offset28.9deg; R100mm; F10N tension; EA1000N per strap. Bottom fixed; top x,y and all rotations fixed, top z free. Internal plates free in all six coordinates. Quadratic fabric shear; two affine triangles per strap; locking, contact, bending, slip and independent wrinkles omitted.',
                energy='sum_edges EA_edge/(2L)*(ell-L)^2 + sum_facets kappa/4*gamma^2 - F*z_top',
                normalization='Lengths/R, energy/(EA_strap R); kappa/(FR) reported as shear ratio.',
                rows=rows,critical_angles=critical,stable_check=stable,
                unstable_check=dict(row=row,energy_changes=changes))
    path=Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(critical=critical,stable=stable,unstable=report['unstable_check']),indent=2))


if __name__=='__main__':
    main()
