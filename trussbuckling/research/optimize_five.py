"""Reproducible search for five-cord alignment stiffness over a folding stroke.

Radii are >=0.6 and <=1; at least one anchor is at radius1. Radius products
are identical, so every candidate has an exact coaxial constant-length path.
All cords share the same EA and sum(EA)/F=6000. H0/R=sqrt(1.2), N=12.
Minimum positive axial-load share is selectable. Exact pair-intersection angles
must exceed80 degrees, giving a5-degree margin beyond the operating stroke.

The result is a numerical best-found design, not a global-optimality proof.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.special import softmax

from .five_cords import contact_angles, make_design, mirror_design, pentagon, stiffness


ANGLES = np.radians([0, 15, 30, 45, 60, 75])


def search_mirror(contact=90.):
    """One-variable search for the practical three-radius mirrored design.

    The contact angle fixes beta-alpha analytically. Only alpha is optimized;
    four cords exchange radii1/.6 and the fifth uses sqrt(.6) at both ends.
    """
    gap = 2*np.arctan(.25*np.tan(np.radians(contact)/2))
    def cost(alpha):
        design = mirror_design(alpha, alpha+gap)
        if design.shares.min() < .05:
            return 1e5
        return -min(stiffness(design, angle) for angle in ANGLES)
    grid = np.linspace(gap/2+1e-5, np.pi/2-gap-1e-5, 161)
    index = int(np.argmin([cost(a) for a in grid]))
    result = minimize_scalar(cost, bounds=(grid[max(0,index-1)], grid[min(len(grid)-1,index+1)]),
                             method="bounded", options={"xatol":1e-12})
    alpha = result.x
    design = mirror_design(alpha, alpha+gap)
    angles = np.radians(np.linspace(0, 75, 1501))
    values = np.array([stiffness(design, angle) for angle in angles])
    return {"family":"reflection and radius swap", "alpha_degrees":float(np.degrees(alpha)),
        "beta_degrees":float(np.degrees(alpha+gap)), "product":.6,
        "angles_degrees":np.degrees(design.angles).tolist(), "bottom_radii":design.bottom.tolist(),
        "top_radii":design.top.tolist(), "load_shares":design.shares.tolist(),
        "worst_stiffness":float(values.min()), "worst_angle_degrees":float(np.degrees(angles[values.argmin()])),
        "first_contact_degrees":float(np.degrees(contact_angles(design).min())),
        "cells":12, "height0_over_radius":float(np.sqrt(1.2)), "total_EA_over_load":6000.}


def decode(values, regular=False):
    if regular:
        angles = np.arange(5)*2*np.pi/5
        fractions, product = values[:4], values[4]
    else:
        gaps = np.radians(1)+np.radians(355)*softmax(np.r_[values[:4], 0.])
        angles = np.r_[0., np.cumsum(gaps)[:4]]
        fractions, product = values[4:8], values[8]
    bottom = np.r_[1., product+(1-product)*np.asarray(fractions)]
    return make_design(angles, bottom, product)


def objective(values, regular=False, angles=ANGLES, minimum_share=.05, penalize=True):
    try:
        design = decode(values, regular)
        if penalize and np.min(design.shares) < minimum_share-1e-10:
            return 10+100*(minimum_share-np.min(design.shares))
        if penalize and np.min(contact_angles(design)) < np.radians(80)-1e-8:
            return 10+100*(np.radians(80)-np.min(contact_angles(design)))
        return -min(stiffness(design, angle) for angle in angles)
    except (ValueError, np.linalg.LinAlgError):
        return 1e5


def search(seed=1, iterations=250, regular=False, minimum_share=.05, start=None):
    base = pentagon()
    fractions = (base.bottom[1:]-.6)/.4
    initial = np.r_[fractions, .6] if regular else np.r_[np.zeros(4), fractions, .6]
    if start is not None:
        prior = json.loads(Path(start).read_text())
        product = prior["product"]
        fractions = (np.asarray(prior["bottom_radii"])[1:]-product)/(1-product)
        if regular:
            initial = np.r_[fractions, product]
        else:
            gaps = np.diff(np.r_[prior["angles_degrees"], 360.])
            initial = np.r_[np.log((gaps[:4]-1)/(gaps[-1]-1)), fractions, product]
    # Preserve positive height at75deg. The optimizer may choose not to use
    # the full allowed radial contrast; product0.6 uses both .6 and1 exactly.
    maximum = (1.2-.02**2)/(4*np.sin(np.radians(75)/2)**2)
    bounds = ([(0, 1)]*4+[(.6, maximum)]) if regular else (
        [(-6, 6)]*4+[(0, 1)]*4+[(.6, maximum)]
    )
    count = 0
    def progress(values, convergence):
        nonlocal count
        count += 1
        if count % 25 == 0:
            print(f"seed={seed}, generation={count}, stiffness={-objective(values, regular, minimum_share=minimum_share):.8f}", flush=True)
    result = differential_evolution(lambda x: objective(x, regular, minimum_share=minimum_share), bounds, x0=initial,
        seed=seed, maxiter=iterations, popsize=10, tol=1e-7, polish=False, callback=progress)
    values = result.x
    # Add any interior-stroke minima found by a much finer independent sweep.
    active_angles = ANGLES.copy()
    dense_angles = np.radians(np.linspace(0, 75, 301))
    constraints = [
        {"type":"ineq", "fun":lambda x:decode(x, regular).shares-minimum_share},
        {"type":"ineq", "fun":lambda x:contact_angles(decode(x, regular))-np.radians(80)},
    ]
    for _ in range(6):
        local = minimize(lambda x: objective(x, regular, active_angles, minimum_share, False), values,
            method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter":400, "ftol":1e-10})
        feasible = min(np.min(c["fun"](local.x)) for c in constraints) >= -1e-7
        if feasible and local.fun < objective(values, regular, active_angles, minimum_share, False):
            values = local.x
        design = decode(values, regular)
        dense = np.array([stiffness(design, a) for a in dense_angles])
        worst = dense_angles[np.argmin(dense)]
        active_score = -objective(values, regular, active_angles, minimum_share, False)
        if active_score-dense.min() < 1e-5:
            break
        active_angles = np.unique(np.r_[active_angles, worst])
    output = {
        "seed":seed, "regular_angles":regular, "iterations":iterations,
        "height0_over_radius":float(np.sqrt(1.2)), "total_EA_over_load":6000., "cells":12,
        "stroke_degrees":[0, 75], "minimum_allowed_load_share":minimum_share,
        "minimum_allowed_contact_degrees":80.,
        "product":float(design.product), "angles_degrees":np.degrees(design.angles).tolist(),
        "bottom_radii":design.bottom.tolist(), "top_radii":design.top.tolist(),
        "load_shares":design.shares.tolist(), "worst_stiffness":float(dense.min()),
        "worst_angle_degrees":float(np.degrees(worst)), "variables":values.tolist(),
        "first_contact_degrees":float(np.degrees(contact_angles(design).min())),
        "sweep_angles_degrees":np.degrees(dense_angles).tolist(), "sweep_stiffness":dense.tolist(),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--regular", action="store_true")
    parser.add_argument("--minimum-share", type=float, default=.05)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--mirror-contact", type=float, help="Use the simple one-variable mirrored search.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = search_mirror(args.mirror_contact) if args.mirror_contact is not None else search(
        args.seed, args.iterations, args.regular, args.minimum_share, args.start)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not k.startswith("sweep_")}, indent=2))


if __name__ == "__main__":
    main()
