import json
from pathlib import Path

import nbformat
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, minimize


root = Path(__file__).resolve().parent
notebook = nbformat.read(root / 'full_stroke_radius_experiment.ipynb', as_version=4)
model = {}
exec(notebook.cells[1].source, model)
length = model['length']
ratio = model['ratio']
radii = np.array(model['radii'])
masses = np.array([0.020, 0.500])
floor = 0.15005
steps = 850
end = np.sqrt(1 - ratio**2)
sample = np.linspace(0, end, 2 * steps + 1)
q = radii[:, None] * sample / np.sqrt(1 - sample**2)
scale = length * end / radii

volts = np.linspace(1, 10, 91)
stall_max = min(model['envelope'](v, 0) * model['voltage'] / v for v in volts)
free_max = min(np.interp(v, np.arange(1, 11), model['free_data']) * model['voltage'] / v
               for v in volts)
bounds = [(0.020, stall_max), (1500, free_max)]


@njit(inline='always')
def rate(k, q, mass, inertia, stall, free, scale):
    omega = np.sqrt(2 * max(k, 0) / (inertia + mass * q * q))
    return scale * (stall * (1 - omega / free) - mass * 9.81 * q)


@njit
def fast(stall, rpm, inertia):
    energy = np.empty((2, 3))
    free = rpm * 2 * np.pi / 60
    step = 1 / steps
    for m in range(2):
        mass = masses[m]
        for r in range(3):
            k = 0.0
            for j in range(steps):
                a = rate(k, q[r, 2*j], mass, inertia, stall, free, scale[r])
                b = rate(k + step*a/2, q[r, 2*j+1], mass, inertia, stall, free, scale[r])
                c = rate(k + step*b/2, q[r, 2*j+1], mass, inertia, stall, free, scale[r])
                d = rate(k + step*c, q[r, 2*j+2], mass, inertia, stall, free, scale[r])
                k += step * (a + 2*b + 2*c + d) / 6
            body = mass * q[r, -1]**2
            energy[m, r] = k * body / (inertia + body)
    return energy


def gaps(energy):
    return np.array([energy[0, 1] / energy[0, 0] - 1,
                     energy[0, 2] / energy[0, 1] - 1,
                     energy[1, 0] / energy[1, 1] - 1,
                     energy[1, 1] / energy[1, 2] - 1])


def exact(parameters, inertia):
    case = {'stall': parameters[0], 'free_rpm': parameters[1], 'inertia': inertia}
    return np.array([[model['energy'](mass, radius, case) for radius in radii]
                     for mass in masses])


def score(parameters, inertia):
    energy = fast(parameters[0], parameters[1], inertia)
    if np.any(energy <= 0):
        return 100.0
    height = energy / (masses[:, None] * 9.81)
    return -gaps(energy).min() + 1000 * max(0, floor - height.min())


results = {}
print('Bounds:', bounds, 'height floor:', floor, flush=True)
for name, original in model['cases'].items():
    inertia = original['inertia']
    runs = []
    for seed in [180, 181, 182]:
        result = differential_evolution(score, bounds, args=(inertia,), seed=seed,
                                        popsize=22, maxiter=240, tol=1e-9,
                                        polish=False)
        runs.append(result)
        print(name, 'seed', seed, 'parameters', result.x,
              'margin', -result.fun, 'evaluations', result.nfev, flush=True)
    result = min(runs, key=lambda run: run.fun)

    # Refine against the adaptive full-stroke solver, including its height constraint.
    def goal(parameters):
        return -gaps(exact(parameters, inertia)).min()

    def height(parameters):
        return (exact(parameters, inertia) / (masses[:, None] * 9.81)).min() - floor

    final = minimize(goal, result.x, method='SLSQP', bounds=bounds,
                     constraints={'type': 'ineq', 'fun': height},
                     options={'ftol': 1e-11, 'maxiter': 100, 'eps': 1e-7})
    parameters = final.x
    energies = exact(parameters, inertia)
    heights = energies / (masses[:, None] * 9.81)
    error = np.max(np.abs(fast(*parameters, inertia) / energies - 1))
    assert heights.min() >= floor - 1e-8
    results[name] = {'inertia': inertia, 'stall': parameters[0],
                     'free_rpm': parameters[1], 'margin': gaps(energies).min(),
                     'gaps': gaps(energies).tolist(), 'heights': heights.tolist(),
                     'fast_relative_error': error, 'refine_success': bool(final.success),
                     'seed_parameters': [run.x.tolist() for run in runs],
                     'seed_margins': [-run.fun for run in runs]}
    print(name, json.dumps(results[name]), flush=True)
    (root / 'results/optimization.json').write_text(json.dumps(results, indent=2) + '\n')
