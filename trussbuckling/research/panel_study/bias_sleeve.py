"""Optimistic two-family spring-network screen of a continuous bias sleeve.

Equal-radius anchors, fixed natural lengths, and tension-only linear fibers.
R=F=1; stiffness is kR/F, not EA/F. No cloth shear, contact, or friction.
Run with python3.11 -m trussbuckling.research.panel_study.bias_sleeve.
"""

import json
from math import pi
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def cross(point):
    x, y, z = point
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def state(angle, bias, stiffness, rest, count=3):
    offsets = np.array([angle+bias, angle-bias])
    chords = 2*(1-np.cos(offsets))

    def force(height):
        lengths = np.sqrt(height*height+chords)
        return count*height*np.sum(stiffness*np.maximum(lengths-rest, 0)/lengths)

    high = max(rest, 1.)
    while force(high) < 1.:
        high *= 2
    height = brentq(lambda h: force(h)-1., 1e-12, high, xtol=1e-13)
    lengths = np.sqrt(height*height+chords)
    tension = stiffness*np.maximum(lengths-rest, 0)
    torque = count*np.sum(tension*np.sin(offsets)/lengths)
    return height, lengths, tension, float(torque)


def model(angle, bias, stiffness, rest, cells=12, count=3, include_energy=False):
    height, lengths, tension, torque = state(angle, bias, stiffness, rest, count)
    size = 6*cells
    matrix = np.zeros((size, size))
    gradient = np.zeros(size)
    base = np.arange(count)*2*pi/count
    arms = np.empty((cells+1, 2, count, 3))
    for j in range(cells+1):
        for family, offset in enumerate((bias, -bias)):
            phase = base+j*(angle+offset)+family*pi/count
            arms[j, family] = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(count)))
    first_rows = []
    for j in range(cells):
        for family in range(2):
            length = lengths[family]
            prestress = tension[family]/length
            material = stiffness if tension[family] > 0 else 0.
            for i in range(count):
                bottom, top = arms[j, family, i], arms[j+1, family, i]
                link = top-bottom+[0, 0, height]
                derivative = np.column_stack((-np.eye(3), cross(bottom), np.eye(3), -cross(top)))
                row = link@derivative/length
                block = prestress*(derivative.T@derivative)+(material-prestress)*np.outer(row, row)
                for start, point, sign in ((3, bottom, -1), (9, top, 1)):
                    block[start:start+3, start:start+3] += sign*prestress*(
                        (np.outer(link, point)+np.outer(point, link))/2-np.dot(link, point)*np.eye(3))
                if j == 0:
                    matrix[:6, :6] += block[6:, 6:]
                    gradient[:6] += tension[family]*row[6:]
                    if material:
                        first_rows.append(row[6:])
                else:
                    where = slice(6*(j-1), 6*(j+1))
                    matrix[where, where] += block
                    gradient[where] += tension[family]*row
    free = np.r_[np.arange(6*(cells-1)), 6*(cells-1)+2]
    gradient[6*(cells-1)+2] -= 1.
    residual = float(np.max(np.abs(gradient[free])))
    assert residual < 1e-10
    hessian = matrix[np.ix_(free, free)]
    smallest = float(eigh(hessian, subset_by_index=[0, 0], eigvals_only=True)[0])
    result = dict(height=float(height), lengths=lengths.tolist(), tension=tension.tolist(),
                  torque=torque, smallest=smallest, stable=smallest > 1e-9,
                  taut=bool(np.all(tension > 1e-9)),
                  strain=((lengths-rest)/rest).tolist(), residual=residual,
                  rank=int(np.linalg.matrix_rank(first_rows)))
    if not include_energy:
        return result

    def energy(values):
        full = np.zeros(size)
        full[free] = values
        motion = full.reshape(cells, 6)
        points = arms.copy()
        points[:, :, :, 2] += np.arange(cells+1)[:, None, None]*height
        for j in range(1, cells+1):
            turn = Rotation.from_rotvec(motion[j-1, 3:]).as_matrix()
            points[j] = arms[j]@turn.T+motion[j-1, :3]+[0., 0., j*height]
        actual = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        return stiffness/2*np.sum(np.maximum(actual-rest, 0)**2)-motion[-1, 2]

    return result, hessian, energy


def verify():
    rng = np.random.default_rng(311)
    error = 0.
    for degrees, bias, stiffness, rest in ((0, 20, 3, 1), (10, 20, 3, 1),
                                         (40, 30, 1, 1.5), (40, 10, 100, 1)):
        result, hessian, energy = model(degrees*pi/180, bias*pi/180, stiffness, rest,
                                        include_energy=True)
        for _ in range(3):
            direction = rng.normal(size=len(hessian))
            direction /= np.linalg.norm(direction)
            step = 2e-4
            measured = (energy(step*direction)+energy(-step*direction)-2*energy(np.zeros(len(hessian))))/step**2
            predicted = float(direction@hessian@direction)
            error = max(error, abs(measured-predicted)/max(1., abs(predicted)))
            assert abs(measured-predicted) < 3e-6*max(1., abs(predicted))
    return error


def sweep():
    records = []
    limited = {limit: None for limit in (.05, .10, .20)}
    for bias in (10, 20, 30):
        for stiffness in (.3, .5, 1, 2, 3, 5, 10, 30, 100):
            for rest in (.5, .75, 1., 1.5, 2., 3.):
                initial = model(0., bias*pi/180, stiffness, rest)
                best = initial
                best_angle = 0
                peak_strain = max(initial['strain'])
                stop = 90
                reason = 'sweep_limit'
                for degrees in range(1, 91):
                    current = model(degrees*pi/180, bias*pi/180, stiffness, rest)
                    if not current['taut'] or not current['stable']:
                        stop = degrees
                        reason = 'slack' if not current['taut'] else 'unstable'
                        break
                    if current['height'] < best['height']:
                        best, best_angle = current, degrees
                    peak_strain = max(peak_strain, *current['strain'])
                    contraction = 1-current['height']/initial['height']
                    for limit, chosen in limited.items():
                        if peak_strain <= limit and (chosen is None or contraction > chosen['contraction']):
                            limited[limit] = dict(bias=bias, stiffness=stiffness, rest=rest,
                                                 initial=initial, chosen=current, angle=degrees,
                                                 contraction=contraction, peak_strain=peak_strain)
                records.append(dict(bias=bias, stiffness=stiffness, rest=rest,
                                    initial=initial, best=best, best_angle=best_angle,
                                    contraction=1-best['height']/initial['height'],
                                    stop=stop, reason=reason))
    for chosen in limited.values():
        record = next(r for r in records if all(r[key] == chosen[key]
                      for key in ('bias', 'stiffness', 'rest')))
        chosen['branch_stop'] = record['stop']
        chosen['branch_stop_reason'] = record['reason']
        _, hessian, _ = model(record['stop']*pi/180, chosen['bias']*pi/180,
                              chosen['stiffness'], chosen['rest'], include_energy=True)
        transverse = [6*j+i for j in range(11) for i in (0, 1, 3, 4)]
        axial = [6*j+i for j in range(11) for i in (2, 5)] + [66]
        chosen['stop_transverse_eigenvalue'] = float(eigh(hessian[np.ix_(transverse, transverse)],
                                                          subset_by_index=[0, 0], eigvals_only=True)[0])
        chosen['stop_axial_twist_eigenvalue'] = float(eigh(hessian[np.ix_(axial, axial)],
                                                           subset_by_index=[0, 0], eigvals_only=True)[0])
    return records, limited


def compact_state(state):
    return {key: state[key] for key in ('height', 'tension', 'strain', 'smallest', 'rank')}


def compact_record(record):
    return {key: compact_state(value) if key in ('initial', 'best', 'chosen') else value
            for key, value in record.items()}


if __name__ == '__main__':
    error = verify()
    records, limited = sweep()
    output = dict(model='12 cells, m=3 per family, R=F=1, k fixed, natural length fixed',
                  assumptions='Straight spring fibers; optimistic network, not a continuum cloth solution. '
                              'No contact, friction, thickness or stitch mechanics. '
                              'Clamps advance each family by +/- beta on every successive plate.',
                  equations='l_pm^2=h^2+2(1-cos(theta+/-beta)); T_pm=k max(l_pm-l0,0); '
                            '1=3h(T_plus/l_plus+T_minus/l_minus)',
                  scan=dict(bias_degrees=[10, 20, 30], stiffness=[.3, .5, 1, 2, 3, 5, 10, 30, 100],
                            rest=[.5, .75, 1., 1.5, 2., 3.], twist_degrees='0 through 90 in 1 degree steps',
                            parameter_sets=len(records), stop='First slack family or nonpositive Hessian eigenvalue',
                            strain_limit='Maximum fiber strain over the entire connected sampled branch'),
                  relative_hessian_error=error,
                  strain_limited={str(limit): compact_record(record) for limit, record in limited.items()},
                  records=[compact_record(record) for record in records])
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(output, separators=(',', ':'))+'\n')
    print(f'Hessian directional finite-difference error: {error:.3g}')
    for limit, record in limited.items():
        print(json.dumps(dict(strain_limit=limit, **compact_record(record))))
