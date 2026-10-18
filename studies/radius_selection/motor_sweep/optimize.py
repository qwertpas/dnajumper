"""Optimize mass, power, and motor-speed endpoints at fixed 180 mm."""
import json
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, NonlinearConstraint
from model import energy, INERTIA, RADII

ROOT = Path(__file__).resolve().parent / 'results/optimization'
ROOT.mkdir(parents=True, exist_ok=True)
MAX_RPM = 21643.4236
MAX_TORQUE = .260

@njit
def endpoint_heights(p, steps=600):
    mass,power,low,high=p[0],10**p[1],10**p[2],10**p[3]
    h=np.empty((2,3))
    for i,rpm in enumerate([low,high]):
        speed=rpm*np.pi/30
        stall=4*power/speed
        for j,r in enumerate(RADII):
            h[i,j]=energy(mass,r,.18,stall,speed,INERTIA,steps)/(mass*9.81)
    return h


def constraints(p):
    h=endpoint_heights(p)
    power,low,high=10**p[1],10**p[2],10**p[3]
    if not np.isfinite(h).all(): return np.full(8,-1.)
    return np.r_[h.ravel()-.15, MAX_TORQUE-4*power/(low*np.pi/30),np.log(high/low)]


def margins(h):
    return np.array([h[0,2]/h[0,1]-1,h[0,1]/h[0,0]-1,
                     h[1,0]/h[1,1]-1,h[1,1]/h[1,2]-1])


def score(p,kind):
    h=endpoint_heights(p)
    if not np.isfinite(h).all():return 1e3
    if kind=='relative':return -min(margins(h))
    return -min(h[0,2]-h[0,1],h[0,1]-h[0,0],h[1,0]-h[1,1],h[1,1]-h[1,2])


if __name__=='__main__':
    out=[]
    for kind,seed in [('relative',41),('relative',78),('absolute',46)]:
        fit=differential_evolution(lambda p:score(p,kind),[(.02,.5),(-1,np.log10(.26*MAX_RPM*np.pi/30/4)),(np.log10(1100),np.log10(MAX_RPM)),(np.log10(1100),np.log10(MAX_RPM))],constraints=NonlinearConstraint(constraints,0,np.inf),popsize=20,maxiter=240,seed=seed,tol=2e-7,polish=False)
        p=fit.x
        h=endpoint_heights(p,2400)
        result={'objective':kind,'seed':seed,'mass':float(p[0]),'power':float(10**p[1]),'rpm_low':float(10**p[2]),'rpm_high':float(10**p[3]),'heights':h.tolist(),'margins':margins(h).tolist(),'min_margin':float(min(margins(h))),'score':float(-fit.fun),'success':bool(fit.success),'evaluations':int(fit.nfev)}
        print(json.dumps(result,indent=2),flush=True)
        out.append(result)
        (ROOT/'optimization_search.json').write_text(json.dumps(out,indent=2))
