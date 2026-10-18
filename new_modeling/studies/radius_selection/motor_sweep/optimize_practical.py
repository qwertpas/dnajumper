import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from new_modeling.studies.radius_selection.motor_sweep.optimize import endpoint_heights, margins

ROOT = Path(__file__).resolve().parent / 'results/optimization'
ROOT.mkdir(parents=True, exist_ok=True)

def run(name,goal,height,cap,max_rpm,seed):
    def limits(p):
        h=endpoint_heights(p)
        power,low,high=10**p[1],10**p[2],10**p[3]
        if not np.isfinite(h).all():return np.full(8,-1.)
        return np.r_[h.ravel()-height,cap-4*power/(low*np.pi/30),np.log(high/low)]
    def objective(p):
        h=endpoint_heights(p)
        if not np.isfinite(h).all():return 1e3
        mm=margins(h)
        if goal=='winners':return -min(mm[[0,2]])
        return -min(mm)
    fit=differential_evolution(objective,[(.02,.5),(-1,np.log10(cap*max_rpm*np.pi/30/4)),(np.log10(1100),np.log10(max_rpm)),(np.log10(1100),np.log10(max_rpm))],constraints=NonlinearConstraint(limits,0,np.inf),popsize=20,maxiter=260,seed=seed,tol=2e-7,polish=False)
    p=fit.x
    h=endpoint_heights(p,2400)
    result={'name':name,'goal':goal,'minimum_height':height,'torque_limit':cap,'maximum_rpm':max_rpm,'mass':float(p[0]),'power':float(10**p[1]),'rpm_low':float(10**p[2]),'rpm_high':float(10**p[3]),'heights':h.tolist(),'margins':margins(h).tolist(),'score':float(-fit.fun),'success':bool(fit.success),'evaluations':int(fit.nfev)}
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    out=[]
    for args in [('winners_only','winners',.15,.26,21643.4236,45),('practical_all','all',.18,.234,20000.,48),('practical_winners','winners',.18,.234,20000.,69)]:
        out.append(run(*args))
        (ROOT/'practical_search.json').write_text(json.dumps(out,indent=2))
