"""40 N audit including imperfect end alignment; lengths normalized by R."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import root, brentq
from scipy.spatial.transform import Rotation
from .cloth_mechanics import analyze, equilibrium, chain_energy

OFFSET=np.deg2rad(28.9)
FORCE=.04


def aligned(angle,shear):
    return analyze(angle,.65,shear,force=FORCE,cells=6,offset=OFFSET)


def imperfect(angle, shift, tilt, shear=.05):
    average=np.deg2rad(angle)
    kappa=shear*FORCE
    pair,_=equilibrium(average,.65,FORCE,kappa,OFFSET)
    energy,size,_,_,_=chain_energy(average,.65,FORCE,kappa,pair,6,OFFSET,
                                  top_shift=(shift/100,0),top_tilt=(0,np.deg2rad(tilt),0))
    def gradient(x):
        t=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        return torch.autograd.grad(energy(t),t)[0].detach().numpy()
    def hessian(x):
        return torch.autograd.functional.hessian(energy,torch.tensor(x,dtype=torch.float64)).numpy()
    result=root(gradient,np.zeros(size),jac=hessian,tol=1e-10)
    residual=float(max(abs(gradient(result.x))))
    assert residual<1e-9, (angle,shift,tilt,residual)
    matrix=hessian(result.x)
    values,vectors=np.linalg.eigh(matrix)
    e0=float(energy(torch.tensor(result.x,dtype=torch.float64)))
    step=1e-4
    change=[float(energy(torch.tensor(result.x+s*step*vectors[:,0],dtype=torch.float64)))-e0 for s in (-1,1)]
    pose=np.zeros((7,6))
    pose[1:6]=result.x[:30].reshape(5,6)
    pose[-1]=[shift/100,0,result.x[-1],0,np.deg2rad(tilt),0]
    heights=np.array([pair[j%2] for j in range(6)])
    angles=np.array([average+(1 if j%2==0 else -1)*pair[2] for j in range(6)])
    phase=np.r_[0,np.cumsum(angles)]
    centers=np.column_stack((np.zeros(7),np.zeros(7),np.r_[0,np.cumsum(heights)]))+pose[:,:3]
    points=[]
    for j in range(7):
        radii=np.where((np.arange(6)+j)%2==0,1.,.65)
        azimuth=np.repeat(np.arange(3)*2*np.pi/3,2)+np.tile([0,OFFSET],3)+phase[j]
        arms=np.column_stack((radii*np.cos(azimuth),radii*np.sin(azimuth),np.zeros(6)))
        points.append(centers[j]+arms@Rotation.from_rotvec(pose[j,3:]).as_matrix().T)
    strains=np.linalg.norm(np.diff(points,axis=0),axis=2)/np.sqrt(1+.65**2)-1
    assert strains.min()>0, 'A fabric edge became slack; bilateral spring result invalid.'
    return dict(angle=angle,top_shift_mm=shift,top_tilt_degrees=tilt,shear_ratio=shear,
                minimum_edge_strain=float(strains.min()),
                residual=residual,minimum_eigenvalue=float(values[0]),
                weakest_mode_energy_changes=change,
                maximum_internal_tilt_degrees=float(np.rad2deg(np.max(np.linalg.norm(result.x[:30].reshape(5,6)[:,3:5],axis=1)))))


def main():
    rows=[]
    for angle in (30,45,60,70,75,80,85):
        for shear in (0,.05):
            row,_,_=aligned(angle,shear)
            rows.append(row)
    critical=[]
    for shear in (0,.05):
        threshold=brentq(lambda a:aligned(a,shear)[0]['minimum_full'],60,89,xtol=1e-6)
        critical.append(dict(shear_ratio=shear,angle=threshold))
    imperfections=[]
    for angle in (45,60,70,75,80):
        for shift,tilt in ((1,0),(5,0),(0,1),(5,1)):
            row=imperfect(angle,shift,tilt)
            imperfections.append(row)
            print(row,flush=True)
    report=dict(force_N=40,radius_mm=100,EA_per_strap_N=1000,
                boundary='Bottom fixed; top lateral offset and tilt prescribed, total twist fixed, top height free. Five internal plates free in all six coordinates.',
                assumptions='Same quadratic two-facet fabric model. Shear ratio=kappa/(FR). Perfect and imperfect cases solve loaded equilibrium, not prescribed GUI heights. No locking, contact, slip, independent wrinkling or strength model.',
                aligned=rows,critical_angles=critical,imperfections=imperfections)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Critical:',critical,flush=True)


if __name__=='__main__':main()
