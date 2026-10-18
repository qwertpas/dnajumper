"""Refine and verify the radius-separation optimum at fixed 180 mm."""
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.optimize import root, brentq, minimize_scalar
from new_modeling.studies.radius_selection.motor_sweep.model import energy, heights, verify, INERTIA, RADII

ROOT = Path(__file__).resolve().parent / 'results/optimization'
ROOT.mkdir(parents=True, exist_ok=True)
MASS=.050
POWER=9.
LOW=1500.
HIGH=20000.


def refine(height,cap,max_rpm,goal,guess):
    def values(p):
        mass,power=p
        low=4*power/(cap*np.pi/30)
        hh=np.array([heights(n,mass=mass,power=power) for n in [low,max_rpm]])
        first=hh[0,2]/hh[0,1]-1
        last=hh[1,0]/hh[1,1]-1 if goal=='winners' else hh[1,1]/hh[1,2]-1
        return [hh[0,0]-height,first-last]
    fit=root(values,guess,tol=1e-10)
    assert fit.success and max(np.abs(values(fit.x)))<1e-8
    mass,power=fit.x
    low=4*power/(cap*np.pi/30)
    hh=np.array([heights(n,mass=mass,power=power) for n in [low,max_rpm]])
    checks=[verify(n,r,power=power,mass=mass) for n in [low,max_rpm] for r in RADII]
    return {'goal':goal,'height_floor_m':height,'torque_limit_nm':cap,
            'mass_kg':mass,'power_w':power,'rpm_range':[low,max_rpm],
            'heights_m':hh.tolist(),'winner_leads':[hh[0,2]/hh[0,1]-1,hh[1,0]/hh[1,1]-1],
            'all_adjacent_margins':[hh[0,2]/hh[0,1]-1,hh[0,1]/hh[0,0]-1,hh[1,0]/hh[1,1]-1,hh[1,1]/hh[1,2]-1],
            'verification':checks}


@njit
def score_grid(masses,powers):
    values=np.full((len(powers),len(masses)),np.nan)
    for i,power in enumerate(powers):
        low=max(1100.,4*power/(.26*np.pi/30))
        if low>=21643.4236:continue
        for j,mass in enumerate(masses):
            best9=-1.
            best5=-1.
            for z in range(121):
                rpm=low*(21643.4236/low)**(z/120)
                speed=rpm*np.pi/30
                stall=4*power/speed
                hh=np.array([energy(mass,r,.18,stall,speed,INERTIA,450)/(mass*9.81) for r in RADII])
                if not np.isfinite(hh).all() or hh.min()<.15:continue
                best9=max(best9,hh[2]/max(hh[0],hh[1])-1)
                best5=max(best5,hh[0]/max(hh[1],hh[2])-1)
            if min(best9,best5)>0:
                values[i,j]=100*min(best9,best5)
    return values


def main():
    optima=[refine(.15,.26,21643.4236,'winners',[.05057,8.553]),
            refine(.15,.26,21643.4236,'all',[.05820,8.713]),
            refine(.18,.234,20000.,'winners',[.04817,8.640]),
            refine(.18,.234,20000.,'all',[.05508,8.816])]
    rpms=np.linspace(LOW,HIGH,801)
    hh=np.array([heights(n,mass=MASS,power=POWER) for n in rpms])
    assert hh.min()>.18
    crossings=[brentq(lambda n:heights(n,mass=MASS,power=POWER)[a]-heights(n,mass=MASS,power=POWER)[b],LOW,HIGH)
               for a,b in [(2,1),(1,0)]]
    mid=minimize_scalar(lambda n:-(heights(n,mass=MASS,power=POWER)[1]/max(heights(n,mass=MASS,power=POWER)[[0,2]])-1),bounds=tuple(crossings),method='bounded')
    cases=[]
    checks=[]
    for n in [LOW,float(mid.x),HIGH]:
        h=heights(n,mass=MASS,power=POWER)
        order=np.argsort(h)
        cases.append({'rpm':n,'stall_nm':4*POWER/(n*np.pi/30),'heights_m':h.tolist(),
                      'winner_mm':float(1000*RADII[order[-1]]),
                      'lead':h[order[-1]]/h[order[-2]]-1,'gap_m':h[order[-1]]-h[order[-2]]})
        checks.extend(verify(n,r,power=POWER,mass=MASS) for r in RADII)
    slope=4.70391607e-5-4.52823176e-6*5.352084
    slack=np.inf
    for n in rpms:
        speeds=np.linspace(0,n,1001)
        requested=4*POWER/(n*np.pi/30)*(1-speeds/n)
        allowed=np.minimum(.26,slope*(21643.4236-speeds))
        slack=min(slack,min(allowed-requested))
    assert slack>.03

    sensitivity=[]
    for n in [LOW,HIGH]:
        winner=np.argmax(heights(n,mass=MASS,power=POWER))
        shared=np.inf
        lower=[]
        upper=[]
        min_height=np.inf
        for r in RADII:
            possible=[]
            for ds,dw,di,dm,dl,dr in itertools.product([.97,1.03],[.97,1.03],[.9,1.1],[-.001,.001],[.1795,.1805],[-.0001,.0001]):
                mass=MASS+dm
                speed=n*np.pi/30*dw
                stall=4*POWER/(n*np.pi/30)*ds
                possible.append(energy(mass,r+dr,dl,stall,speed,INERTIA*di)/(mass*9.81))
            lower.append(min(possible))
            upper.append(max(possible))
            min_height=min(min_height,min(possible))
        independent=lower[winner]/max(np.delete(upper,winner))-1
        for ds,dw,di,dm,dl in itertools.product([.97,1.03],[.97,1.03],[.9,1.1],[-.001,.001],[.1795,.1805]):
            mass=MASS+dm
            speed=n*np.pi/30*dw
            stall=4*POWER/(n*np.pi/30)*ds
            h=np.array([energy(mass,r,dl,stall,speed,INERTIA*di)/(mass*9.81) for r in RADII])
            shared=min(shared,h[winner]/max(np.delete(h,winner))-1)
        sensitivity.append({'rpm':n,'common_error_min_lead':shared,'independent_error_min_lead':independent,
                            'minimum_perturbed_height_m':min_height,'independent_height_ranges_m':np.c_[lower,upper].tolist()})

    recommendation={'mass_kg':MASS,'power_w':POWER,'length_m':.18,'rpm_range':[LOW,HIGH],
                    'cases':cases,'crossovers_rpm':crossings,'middle_max_lead':-mid.fun,
                    'height_range_m':[float(hh.min()),float(hh.max())],
                    'envelope_slack_nm':slack,'verification':checks,'sensitivity':sensitivity}
    (ROOT/'optimum_results.json').write_text(json.dumps({'optima':optima,'recommendation':recommendation},indent=2))
    np.savetxt(ROOT/'optimum_sweep.csv',np.c_[rpms,4*POWER/(rpms*np.pi/30),hh],delimiter=',',header='free_rpm,stall_nm,height_r5_m,height_r7_m,height_r9_m',comments='')
    print(json.dumps({'optima':[{k:v for k,v in x.items() if k!='verification'} for x in optima],
                      'recommendation':{k:v for k,v in recommendation.items() if k!='verification'}},indent=2),flush=True)

    masses=np.linspace(.02,.16,45)
    powers=np.geomspace(2,25,45)
    scores=score_grid(masses,powers)
    np.savez(ROOT/'mass_power_grid.npz',mass_kg=masses,power_w=powers,margin_percent=scores)
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    img=axes[0].pcolormesh(masses*1000,powers,np.ma.masked_invalid(scores),shading='nearest',cmap='viridis',vmin=0,vmax=33)
    fig.colorbar(img,ax=axes[0],label='Smaller endpoint winning margin (%)')
    axes[0].set(yscale='log',xlabel='Body mass (g)',ylabel='Peak motor power (W)',title='Search at fixed 180 mm; every radius above 0.15 m')
    for m,p,label,color,marker in [(optima[0]['mass_kg'],optima[0]['power_w'],'Numerical optimum','white','*'),(MASS,POWER,'Rounded setup','#F8CE50','o'),(.1,15,'Previous setup','#ED6959','s')]:
        axes[0].scatter(m*1000,p,c=color,edgecolor='black',marker=marker,s=90,label=label,zorder=5)
    axes[0].legend(loc='upper right',framealpha=.95,fontsize=9)
    colors=['#1565C0','#D47900','#28834A']
    for j,r in enumerate(RADII):
        axes[1].plot(rpms/1000,hh[:,j],lw=2.5,color=colors[j],label=f'{1000*r:.0f} mm')
    for x in crossings:axes[1].axvline(x/1000,color='#777777',ls=':',lw=1)
    for case in cases[::2]:axes[1].scatter([case['rpm']/1000]*3,case['heights_m'],c=colors,s=25,zorder=4)
    axes[1].axhline(.15,color='#777777',ls='--',lw=1)
    axes[1].set(xlabel='Motor free speed (thousand RPM)',ylabel='Ballistic height above takeoff (m)',title='Recommended: 50 g body, 9 W peak power',xlim=(1.5,20),ylim=(.15,1.22))
    axes[1].legend(title='Transmission radius',frameon=False)
    axes[1].grid(alpha=.18)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'optimum.png',dpi=190)
    fig.savefig(ROOT/'optimum.svg')
    print('Grid check best:',float(np.nanmax(scores)),'percent',flush=True)

if __name__=='__main__':main()
