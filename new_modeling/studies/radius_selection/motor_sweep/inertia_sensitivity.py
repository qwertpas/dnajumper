"""Compare the three designs under shared and radius-specific inertia uncertainty."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from new_modeling.studies.radius_selection.motor_sweep.model import CASES
from new_modeling.studies.radius_selection.motor_sweep.model import heights, RADII, verify

ROOT = Path(__file__).resolve().parent / 'results'
ROOT.mkdir(parents=True, exist_ok=True)
LOW=1.86e-6
HIGH=1.82396e-5
STRESS=3e-5


def main():
    inertias=np.geomspace(LOW,STRESS,301)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),sharex=True,sharey=True,layout='constrained')
    colors=['#1565C0','#D47900','#28834A']
    results=[]
    for (mass_g,power,lo,hi),color in zip(CASES,colors):
        mass=mass_g/1000
        case={'body_mass_g':mass_g,'power_w':power,'rpm_range':[lo,hi],'endpoints':[]}
        for ax,rpm,winner in zip(axes,[lo,hi],[2,0]):
            h=np.array([heights(rpm,mass=mass,inertia=I,power=power) for I in inertias])
            lead=h[:,winner]/np.max(np.delete(h,winner,axis=1),axis=1)-1
            ax.plot(inertias*1e6,100*lead,lw=2.3,color=color,label=f'{mass_g} g, {power:g} W')
            def values(logI):return heights(rpm,mass=mass,inertia=np.exp(logI),power=power)
            def margin(logI):
                hh=values(logI)
                return hh[winner]/max(np.delete(hh,winner))-1
            bounds=(np.log(LOW),np.log(HIGH))
            fit=minimize_scalar(margin,bounds=bounds,method='bounded',options={'xatol':1e-12})
            trial=[(margin(bounds[0]),LOW),(margin(bounds[1]),HIGH),(fit.fun,np.exp(fit.x))]
            worst,Iworst=min(trial)
            ranges=[]
            for j in range(3):
                fmin=minimize_scalar(lambda x:values(x)[j],bounds=bounds,method='bounded')
                fmax=minimize_scalar(lambda x:-values(x)[j],bounds=bounds,method='bounded')
                samples=[values(bounds[0])[j],values(bounds[1])[j]]
                ranges.append([min(*samples,fmin.fun),max(*samples,-fmax.fun)])
            ranges=np.array(ranges)
            independent=ranges[winner,0]/max(np.delete(ranges[:,1],winner))-1
            roots=[]
            for threshold in [.1,0.]:
                changes=np.flatnonzero(np.diff(np.sign(lead-threshold))!=0)
                for i in changes:
                    root=brentq(lambda I:margin(np.log(I))-threshold,inertias[i],inertias[i+1],xtol=1e-15)
                    roots.append({'threshold':threshold,'inertia':root})
            checks=[verify(rpm,r,power=power,mass=mass,inertia=Iworst) for r in RADII]
            case['endpoints'].append({'rpm':rpm,'target_winner_mm':float(RADII[winner]*1000),
                'lead_at_low_inertia':float(lead[0]),'lead_at_high_inertia':float(margin(np.log(HIGH))),
                'worst_shared_inertia_lead':float(worst),'worst_shared_inertia':float(Iworst),
                'worst_independent_inertia_lead':float(independent),
                'individual_radius_height_ranges_m':ranges.tolist(),'stress_crossings':roots,
                'max_independent_solver_error':max(c['solver_error'] for c in checks)})
        case['worst_shared_winning_lead']=min(e['worst_shared_inertia_lead'] for e in case['endpoints'])
        case['worst_independent_winning_lead']=min(e['worst_independent_inertia_lead'] for e in case['endpoints'])
        results.append(case)
    for ax,title in zip(axes,['Slow motor: 9 mm lead over nearest competitor','Fast motor: 5 mm lead over nearest competitor']):
        ax.axhline(10,color='#777777',ls='--',lw=1)
        ax.axhline(0,color='#AAAAAA',lw=.8)
        ax.axvline(HIGH*1e6,color='#777777',ls=':',lw=1)
        ax.axvspan(HIGH*1e6,STRESS*1e6,color='#777777',alpha=.06)
        ax.set(xscale='log',xlabel='Shared shaft inertia (10⁻⁶ kg m²)',title=title,xlim=(LOW*1e6,STRESS*1e6))
        ax.set_xticks([1.86,5,10,18.2396,30],labels=['1.86','5','10','18.24','30'])
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Predicted height advantage (%)')
    axes[0].legend(frameon=False,loc='lower left')
    fig.suptitle('Fixed 180 mm; same inertia for all radii within each scenario',fontsize=12)
    fig.savefig(ROOT/'inertia_sensitivity.png',dpi=190)
    fig.savefig(ROOT/'inertia_sensitivity.svg')
    (ROOT/'inertia_sensitivity.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
