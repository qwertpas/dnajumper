"""Motor families with jump-height sweeps at the two requested inertias."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import brentq
from model import CASES, envelope
from model import heights, verify, RADII

ROOT = Path(__file__).resolve().parent / 'results'
ROOT.mkdir(parents=True, exist_ok=True)
INERTIAS=[1.86e-6,1.82396e-5]
COLORS=['#1565C0','#D47900','#28834A']


def main():
    fig=plt.figure(figsize=(15,11),layout='constrained')
    grid=fig.add_gridspec(3,4,width_ratios=[1,1,1,.035],height_ratios=[1,1,1])
    axes=np.array([[fig.add_subplot(grid[i,j]) for j in range(3)] for i in range(3)])
    norm=LogNorm(1500,20000)
    cmap=plt.get_cmap('viridis')
    rpm_format=FuncFormatter(lambda x,_:f'{x/1000:g}k' if x else '0')
    motor_speeds=np.linspace(0,21643.4236,1001)
    results=[]
    data=[]
    height_max=0.
    for column,(mass_g,power,lo,hi) in enumerate(CASES):
        mass=mass_g/1000
        ax=axes[0,column]
        ax.fill_between(motor_speeds,0,envelope(motor_speeds),color='#E4E7EB',label='Motor envelope')
        ax.plot(motor_speeds,envelope(motor_speeds),color='#626973',lw=1.4)
        for k,free in enumerate(np.geomspace(lo,hi,21)):
            stall=4*power/(free*np.pi/30)
            speed=np.linspace(0,free,301)
            torque=stall*(1-speed/free)
            assert np.all(torque<=envelope(speed)+1e-12)
            ax.plot(speed,torque,color=cmap(norm(free)),lw=2 if k in [0,20] else .9,
                    alpha=1 if k in [0,20] else .8)
        ax.set(title=f'{mass_g} g, {power:g} W\nFree speed {lo:,.0f}–{hi:,.0f} RPM',
               xlabel='Motor speed (RPM)',ylim=(0,.28))
        if column==0:
            ax.set_ylabel('Motor shaft torque (N m)')
            ax.legend(loc='upper right',frameon=False,fontsize=9)
        else:
            ax.tick_params(labelleft=False)

        free_speeds=np.linspace(lo,hi,601)
        for row,inertia in enumerate(INERTIAS,start=1):
            h=np.array([heights(n,mass=mass,inertia=inertia,power=power) for n in free_speeds])
            assert np.isfinite(h).all() and h.min()>0
            height_max=max(height_max,float(h.max()))
            ax=axes[row,column]
            for j,r in enumerate(RADII):
                ax.plot(free_speeds,h[:,j],color=COLORS[j],lw=2.3,label=f'{r*1000:.0f} mm')
            ax.axhline(.15,color='#8B8F95',ls='--',lw=1)
            ax.set_xlabel('Motor free speed (RPM)')
            if column==0:
                value='1.86 × 10⁻⁶' if row==1 else '1.82396 × 10⁻⁵'
                ax.set_ylabel(f'Ballistic height (m)\nI = {value} kg m²')
            else:
                ax.tick_params(labelleft=False)
            winners=np.argmax(h,axis=1)
            changes=np.flatnonzero(np.diff(winners)!=0)
            crossings=[]
            for i in changes:
                a,b=int(winners[i]),int(winners[i+1])
                crossing=brentq(lambda n:heights(n,mass=mass,inertia=inertia,power=power)[a]-heights(n,mass=mass,inertia=inertia,power=power)[b],free_speeds[i],free_speeds[i+1])
                crossings.append({'free_rpm':crossing,'from_mm':float(RADII[a]*1000),
                                  'to_mm':float(RADII[b]*1000)})
            checks=[verify(n,r,power=power,mass=mass,inertia=inertia)
                    for n in [lo,float(np.sqrt(lo*hi)),hi] for r in RADII]
            results.append({'body_mass_g':mass_g,'peak_power_w':power,'inertia_kg_m2':inertia,
                            'free_rpm_range':[lo,hi],'height_range_m':[float(h.min()),float(h.max())],
                            'endpoint_heights_m':[h[0].tolist(),h[-1].tolist()],
                            'winner_crossovers':crossings,
                            'max_solver_relative_error':max(x['solver_error'] for x in checks),
                            'max_energy_relative_error':max(x['energy_error'] for x in checks)})
            data.extend(np.c_[np.full(len(h),mass_g),np.full(len(h),power),np.full(len(h),inertia),free_speeds,h].tolist())
    for row in [1,2]:
        for ax in axes[row]:
            ax.set_ylim(0,1.07*height_max)
    for row in axes:
        for ax in row:
            ax.set_xlim(0,22000)
            ax.set_xticks([0,5000,10000,15000,20000])
            ax.xaxis.set_major_formatter(rpm_format)
            ax.grid(alpha=.16)
            ax.spines[['top','right']].set_visible(False)
    axes[1,0].legend(title='Transmission radius',loc='upper right',frameon=False,fontsize=9)
    axes[2,0].text(21000,.15,'0.15 m',ha='right',va='bottom',fontsize=9,color='#777777')
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_subplot(grid[0,3]))
    bar.set_label('Motor-curve free speed (RPM)')
    bar.set_ticks([1500,2500,5000,10000,20000])
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'))
    bar.minorticks_off()
    fig.suptitle('180 mm total length · dead/total = 49.5/157.5 · zero foot mass',fontsize=12)
    fig.savefig(ROOT/'motor_curves_and_heights.png',dpi=190)
    fig.savefig(ROOT/'motor_curves_and_heights.svg')
    (ROOT/'motor_height_results.json').write_text(json.dumps(results,indent=2))
    np.savetxt(ROOT/'motor_height_sweeps.csv',data,delimiter=',',
               header='body_mass_g,peak_power_w,inertia_kg_m2,free_rpm,height_r5_m,height_r7_m,height_r9_m',comments='')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
