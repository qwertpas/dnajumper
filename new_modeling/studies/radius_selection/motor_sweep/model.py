"""Dynamics and independent verification for equal-power linear motors."""

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

DEAD = 49.5 / 157.5
INERTIA = 1.86e-6
MASS = .100
LENGTH = .180
POWER = 10.
RADII = np.array([.005, .007, .009])

@njit
def rate(x, e, mass, r, length, stall, speed, inertia):
    q = r*x/np.sqrt(1-x*x)
    w = np.sqrt(2*max(e, 0)/(inertia+mass*q*q))
    return length/r*(stall*(1-w/speed)-mass*9.81*q)

@njit
def energy(mass, r, length, stall, speed, inertia=INERTIA, steps=1800):
    end = np.sqrt(1-DEAD*DEAD)
    e = 0.
    for i in range(steps):
        x = end*(i/steps)**2
        xx = end*((i+1)/steps)**2
        h = xx-x
        a = rate(x,e,mass,r,length,stall,speed,inertia)
        b = rate(x+h/2,e+h*a/2,mass,r,length,stall,speed,inertia)
        c = rate(x+h/2,e+h*b/2,mass,r,length,stall,speed,inertia)
        d = rate(xx,e+h*c,mass,r,length,stall,speed,inertia)
        e += h*(a+2*b+2*c+d)/6
        if e <= 0:
            return np.nan
    q = r*end/DEAD
    return e*mass*q*q/(inertia+mass*q*q)


def heights(rpm, mass=MASS, length=LENGTH, inertia=INERTIA, power=POWER):
    speed = rpm*np.pi/30
    stall = 4*power/speed
    return np.array([energy(mass,r,length,stall,speed,inertia)/(mass*9.81) for r in RADII])


def verify(rpm, r, power=POWER, mass=MASS, length=LENGTH, inertia=INERTIA):
    speed = rpm*np.pi/30
    stall = 4*power/speed
    end = length/r*np.sqrt(1-DEAD**2)

    def geometry(theta):
        root = np.sqrt(length**2-(r*theta)**2)
        return length-root, r*r*theta/root, r*r*length*length/root**3

    def dynamics(t, state):
        theta,w = state[:2]
        y,q,qp = geometry(theta)
        torque = stall*(1-w/speed)
        alpha = (torque-mass*9.81*q-mass*q*qp*w*w)/(inertia+mass*q*q)
        return [w,alpha,torque*w]

    def finish(t, state):
        return state[0]-end
    finish.terminal = True
    finish.direction = 1
    sol = solve_ivp(dynamics,(0,3),[0,0,0],events=finish,dense_output=True,
                    method='DOP853',rtol=2e-11,atol=1e-12,max_step=.00025)
    assert sol.success and len(sol.t_events[0]) == 1
    duration = sol.t_events[0][0]
    t = np.linspace(0,duration,16001)
    theta,w,shaft_work = sol.sol(t)
    y,q,qp = geometry(theta)
    alpha = dynamics(t,[theta,w])[1]
    force = mass*(9.81+q*alpha+qp*w*w)
    body = .5*mass*(q*w)**2
    rotor = .5*inertia*w*w
    work = shaft_work[-1]
    final = body[-1]+rotor[-1]+mass*9.81*y[-1]
    h = body[-1]/(mass*9.81)
    fast_h = energy(mass,r,length,stall,speed,inertia)/(mass*9.81)
    fine_h = energy(mass,r,length,stall,speed,inertia,steps=7200)/(mass*9.81)
    assert abs(h/fast_h-1) < 2e-5
    assert abs(h/fine_h-1) < 2e-6
    assert abs(work/final-1) < 2e-6
    assert force.min() > 0 and w.max() < speed
    return {'rpm':rpm,'radius_mm':1000*r,'height_m':h,'time_s':duration,
            'takeoff_speed_m_s':float(q[-1]*w[-1]),'body_energy_j':body[-1],
            'rotor_energy_j':rotor[-1],'motor_work_j':work,
            'energy_error':abs(work/final-1),'solver_error':abs(h/fast_h-1),
            'minimum_force_n':force.min(),'maximum_speed_rpm':w.max()*30/np.pi}



CASES=[(100,15.,2500.,15000.),(50,9.,1500.,20000.),(60,16.7,2500.,15000.)]
TORQUE_CAP=.260
FREE_MAX=21643.4236
SLOPE=4.70391607e-5-4.52823176e-6*5.352084


def envelope(rpm):
    return np.maximum(0,np.minimum(TORQUE_CAP,SLOPE*(FREE_MAX-rpm)))

