"""Quick case: 1x motor, m=100g, L=180mm, low inertia"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Base parameters
BASE_TAU_MAX = 0.2795
BASE_W_MAX = 1704
I_BASE = 0.5 * (37.4e-3/2) * (28.4e-3/2)**2


def get_optimal_pulley_radius(params):
    """Find optimal pulley radius that maximizes final velocity."""
    m = params['m']
    I = params['I']
    tau_max = params['tau_max']
    w_max = params['w_max']
    g = params['g']
    s_target = -params['stroke']
    
    def get_vf_pulley(r_val):
        if r_val <= 1e-6:
            return 0.0
        G = -1 / r_val
        m_eff = m + I * G**2
        a = (G * tau_max + m * g) / m_eff
        b = (G**2 * tau_max / w_max) / m_eff
        if a >= 0:
            return 0.0
        arg = -np.exp(-(1 + (b**2 * s_target) / a))
        return (a / b) * (1 + np.real(lambertw(arg)))
    
    r_crit = tau_max / (m * g)
    res = minimize_scalar(lambda r: get_vf_pulley(r), bounds=(0.001, r_crit), method='bounded')
    return res.x, abs(res.fun)

def integrate_theta(get_y, get_tau, params, theta0=0, theta_dot0=0):
    m = params['m']
    I = params['I']
    stroke = params['stroke']
    g = params['g']
   
    def dynamics(t, state):
        theta, theta_dot = state
        y, dy_dtheta, d2y_dtheta2 = get_y(theta)
        tau = get_tau(theta, theta_dot, t)
        theta_ddot = (-theta_dot**2 * m * dy_dtheta * d2y_dtheta2 - m*g*dy_dtheta + tau) / (I + m * (dy_dtheta)**2)
        return [theta_dot, theta_ddot]
    
    def y_target_event(t, state):
        theta, theta_dot = state
        y, _, _ = get_y(theta)
        return y - stroke
    
    y_target_event.terminal = True
    y_target_event.direction = 0

    t_max = 2.0
    t_eval = np.arange(0, t_max, 0.0001)
    
    sol = solve_ivp(dynamics, t_span=(0, t_max), y0=[theta0, theta_dot0],
                    method='RK45', t_eval=t_eval, events=y_target_event,
                    dense_output=True, rtol=1e-6, atol=1e-9)
    
    t = sol.t
    theta = sol.y[0]
    theta_d = sol.y[1]
    
    y = np.zeros_like(theta)
    y_d = np.zeros_like(theta)
    tau = np.zeros_like(theta)
    for i, (th, th_d) in enumerate(zip(theta, theta_d)):
        y_val, dy_dtheta, _ = get_y(th)
        y[i] = y_val
        y_d[i] = dy_dtheta * th_d
        tau[i] = get_tau(th, th_d, t[i])
    
    return {'t': t, 'theta': theta, 'theta_d': theta_d, 'y': y, 'y_d': y_d, 'tau': tau}

# Load CSV and find case
df = pd.read_csv('sweep_motor_1p0x.csv')
df_candidates = df[(df['success'] == True) & (df['m'] == 0.1) & (df['sweep_type'] == 'inertia')]
df_candidates = df_candidates.copy()
df_candidates['L_diff'] = np.abs(df_candidates['opt_L'] - 0.18)
case = df_candidates.nsmallest(1, 'L_diff').iloc[0]

L = case['opt_L']
r = case['opt_r']
I_ratio = case['I_ratio']
I_actual = case['I']

params = {
    'm': 0.100,
    'I': I_actual,
    'stroke': 0.080,
    'g': 9.81,
    'tau_max': BASE_TAU_MAX * 1.0,
    'w_max': BASE_W_MAX * 1.0,
}

print(f"Case: Motor 1.0x, m=100g, I_ratio={I_ratio:.3f}")
print(f"Optimal: L={L*1000:.1f}mm, r={r*1000:.3f}mm, T={case['opt_T']*1000:.1f}ms")
print(f"Expected ydot: {case['ydot_final']:.3f} m/s")

# Run simulations
def get_tau(theta, theta_dot, t):
    return params['tau_max'] * (1 - theta_dot / params['w_max'])

# Optimal pulley radius
r_pulley_opt, vf_pulley_expected = get_optimal_pulley_radius(params)
print(f"Optimal pulley radius: {r_pulley_opt*1000:.3f}mm, expected vf={vf_pulley_expected:.3f} m/s")

def get_y_pulley(theta):
    y = r_pulley_opt * theta
    return y, r_pulley_opt, 0
res_pulley = integrate_theta(get_y_pulley, get_tau, params)

y_offset = 0.00
offset_direction = 1
theta_offset = offset_direction * np.sqrt(L**2 - (L - y_offset)**2) / r

def get_y_tsa(theta):
    val = L**2 - (r*theta)**2
    if val < 1e-12:
        val = 1e-12
    y = L - np.sqrt(val)
    dy_dtheta = (r**2 * theta) / np.sqrt(val)
    d2y_dtheta2 = (r**2 * L**2) / (val)**(3/2)
    return y, dy_dtheta, d2y_dtheta2
res_tsa = integrate_theta(get_y_tsa, get_tau, params, theta0=theta_offset, theta_dot0=0)

def get_y_const_speed(theta):
    P_max = params['tau_max'] * params['w_max'] / 4
    m = params['m']
    w_max = params['w_max']
    w_opt = w_max / 2
    A = (2/3) * np.sqrt(2 * P_max / m)
    if theta < 1e-9:
        theta = 1e-9
    y = A * (theta / w_opt)**(3/2)
    dy_dtheta = A * (3/2) * (theta / w_opt)**(1/2) * (1 / w_opt)
    d2y_dtheta2 = A * (3/2) * (1/2) * (theta / w_opt)**(-1/2) * (1 / w_opt)**2
    return y, dy_dtheta, d2y_dtheta2
res_const_speed = integrate_theta(get_y_const_speed, get_tau, params, 
                                   theta0=0, theta_dot0=params['w_max']/2)

res_all = {'pulley': res_pulley, 'tsa': res_tsa, 'const_speed': res_const_speed}

print(f"\nSimulated ydot:")
print(f"  Best TSA (r={r*1000:.2f}mm): {res_tsa['y_d'][-1]:.3f} m/s")
print(f"  Best Pulley (r={r_pulley_opt*1000:.2f}mm): {res_pulley['y_d'][-1]:.3f} m/s")
print(f"  ConstSpeed: {res_const_speed['y_d'][-1]:.3f} m/s")

# Plot 4x2
fig, axes = plt.subplots(4, 2, figsize=(10, 12))
axes = axes.flatten()

colors = {'pulley': 'b', 'tsa': 'r', 'const_speed': 'g'}
labels = {
    'pulley': f'Best Pulley (r={r_pulley_opt*1000:.2f}mm)',
    'tsa': f'Best TSA (r={r*1000:.2f}mm, L={L*1000:.0f}mm)',
    'const_speed': 'Constant Speed'
}

# Find max time to determine axis formatting
max_time = max([res_all[key]['t'][-1] for key in res_all.keys()])

# Format time axis: use ms if max_time < 0.1s, otherwise use seconds
use_ms = max_time < 0.1
if use_ms:
    time_scale = 1000  # convert to ms
    time_label = 'Time (ms)'
    # Set reasonable number of ticks (4-6) for better spacing
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
else:
    time_scale = 1
    time_label = 'Time (s)'

# Helper to get scaled time
def get_scaled_time(t):
    return t * time_scale

for key, res in res_all.items():
    axes[0].plot(get_scaled_time(res['t']), res['theta'], color=colors[key], linewidth=2, label=labels[key])
axes[0].set_xlabel(time_label, fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Angle (rad)', fontsize=14, fontweight='bold')
axes[0].legend()

for key, res in res_all.items():
    axes[1].plot(get_scaled_time(res['t']), res['theta_d'], color=colors[key], linewidth=2, label=labels[key])
axes[1].set_xlabel(time_label, fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')

for key, res in res_all.items():
    axes[2].plot(get_scaled_time(res['t']), res['y'], color=colors[key], linewidth=2, label=labels[key])
axes[2].set_xlabel(time_label, fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].set_title('Linear Position (m)', fontsize=14, fontweight='bold')

for key, res in res_all.items():
    axes[3].plot(get_scaled_time(res['t']), res['y_d'], color=colors[key], linewidth=2, label=labels[key])
    axes[3].text(get_scaled_time(res['t'][-1]), res['y_d'][-1], f"{res['y_d'][-1]:.2f}",
                fontsize=10, color=colors[key], va='bottom', ha='right', fontweight='bold')
    if key == 'tsa':
        axes[3].set_ylim(0, res['y_d'][-1]*1.1)
axes[3].set_xlabel(time_label, fontsize=12)
axes[3].grid(True, alpha=0.3)
axes[3].set_title('Linear Velocity (m/s)', fontsize=14, fontweight='bold')

for key, res in res_all.items():
    I = params['I']
    m = params['m']
    KE_rot = 0.5 * I * np.array(res['theta_d'])**2
    KE_trans = 0.5 * m * np.array(res['y_d'])**2
    axes[4].plot(get_scaled_time(res['t']), KE_rot, color=colors[key], linestyle='--', linewidth=2)
    axes[4].plot(get_scaled_time(res['t']), KE_trans, color=colors[key], linewidth=2)
axes[4].set_xlabel(time_label, fontsize=12)
axes[4].grid(True, alpha=0.3)
axes[4].set_title('Kinetic Energy', fontsize=14, fontweight='bold')
custom_lines = [Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Rotational KE'),
                Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Translational KE')]
axes[4].legend(handles=custom_lines, fontsize=10)

for key, res in res_all.items():
    tau = np.array(res['tau'])
    theta_d = np.array(res['theta_d'])
    power = tau * theta_d
    t = np.array(res['t'])
    KE_rot = 0.5 * params['I'] * theta_d**2
    y_d = np.array(res['y_d'])
    KE_trans = 0.5 * params['m'] * y_d**2
    KE_total = KE_rot + KE_trans
    dKE_dt = np.gradient(KE_total, t)
    axes[5].plot(get_scaled_time(t), power, color=colors[key], linewidth=2, label=labels[key])
    # axes[5].plot(get_scaled_time(t), dKE_dt, color=colors[key], linestyle='--', linewidth=3)
axes[5].set_xlabel(time_label, fontsize=12)
axes[5].grid(True, alpha=0.3)
axes[5].set_title('Motor Power (W)', fontsize=14, fontweight='bold')

for key, res in res_all.items():
    y_d = np.array(res['y_d'])
    theta_d = np.array(res['theta_d'])
    ratio = np.divide(y_d, theta_d, out=np.full_like(y_d, np.nan), where=theta_d!=0)
    axes[6].plot(get_scaled_time(res['t']), ratio, color=colors[key], linewidth=2, label=labels[key])
axes[6].set_xlabel(time_label, fontsize=12)
axes[6].grid(True, alpha=0.3)
axes[6].set_title('Linear Velocity / Angular Velocity', fontsize=14, fontweight='bold')

for key, res in res_all.items():
    tau = np.array(res['tau'])
    axes[7].plot(get_scaled_time(res['t']), tau, color=colors[key], linewidth=2, label=labels[key])
axes[7].set_xlabel(time_label, fontsize=12)
axes[7].grid(True, alpha=0.3)
axes[7].set_title('Motor Torque (Nm)', fontsize=14, fontweight='bold')
axes[7].axhline(y=params['tau_max'], color='k', linestyle='--', linewidth=1)
axes[7].axhline(y=0, color='k', linestyle='--', linewidth=1)

title = f"Motor 1.0x | Mass 100g | Inertia {I_ratio:.3f}x"
fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('optimal_case_plots/case_180mm.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: quick_case_180mm.png")
plt.close()

