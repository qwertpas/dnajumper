"""
Generate 4x2 comparison plots for specific optimization cases.
Uses optimal parameters from CSV sweep files and runs numerical integration.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pathlib import Path
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
SHOW_PLOTS = False  # Set to True to display instead of save
OUTPUT_DIR = "optimal_case_plots"

# Base motor parameters (before scaling)
# BASE_TAU_MAX = 0.2795  # Nm
BASE_TAU_MAX = 0.09555555556 * 3.5  # Nm
BASE_W_MAX = 2135*2*np.pi/60 * 3.5      # rad/s

# Base inertia
# I_BASE = 0.5 * (37.4e-3/2) * (28.4e-3/2)**2  # kg·m²
I_BASE = 4.5e-6 # from free speed fits at 3V


# ============================================================================
# NUMERICAL INTEGRATION (from optimal_comparison_theta.ipynb)
# ============================================================================
def integrate_theta(get_y, get_tau, params, theta0=0, theta_dot0=0):
    """Numerical integration of CVT dynamics."""
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
    
    sol = solve_ivp(
        dynamics,
        t_span=(0, t_max),
        y0=[theta0, theta_dot0],
        method='BDF',
        t_eval=t_eval,
        events=y_target_event,
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )
    
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
    
    return {
        't': t,
        'theta': theta,
        'theta_d': theta_d,
        'y': y,
        'y_d': y_d,
        'tau': tau,
    }


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
    res = minimize_scalar(lambda r: get_vf_pulley(r), bounds=(0.001, min(r_crit, 0.05)), method='bounded')
    return res.x, abs(res.fun)


def run_simulations(params, L, r):
    """Run pulley, TSA, and const_speed simulations."""
    
    def get_tau(theta, theta_dot, t):
        return params['tau_max'] * (1 - theta_dot / params['w_max'])
    
    # Optimal pulley radius
    r_pulley, vf_pulley_expected = get_optimal_pulley_radius(params)
    print(params)
    print(f"Optimal pulley radius: {r_pulley*1000:.3f}mm, expected vf={vf_pulley_expected:.3f} m/s")
    
    def get_y_pulley(theta):
        y = r_pulley * theta
        dy_dtheta = r_pulley
        d2y_dtheta2 = 0
        return y, dy_dtheta, d2y_dtheta2
    res_pulley = integrate_theta(get_y_pulley, get_tau, params)
    
    # TSA with optimal L, r
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
    
    # Constant speed theoretical
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
    
    return {
        'pulley': res_pulley,
        'tsa': res_tsa,
        'const_speed': res_const_speed,
        'r_pulley': r_pulley,
        'r_tsa': r,
        'L': L,
    }


def plot_4x2(res_all, params, title, save_path=None):
    """Generate the 4x2 comparison plot."""
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    axes = axes.flatten()
    
    # Get radii from res_all
    r_pulley = res_all.get('r_pulley', 0.006)
    r_tsa = res_all.get('r_tsa', 0.006)
    L = res_all.get('L', 0.180)
    
    colors = {
        'pulley': 'b',
        'tsa': 'r',
        'const_speed': 'g',
    }
    labels = {
        'pulley': f'Best Pulley (r={r_pulley*1000:.2f}mm)',
        'tsa': f'Best TSA (r={r_tsa*1000:.2f}mm, L={L*1000:.0f}mm)',
        'const_speed': 'Constant Speed',
    }
    
    # Filter out non-result keys
    plot_keys = ['pulley', 'tsa', 'const_speed']
    
    # Find max time to determine axis formatting
    max_time = max([res_all[key]['t'][-1] for key in plot_keys])
    
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
    
    # 0: theta vs t
    for key in plot_keys:
        res = res_all[key]
        axes[0].plot(get_scaled_time(res['t']), res['theta'], color=colors[key], linewidth=2, label=labels[key])
    axes[0].set_xlabel(time_label, fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Angle (rad)', fontsize=14, fontweight='bold')
    axes[0].legend()
    
    # 1: theta_dot vs t
    for key in plot_keys:
        res = res_all[key]
        axes[1].plot(get_scaled_time(res['t']), res['theta_d'], color=colors[key], linewidth=2, label=labels[key])
    axes[1].set_xlabel(time_label, fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Angular Velocity (rad/s)', fontsize=14, fontweight='bold')
    
    # 2: y vs t
    for key in plot_keys:
        res = res_all[key]
        axes[2].plot(get_scaled_time(res['t']), res['y'], color=colors[key], linewidth=2, label=labels[key])
    axes[2].set_xlabel(time_label, fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Linear Position (m)', fontsize=14, fontweight='bold')
    
    # 3: y_dot vs t
    for key in plot_keys:
        res = res_all[key]
        axes[3].plot(get_scaled_time(res['t']), res['y_d'], color=colors[key], linewidth=2, label=labels[key])
        axes[3].text(get_scaled_time(res['t'][-1]), res['y_d'][-1], f"{res['y_d'][-1]:.2f}",
                    fontsize=10, color=colors[key], va='bottom', ha='right', fontweight='bold')
        if key == 'tsa':
            axes[3].set_ylim(0, res['y_d'][-1]*1.1)
    axes[3].set_xlabel(time_label, fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Linear Velocity (m/s)', fontsize=14, fontweight='bold')
    
    # 4: Kinetic Energy
    for key in plot_keys:
        res = res_all[key]
        I = params['I']
        m = params['m']
        KE_rot = 0.5 * I * np.array(res['theta_d'])**2
        KE_trans = 0.5 * m * np.array(res['y_d'])**2
        axes[4].plot(get_scaled_time(res['t']), KE_rot, color=colors[key], linestyle='--', linewidth=2)
        axes[4].plot(get_scaled_time(res['t']), KE_trans, color=colors[key], linewidth=2)
    axes[4].set_xlabel(time_label, fontsize=12)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_title('Kinetic Energy', fontsize=14, fontweight='bold')
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Rotational KE'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Translational KE'),
    ]
    axes[4].legend(handles=custom_lines, fontsize=10)
    
    # 5: Motor Power
    for key in plot_keys:
        res = res_all[key]
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
    
    # 6: Linear/Angular Velocity ratio
    for key in plot_keys:
        res = res_all[key]
        y_d = np.array(res['y_d'])
        theta_d = np.array(res['theta_d'])
        ratio = np.divide(y_d, theta_d, out=np.full_like(y_d, np.nan), where=theta_d!=0)
        axes[6].plot(get_scaled_time(res['t']), ratio, color=colors[key], linewidth=2, label=labels[key])
    axes[6].set_xlabel(time_label, fontsize=12)
    axes[6].grid(True, alpha=0.3)
    axes[6].set_title('Linear Velocity / Angular Velocity', fontsize=14, fontweight='bold')
    
    # 7: Torque
    for key in plot_keys:
        res = res_all[key]
        tau = np.array(res['tau'])
        axes[7].plot(get_scaled_time(res['t']), tau, color=colors[key], linewidth=2, label=labels[key])
    axes[7].set_xlabel(time_label, fontsize=12)
    axes[7].grid(True, alpha=0.3)
    axes[7].set_title('Motor Torque (Nm)', fontsize=14, fontweight='bold')
    axes[7].axhline(y=params['tau_max'], color='k', linestyle='--', linewidth=1)
    axes[7].axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path and not SHOW_PLOTS:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show(block=False)
        print(f"  Displayed: {title}")
    
    return fig


def load_csv(motor_scale):
    """Load CSV file for given motor scale."""
    scale_str = f"{motor_scale:.1f}".replace('.', 'p')
    filename = f"sweep_motor_{scale_str}x.csv"
    path = Path(__file__).parent / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def get_optimal_params_mass(df, motor_scale, target_mass):
    """Get optimal L, r, T for a specific mass from mass sweep."""
    mass_df = df[(df['sweep_type'] == 'mass') & (df['success'] == True)]
    
    # Find closest mass
    masses = mass_df['m'].unique()
    closest_mass = masses[np.argmin(np.abs(masses - target_mass))]
    
    # Get best result for this mass
    subset = mass_df[mass_df['m'] == closest_mass]
    best_idx = subset['ydot_final'].idxmax()
    best = subset.loc[best_idx]
    
    return {
        'L': best['opt_L'],
        'r': best['opt_r'],
        'T': best['opt_T'],
        'ydot': best['ydot_final'],
        'm': best['m'],
    }


def get_optimal_params_inertia(df, motor_scale, target_I_ratio):
    """Get optimal L, r, T for a specific inertia ratio from inertia sweep."""
    inertia_df = df[(df['sweep_type'] == 'inertia') & (df['success'] == True)]
    
    # Find closest I_ratio
    ratios = inertia_df['I_ratio'].unique()
    closest_ratio = ratios[np.argmin(np.abs(ratios - target_I_ratio))]
    
    # Get best result for this ratio
    subset = inertia_df[inertia_df['I_ratio'] == closest_ratio]
    best_idx = subset['ydot_final'].idxmax()
    best = subset.loc[best_idx]
    
    return {
        'L': best['opt_L'],
        'r': best['opt_r'],
        'T': best['opt_T'],
        'ydot': best['ydot_final'],
        'I': best['I'],
        'I_ratio': best['I_ratio'],
    }


def get_best_inertia_ratio(df):
    """Find the inertia ratio that gives the best ydot_final."""
    inertia_df = df[(df['sweep_type'] == 'inertia') & (df['success'] == True)]
    if len(inertia_df) == 0:
        return 1.0
    
    # Group by I_ratio and find max ydot for each
    best_per_ratio = inertia_df.loc[inertia_df.groupby('I_ratio')['ydot_final'].idxmax()]
    
    # Find overall best
    best_idx = best_per_ratio['ydot_final'].idxmax()
    best = best_per_ratio.loc[best_idx]
    
    return best['I_ratio']


def generate_case_plot(motor_scale, mass, I_ratio, label_suffix=""):
    """Generate a single case plot."""
    df = load_csv(motor_scale)
    
    # Get optimal params - use mass sweep for m, but use specified I_ratio
    # Actually, we need to look at the inertia sweep for I_ratio variations
    if I_ratio == 1.0:
        # Use mass sweep (which has I_ratio=1.0)
        opt = get_optimal_params_mass(df, motor_scale, mass)
        I_actual = I_BASE
    else:
        # Use inertia sweep
        opt = get_optimal_params_inertia(df, motor_scale, I_ratio)
        I_actual = opt['I']
    
    # Build params for simulation
    params = {
        'm': mass,
        'I': I_actual,
        'stroke': 0.050,
        'g': 9.81,
        'tau_max': BASE_TAU_MAX * motor_scale,
        'w_max': BASE_W_MAX * motor_scale,
    }
    
    print(f"\nCase: motor={motor_scale}x, m={mass*1000:.0f}g, I_ratio={I_ratio:.2f}{label_suffix}")
    print(f"  TSA Optimal: L={opt['L']*1000:.2f}mm, r={opt['r']*1000:.3f}mm, T={opt['T']*1000:.1f}ms")
    print(f"  Expected ydot: {opt['ydot']:.3f} m/s")
    print(f"  Actual params: {params}")
    
    # Run simulations
    res_all = run_simulations(params, opt['L'], opt['r'])
    
    # Print actual final velocities
    print(f"  Optimal Pulley: r={res_all['r_pulley']*1000:.3f}mm")
    print(f"  Simulated ydot: TSA={res_all['tsa']['y_d'][-1]:.3f}, "
          f"Pulley={res_all['pulley']['y_d'][-1]:.3f}, "
          f"ConstSpeed={res_all['const_speed']['y_d'][-1]:.3f}")
    
    # Generate title
    I_ratio_str = f"{I_ratio:.2f}x" if I_ratio != 1.0 else "1x"
    title = (f"Motor {motor_scale}x | Mass {mass*1000:.0f}g | Inertia {I_ratio_str}")
    
    # Create safe filename
    motor_str = f"{motor_scale:.1f}".replace('.', 'p')
    I_str = f"{I_ratio:.2f}".replace('.', 'p')
    filename = f"case_motor{motor_str}x_m{int(mass*1000)}g_I{I_str}x.png"
    
    return params, opt, res_all, title, filename


def main():
    """Generate all case plots."""
    print("="*70)
    print("Generating Optimal Case Comparison Plots")
    print("="*70)
    
    # Create output directory
    output_dir = Path(__file__).parent / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define cases: (motor_scale, mass, I_ratio_mode)
    # I_ratio_mode: 1.0 for baseline, or 'best' to find optimal
    cases = [
        (0.5, 0.100, 1.0),
        (0.5, 0.100, 'best'),
        (1.0, 0.100, 1.0),
        (1.0, 0.100, 'best'),
        (1.0, 0.200, 1.0),
        (1.0, 0.200, 'best'),
        (2.0, 0.100, 1.0),
        (2.0, 0.100, 'best'),
    ]
    
    for motor_scale, mass, I_ratio_mode in cases:
        df = load_csv(motor_scale)
        
        if I_ratio_mode == 'best':
            I_ratio = get_best_inertia_ratio(df)
            label_suffix = " (BEST I)"
        else:
            I_ratio = I_ratio_mode
            label_suffix = ""
        
        params, opt, res_all, title, filename = generate_case_plot(
            motor_scale, mass, I_ratio, label_suffix
        )
        
        save_path = output_dir / filename
        plot_4x2(res_all, params, title, save_path)
    
    if SHOW_PLOTS:
        print("\nAll plots displayed! Close windows to exit.")
        plt.show(block=True)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()

