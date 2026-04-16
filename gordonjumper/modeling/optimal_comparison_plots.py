"""
Parameter sweep for TSA optimization with CSV output.
Saves all guess results for later analysis.
"""
import casadi as ca
import numpy as np
import csv
from datetime import datetime

# Parameter bounds
L_BOUNDS = (0.100, 0.300)
R_BOUNDS = (0.001, 0.015)
T_BOUNDS = (0.01, 0.5)

# Initial guesses
GUESSES = [
    [0.12, 0.004, 0.03],
    [0.12, 0.010, 0.03],
    [0.18, 0.004, 0.03],
    [0.18, 0.010, 0.03],
]


def build_problem(params, N, theta_margin=0.95):
    """Build optimization with y_offset=0 fixed."""
    stroke = params['stroke']
    m, I = params['m'], params['I']
    tau_max, w_max, g = params['tau_max'], params['w_max'], params['g']
    L_min = params.get('L_min', L_BOUNDS[0])
    L_max = params.get('L_max', L_BOUNDS[1])

    eps = 1e-9
    sqrt_eps = 1e-12

    opti = ca.Opti()

    # Design variables (NO y_offset - fixed to 0)
    L = opti.variable()
    r = opti.variable()
    T = opti.variable()
    dt = T / N

    # State trajectory
    X = opti.variable(2, N+1)
    theta, theta_dot = X[0, :], X[1, :]

    def safe_sqrt(x):
        return ca.sqrt(ca.fmax(x, sqrt_eps))

    def y_from_theta(th):
        return L - safe_sqrt(L**2 - (r*th)**2)

    def dy_dtheta(th):
        return (r**2 * th) / safe_sqrt(L**2 - (r*th)**2)

    def dynamics(x):
        th, thd = x[0], x[1]
        denom = L**2 - (r*th)**2
        sqrt_denom = safe_sqrt(denom)
        dyd = (r**2 * th) / sqrt_denom
        d2yd = (r**2 * L**2) / (ca.fmax(denom, sqrt_eps)**1.5)
        tau = tau_max * (1 - thd / w_max)
        M = I + m * dyd**2
        thdd = (tau - m*g*dyd - m*(thd**2)*dyd*d2yd) / (M + eps)
        return ca.vertcat(thd, thdd)

    def rk4_step(xk):
        k1 = dynamics(xk)
        k2 = dynamics(xk + dt/2 * k1)
        k3 = dynamics(xk + dt/2 * k2)
        k4 = dynamics(xk + dt * k3)
        return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # Variable bounds
    singularity_margin = min(0.98, theta_margin + 0.02)
    opti.subject_to(opti.bounded(L_min, L, L_max))
    opti.subject_to(opti.bounded(R_BOUNDS[0], r, R_BOUNDS[1]))
    opti.subject_to(opti.bounded(T_BOUNDS[0], T, T_BOUNDS[1]))
    
    # Geometric: stroke <= singularity_margin * L (since y_offset=0)
    opti.subject_to(stroke <= singularity_margin * L)

    # Initial conditions: y_offset=0 means theta[0]=0
    opti.subject_to(theta[0] == 0)
    opti.subject_to(theta_dot[0] == 0)

    # Path constraints
    opti.subject_to(theta >= 0)
    opti.subject_to(r * theta <= theta_margin * L)
    opti.subject_to(theta_dot >= 0)
    opti.subject_to(theta_dot <= 0.999 * w_max)

    # y trajectory: from 0 to stroke
    y_nodes = ca.horzcat(*[y_from_theta(theta[k]) for k in range(N+1)])
    opti.subject_to(y_nodes >= 0)
    opti.subject_to(y_nodes <= stroke)
    opti.subject_to(y_nodes[-1] == stroke)

    # Dynamics
    for k in range(N):
        opti.subject_to(X[:, k+1] == rk4_step(X[:, k]))

    # Objective
    ydotN = dy_dtheta(theta[-1]) * theta_dot[-1]
    opti.minimize(-ydotN)

    return opti, {'L': L, 'r': r, 'T': T, 'theta': theta, 'theta_dot': theta_dot,
                  'y_nodes': y_nodes, 'ydotN': ydotN, 'theta_margin': theta_margin}


def solve_once(params, guess, N, max_iter, tol, theta_margin):
    """Single solve. guess = [L, r, T]"""
    opti, syms = build_problem(params, N, theta_margin)
    L0, r0, T0 = guess[:3]
    stroke = params['stroke']
    w_max = params['w_max']
    
    opti.set_initial(syms['L'], L0)
    opti.set_initial(syms['r'], r0)
    opti.set_initial(syms['T'], T0)
    
    # Theta trajectory guess: from 0 to theta at y=stroke
    def theta_at_y(Lv, rv, yv):
        yv = min(yv, 0.85 * Lv)
        val = max(Lv**2 - (Lv - yv)**2, 0.0)
        return np.sqrt(val) / max(rv, 1e-9)
    
    th_end = theta_at_y(L0, r0, min(stroke, 0.85 * L0))
    th_max = 0.85 * theta_margin * L0 / max(r0, 1e-9)
    N_nodes = syms['theta'].shape[1]
    opti.set_initial(syms['theta'], np.linspace(0, min(th_end, th_max), N_nodes))
    opti.set_initial(syms['theta_dot'], np.linspace(0, 0.2 * w_max, N_nodes))

    opti.solver("ipopt", {
        "print_time": 0, "ipopt.print_level": 0,
        "ipopt.max_iter": max_iter, "ipopt.tol": tol,
        "ipopt.mu_strategy": "adaptive",
    })

    try:
        sol = opti.solve()
        result = {
            "success": True,
            "L": sol.value(syms['L']),
            "r": sol.value(syms['r']),
            "T": sol.value(syms['T']),
            "y_final": sol.value(syms['y_nodes'][-1]),
            "ydot_final": sol.value(syms['ydotN']),
        }
        if any(np.isnan(v) for k, v in result.items() if k != 'success'):
            return {"success": False, "error": "NaN"}
        return result
    except RuntimeError as e:
        return {"success": False, "error": str(e)[:50]}


def solve_all_guesses(params, N=40, theta_margin=0.98, max_iter=1000, tol=1e-6):
    """Solve with all guesses and return all results."""
    results = []
    for i, guess in enumerate(GUESSES):
        result = solve_once(params, guess, N, max_iter, tol, theta_margin)
        result['guess_idx'] = i
        result['guess_L'] = guess[0]
        result['guess_r'] = guess[1]
        result['guess_T'] = guess[2]
        results.append(result)
    return results


def run_sweep_and_save(base_params, motor_scale, I_base, filename):
    """Run mass and inertia sweeps and save to CSV."""
    
    # Apply motor scaling
    params = base_params.copy()
    params['tau_max'] = base_params['tau_max'] * motor_scale
    params['w_max'] = base_params['w_max'] * motor_scale
    
    print(f"\n{'='*70}")
    print(f"Motor scaling: {motor_scale}x")
    print(f"  tau_max = {params['tau_max']:.4f} N·m")
    print(f"  w_max   = {params['w_max']:.1f} rad/s")
    print(f"Output file: {filename}")
    print('='*70)
    
    # Sweep parameters
    mass_values = np.linspace(0.050, 0.500, 30)
    inertia_ratios = np.logspace(-1, 1, 30)
    inertia_values = I_base * inertia_ratios
    
    all_rows = []
    
    # Mass sweep
    print(f"\nMass Sweep (50g to 500g) - 30 points:")
    print("-"*70)
    n_total = len(mass_values)
    for i, m in enumerate(mass_values):
        print(f"  [{i+1:2d}/{n_total}] m={m*1e3:6.1f}g", end="", flush=True)
        sweep_params = params.copy()
        sweep_params['m'] = m
        
        results = solve_all_guesses(sweep_params)
        
        # Find best for display
        best_ydot = -np.inf
        for res in results:
            if res['success'] and res['ydot_final'] > best_ydot:
                best_ydot = res['ydot_final']
        
        n_success = sum(1 for r in results if r['success'])
        print(f" -> {n_success}/4 converged", end="")
        if best_ydot > -np.inf:
            print(f", best ydot={best_ydot:.4f}m/s")
        else:
            print(" FAILED")
        
        # Save all results
        for res in results:
            row = {
                'sweep_type': 'mass',
                'motor_scale': motor_scale,
                'tau_max': params['tau_max'],
                'w_max': params['w_max'],
                'm': m,
                'I': params['I'],
                'I_ratio': 1.0,
                'stroke': params['stroke'],
                'g': params['g'],
                'L_min': params['L_min'],
                'L_max': params['L_max'],
                'guess_idx': res['guess_idx'],
                'guess_L': res['guess_L'],
                'guess_r': res['guess_r'],
                'guess_T': res['guess_T'],
                'success': res['success'],
                'opt_L': res.get('L', np.nan),
                'opt_r': res.get('r', np.nan),
                'opt_T': res.get('T', np.nan),
                'y_final': res.get('y_final', np.nan),
                'ydot_final': res.get('ydot_final', np.nan),
                'error': res.get('error', ''),
            }
            all_rows.append(row)
    
    # Inertia sweep
    print(f"\nInertia Sweep (0.1× to 10× base) - 30 points:")
    print("-"*70)
    n_total = len(inertia_values)
    for i, (I, I_ratio) in enumerate(zip(inertia_values, inertia_ratios)):
        print(f"  [{i+1:2d}/{n_total}] I={I_ratio:5.2f}x", end="", flush=True)
        sweep_params = params.copy()
        sweep_params['I'] = I
        
        results = solve_all_guesses(sweep_params)
        
        # Find best for display
        best_ydot = -np.inf
        for res in results:
            if res['success'] and res['ydot_final'] > best_ydot:
                best_ydot = res['ydot_final']
        
        n_success = sum(1 for r in results if r['success'])
        print(f" -> {n_success}/4 converged", end="")
        if best_ydot > -np.inf:
            print(f", best ydot={best_ydot:.4f}m/s")
        else:
            print(" FAILED")
        
        # Save all results
        for res in results:
            row = {
                'sweep_type': 'inertia',
                'motor_scale': motor_scale,
                'tau_max': params['tau_max'],
                'w_max': params['w_max'],
                'm': params['m'],
                'I': I,
                'I_ratio': I_ratio,
                'stroke': params['stroke'],
                'g': params['g'],
                'L_min': params['L_min'],
                'L_max': params['L_max'],
                'guess_idx': res['guess_idx'],
                'guess_L': res['guess_L'],
                'guess_r': res['guess_r'],
                'guess_T': res['guess_T'],
                'success': res['success'],
                'opt_L': res.get('L', np.nan),
                'opt_r': res.get('r', np.nan),
                'opt_T': res.get('T', np.nan),
                'y_final': res.get('y_final', np.nan),
                'ydot_final': res.get('ydot_final', np.nan),
                'error': res.get('error', ''),
            }
            all_rows.append(row)
    
    # Write CSV
    print(f"\nSaving {len(all_rows)} rows to {filename}...", flush=True)
    fieldnames = list(all_rows[0].keys())
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Done! Saved {filename}")
    
    return all_rows


if __name__ == "__main__":
    # Base inertia value
    # I_base = 0.5 * (37.4e-3/2) * (28.4e-3/2)**2
    I_base = 4.5e-6 #from free speed fits at 3V

    voltage_base = 3.5
    
    # Base parameters (before motor scaling)
    base_params = {
        'm': 0.100,
        'I': I_base,
        'stroke': 0.050,
        'g': 9.81,
        # 'tau_max': 0.2795,  # Will be scaled
        'tau_max': 0.09555555556 * voltage_base,  # Will be scaled
        'w_max': 2135*2*np.pi/60 * voltage_base,       # Will be scaled
        'L_min': 0.100,
        'L_max': 0.180,
    }
    
    print("="*70)
    print("TSA Optimization Parameter Sweeps with CSV Output")
    print("="*70)
    print(f"Base params: stroke={base_params['stroke']*1e3:.0f}mm, "
          f"L∈[{base_params['L_min']*1e3:.0f},{base_params['L_max']*1e3:.0f}]mm")
    print(f"Base inertia: I_base = {I_base:.6e} kg·m²")
    print(f"Base tau_max: {base_params['tau_max']:.4f} N·m")
    print(f"Base w_max: {base_params['w_max']:.1f} rad/s")
    print(f"Guesses: {len(GUESSES)}")
    print(f"Points per sweep: 30")
    print(f"Motor scales: 0.5x, 1.0x, 2.0x")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run for each motor scaling
    for scale in [0.5, 1.0, 2.0]:
        scale_str = f"{scale:.1f}".replace('.', 'p')
        filename = f"sweep_motor_{scale_str}x.csv"
        run_sweep_and_save(base_params, scale, I_base, filename)
    
    print(f"\n{'='*70}")
    print(f"All sweeps complete! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
