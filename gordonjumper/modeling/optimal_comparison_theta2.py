"""
TSA (Twisted String Actuator) Optimization using CasADi.
Maximizes final linear velocity (ydot) for a given stroke.
"""
import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

# Parameter bounds (in meters)
L_BOUNDS = (0.100, 0.300)       # 100-300mm string length
R_BOUNDS = (0.001, 0.015)       # 1-15mm string radius
T_BOUNDS = (0.01, 0.5)          # 10-500ms time horizon
Y_OFFSET_BOUNDS = (0.0, 0.020)  # 0-20mm starting offset


def build_opti_problem(params, N, theta_margin=0.95):
    """
    Build the CasADi optimization problem for TSA.
    
    Args:
        params: dict with keys: m, I, stroke, g, tau_max, w_max, L_min, L_max, y_offset_max
        N: number of shooting intervals
        theta_margin: fraction of singularity to stay within (r*theta <= theta_margin*L)
    
    Returns:
        opti: CasADi Opti object
        syms: dict of symbolic variables and expressions
    """
    stroke = params['stroke']
    m, I = params['m'], params['I']
    tau_max, w_max, g = params['tau_max'], params['w_max'], params['g']
    L_min = params.get('L_min', L_BOUNDS[0])
    L_max = params.get('L_max', L_BOUNDS[1])
    y_offset_max = params.get('y_offset_max', Y_OFFSET_BOUNDS[1])

    eps = 1e-9
    sqrt_eps = 1e-12

    opti = ca.Opti()

    # Design variables
    L = opti.variable()
    r = opti.variable()
    T = opti.variable()
    y_offset = opti.variable()
    dt = T / N

    # State trajectory: [theta, theta_dot] at N+1 nodes
    X = opti.variable(2, N+1)
    theta, theta_dot = X[0, :], X[1, :]

    # Safe sqrt to avoid NaN during solver exploration
    def safe_sqrt(x):
        return ca.sqrt(ca.fmax(x, sqrt_eps))

    # TSA kinematics: y = L - sqrt(L^2 - (r*theta)^2)
    def y_from_theta(th):
        return L - safe_sqrt(L**2 - (r*th)**2)

    def theta_from_y(y):
        return safe_sqrt(L**2 - (L - y)**2) / r

    def dy_dtheta(th):
        return (r**2 * th) / safe_sqrt(L**2 - (r*th)**2)

    # TSA dynamics
    def dynamics(x):
        th, thd = x[0], x[1]
        denom = L**2 - (r*th)**2
        sqrt_denom = safe_sqrt(denom)
        dyd = (r**2 * th) / sqrt_denom
        d2yd = (r**2 * L**2) / (ca.fmax(denom, sqrt_eps)**1.5)
        tau = tau_max * (1 - thd / w_max)  # Linear torque-speed curve
        M = I + m * dyd**2                  # Effective inertia
        thdd = (tau - m*g*dyd - m*(thd**2)*dyd*d2yd) / (M + eps)
        return ca.vertcat(thd, thdd)

    # RK4 integrator
    def rk4_step(xk):
        k1 = dynamics(xk)
        k2 = dynamics(xk + dt/2 * k1)
        k3 = dynamics(xk + dt/2 * k2)
        k4 = dynamics(xk + dt * k3)
        return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # Constraints
    singularity_margin = min(0.98, theta_margin + 0.02)
    
    opti.subject_to(opti.bounded(L_min, L, L_max))
    opti.subject_to(opti.bounded(R_BOUNDS[0], r, R_BOUNDS[1]))
    opti.subject_to(opti.bounded(T_BOUNDS[0], T, T_BOUNDS[1]))
    opti.subject_to(opti.bounded(0.0, y_offset, y_offset_max))
    opti.subject_to(y_offset + stroke <= singularity_margin * L)

    # Initial conditions
    opti.subject_to(theta[0] == theta_from_y(y_offset))
    opti.subject_to(theta_dot[0] == 0)

    # Path constraints
    opti.subject_to(theta >= 0)
    opti.subject_to(r * theta <= theta_margin * L)
    opti.subject_to(theta_dot >= 0)
    opti.subject_to(theta_dot <= 0.999 * w_max)

    # Trajectory bounds
    y_nodes = ca.horzcat(*[y_from_theta(theta[k]) for k in range(N+1)])
    opti.subject_to(y_nodes >= y_offset)
    opti.subject_to(y_nodes <= y_offset + stroke)
    opti.subject_to(y_nodes[-1] == y_offset + stroke)

    # Multiple shooting dynamics
    for k in range(N):
        opti.subject_to(X[:, k+1] == rk4_step(X[:, k]))

    # Objective: maximize final velocity
    ydotN = dy_dtheta(theta[-1]) * theta_dot[-1]
    opti.minimize(-ydotN)

    return opti, {
        'L': L, 'r': r, 'T': T, 'y_offset': y_offset,
        'theta': theta, 'theta_dot': theta_dot,
        'y_nodes': y_nodes, 'ydotN': ydotN,
        'theta_margin': theta_margin,
    }


def solve_once(params, guess, N, max_iter, tol, theta_margin):
    """Single solve attempt. Returns (success, result_dict)."""
    opti, syms = build_opti_problem(params, N, theta_margin)
    L0, r0, T0, y_off0 = guess[:4]
    stroke = params['stroke']
    w_max = params['w_max']
    
    # Set initial values for design variables
    opti.set_initial(syms['L'], L0)
    opti.set_initial(syms['r'], r0)
    opti.set_initial(syms['T'], T0)
    opti.set_initial(syms['y_offset'], y_off0)
    
    # Generate trajectory initial guess
    def theta_at_y(Lv, rv, yv):
        yv = min(yv, 0.85 * Lv)
        val = max(Lv**2 - (Lv - yv)**2, 0.0)
        return np.sqrt(val) / max(rv, 1e-9)

    th_start = theta_at_y(L0, r0, y_off0)
    th_end = theta_at_y(L0, r0, min(y_off0 + stroke, 0.85 * L0))
    th_max = 0.85 * theta_margin * L0 / max(r0, 1e-9)
    
    N_nodes = syms['theta'].shape[1]
    opti.set_initial(syms['theta'], np.linspace(th_start, min(th_end, th_max), N_nodes))
    opti.set_initial(syms['theta_dot'], np.linspace(0.0, 0.2 * w_max, N_nodes))

    # Solver options
    opti.solver("ipopt", {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.max_iter": max_iter,
        "ipopt.tol": tol,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.bound_push": 1e-6,
        "ipopt.bound_frac": 1e-6,
    })

    try:
        sol = opti.solve()
        result = {
            "L": sol.value(syms['L']),
            "r": sol.value(syms['r']),
            "T": sol.value(syms['T']),
            "y_offset": sol.value(syms['y_offset']),
            "y_final": sol.value(syms['y_nodes'][-1]),
            "ydot_final": sol.value(syms['ydotN']),
        }
        if any(np.isnan(v) for v in result.values()):
            return False, {"error": "NaN in solution"}
        return True, result
    except RuntimeError as e:
        return False, {"error": str(e)}


def compute_min_theta_margin(L_val, y_final):
    """Compute minimum theta_margin needed to reach y_final with string length L."""
    if y_final >= L_val:
        return 1.0
    ratio = (L_val - y_final) / L_val
    return np.sqrt(1 - ratio**2)


def optimize_tsa(params, guess=None, N=40, max_attempts=15, verbose=False):
    """
    Robust TSA optimization with multiple attempts.
    
    Args:
        params: dict with physical parameters
        guess: optional [L, r, T, y_offset] initial guess
        N: target number of shooting intervals
        max_attempts: maximum solve attempts
        verbose: print progress
    
    Returns:
        dict with optimal L, r, T, y_offset, y_final, ydot_final
    """
    stroke = params['stroke']
    L_min = params.get('L_min', L_BOUNDS[0])
    L_max = params.get('L_max', L_BOUNDS[1])
    y_off_max = params.get('y_offset_max', Y_OFFSET_BOUNDS[1])

    # Attempt configurations: (N, max_iter, tol)
    # theta_margin will be computed dynamically for each (L, y_offset) combination
    configs = [
        (15, 500, 1e-4),
        (15, 500, 1e-4),
        (20, 600, 1e-4),
        (20, 600, 1e-4),
        (25, 700, 1e-5),
        (30, 800, 1e-5),
        (35, 1000, 1e-6),
        (40, 1200, 1e-6),
        (40, 1500, 1e-6),
        (50, 1500, 1e-6),
        (50, 2000, 1e-6),
        (60, 2500, 1e-6),
    ]

    # Default or extract base guess
    if guess is None:
        base_L = (L_min + L_max) / 2
        base_r = 0.005
        base_T = 0.05
        base_y_off = 0.0
    else:
        base_L, base_r, base_T = guess[0], guess[1], guess[2]
        base_y_off = guess[3] if len(guess) > 3 else 0.0

    r_grid = [0.003, 0.005, 0.007, 0.009, 0.011, 0.013]
    best_result = None
    best_ydot = -np.inf

    for idx in range(min(max_attempts, len(configs))):
        N_try, max_iter, tol = configs[idx]

        # Vary guesses systematically - explore full L range including L_min
        L_try = L_min + (L_max - L_min) * ((idx % 5) / 4)  # 0, 0.25, 0.5, 0.75, 1.0 of range
        r_try = r_grid[idx % len(r_grid)]
        T_try = base_T * (0.6 + 0.6 * ((idx % 5) / 4))
        
        L_try = np.clip(L_try, L_min, L_max)
        r_try = np.clip(r_try, R_BOUNDS[0], R_BOUNDS[1])
        T_try = np.clip(T_try, T_BOUNDS[0], T_BOUNDS[1])
        
        # Compute y_offset: ensure geometric feasibility
        max_y_off = min(y_off_max, max(0.0, 0.88 * L_try - stroke))
        y_off_try = 0.0 if idx % 2 == 0 else min(base_y_off, max_y_off * 0.3)
        y_off_try = np.clip(y_off_try, 0.0, max_y_off)
        
        # Compute theta_margin dynamically based on actual (L, y_offset) being tried
        # This is the key fix: use actual y_final, not worst-case y_off_max
        y_final_try = y_off_try + stroke
        min_tm = compute_min_theta_margin(L_try, y_final_try)
        theta_margin = min(0.98, min_tm + 0.02)

        current_guess = [L_try, r_try, T_try, y_off_try]

        if verbose:
            print(f"Attempt {idx+1}: N={N_try}, tm={theta_margin:.2f}, "
                  f"[L={L_try*1e3:.1f}, r={r_try*1e3:.2f}, T={T_try*1e3:.0f}, y0={y_off_try*1e3:.1f}]mm")

        success, result = solve_once(params, current_guess, N_try, max_iter, tol, theta_margin)

        if success:
            ydot = result["ydot_final"]
            if verbose:
                print(f"  -> Success! ydot={ydot:.3f} m/s")

            if ydot > best_ydot:
                best_ydot = ydot
                best_result = result

            # Refine with target N if early success
            if idx < 5 and N_try < N:
                refined_guess = [result["L"], result["r"], result["T"], result["y_offset"]]
                ok, ref = solve_once(params, refined_guess, N, 800, 1e-6, theta_margin)
                if ok and ref["ydot_final"] > best_ydot:
                    best_ydot = ref["ydot_final"]
                    best_result = ref
                    if verbose:
                        print(f"  -> Refined: ydot={best_ydot:.3f} m/s")

            return best_result
        elif verbose:
            print(f"  -> Failed")

    if best_result:
        return best_result
    raise RuntimeError(f"Optimization failed after {max_attempts} attempts")


def verify_solution(params, result, verbose=True):
    """
    Verify optimization result by numerically integrating the dynamics.
    
    Returns dict with:
        - numeric_y_final: final y from numerical integration
        - numeric_ydot_final: final ydot from numerical integration  
        - casadi_y_final: y_final from CasADi result
        - casadi_ydot_final: ydot_final from CasADi result
        - y_error: absolute error in y
        - ydot_error: absolute error in ydot
        - valid: True if errors are within tolerance
    """
    L = result['L']
    r = result['r']
    y_offset = result['y_offset']
    stroke = params['stroke']
    m, I = params['m'], params['I']
    g = params['g']
    tau_max, w_max = params['tau_max'], params['w_max']
    
    # Compute initial theta from y_offset
    theta0 = np.sqrt(L**2 - (L - y_offset)**2) / r if y_offset > 0 else 0.0
    
    # TSA kinematics
    def get_y_tsa(theta):
        rt = r * theta
        if rt >= L:
            # At or past singularity
            return L, np.inf, np.inf
        denom = L**2 - rt**2
        sqrt_denom = np.sqrt(denom)
        y = L - sqrt_denom
        dy_dtheta = (r**2 * theta) / sqrt_denom
        d2y_dtheta2 = (r**2 * L**2) / (denom**1.5)
        return y, dy_dtheta, d2y_dtheta2
    
    # Motor torque (linear torque-speed)
    def get_tau(theta_dot):
        return tau_max * (1 - theta_dot / w_max)
    
    # Dynamics: d/dt [theta, theta_dot] = [theta_dot, theta_ddot]
    def dynamics(t, state):
        theta, theta_dot = state
        y, dy_dtheta, d2y_dtheta2 = get_y_tsa(theta)
        tau = get_tau(theta_dot)
        
        M = I + m * dy_dtheta**2
        theta_ddot = (tau - m*g*dy_dtheta - m*(theta_dot**2)*dy_dtheta*d2y_dtheta2) / M
        return [theta_dot, theta_ddot]
    
    # Event: stop when y reaches y_offset + stroke
    def y_target_event(t, state):
        theta, _ = state
        y, _, _ = get_y_tsa(theta)
        return y - (y_offset + stroke)
    
    y_target_event.terminal = True
    y_target_event.direction = 1  # Only trigger when increasing
    
    # Integrate
    t_max = 1.0
    sol = solve_ivp(
        dynamics,
        t_span=(0, t_max),
        y0=[theta0, 0.0],
        method='RK45',
        events=y_target_event,
        dense_output=True,
        rtol=1e-9,
        atol=1e-12,
    )
    
    # Extract final state
    theta_final = sol.y[0, -1]
    theta_dot_final = sol.y[1, -1]
    y_final, dy_dtheta_final, _ = get_y_tsa(theta_final)
    ydot_final = dy_dtheta_final * theta_dot_final
    
    # Compare with CasADi result
    casadi_y_final = result['y_final']
    casadi_ydot_final = result['ydot_final']
    
    y_error = abs(y_final - casadi_y_final)
    ydot_error = abs(ydot_final - casadi_ydot_final)
    
    # Check if y reached target (y_offset + stroke)
    y_target = y_offset + stroke
    y_target_error = abs(y_final - y_target)
    
    # Tolerance for validation
    tol_y = 1e-4  # 0.1mm
    tol_ydot = 0.05  # 0.05 m/s
    valid = y_error < tol_y and ydot_error < tol_ydot and y_target_error < tol_y
    
    if verbose:
        print(f"Verification (L={L*1e3:.1f}mm, r={r*1e3:.2f}mm, y0={y_offset*1e3:.2f}mm):")
        print(f"  Numeric:  y_final={y_final*1e3:.3f}mm, ydot_final={ydot_final:.4f} m/s, t={sol.t[-1]*1e3:.2f}ms")
        print(f"  CasADi:   y_final={casadi_y_final*1e3:.3f}mm, ydot_final={casadi_ydot_final:.4f} m/s, T={result['T']*1e3:.2f}ms")
        print(f"  Target:   y_target={y_target*1e3:.3f}mm (stroke={stroke*1e3:.1f}mm)")
        print(f"  Errors:   y={y_error*1e3:.4f}mm, ydot={ydot_error:.5f} m/s")
        print(f"  Valid:    {'✓' if valid else '✗'}")
    
    return {
        'numeric_y_final': y_final,
        'numeric_ydot_final': ydot_final,
        'numeric_t_final': sol.t[-1],
        'casadi_y_final': casadi_y_final,
        'casadi_ydot_final': casadi_ydot_final,
        'casadi_T': result['T'],
        'y_error': y_error,
        'ydot_error': ydot_error,
        'y_target_error': y_target_error,
        'valid': valid,
        'sol': sol,
    }


def test_random_guesses(params, n_tests=10, seed=42):
    """
    Test optimization robustness with randomized initial guesses.
    """
    np.random.seed(seed)
    L_min = params.get('L_min', L_BOUNDS[0])
    L_max = params.get('L_max', L_BOUNDS[1])
    y_off_max = params.get('y_offset_max', Y_OFFSET_BOUNDS[1])
    stroke = params['stroke']
    
    print(f"Testing {n_tests} randomized guesses...")
    print(f"Params: L∈[{L_min*1e3:.0f},{L_max*1e3:.0f}]mm, stroke={stroke*1e3:.0f}mm")
    print("-" * 70)
    
    results = []
    for i in range(n_tests):
        # Random guess within bounds
        L_g = np.random.uniform(L_min, L_max)
        r_g = np.random.uniform(R_BOUNDS[0], R_BOUNDS[1])
        T_g = np.random.uniform(T_BOUNDS[0], 0.15)
        y_off_g = np.random.uniform(0, min(y_off_max, max(0, 0.85*L_g - stroke)))
        
        guess = [L_g, r_g, T_g, y_off_g]
        print(f"Test {i+1:2d}: guess=[L={L_g*1e3:.1f}, r={r_g*1e3:.2f}, T={T_g*1e3:.0f}, y0={y_off_g*1e3:.1f}]mm ", end="")
        
        try:
            result = optimize_tsa(params, guess, verbose=False)
            print(f"-> L={result['L']*1e3:.1f}mm, r={result['r']*1e3:.2f}mm, "
                  f"ydot={result['ydot_final']:.3f} m/s ✓")
            results.append(result)
        except RuntimeError:
            print("-> FAILED ✗")
            results.append(None)
    
    # Summary
    successes = [r for r in results if r is not None]
    print("-" * 70)
    print(f"Success rate: {len(successes)}/{n_tests}")
    
    if successes:
        ydots = [r['ydot_final'] for r in successes]
        Ls = [r['L'] for r in successes]
        rs = [r['r'] for r in successes]
        print(f"ydot: mean={np.mean(ydots):.3f}, std={np.std(ydots):.4f}, "
              f"range=[{min(ydots):.3f}, {max(ydots):.3f}]")
        print(f"L:    mean={np.mean(Ls)*1e3:.1f}mm, std={np.std(Ls)*1e3:.2f}mm")
        print(f"r:    mean={np.mean(rs)*1e3:.2f}mm, std={np.std(rs)*1e3:.3f}mm")
    
    return results


if __name__ == "__main__":
    import sys

    # Default parameters
    params = {
        'm': 0.100,
        'I': 0.5 * (37.4e-3/2) * (28.4e-3/2)**2,
        'stroke': 0.080,
        'g': 9.81,
        'tau_max': 0.2795,
        'w_max': 1704,
        'L_min': 0.100,
        'L_max': 0.162,
        'y_offset_max': 0.020,
    }

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run randomized tests
        test_random_guesses(params, n_tests=10)
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        # Run optimization and verify
        print("Optimizing...")
        result = optimize_tsa(params, verbose=False)
        print(f"Optimal: L={result['L']*1e3:.1f}mm, r={result['r']*1e3:.2f}mm, "
              f"T={result['T']*1e3:.1f}ms, y_offset={result['y_offset']*1e3:.2f}mm")
        print(f"         ydot_final={result['ydot_final']:.3f} m/s\n")
        verify_solution(params, result, verbose=True)
    else:
        # Single optimization with verification
        result = optimize_tsa(params, verbose=True)
        print(f"\nOptimal: L={result['L']*1e3:.1f}mm, r={result['r']*1e3:.2f}mm, "
              f"T={result['T']*1e3:.1f}ms, y_offset={result['y_offset']*1e3:.2f}mm")
        print(f"         ydot_final={result['ydot_final']:.3f} m/s\n")
        verify_solution(params, result, verbose=True)
