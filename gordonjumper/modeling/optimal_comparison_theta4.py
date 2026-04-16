"""
Simplified TSA Optimization with y_offset=0 fixed.
Tests whether local optima persist without y_offset as a variable.
"""
import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

# Parameter bounds
L_BOUNDS = (0.100, 0.300)
R_BOUNDS = (0.001, 0.015)
T_BOUNDS = (0.01, 0.5)


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
            "L": sol.value(syms['L']),
            "r": sol.value(syms['r']),
            "T": sol.value(syms['T']),
            "y_final": sol.value(syms['y_nodes'][-1]),
            "ydot_final": sol.value(syms['ydotN']),
        }
        if any(np.isnan(v) for v in result.values()):
            return False, {"error": "NaN"}
        return True, result
    except RuntimeError as e:
        return False, {"error": str(e)}


def verify_solution(params, result):
    """Verify with numerical integration."""
    L, r = result['L'], result['r']
    stroke = params['stroke']
    m, I, g = params['m'], params['I'], params['g']
    tau_max, w_max = params['tau_max'], params['w_max']
    
    def dynamics(t, state):
        theta, theta_dot = state
        rt = r * theta
        if rt >= L:
            return [0, 0]
        denom = L**2 - rt**2
        sqrt_denom = np.sqrt(denom)
        dyd = (r**2 * theta) / sqrt_denom
        d2yd = (r**2 * L**2) / (denom**1.5)
        tau = tau_max * (1 - theta_dot / w_max)
        M = I + m * dyd**2
        thdd = (tau - m*g*dyd - m*(theta_dot**2)*dyd*d2yd) / M
        return [theta_dot, thdd]
    
    def event(t, state):
        theta = state[0]
        rt = r * theta
        if rt >= L:
            return -1
        y = L - np.sqrt(L**2 - rt**2)
        return y - stroke
    event.terminal = True
    event.direction = 1
    
    sol = solve_ivp(dynamics, (0, 1), [0, 0], events=event, rtol=1e-9, atol=1e-12)
    
    theta_f = sol.y[0, -1]
    theta_dot_f = sol.y[1, -1]
    rt = r * theta_f
    y_f = L - np.sqrt(L**2 - rt**2)
    dyd_f = (r**2 * theta_f) / np.sqrt(L**2 - rt**2)
    ydot_f = dyd_f * theta_dot_f
    
    return {
        'numeric_y': y_f, 'numeric_ydot': ydot_f, 'numeric_t': sol.t[-1],
        'casadi_ydot': result['ydot_final'],
        'ydot_error': abs(ydot_f - result['ydot_final']),
        'valid': abs(ydot_f - result['ydot_final']) < 0.05
    }


def test_rk4_convergence(params):
    """Test if solution converges as N increases (proves RK4 accuracy)."""
    print("Testing RK4 convergence (same guess, varying N):")
    print("="*70)
    
    guess = [0.14, 0.005, 0.03]
    theta_margin = 0.98
    
    results = []
    for N in [10, 15, 20, 30, 40, 60]:
        success, result = solve_once(params, guess, N, max_iter=1000, tol=1e-6, theta_margin=theta_margin)
        if success:
            v = verify_solution(params, result)
            results.append((N, result, v))
            print(f"N={N:3d}: L={result['L']*1e3:6.2f}mm, r={result['r']*1e3:5.3f}mm, "
                  f"ydot={result['ydot_final']:.5f}, err={v['ydot_error']:.2e}")
        else:
            print(f"N={N:3d}: FAILED")
    
    if len(results) >= 2:
        ydots = [r[1]['ydot_final'] for r in results]
        print(f"\nydot range: [{min(ydots):.5f}, {max(ydots):.5f}], spread: {max(ydots)-min(ydots):.6f}")
        print("If spread is small (<0.001), RK4 is converged and NOT the source of variance.")
    return results


def test_guess_sensitivity(params):
    """Test if different guesses converge to different optima."""
    print("\nTesting guess sensitivity (same N=40, different guesses):")
    print("="*70)
    
    theta_margin = 0.98
    N = 40
    
    # Grid of guesses
    solutions = []
    for L_g in [0.12, 0.14, 0.16]:
        for r_g in [0.003, 0.006, 0.010]:
            guess = [L_g, r_g, 0.03]
            success, result = solve_once(params, guess, N, max_iter=1000, tol=1e-6, theta_margin=theta_margin)
            if success:
                v = verify_solution(params, result)
                solutions.append((guess, result, v))
                print(f"Guess [L={L_g*1e3:.0f}, r={r_g*1e3:.1f}]mm -> "
                      f"L={result['L']*1e3:6.2f}mm, r={result['r']*1e3:5.3f}mm, ydot={result['ydot_final']:.4f}")
            else:
                print(f"Guess [L={L_g*1e3:.0f}, r={r_g*1e3:.1f}]mm -> FAILED")
    
    if solutions:
        ydots = [s[1]['ydot_final'] for s in solutions]
        Ls = [s[1]['L'] for s in solutions]
        rs = [s[1]['r'] for s in solutions]
        print(f"\nResults spread:")
        print(f"  ydot: [{min(ydots):.4f}, {max(ydots):.4f}], range={max(ydots)-min(ydots):.4f}")
        print(f"  L:    [{min(Ls)*1e3:.1f}, {max(Ls)*1e3:.1f}]mm")
        print(f"  r:    [{min(rs)*1e3:.2f}, {max(rs)*1e3:.2f}]mm")
        print("\nIf ydot spread is large (>0.1), problem has multiple local optima.")
    
    return solutions


if __name__ == "__main__":
    params = {
        'm': 0.100,
        'I': 0.5 * (37.4e-3/2) * (28.4e-3/2)**2,
        'stroke': 0.080,
        'g': 9.81,
        'tau_max': 0.2795,
        'w_max': 1704,
        'L_min': 0.100,
        'L_max': 0.162,
    }
    
    print("Simplified TSA Optimization (y_offset=0 fixed)")
    print("="*70)
    print(f"Params: stroke={params['stroke']*1e3:.0f}mm, L∈[{params['L_min']*1e3:.0f},{params['L_max']*1e3:.0f}]mm")
    print()
    
    test_rk4_convergence(params)
    print()
    test_guess_sensitivity(params)

