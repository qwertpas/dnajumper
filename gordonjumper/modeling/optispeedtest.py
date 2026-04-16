import casadi as ca
import numpy as np
import time

def get_ode(state, L, r, T, params):
    # Unpack state
    y = state[0]
    yd = state[1]
    
    # Unpack params
    stroke = params[0]
    m = params[1]
    I = params[2]
    tau_max = params[3]
    w_max = params[4]
    g = params[5]

    # Dynamics
    G = -y/(r*ca.sqrt(L**2 - y**2))
    ydd = (G * tau_max*(1 - G*yd/w_max) - m*g) / (m + I*G**2)
    
    # Derivatives w.r.t. normalized time tau (0 to 1), so multiply by T
    return ca.vertcat(T*yd, T*ydd)

def run_rk4_step(state, L, r, T, params, dt):
    k1 = get_ode(state, L, r, T, params)
    k2 = get_ode(state + dt/2 * k1, L, r, T, params)
    k3 = get_ode(state + dt/2 * k2, L, r, T, params)
    k4 = get_ode(state + dt * k3, L, r, T, params)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def setup_common_vars_and_constraints(opti, params, guess):
    stroke = params[0]
    
    L_var = opti.variable()
    r_var = opti.variable()
    y0_var = opti.variable()
    T_var = opti.variable()
    
    # Initial Guess
    opti.set_initial(L_var, guess[0])
    opti.set_initial(r_var, guess[1])
    opti.set_initial(y0_var, guess[2])
    opti.set_initial(T_var, guess[3])

    # Constraints
    opti.subject_to(L_var > stroke)
    opti.subject_to(L_var < 1.2345)
    opti.subject_to(r_var > 0.0001)
    opti.subject_to(r_var < 0.050)
    opti.subject_to(y0_var > stroke)
    opti.subject_to(y0_var < L_var - 0.001)
    
    opti.subject_to(T_var >= 0.001)
    opti.subject_to(T_var <= 2.0)
    
    return L_var, r_var, y0_var, T_var

def get_solution_values(sol, vars_list):
    return [sol.value(v) for v in vars_list]

def solve_baseline(params, guess, N=20):
    opti = ca.Opti()
    L, r, y0, T = setup_common_vars_and_constraints(opti, params, guess)
    
    # Integrator setup
    state = ca.MX.sym('state', 2)
    L_sym = ca.MX.sym('L')
    r_sym = ca.MX.sym('r')
    T_sym = ca.MX.sym('T')
    
    ode = get_ode(state, L_sym, r_sym, T_sym, params)
    
    # p must match the order passed in later
    dae = {'x': state, 'p': ca.vertcat(L_sym, r_sym, T_sym), 'ode': ode}
    opts = {'number_of_finite_elements': N}
    integrator = ca.integrator('F', 'rk', dae, opts)
    
    # Call integrator: x0 is [y0, 0]
    # p is [L, r, T]
    res = integrator(x0=ca.vertcat(y0, 0), p=ca.vertcat(L, r, T))
    y_final = res['xf'][0]
    yd_final = res['xf'][1]
    
    opti.minimize(yd_final)
    
    stroke = params[0]
    opti.subject_to(y_final > y0 - stroke)
    opti.subject_to(y_final < y0)
    
    opts_solver = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts_solver)
    
    sol = opti.solve()
    return get_solution_values(sol, [L, r, y0, T])

def solve_manual_rk4_single(params, guess, N=20):
    opti = ca.Opti()
    L, r, y0, T = setup_common_vars_and_constraints(opti, params, guess)
    
    state = ca.vertcat(y0, 0) # Initial state
    dt = 1.0 / N
    
    # Symbolic loop to build the expression graph
    for _ in range(N):
        state = run_rk4_step(state, L, r, T, params, dt)
        
    y_final = state[0]
    yd_final = state[1]
    
    opti.minimize(yd_final)
    
    stroke = params[0]
    opti.subject_to(y_final > y0 - stroke)
    opti.subject_to(y_final < y0)
    
    opts_solver = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts_solver)
    
    sol = opti.solve()
    return get_solution_values(sol, [L, r, y0, T])

def solve_manual_rk4_multiple(params, guess, N=20):
    opti = ca.Opti()
    L, r, y0, T = setup_common_vars_and_constraints(opti, params, guess)
    
    # Decision variables for states at each step: X = [y, yd] at each node
    # N steps means N+1 nodes (0 to N)
    X = opti.variable(2, N + 1)
    
    # Initial state constraint
    opti.subject_to(X[:, 0] == ca.vertcat(y0, 0))
    
    dt = 1.0 / N
    
    # Gap closing constraints
    for k in range(N):
        state_k = X[:, k]
        state_next_predicted = run_rk4_step(state_k, L, r, T, params, dt)
        opti.subject_to(X[:, k+1] == state_next_predicted)
        
    y_final = X[0, N]
    yd_final = X[1, N]
    
    opti.minimize(yd_final)
    
    stroke = params[0]
    opti.subject_to(y_final > y0 - stroke)
    opti.subject_to(y_final < y0)
    
    opts_solver = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts_solver)
    
    sol = opti.solve()
    return get_solution_values(sol, [L, r, y0, T])

def evaluate_method(method_func, name, params, guess, N=20, runs=10):
    times = []
    vals = None
    print(f"Evaluating {name}...", end='', flush=True)
    try:
        # Warmup
        method_func(params, guess, N)
        
        for _ in range(runs):
            t0 = time.time()
            vals = method_func(params, guess, N)
            t1 = time.time()
            times.append(t1 - t0)
        
        avg_time = np.mean(times)
        print(f" Done. Avg time: {avg_time:.4f}s")
        return vals, avg_time
    except Exception as e:
        print(f" Failed: {e}")
        return None, None

if __name__ == "__main__":
    guess = [0.050, 0.002, 0.045, 0.02367]
    params = [
        0.040, # stroke
        0.4,  # m
        0.5 * (37.4/1000/2) * (28.4/1000/2)**2, # I rotor inertia
        0.2795,  # tau_max
        1000,  # w_max
        9.81 # g
    ]
    
    N = 20
    
    sol_baseline, time_baseline = evaluate_method(solve_baseline, "Baseline (ca.integrator)", params, guess, N)
    sol_m1, time_m1 = evaluate_method(solve_manual_rk4_single, "Method 1 (Manual RK4 Single Expr)", params, guess, N)
    sol_m2, time_m2 = evaluate_method(solve_manual_rk4_multiple, "Method 2 (Manual RK4 Multiple Shooting)", params, guess, N)
    
    print("\n--- Results Comparison ---")
    labels = ['L', 'r', 'y0', 'T']
    
    # Header
    header = f"{'Variable':<10} | {'Baseline':<12} | {'Method 1':<12} | {'Method 2':<12} | {'Diff M1(%)':<12} | {'Diff M2(%)':<12}"
    print(header)
    print("-" * len(header))
    
    if sol_baseline and sol_m1 and sol_m2:
        for i, label in enumerate(labels):
            v_b = sol_baseline[i]
            v_1 = sol_m1[i]
            v_2 = sol_m2[i]
            
            diff_1 = abs(v_1 - v_b) / abs(v_b) * 100 if v_b != 0 else abs(v_1 - v_b)
            diff_2 = abs(v_2 - v_b) / abs(v_b) * 100 if v_b != 0 else abs(v_2 - v_b)
            
            print(f"{label:<10} | {v_b:<12.6f} | {v_1:<12.6f} | {v_2:<12.6f} | {diff_1:<12.4e} | {diff_2:<12.4e}")

    print("\n--- Timing Comparison ---")
    print(f"Baseline: {time_baseline:.4f}s")
    if time_m1: print(f"Method 1: {time_m1:.4f}s (x{time_m1/time_baseline:.2f})")
    if time_m2: print(f"Method 2: {time_m2:.4f}s (x{time_m2/time_baseline:.2f})")
