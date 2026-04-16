import casadi as ca

def dynamics(x, p):
    theta = x[0]
    theta_d = x[1]
    
    # Unpack parameters
    L_sym = p[0]
    tau_max_sym = p[1]
    r_sym = p[2]
    m_sym = p[3]
    w_max_sym = p[4]
    
    theta_dd = (L_sym * tau_max_sym) / (r_sym**2 * m_sym * theta) * (1 - theta_d / w_max_sym)
    return ca.vertcat(theta_d, theta_dd)

def rk4_integrate(T, params, N=50, theta_init=1e-5):
    """
    Integrates the system dynamics using RK4.
    
    Args:
        T: Total time (CasADi expression or float)
        params: Parameter vector [L, tau_max, r, m, w_max]
        N: Number of steps
        theta_init: Initial angle
        
    Returns:
        tuple: (theta_final, theta_d_final) as CasADi expressions
    """
    dt = T / N
    current_x = ca.vertcat(theta_init, 0.0) # [theta, theta_d]
    
    for _ in range(N):
        k1 = dynamics(current_x, params)
        k2 = dynamics(current_x + 0.5 * dt * k1, params)
        k3 = dynamics(current_x + 0.5 * dt * k2, params)
        k4 = dynamics(current_x + dt * k3, params)
        
        current_x = current_x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return current_x[0], current_x[1]

def get_symbolic_expression(N=50, theta_init=1e-5):
    """
    Creates a symbolic CasADi expression for the final state of the system
    after integrating for time T using RK4.
    """
    # Define symbols for parameters
    L = ca.MX.sym('L')
    tau_max = ca.MX.sym('tau_max')
    r = ca.MX.sym('r')
    m = ca.MX.sym('m')
    w_max = ca.MX.sym('w_max')
    
    # Parameter vector
    params = ca.vertcat(L, tau_max, r, m, w_max)
    
    # Total integration time (unknown, will be optimization variable)
    T = ca.MX.sym('T')
    
    theta_final, theta_d_final = rk4_integrate(T, params, N, theta_init)
    
    return {
        'theta_f': theta_final,
        'theta_d_f': theta_d_final,
        'T': T,
        'L': L, 'tau_max': tau_max, 'r': r, 'm': m, 'w_max': w_max,
        'params': params,
        'param_names': ['L', 'tau_max', 'r', 'm', 'w_max']
    }

def setup_nlp(theta_max=100.0, N=50, tau_max_val=0.2795, m_val=0.4, w_max_val=1704.0):
    """
    Sets up a Nonlinear Programming (NLP) problem to maximize final angular velocity
    subject to a final angle constraint using standard nlpsol interface.
    tau_max, m, and w_max are treated as constant parameters.
    """
    syms = get_symbolic_expression(N=N)
    
    theta_f = syms['theta_f']
    theta_d_f = syms['theta_d_f']
    
    # Decision variables: [T, L, r]
    w = ca.vertcat(syms['T'], syms['L'], syms['r'])
    
    # Parameters: [tau_max, m, w_max]
    p = ca.vertcat(syms['tau_max'], syms['m'], syms['w_max'])
    
    # Objective: Maximize final velocity (Minimize negative final velocity)
    J = -theta_d_f
    
    # Constraints: theta_f = theta_max
    g = ca.vertcat(theta_f - theta_max)
    
    nlp = {'x': w, 'p': p, 'f': J, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    return solver, syms, [tau_max_val, m_val, w_max_val]

def setup_opti(theta_max=100.0, N=50, tau_max_val=0.2795, m_val=0.4, w_max_val=1704.0):
    """
    Sets up the optimization problem using CasADi Opti stack.
    tau_max, m, and w_max are treated as constant parameters.
    """
    opti = ca.Opti()
    
    # Variables
    T = opti.variable()
    L = opti.variable()
    r = opti.variable()
    
    # Parameters
    tau_max = opti.parameter()
    m = opti.parameter()
    w_max = opti.parameter()
    
    # Set parameter values
    opti.set_value(tau_max, tau_max_val)
    opti.set_value(m, m_val)
    opti.set_value(w_max, w_max_val)
    
    params = ca.vertcat(L, tau_max, r, m, w_max)
    
    # Dynamics constraints
    theta_f, theta_d_f = rk4_integrate(T, params, N=N)
    
    # Objective
    opti.minimize(-theta_d_f)
    
    # Constraints
    opti.subject_to(theta_f == theta_max)
    
    # Bounds
    opti.subject_to(T >= 0.01)
    opti.subject_to(T <= 2.0)
    opti.subject_to(L >= 0.01)
    opti.subject_to(L <= 1.0)
    opti.subject_to(r >= 0.001)
    opti.subject_to(r <= 0.05)
    
    # Initial guess
    opti.set_initial(T, 0.5)
    opti.set_initial(L, 0.050)
    opti.set_initial(r, 0.002)
    
    # Solver options
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', opts)
    
    return opti, {'T': T, 'L': L, 'tau_max': tau_max, 'r': r, 'm': m, 'w_max': w_max}

if __name__ == "__main__":
    print("Testing setup_opti with fixed parameters...")
    opti, vars_dict = setup_opti(theta_max=100.0)
    
    try:
        sol = opti.solve()
        print("\nSolution found with Opti:")
        print(f"Final velocity: {-sol.value(opti.f):.4f} rad/s")
        print(f"T: {sol.value(vars_dict['T']):.4f}")
        for name in ['L', 'tau_max', 'r', 'm', 'w_max']:
            print(f"{name}: {sol.value(vars_dict[name]):.4f}")
            
    except Exception as e:
        print(f"Solver failed: {e}")
        
    print("\nTesting setup_nlp with fixed parameters...")
    solver, syms, p_vals = setup_nlp(theta_max=100.0)
    
    # Initial guess for [T, L, r]
    w0 = [0.5, 0.050, 0.002]
    
    # Bounds for [T, L, r]
    lbw = [0.01, 0.01, 0.001]
    ubw = [2.0, 1.0, 0.05]
    lbg = [0.0]
    ubg = [0.0]
    
    sol = solver(x0=w0, p=p_vals, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    
    w_opt = sol['x'].full().flatten()
    print("\nSolution found with NLP:")
    print(f"Final velocity: {-sol['f'].full().item():.4f} rad/s")
    print(f"T: {w_opt[0]:.4f}")
    print(f"L: {w_opt[1]:.4f}")
    print(f"r: {w_opt[2]:.4f}")
