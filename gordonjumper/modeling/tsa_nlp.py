import casadi as ca
import numpy as np
from tsa_casadi import get_symbolic_expression

def solve_nlp(theta_max=100.0):
    # Get symbolic expressions
    # N=50 is default in get_symbolic_expression
    syms = get_symbolic_expression(N=50)
    
    theta_f = syms['theta_f']
    theta_d_f = syms['theta_d_f']
    T = syms['T']
    params = syms['params']
    param_names = syms['param_names']
    
    # Decision variables: [T, params...]
    # params is [L, tau_max, r, m, w_max]
    w = ca.vertcat(T, params)
    
    # Objective: Maximize final velocity (Minimize negative final velocity)
    J = -theta_d_f
    
    # Constraints
    # 1. theta_f = theta_max
    g = ca.vertcat(theta_f - theta_max)
    
    # NLP dictionary
    nlp = {'x': w, 'f': J, 'g': g}
    
    # Solver options
    opts = {'ipopt.print_level': 5, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # Bounds and initial guess
    # w = [T, L, tau_max, r, m, w_max]
    
    # Initial guess (from previous notebooks)
    # T=0.5, L=0.05, tau_max=0.2795, r=0.002, m=0.4, w_max=1704
    w0 = [0.5, 0.050, 0.2795, 0.002, 0.4, 1704]
    
    # Lower bounds (must be positive generally)
    lbw = [0.01,  # T > 0
           0.01,  # L
           0.1,   # tau_max
           0.001, # r
           0.1,   # m
           100.0] # w_max
           
    # Upper bounds (arbitrary reasonable limits to keep solver happy, or inf)
    ubw = [2.0,   # T
           1.0,   # L
           10.0,  # tau_max
           0.05,  # r
           10.0,  # m
           10000.0] # w_max
           
    # Constraints bounds (equality constraint)
    lbg = [0.0]
    ubg = [0.0]
    
    # Solve
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    
    w_opt = sol['x'].full().flatten()
    theta_d_f_opt = -sol['f'].full().item()
    
    print("\nOptimization Results:")
    print(f"Final angular velocity: {theta_d_f_opt:.4f} rad/s")
    print("-" * 30)
    print(f"{'Variable':<10} {'Value':<10}")
    print("-" * 30)
    print(f"{'T':<10} {w_opt[0]:.4f}")
    for name, val in zip(param_names, w_opt[1:]):
        print(f"{name:<10} {val:.4f}")
        
    return w_opt

if __name__ == "__main__":
    solve_nlp()
