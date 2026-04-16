import casadi as ca
import numpy as np


params = {
    'm': 0.100, # mass
    'I': 0.5 * (37.4/1000/2) * (28.4/1000/2)**2,  #kg m^2
    'stroke': 0.080, # y starts from zero and stops at stroke
    'g': 9.81, #m/s^2
    'tau_max': 0.2795, #Nm
    'w_max': 1704, #rad/s
}



def optimize_tsa(params, guess, N=40, y_offset=0.0):
    """
    Optimize TSA parameters for maximum takeoff velocity.
    y starts at y_offset and must reach at least stroke.
    """
    stroke = params['stroke']
    m = params['m']
    I = params['I']
    tau_max = params['tau_max']
    w_max = params['w_max']
    g = params['g']
    eps = 1e-6
    theta_margin = 0.99

    opti = ca.Opti()
    L = opti.variable()
    r = opti.variable()
    T = opti.variable()
    
    def safe_sqrt(x):
        return ca.sqrt(x + eps**2)

    # Initial state: y starts at y_offset
    # y = L - sqrt(L² - (r*theta)²)  =>  theta = sqrt(L² - (L-y)²) / r
    theta_offset = safe_sqrt(L**2 - (L - y_offset)**2) / r
    state = ca.vertcat(theta_offset, 0)

    def get_theta_ddot(theta, theta_dot, L, r):
        theta_max = theta_margin * L / r
        theta_safe = ca.fmin(ca.fmax(theta, 0), theta_max)
        
        denom_inner = L**2 - (r*theta_safe)**2 + eps**2
        dy_dtheta = (r**2 * theta_safe) / safe_sqrt(denom_inner)
        d2y_dtheta2 = (r**2 * L**2) / (denom_inner**(1.5))
        
        tau = tau_max * (1 - theta_dot / w_max)
        theta_ddot = (-theta_dot**2 * m * dy_dtheta * d2y_dtheta2 - m * g * dy_dtheta + tau) / (I + m * dy_dtheta**2 + eps)
        return theta_ddot

    def f(x, L, r):
        theta = x[0]
        theta_dot = x[1]
        return ca.vertcat(theta_dot, get_theta_ddot(theta, theta_dot, L, r))

    # Manual RK4 integration
    dt = T / N
    x = state
    for i in range(N):
        k1 = f(x, L, r)
        k2 = f(x + dt/2 * k1, L, r)
        k3 = f(x + dt/2 * k2, L, r)
        k4 = f(x + dt * k3, L, r)
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    theta_final = x[0]
    theta_dot_final = x[1]

    theta_max = theta_margin * L / r
    theta_final_safe = ca.fmin(theta_final, theta_max)
    
    denom_final = L**2 - (r*theta_final_safe)**2 + eps**2
    dy_dtheta_final = (r**2 * theta_final_safe) / safe_sqrt(denom_final)
    y_dot_final = dy_dtheta_final * theta_dot_final
    y_final = L - safe_sqrt(denom_final)

    opti.minimize(-y_dot_final)

    opti.subject_to(opti.bounded(stroke, L, 1.0))
    opti.subject_to(opti.bounded(0.001, r, 0.05))
    opti.subject_to(opti.bounded(0.01, T, 0.5))
    opti.subject_to(y_final >= stroke)  # Must reach at least stroke
    opti.subject_to(r * theta_final <= theta_margin * L)

    opti.set_initial(L, guess[0])
    opti.set_initial(r, guess[1])
    opti.set_initial(T, guess[2])

    opts = {
        'ipopt.print_level': 0, 
        'print_time': 0,
        'ipopt.max_iter': 2000,
        'ipopt.tol': 1e-6,
    }
    opti.solver('ipopt', opts)
    sol = opti.solve()

    return [sol.value(L), sol.value(r), sol.value(T), y_offset, sol.value(y_final), sol.value(y_dot_final)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    labels = ['m', 'L', 'r', 'T', 'y0', 'y_f', 'yd_f']
    print("{:<7} {:<7} {:<9} {:<7} {:<7} {:<7} {:<9}".format(*labels))
    
    masses = []
    vfs = []
    
    guess = [0.5, 0.007, 0.15]  # L, r, T
    current_guess = list(guess)
    y_offset = 0.0  # Starting y position (can be changed)
    
    # Iterate from heavy to light
    mass_values = np.linspace(0.50, 0.050, 10)
    
    for m_val in mass_values:
        params['m'] = m_val
        try:
            variables = optimize_tsa(params, current_guess, y_offset=y_offset)
            # variables = [L, r, T, y_offset, y_final, y_dot_final]
            vf = variables[5]
            masses.append(m_val)
            vfs.append(vf)
            current_guess = variables[:3]  # L, r, T
            print("{:<7.3f} {:<7.3f} {:<9.6f} {:<7.3f} {:<7.3f} {:<7.3f} {:<9.3f}".format(m_val, *variables))
        except Exception as e:
            print(f"Failed for m={m_val:.3f}")
    
    # Reverse to plot in ascending mass order
    masses = masses[::-1]
    vfs = vfs[::-1]
    
    plt.figure()
    plt.plot(masses, vfs, marker='o')
    plt.xlabel('Mass (kg)')
    plt.ylabel('vf (m/s)')
    plt.title('vf vs Mass')
    plt.grid(True)
    plt.savefig('modeling/optimal_comparison_theta.png', dpi=150)
    plt.show()
