from typing import Dict, Any, List

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy.optimize import minimize_scalar

# Base motor parameters (mirrors plot_optimal_cases.py)
BASE_TAU_MAX = 0.09555555556 * 3.5  # Nm
BASE_W_MAX = 2135 * 2 * np.pi / 60 * 3.5  # rad/s
I_BASE = 4.5e-6  # kg·m²


def integrate_theta(get_y, get_tau, params, theta0=0, theta_dot0=0):
    m = params["m"]
    I = params["I"]
    stroke = params["stroke"]
    g = params["g"]

    def dynamics(t, state):
        theta, theta_dot = state
        y, dy_dtheta, d2y_dtheta2 = get_y(theta)
        tau = get_tau(theta, theta_dot, t)
        theta_ddot = (-theta_dot**2 * m * dy_dtheta * d2y_dtheta2 - m * g * dy_dtheta + tau) / (
            I + m * dy_dtheta**2
        )
        return [theta_dot, theta_ddot]

    def y_target_event(t, state):
        theta, theta_dot = state
        y, _, _ = get_y(theta)
        return y - stroke

    y_target_event.terminal = True
    y_target_event.direction = 0

    t_max = 2.0
    t_eval = np.arange(0, t_max, 0.00005)

    sol = solve_ivp(
        dynamics,
        t_span=(0, t_max),
        y0=[theta0, theta_dot0],
        method="BDF",
        t_eval=t_eval,
        events=y_target_event,
        dense_output=True,
        rtol=1e-9,
        atol=1e-12,
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

    return {"t": t, "theta": theta, "theta_d": theta_d, "y": y, "y_d": y_d, "tau": tau}


def integrate_theta_fine(get_y, get_tau, params, theta0=0, theta_dot0=0):
    m = params["m"]
    I = params["I"]
    stroke = params["stroke"]
    g = params["g"]

    def dynamics(t, state):
        theta, theta_dot = state
        y, dy_dtheta, d2y_dtheta2 = get_y(theta)
        tau = get_tau(theta, theta_dot, t)
        theta_ddot = (-theta_dot**2 * m * dy_dtheta * d2y_dtheta2 - m * g * dy_dtheta + tau) / (
            I + m * dy_dtheta**2
        )
        return [theta_dot, theta_ddot]

    def y_target_event(t, state):
        theta, theta_dot = state
        y, _, _ = get_y(theta)
        return y - stroke

    y_target_event.terminal = True
    y_target_event.direction = 0

    t_max = 2.0
    t_eval = np.arange(0, t_max, 0.00001)

    sol = solve_ivp(
        dynamics,
        t_span=(0, t_max),
        y0=[theta0, theta_dot0],
        method="BDF",
        t_eval=t_eval,
        events=y_target_event,
        dense_output=True,
        rtol=1e-11,
        atol=1e-14,
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

    return {"t": t, "theta": theta, "theta_d": theta_d, "y": y, "y_d": y_d, "tau": tau}


def get_optimal_pulley_radius(params):
    m = params["m"]
    I = params["I"]
    tau_max = params["tau_max"]
    w_max = params["w_max"]
    g = params["g"]
    s_target = -params["stroke"]

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
    res = minimize_scalar(lambda r: get_vf_pulley(r), bounds=(0.001, min(r_crit, 0.05)), method="bounded")
    return res.x, abs(res.fun)


def verify_case(mass: float, motor_scale: float, stroke: float, I_ratio: float, integrator) -> Dict[str, Any]:
    params = {
        "m": mass,
        "I": I_BASE * I_ratio,
        "stroke": stroke,
        "g": 9.81,
        "tau_max": BASE_TAU_MAX * motor_scale,
        "w_max": BASE_W_MAX * motor_scale,
    }
    r_opt, vf_expected = get_optimal_pulley_radius(params)

    def get_tau(theta, theta_dot, t):
        return params["tau_max"] * (1 - theta_dot / params["w_max"])

    def get_y_pulley(theta):
        y = r_opt * theta
        return y, r_opt, 0.0

    res = integrator(get_y_pulley, get_tau, params)
    vf_numeric = float(res["y_d"][-1])
    abs_err = abs(vf_numeric - vf_expected)
    rel_err = abs_err / max(1e-9, abs(vf_expected))

    return {
        "mass": mass,
        "motor_scale": motor_scale,
        "stroke": stroke,
        "I_ratio": I_ratio,
        "r_opt": r_opt,
        "vf_expected": vf_expected,
        "vf_numeric": vf_numeric,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }


def main() -> None:
    tests: List[Dict[str, Any]] = [
        {"mass": 0.10, "motor_scale": 1.0, "stroke": 0.05, "I_ratio": 1.0},
        {"mass": 0.20, "motor_scale": 1.0, "stroke": 0.05, "I_ratio": 1.0},
        {"mass": 0.10, "motor_scale": 0.5, "stroke": 0.05, "I_ratio": 1.0},
        {"mass": 0.10, "motor_scale": 2.0, "stroke": 0.05, "I_ratio": 1.0},
        {"mass": 0.10, "motor_scale": 1.0, "stroke": 0.08, "I_ratio": 1.0},
        {"mass": 0.10, "motor_scale": 1.0, "stroke": 0.05, "I_ratio": 0.5},
        {"mass": 0.10, "motor_scale": 1.0, "stroke": 0.05, "I_ratio": 2.0},
    ]

    print("LambertW pulley vs numerical integration (default)")
    res_default = [verify_case(integrator=integrate_theta, **t) for t in tests]
    for r in res_default:
        print(
            f"m={r['mass']*1000:4.0f}g, motor={r['motor_scale']:3.1f}x, "
            f"stroke={r['stroke']*1000:4.0f}mm, I={r['I_ratio']:3.2f}, "
            f"r={r['r_opt']*1000:5.2f}mm, vf_th={r['vf_expected']:6.3f}, "
            f"vf_num={r['vf_numeric']:6.3f}, abs={r['abs_err']:7.4f}, "
            f"rel={r['rel_err']*100:6.3f}%"
        )
    print(
        f"\nmax_abs_err = {max(r['abs_err'] for r in res_default):.6f} m/s, "
        f"max_rel_err = {max(r['rel_err'] for r in res_default)*100:.6f} %"
    )

    print("\nLambertW pulley vs numerical integration (fine)")
    res_fine = [verify_case(integrator=integrate_theta_fine, **t) for t in tests]
    for r in res_fine:
        print(
            f"m={r['mass']*1000:4.0f}g, motor={r['motor_scale']:3.1f}x, "
            f"stroke={r['stroke']*1000:4.0f}mm, I={r['I_ratio']:3.2f}, "
            f"r={r['r_opt']*1000:5.2f}mm, vf_th={r['vf_expected']:6.3f}, "
            f"vf_num={r['vf_numeric']:6.3f}, abs={r['abs_err']:7.4f}, "
            f"rel={r['rel_err']*100:6.3f}%"
        )
    print(
        f"\nmax_abs_err = {max(r['abs_err'] for r in res_fine):.6f} m/s, "
        f"max_rel_err = {max(r['rel_err'] for r in res_fine)*100:.6f} %"
    )


if __name__ == "__main__":
    main()

