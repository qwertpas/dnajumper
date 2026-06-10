from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from springchain import Params, Result, get_motor_torque


def segment_geometry(theta, p):
    upper_theta = np.r_[theta[1:], 0.0]
    delta = upper_theta - theta
    z2 = p.l0 * p.l0 - 2 * p.R * p.R * (1 - np.cos(delta))
    if np.any(z2 <= 0):
        raise RuntimeError("fixed-length geometry became infeasible")

    z = np.sqrt(z2)
    dz = -p.R * p.R * np.sin(delta) / z
    ddz = -p.R * p.R * np.cos(delta) / z - p.R**4 * np.sin(delta) ** 2 / z**3
    return delta, z, dz, ddz


def kinematics(theta, theta_dot, p):
    n = p.n
    _, z, dz, ddz = segment_geometry(theta, p)
    h = np.zeros(n)
    h_dot = np.zeros(n)
    jac = np.zeros((n, n))
    bias = np.zeros(n)

    h_upper = n * p.l0
    h_dot_upper = 0.0
    jac_upper = np.zeros(n)
    bias_upper = 0.0

    for i in range(n - 1, -1, -1):
        upper = i + 1 if i + 1 < n else None
        upper_dot = theta_dot[upper] if upper is not None else 0.0
        delta_dot = upper_dot - theta_dot[i]

        h[i] = h_upper - z[i]
        h_dot[i] = h_dot_upper - dz[i] * delta_dot
        jac[i] = jac_upper
        if upper is not None:
            jac[i, upper] -= dz[i]
        jac[i, i] += dz[i]
        bias[i] = bias_upper - ddz[i] * delta_dot * delta_dot

        h_upper = h[i]
        h_dot_upper = h_dot[i]
        jac_upper = jac[i].copy()
        bias_upper = bias[i]

    return h, h_dot, jac, bias


def fixed_dynamics(t, y, p):
    theta = y[: p.n]
    theta_dot = y[p.n :]
    _, _, jac, bias = kinematics(theta, theta_dot, p)

    mass = np.full(p.n, p.m)
    inertia = np.full(p.n, p.I)
    mass[0] = p.M0
    inertia[0] = p.I0

    mat = np.diag(inertia) + jac.T @ (mass[:, None] * jac)
    rhs = -jac.T @ (mass * bias)
    rhs[0] += get_motor_torque(theta[0], theta_dot[0], t)
    theta_ddot = np.linalg.solve(mat, rhs)
    return np.r_[theta_dot, theta_ddot]


def simulate_fixed(p=Params()):
    p = replace(p, k=0.0, c=0.0)
    y0 = np.zeros(2 * p.n)

    end_condition = p.end_condition.lower()
    if end_condition == "spacer distance":
        end_condition = "motor height"
    if end_condition not in ("motor angle", "motor height"):
        raise ValueError("end_condition must be 'motor angle' or 'motor height'")

    def stop_at_target(t, y):
        theta = y[: p.n]
        theta_dot = y[p.n :]
        if end_condition == "motor height":
            h, _, _, _ = kinematics(theta, theta_dot, p)
            return h[0] - p.spacer_end_height
        return theta[0] - p.motor_stop_angle

    stop_at_target.terminal = True
    stop_at_target.direction = 1

    solve = solve_ivp(
        lambda t, y: fixed_dynamics(t, y, p),
        (0.0, p.max_time),
        y0,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        events=stop_at_target,
    )
    if not solve.success:
        raise RuntimeError(solve.message)
    if len(solve.t_events[0]) == 0:
        raise RuntimeError(f"fixed-length simulation did not reach {p.end_condition}")

    t_end = solve.t_events[0][0]
    t = np.linspace(0.0, t_end, p.samples)
    y = solve.sol(t)
    theta = y[: p.n]
    theta_dot = y[p.n :]
    h = np.zeros_like(theta)
    h_dot = np.zeros_like(theta)
    for i in range(t.size):
        h[:, i], h_dot[:, i], _, _ = kinematics(theta[:, i], theta_dot[:, i], p)

    return Result(p, t, h, theta, h_dot, theta_dot, solve)


def sample_fixed(result, t):
    y = result.solve.sol(t)
    theta = y[: result.params.n]
    theta_dot = y[result.params.n :]
    h = np.zeros_like(theta)
    h_dot = np.zeros_like(theta)
    for i in range(t.size):
        h[:, i], h_dot[:, i], _, _ = kinematics(theta[:, i], theta_dot[:, i], result.params)
    return h, theta, h_dot, theta_dot


def main():
    result = simulate_fixed()
    print(f"fixed-length reached target in {result.t[-1] * 1000:.2f} ms")
    print(f"motor angle: {result.theta[0, -1]:.2f} rad")
    print(f"bottom height: {result.h[0, -1] * 1000:.2f} mm")
    print(f"bottom velocity: {result.h_dot[0, -1]:.2f} m/s")


if __name__ == "__main__":
    main()
