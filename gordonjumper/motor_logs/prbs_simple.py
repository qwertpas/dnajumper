#!/usr/bin/env python3
"""
Simple motor sysid: no inductance, with friction.
Model: I * dw/dt = stall_torque_1V * V - (stall_torque_1V / free_speed_1V) * w - b*w - tau_c*tanh(w/w_eps)

Fit variables: stall_torque_1V, I, b (viscous), tau_c (Coulomb)
Fixed: free_speed_1V (from Kv)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Motor constant (from Kv)
Kv_rpm_per_V = 2135.0
FREE_SPEED_1V = Kv_rpm_per_V * 2 * np.pi / 60  # rad/s per volt
W_EPS = 5.0  # smoothing for tanh (rad/s)


def load_data(filenames):
    """Load multiple CSV files into list of trials."""
    trials = []
    for fn in filenames:
        df = pd.read_csv(fn)
        df = df[df["time_ms"] < 9000.0]
        t = df['time_us'].values * 1e-6
        w = df['vel'].values
        V = df['set_volts'].values
        trials.append({'t': t, 'w': w, 'V': V, 'file': fn})
    return trials


def motor_ode(t, w, t_grid, V_grid, stall_torque_1V, I, b_visc, tau_c):
    """
    dw/dt = (stall_torque_1V/I)*V - (stall_torque_1V/(free_speed_1V*I))*w - (b_visc/I)*w - (tau_c/I)*tanh(w/w_eps)
    """
    # Zero-order hold for voltage
    idx = np.searchsorted(t_grid, t, side='right') - 1
    idx = np.clip(idx, 0, len(V_grid) - 1)
    V = V_grid[idx]

    tau_motor = stall_torque_1V * V - (stall_torque_1V / FREE_SPEED_1V) * w
    tau_fric = b_visc * w + tau_c * np.tanh(w / W_EPS)
    dw_dt = (tau_motor - tau_fric) / I
    return dw_dt


def simulate(t, V, w0, stall_torque_1V, I, b_visc, tau_c):
    """Simulate the motor model with friction."""
    sol = solve_ivp(
        fun=lambda tt, ww: motor_ode(tt, ww, t, V, stall_torque_1V, I, b_visc, tau_c),
        t_span=(t[0], t[-1]),
        y0=[w0],
        t_eval=t,
        method='RK45',
        max_step=0.002,
    )
    return sol.y[0]


def fit_model(trials, downsample=1):
    """Fit stall_torque_1V, I, b_visc, tau_c to minimize velocity error."""

    def residuals(p):
        stall_torque_1V, I, b_visc, tau_c = p
        if stall_torque_1V <= 0 or I <= 0:
            return np.ones(10000) * 1e6

        res_all = []
        for trial in trials:
            t = trial['t'][::downsample]
            w = trial['w'][::downsample]
            V = trial['V'][::downsample]

            w_sim = simulate(t, V, w[0], stall_torque_1V, I, b_visc, tau_c)
            res_all.append(w_sim - w)

        return np.concatenate(res_all)

    # Initial guess
    # stall_torque_1V ~ Kt/R ~ 0.0043/0.045 ~ 0.095 Nm/V
    # I ~ 1e-5 kg*m^2
    # b_visc ~ 1e-6 Nm*s/rad
    # tau_c ~ 1e-4 Nm
    p0 = [0.095, 1e-5, 1e-6, 1e-4]
    lb = [0.01, 1e-7, 0.0, 0.0]
    ub = [0.5, 1e-3, 1e-3, 0.1]

    result = least_squares(
        residuals, p0,
        bounds=(lb, ub),
        loss='soft_l1',
        f_scale=20.0,
        max_nfev=200,
        verbose=2
    )
    return result


def main():
    files = ['prbs2.csv', 'prbs3.csv']
    trials = load_data(files)

    print(f"Loaded {len(trials)} trials")
    print(f"Free speed @ 1V = {FREE_SPEED_1V:.1f} rad/s (Kv={Kv_rpm_per_V} rpm/V)")
    print()

    result = fit_model(trials, downsample=1)
    stall_torque_1V, I, b_visc, tau_c = result.x

    print("\n" + "=" * 50)
    print("Fit Results (with friction, no inductance)")
    print("=" * 50)
    print(f"stall_torque_1V = {stall_torque_1V:.6f} Nm/V")
    print(f"I               = {I*1e6:.2f} g·mm² = {I:.6e} kg·m²")
    print(f"b_visc          = {b_visc:.6e} Nm·s/rad")
    print(f"tau_c           = {tau_c:.6e} Nm")
    print()
    print("Derived quantities:")
    tau_eff = I * FREE_SPEED_1V / stall_torque_1V
    print(f"  Time constant tau = {tau_eff * 1000:.2f} ms")
    print(f"  Free speed @ 1V = {FREE_SPEED_1V:.1f} rad/s")

    # Compute fit quality
    for trial in trials:
        t, w, V = trial['t'], trial['w'], trial['V']
        w_sim = simulate(t, V, w[0], stall_torque_1V, I, b_visc, tau_c)
        rmse = np.sqrt(np.mean((w_sim - w)**2))
        ss_res = np.sum((w_sim - w)**2)
        ss_tot = np.sum((w - np.mean(w))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"\n{trial['file']}: RMSE={rmse:.1f} rad/s, R²={r2:.4f}")

    # Plot
    _, axes = plt.subplots(len(trials), 1, figsize=(12, 4*len(trials)), sharex=True)
    if len(trials) == 1:
        axes = [axes]

    for ax, trial in zip(axes, trials):
        t, w, V = trial['t'], trial['w'], trial['V']
        w_sim = simulate(t, V, w[0], stall_torque_1V, I, b_visc, tau_c)

        ax.plot(t * 1000, w, 'b-', alpha=0.7, label='Measured')
        ax.plot(t * 1000, w_sim, 'r--', alpha=0.7, label='Simulated')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title(trial['file'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()
    plt.savefig('prbs_simple_fit.png', dpi=150)
    plt.close()
    print("\nPlot saved to: prbs_simple_fit.png")


if __name__ == "__main__":
    main()
