#!/usr/bin/env python3
"""
PRBS System Identification for Motor Inertia
Model: I * dw/dt = K*V - c*w
Rearranged: dw/dt = a*V - b*w  where a=K/I, b=c/I
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def load_data(filename):
    df = pd.read_csv(filename)
    t = df['time_us'].values * 1e-6  # to seconds
    w = df['vel'].values  # rad/s
    V = df['set_volts'].values  # volts
    return t, w, V

def estimate_dw_dt(t, w):
    """Central difference for dw/dt"""
    dw_dt = np.zeros_like(w)
    dw_dt[1:-1] = (w[2:] - w[:-2]) / (t[2:] - t[:-2])
    dw_dt[0] = (w[1] - w[0]) / (t[1] - t[0])
    dw_dt[-1] = (w[-1] - w[-2]) / (t[-1] - t[-2])
    return dw_dt

def identify_params(t, w, V):
    """
    Least squares fit: dw/dt = a*V - b*w
    Returns a=K/I, b=c/I
    """
    dw_dt = estimate_dw_dt(t, w)

    # Skip first few samples (noisy derivatives at start)
    skip = 5
    dw_dt = dw_dt[skip:]
    V_fit = V[skip:]
    w_fit = w[skip:]

    # Build regressor matrix: [V, -w]
    A = np.column_stack([V_fit, -w_fit])

    # Least squares: dw/dt = A @ [a, b]
    params, residuals, rank, s = np.linalg.lstsq(A, dw_dt, rcond=None)
    a, b = params

    return a, b, dw_dt, V_fit, w_fit

def simulate_model(t, V, w0, a, b):
    """Simulate: dw/dt = a*V - b*w using Euler integration"""
    w_sim = np.zeros_like(t)
    w_sim[0] = w0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dw = a * V[i-1] - b * w_sim[i-1]
        w_sim[i] = w_sim[i-1] + dw * dt
    return w_sim

def analyze(filename):
    t, w, V = load_data(filename)

    # Split into train (first half) and test (second half)
    n = len(t)
    mid = n // 2

    t_train, w_train, V_train = t[:mid], w[:mid], V[:mid]
    t_test, w_test, V_test = t[mid:], w[mid:], V[mid:]

    # Identify on training data
    a, b, dw_dt_train, V_fit, w_fit = identify_params(t_train, w_train, V_train)

    print("=" * 50)
    print("PRBS System Identification Results")
    print("=" * 50)
    print(f"Model: I * dw/dt = K*V - c*w")
    print(f"Identified: dw/dt = {a:.4f}*V - {b:.4f}*w")
    print()
    print(f"  a = K/I = {a:.4f} rad/s^2/V")
    print(f"  b = c/I = {b:.4f} 1/s")
    print(f"  Time constant tau = 1/b = {1/b*1000:.2f} ms")
    print(f"  Steady-state gain = a/b = {a/b:.2f} (rad/s)/V")
    print()

    # If you know Kt (torque constant), you can get I:
    # Example: Kt = 0.01 Nm/A, R = 1 ohm => K = 0.01
    # Then I = K/a
    # Motor constants (Vertiq 2306)
    Kt = 0.0043  # Nm/A
    R = 0.045    # ohms
    K = Kt / R

    I_model = K / a
    c_model = b * I_model

    # Corrected estimate using observed steady-state
    # Find approx steady-state from data
    w_max = np.max(np.abs(w))
    V_max = np.max(np.abs(V))
    c_observed = (Kt / R) * V_max / (w_max * 0.95)  # 0.95 factor for not quite reaching SS
    I_observed = c_observed / b

    print(f"Motor: Kt={Kt} Nm/A, R={R} ohms, K=Kt/R={K:.4f} Nm/V")
    print()
    print("Inertia estimates:")
    print(f"  From model constants: I = {I_model*1e6:.2f} g·mm²")
    print(f"  From observed SS:     I = {I_observed*1e6:.2f} g·mm²")
    print()

    # Validate on test data
    t_test_rel = t_test - t_test[0]
    w_sim_test = simulate_model(t_test_rel, V_test, w_test[0], a, b)

    # Metrics
    residuals = w_test - w_sim_test
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((w_test - np.mean(w_test))**2)
    r2 = 1 - ss_res / ss_tot

    print("Validation (test set):")
    print(f"  RMSE: {rmse:.2f} rad/s")
    print(f"  R^2:  {r2:.4f}")
    print("=" * 50)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Full data with train/test split
    ax1 = axes[0]
    ax1.plot(t * 1000, w, 'b-', label='Measured velocity', alpha=0.7)
    ax1.axvline(t[mid] * 1000, color='r', linestyle='--', label='Train/Test split')
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend()
    ax1.set_title('PRBS Response - Full Dataset')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test set - measured vs simulated
    ax2 = axes[1]
    t_test_ms = (t_test - t_test[0]) * 1000
    ax2.plot(t_test_ms, w_test, 'b-', label='Measured', alpha=0.7)
    ax2.plot(t_test_ms, w_sim_test, 'r--', label='Simulated', alpha=0.7)
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.legend()
    ax2.set_title(f'Validation: Measured vs Simulated (R²={r2:.4f}, RMSE={rmse:.1f})')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Input voltage
    ax3 = axes[2]
    ax3.plot(t * 1000, V, 'g-', alpha=0.7)
    ax3.set_ylabel('Voltage (V)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Input Voltage (PRBS)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = filename.replace('.csv', '_sysid.png')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Plot saved to: {outfile}")

    return a, b

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "prbs2.csv"
    analyze(filename)
