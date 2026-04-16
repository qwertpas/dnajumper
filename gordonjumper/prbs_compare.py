import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------
# Load one PRBS log
# -----------------------
fn = "motor_logs/prbs3.csv"   # <-- change
df = pd.read_csv(fn)

df = df[df["time_ms"] < 9000.0]
# time vector (seconds)
if "time_us" in df.columns:
    t = df["time_us"].to_numpy(dtype=float) * 1e-6
else:
    t = df["time_ms"].to_numpy(dtype=float) * 1e-3

w_meas = df["vel"].to_numpy(dtype=float)          # rad/s
V_cmd  = df["set_volts"].to_numpy(dtype=float)    # volts (you said true terminal volts)

# -----------------------
# Parameters (fill with yours)
# -----------------------
# From your latest fit output screenshot:
# I_hat    = 4.711649e-06
# b_hat    = 1.5e-21          # basically 0
# tau_c_hat= 6.382151e-04
# tau0_hat = 5.589221e-05

# Ke = 4.472738e-03           # V*s/rad
# Kt = 4.472738e-03           # N*m/A
# R  = 5.134004e-02           # ohm
# L  = 7.334291e-05           # H


I_hat  = 4.599210e-06  # N*m*s^2/rad
b_hat  = 3.688781e-10  # N*m*s/rad
tau_c_hat= 1.024093e-03  # N*m
tau0_hat = 1.118110e-05  # N*m

# Electrical params from Kv + tau_stall_per_V + k_elec:
Ke = 4.472738e-03 # V*s/rad   (Kv=2135.0 rpm/V)
Kt = 4.472738e-03 # N*m/A
R  = 5.134004e-02 # ohm
L  = 1.734461e-04 # H


# for I_hat in np.linspace(3.8e-06, 4.8e-06, 20):
for I_hat in [4.58421052631579e-06]:
    print(f"I_hat: {I_hat}")

    # For the simplified torque-speed model:
    Kv_rpm_per_V = 2135.0
    wfree_per_V = Kv_rpm_per_V * 2*np.pi/60.0     # rad/s per volt

    # IMPORTANT: choose the stall-torque-per-volt you want to compare
    # tau_stall_per_V = 0.0871   # N*m/V  
    tau_stall_per_V = 0.0955   # N*m/V 

    # Coulomb smoothing for full model
    w_eps = 5.0  # rad/s (increase if you see chatter around zero)

    # Interpolated input voltage
    def V_of_t(tt):
        return np.interp(tt, t, V_cmd)

    # -----------------------
    # Full model: states [i, w]
    # -----------------------
    def ode_full(tt, x):
        i, w = x
        V = V_of_t(tt)
        i_dot = (V - R*i - Ke*w) / L
        tau_fric = b_hat*w + tau_c_hat*np.tanh(w / w_eps) + tau0_hat
        w_dot = (Kt*i - tau_fric) / I_hat
        return [i_dot, w_dot]

    # initial condition: w0 from data, i0 from quasi-steady i ~ (V - Ke*w)/R
    w0 = float(w_meas[0])
    i0 = (float(V_cmd[0]) - Ke*w0) / R
    x0_full = [i0, w0]

    sol_full = solve_ivp(
        ode_full, (t[0], t[-1]), x0_full,
        t_eval=t, rtol=1e-7, atol=1e-9, max_step=0.002
    )
    w_full = sol_full.y[1]

    # -----------------------
    # Simplified model: no inductance, no friction
    # torque = tau_stall_per_V * V  - (tau_stall_per_V/wfree_per_V)*w
    # wdot = torque / I
    # -----------------------
    a = tau_stall_per_V / I_hat                       # (rad/s^2) per volt
    c = (tau_stall_per_V / wfree_per_V) / I_hat       # 1/s

    def ode_simple(tt, w):
        V = V_of_t(tt)
        return a*V - c*w

    sol_simple = solve_ivp(
        lambda tt, ww: ode_simple(tt, ww),
        (t[0], t[-1]), [w0],
        t_eval=t, rtol=1e-7, atol=1e-9, max_step=0.002
    )
    w_simple = sol_simple.y[0]

    # rms_res_full = np.sqrt(np.mean((w_full - w_meas)**2))
    # print(f"RMS error (full model):   {rms_res_full:.3f} rad/s")

    # Apply 2.4 ms time shift to w_simple for residuals calculation
    time_shift = 0.003
    from scipy.interpolate import interp1d
    interp_simple = interp1d(t + time_shift, w_simple, kind='linear', bounds_error=False, fill_value='extrapolate')
    w_simple_shifted = interp_simple(t)
    rms_res_simple = np.sqrt(np.mean((w_simple_shifted - w_meas)**2))
    print(f"RMS error (simple model, shifted {time_shift*1000:.1f} ms): {rms_res_simple:.3f} rad/s")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(t, w_meas, label="measured", linewidth=1.0, alpha=0.6)
# plt.plot(t, w_full, label="full model (L + friction)", linestyle="--", linewidth=2.0)
plt.plot(t+0.003, w_simple, label="simple (no L, no friction)", linestyle=":", linewidth=2.0)
plt.xlabel("Time (s)")
plt.ylabel("Speed ω (rad/s)")
plt.title(f"Measured vs Full vs Simple Motor Model {I_hat:.6e} N*m*s^2/rad")
plt.legend()
plt.tight_layout()
plt.show()

# # Residuals (optional)
# plt.figure(figsize=(10,3))
# plt.plot(t, w_full - w_meas, label="full - meas")
# plt.plot(t, w_simple - w_meas, label="simple - meas")


# plt.axhline(0, linewidth=1)
# plt.xlabel("Time (s)")
# plt.ylabel("Residual (rad/s)")
# plt.title("Residuals")
# plt.legend()
# plt.tight_layout()
# plt.show()
