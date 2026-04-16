import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# -----------------------------
# Put your fitted params here
# -----------------------------

'''
Fitted for interpolation, 17.545 rad/s RMSE:
I      = 4.689381e-06  N*m*s^2/rad
b      = 1.044140e-08  N*m*s/rad
tau_c  = 9.994011e-04  N*m
tau0   = -1.150287e-05  N*m

Fitted for zero-order hold datasheet tau_stall of 0.095556, 19.558 rad/s RMSE:
I      = 4.531964e-06  N*m*s^2/rad
b      = 1.115962e-06  N*m*s/rad
tau_c  = 2.148351e-04  N*m
tau0   = 6.394837e-04  N*m

Fitted for zero-order hold 1V pulley tau_stall of 0.08711988, 19.843 rad/s RMSE:
I      = 4.127277e-06  N*m*s^2/rad
b      = 4.779008e-09  N*m*s/rad
tau_c  = 1.014041e-03  N*m
tau0   = -2.659378e-05  N*m

Fitted for zoh, 2V and 3V datasheet tau_stall:
I      = 5.044257e-06  N*m*s^2/rad
b      = 8.077734e-10  N*m*s/rad
tau_c  = 1.168439e-03  N*m
tau0   = 1.404776e-04  N*m

Fitted for zoh, 2V and 3V, 1V tau_stall=0.087, k_elec=270:
I      = 4.599210e-06  N*m*s^2/rad
b      = 3.688781e-10  N*m*s/rad
tau_c  = 1.024093e-03  N*m
tau0   = 1.118110e-05  N*m

'''
I_hat    = 4.599210e-06
b_hat    = 3.688781e-10
tau_c_hat= 1.024093e-03
tau0_hat = 1.118110e-05

# -----------------------------
# Motor constants
# -----------------------------
@dataclass
class MotorParams:
    Kv_rpm_per_V: float = 2135.0
    # tau_stall_per_V: float = 0.09555555556 # Nm per V (datasheet)
    tau_stall_per_V: float = 0.0871198895 # Nm per V (fitted to 1V)
    # k_elec: float = 296.0                    # 1/s  (R/L) from claude's fit
    k_elec: float = 700                        # 1.43 ms electrical time constant
    w_eps: float = 5.0                       # rad/s smoothing for coulomb

    def Ke(self):
        Kv_rad_s_per_V = self.Kv_rpm_per_V * 2*np.pi/60.0
        return 1.0 / Kv_rad_s_per_V

    def Kt(self):
        return self.Ke()

    def R(self):
        # tau_stall_per_V = Kt / R
        return self.Kt() / self.tau_stall_per_V

    def L(self):
        return self.R() / self.k_elec

mp = MotorParams()

# -----------------------------
# Voltage mapping
# -----------------------------
def compute_V_in(g: pd.DataFrame):
    Vcmd = g["set_volts"].to_numpy(dtype=float)

    # If set_volts is truly motor terminal volts, use directly:
    V_in = Vcmd

    # If set_volts is a "normalized" command and you want to scale by battery, try this instead:
    # if "vbat" in g.columns:
    #     V_in = Vcmd * (g["vbat"].to_numpy(dtype=float) / 7.0)

    return V_in

# -----------------------------
# ODE + simulation
# -----------------------------
def motor_ode(t, x, t_grid, V_grid, p, mp: MotorParams):
    i, w = x
    I, b, tau_c, tau0 = p

    V = np.interp(t, t_grid, V_grid)

    # # Zero-order hold instead of linear interpolation
    # idx = np.searchsorted(t_grid, t, side='right') - 1
    # idx = np.clip(idx, 0, len(V_grid) - 1)
    # V = V_grid[idx]

    R = mp.R()
    L = mp.L()
    Ke = mp.Ke()
    Kt = mp.Kt()

    i_dot = (V - R*i - Ke*w) / L
    tau_fric = b*w + tau_c*np.tanh(w/mp.w_eps) + tau0
    w_dot = (Kt*i - tau_fric) / I
    return [i_dot, w_dot]

def simulate_trial(t, V_in, w_meas, p, mp: MotorParams):
    # init from data + quasi-steady current
    w0 = float(w_meas[0])
    R  = mp.R()
    Ke = mp.Ke()
    i0 = (float(V_in[0]) - Ke*w0) / R

    sol = solve_ivp(
        fun=lambda tt, xx: motor_ode(tt, xx, t, V_in, p, mp),
        t_span=(float(t[0]), float(t[-1])),
        y0=[i0, w0],
        t_eval=t,
        method="RK45",
        max_step=0.002,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    i_sim = sol.y[0]
    w_sim = sol.y[1]
    return i_sim, w_sim

# -----------------------------
# Load your logs
# -----------------------------
dfs = []
for k, fn in enumerate(sorted(glob.glob("motor_logs/*prbs*.csv"))):
    d = pd.read_csv(fn)
    d = d[d["time_ms"] < 9000.0]
    d["trial"] = k
    d["file"] = fn
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)

if "time_us" in df.columns:
    df["t"] = df["time_us"].astype(float) * 1e-6
else:
    df["t"] = df["time_ms"].astype(float) * 1e-3

df = df.sort_values(["trial", "t"]).reset_index(drop=True)

# Optional: downsample to speed up plotting/sim
DOWNSAMPLE = 1

p = np.array([I_hat, b_hat, tau_c_hat, tau0_hat], dtype=float)

# -----------------------------
# Sim + collect residuals
# -----------------------------
trial_stats = []
all_res = []

plt.figure(figsize=(10, 5))
for trial, g in df.groupby("trial"):
    t = g["t"].to_numpy(dtype=float)[::DOWNSAMPLE]
    w = g["vel"].to_numpy(dtype=float)[::DOWNSAMPLE]
    V_in = compute_V_in(g)[::DOWNSAMPLE]

    _, w_sim = simulate_trial(t, V_in, w, p, mp)
    res = w_sim - w

    rmse = float(np.sqrt(np.mean(res**2)))
    mae  = float(np.mean(np.abs(res)))
    w_span = float(np.max(w) - np.min(w))

    trial_stats.append((trial, g["set_volts"].iloc[0], rmse, mae, w_span))
    all_res.append(res)

    # overlay speed
    plt.plot(t, w, alpha=0.5, linewidth=1.0, label=f"meas T{trial} ({g['set_volts'].iloc[0]:.1f}V)")
    plt.plot(t, w_sim, alpha=0.9, linewidth=1.5, linestyle="--", label=f"sim  T{trial}")

plt.xlabel("Time (s)")
plt.ylabel("Speed ω (rad/s)")
plt.title("Measured vs Simulated Speed (all trials)")
plt.legend(ncols=2, fontsize=8)
plt.tight_layout()
plt.show()

# Residuals plot (all trials, concatenated)
all_res = np.concatenate(all_res)

plt.figure(figsize=(10, 4))
plt.plot(all_res, linewidth=0.8)
plt.xlabel("Sample index (concatenated)")
plt.ylabel("Residual (ω_sim - ω_meas) [rad/s]")
plt.title(f"Residuals (all trials) — RMSE={np.sqrt(np.mean(all_res**2)):.3f} rad/s")
plt.tight_layout()
plt.show()

# Residual histogram
plt.figure(figsize=(6, 4))
plt.hist(all_res, bins=80)
plt.xlabel("Residual (rad/s)")
plt.ylabel("Count")
plt.title("Residual Histogram")
plt.tight_layout()
plt.show()

# Print per-trial summary
print("\nPer-trial error summary:")
print("trial  setV   RMSE(rad/s)   MAE(rad/s)    span(rad/s)")
for trial, setV, rmse, mae, span in trial_stats:
    print(f"{trial:5d}  {setV:4.1f}   {rmse:10.3f}   {mae:10.3f}   {span:10.3f}")

# Sanity print of motor constants
print("\nMotor constants used:")
print(f"Ke = {mp.Ke():.6e}  V*s/rad")
print(f"Kt = {mp.Kt():.6e}  N*m/A")
print(f"R  = {mp.R():.6e}  ohm")
print(f"L  = {mp.L():.6e}  H")
