import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- load + concat ---
dfs = []
for k, fn in enumerate(sorted(glob.glob("motor_logs/*free*.csv"))):
    d = pd.read_csv(fn)
    d["trial"] = k
    # extract voltage from filename
    d["V"] = d["set_volts"].iloc[0]
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

df["t"] = df["time_ms"] * 1e-3
df = df.sort_values(["trial", "t"]).reset_index(drop=True)

'''
#V is set voltage
tau_stall = 0.09555555556 * V
omega_free = 2136 * 2*np.pi/60 * V
tau_m = tau_stall * (1.0 - w / omega_free)

#motor model: tau_m = I*wdot + b*w + tau0
'''

# --- filter and differentiate per trial ---
# savgol params: window_length, polyorder
window_length = 7  # try smaller window for sharper transient
polyorder = 2

filtered_trials = []
for trial_id in df["trial"].unique():
    trial_df = df[df["trial"] == trial_id].copy()
    trial_df = trial_df.sort_values("t").reset_index(drop=True)

    # filter velocity
    vel = trial_df["vel"].values
    t = trial_df["t"].values

    # ensure window_length is odd and less than data length
    wl = min(window_length, len(vel) - 1)
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        wl = 3

    vel_filt = savgol_filter(vel, wl, polyorder)
    trial_df["vel_filt"] = vel_filt

    # differentiate using gradient
    wdot = np.gradient(vel_filt, t)
    trial_df["wdot"] = wdot

    filtered_trials.append(trial_df)

df = pd.concat(filtered_trials, ignore_index=True)

# --- plot vel vs filtered vel for a few trials ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    ax = axes[i]
    ax.plot(trial_df["t"], trial_df["vel"], 'b-', alpha=0.5, label='raw')
    ax.plot(trial_df["t"], trial_df["vel_filt"], 'r-', label='filtered')
    ax.set_title(f"Trial {trial_id}, V={trial_df['V'].iloc[0]:.1f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sysid_vel_filter.png", dpi=150)
plt.close()
print("Saved sysid_vel_filter.png")

# --- plot wdot to check if reasonable ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    ax = axes[i]
    ax.plot(trial_df["t"], trial_df["wdot"], 'g-')
    ax.set_title(f"Trial {trial_id}, V={trial_df['V'].iloc[0]:.1f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("wdot (rad/s^2)")

plt.tight_layout()
plt.savefig("sysid_wdot.png", dpi=150)
plt.close()
print("Saved sysid_wdot.png")

# --- use first 60% of each trial for transients ---
transient_dfs = []
for trial_id in df["trial"].unique():
    trial_df = df[df["trial"] == trial_id].copy()
    n = len(trial_df)
    n_use = int(0.6 * n)
    transient_dfs.append(trial_df.iloc[:n_use])

df_transient = pd.concat(transient_dfs, ignore_index=True)

# exclude low voltage trials (too noisy)
df_transient = df_transient[df_transient["V"] >= 1.0].reset_index(drop=True)
print(f"Using {len(df_transient)} data points from trials with V >= 1.0")

# --- compute motor torque from model ---
# tau_stall = 0.09555555556 * V  (original)
# tau_stall = 0.0871 * V  (adjusted)
# omega_free = 2136 * 2*pi/60 * V = 223.7 * V (rad/s)
# tau_m = tau_stall * (1 - w / omega_free)

V = df_transient["V"].values
w = df_transient["vel_filt"].values
wdot = df_transient["wdot"].values

tau_stall = 0.0871 * V  # adjusted value
omega_free = 2136 * 2 * np.pi / 60 * V
tau_m = tau_stall * (1.0 - w / omega_free)

# --- least squares: tau_m = I*wdot + b*w + tau0 ---
# Ax = y where x = [I, b, tau0]^T
# A = [wdot, w, 1], y = tau_m

A = np.column_stack([wdot, w, np.ones_like(wdot)])
y = tau_m

# solve least squares (unconstrained)
x, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

I_est, b_est, tau0_est = x

print("\n=== System Identification Results (unconstrained) ===")
print(f"Inertia I     = {I_est:.6e} kg*m^2")
print(f"Viscous drag b = {b_est:.6e} N*m*s/rad")
print(f"Friction tau0 = {tau0_est:.6e} N*m")

# Also try with b=0 (no viscous drag)
A2 = np.column_stack([wdot, np.ones_like(wdot)])
x2, _, _, _ = np.linalg.lstsq(A2, y, rcond=None)
I_est2, tau0_est2 = x2

print("\n=== With b=0 (no viscous drag) ===")
print(f"Inertia I     = {I_est2:.6e} kg*m^2")
print(f"Friction tau0 = {tau0_est2:.6e} N*m")

y_pred2 = A2 @ x2
ss_res2 = np.sum((y - y_pred2)**2)
r2_b0 = 1 - ss_res2 / np.sum((y - np.mean(y))**2)
print(f"R^2 (b=0)     = {r2_b0:.4f}")

# Constrained optimization: I > 0, b >= 0, tau0 >= 0
def objective(params):
    I, b, tau0 = params
    pred = I * wdot + b * w + tau0
    return np.sum((y - pred)**2)

res = minimize(objective, x0=[1e-5, 1e-5, 1e-3],
               bounds=[(1e-10, None), (0, None), (0, None)],
               method='L-BFGS-B')
I_con, b_con, tau0_con = res.x

print("\n=== Constrained (b>=0, tau0>=0) ===")
print(f"Inertia I     = {I_con:.6e} kg*m^2")
print(f"Viscous drag b = {b_con:.6e} N*m*s/rad")
print(f"Friction tau0 = {tau0_con:.6e} N*m")

y_pred_con = I_con * wdot + b_con * w + tau0_con
ss_res_con = np.sum((y - y_pred_con)**2)
r2_con = 1 - ss_res_con / np.sum((y - np.mean(y))**2)
print(f"R^2           = {r2_con:.4f}")

# Alternative: fit linear motor model directly
# Motor: tau = Kt*V - Kv*w  (Kt = torque const, Kv = back-emf const)
# Load:  tau = I*wdot + b*w + tau0
# Combined: Kt*V = I*wdot + (b + Kv)*w + tau0
# Let d = b + Kv (total damping: mechanical + electrical)
# Fit: Kt*V = I*wdot + d*w + tau0
# Rearrange: wdot = (Kt/I)*V - (d/I)*w - tau0/I

# Linear fit: Kt*V - I*wdot - d*w - tau0 = 0
# A @ [Kt, I, d, tau0] = 0  doesn't work directly...
# Instead: I*wdot = Kt*V - d*w - tau0
# So y = wdot, and we fit Kt*V - d*w - tau0 = I*wdot
# Divide by I: wdot = (Kt/I)*V - (d/I)*w - (tau0/I)
# Let a = Kt/I, c = d/I, f = tau0/I
# wdot = a*V - c*w - f

A_motor = np.column_stack([V, -w, -np.ones_like(w)])
y_motor = wdot

x_motor, _, _, _ = np.linalg.lstsq(A_motor, y_motor, rcond=None)
a_fit, c_fit, f_fit = x_motor

print("\n=== Linear motor model: wdot = a*V - c*w - f ===")
print(f"a (Kt/I)       = {a_fit:.2f} rad/s^2/V")
print(f"c (d/I)        = {c_fit:.2f} 1/s")
print(f"f (tau0/I)     = {f_fit:.2f} rad/s^2")
print(f"Time constant  = {1/c_fit:.4f} s")

# Using constrained I estimate to back out other params
I_best = I_con
Kt_est = a_fit * I_best
d_est = c_fit * I_best
tau0_lin = f_fit * I_best

print(f"\nUsing I = {I_best:.6e} kg*m^2 (from constrained fit):")
print(f"Kt (torque constant)    = {Kt_est:.6e} N*m/V")
print(f"d (total damping)       = {d_est:.6e} N*m*s/rad")
print(f"tau0 (friction)         = {tau0_lin:.6e} N*m")

# R^2 for linear model
wdot_pred_lin = a_fit * V - c_fit * w - f_fit
ss_res_lin = np.sum((wdot - wdot_pred_lin)**2)
ss_tot_lin = np.sum((wdot - np.mean(wdot))**2)
r2_lin = 1 - ss_res_lin / ss_tot_lin
print(f"R^2           = {r2_lin:.4f}")

# Also try with tau0 = 0 constraint
A_motor2 = np.column_stack([V, -w])
x_motor2, _, _, _ = np.linalg.lstsq(A_motor2, wdot, rcond=None)
a_fit2, c_fit2 = x_motor2

print("\n=== Linear motor model (tau0=0): wdot = a*V - c*w ===")
print(f"a (Kt/I)       = {a_fit2:.2f} rad/s^2/V")
print(f"c (d/I)        = {c_fit2:.2f} 1/s")
print(f"Time constant  = {1/c_fit2:.4f} s")
Kt_est2 = a_fit2 * I_best
d_est2 = c_fit2 * I_best
print(f"Kt = {Kt_est2:.6e} N*m/V")
print(f"d  = {d_est2:.6e} N*m*s/rad")

wdot_pred2 = a_fit2 * V - c_fit2 * w
r2_lin2 = 1 - np.sum((wdot - wdot_pred2)**2) / ss_tot_lin
print(f"R^2           = {r2_lin2:.4f}")

# Try without d (no damping): wdot = a*V
A_motor3 = V.reshape(-1, 1)
x_motor3, _, _, _ = np.linalg.lstsq(A_motor3, wdot, rcond=None)
a_fit3 = x_motor3[0]

print("\n=== Linear motor model (d=0, tau0=0): wdot = a*V ===")
print(f"a (Kt/I)       = {a_fit3:.2f} rad/s^2/V")
Kt_est3 = a_fit3 * I_best
print(f"Kt = {Kt_est3:.6e} N*m/V")

wdot_pred3 = a_fit3 * V
r2_lin3 = 1 - np.sum((wdot - wdot_pred3)**2) / ss_tot_lin
print(f"R^2           = {r2_lin3:.4f}")

# --- Weighted fit: emphasize early points (low w) ---
# Weight inversely proportional to velocity (more weight on startup)
w_max = w.max()
weights = 1.0 + 2.0 * (1.0 - w / w_max)  # weight 3x at w=0, 1x at w=w_max

# Weighted least squares: wdot = a*V - c*w
W = np.diag(weights)
A_weighted = np.column_stack([V, -w])
x_weighted = np.linalg.lstsq(W @ A_weighted, W @ wdot, rcond=None)[0]
a_w, c_w = x_weighted

print("\n=== Weighted fit (emphasize startup): wdot = a*V - c*w ===")
print(f"a (Kt/I)       = {a_w:.2f} rad/s^2/V")
print(f"c (d/I)        = {c_w:.2f} 1/s")
print(f"Time constant  = {1/c_w:.4f} s")
Kt_w = a_w * I_best
d_w = c_w * I_best
print(f"Kt = {Kt_w:.6e} N*m/V")
print(f"d  = {d_w:.6e} N*m*s/rad")

wdot_pred_w = a_w * V - c_w * w
r2_w = 1 - np.sum((wdot - wdot_pred_w)**2) / ss_tot_lin
print(f"R^2           = {r2_w:.4f}")

# --- Nonlinear model with startup boost: wdot = a*V - c*w + boost*exp(-w/w_scale) ---
def objective_boost(params):
    a, c, boost, w_scale = params
    pred = a * V - c * w + boost * V * np.exp(-w / w_scale)
    return np.sum((wdot - pred)**2)

res_boost = minimize(objective_boost, x0=[a_fit2, c_fit2, 1000, 100],
                     bounds=[(1000, 30000), (10, 200), (0, 10000), (10, 500)],
                     method='L-BFGS-B')
a_b, c_b, boost_b, w_scale_b = res_boost.x

print("\n=== Nonlinear with startup boost: wdot = a*V - c*w + boost*V*exp(-w/w_scale) ===")
print(f"a (Kt/I)       = {a_b:.2f} rad/s^2/V")
print(f"c (d/I)        = {c_b:.2f} 1/s")
print(f"boost          = {boost_b:.2f} rad/s^2/V")
print(f"w_scale        = {w_scale_b:.2f} rad/s")
print(f"Time constant  = {1/c_b:.4f} s")

wdot_pred_b = a_b * V - c_b * w + boost_b * V * np.exp(-w / w_scale_b)
r2_b = 1 - np.sum((wdot - wdot_pred_b)**2) / ss_tot_lin
print(f"R^2           = {r2_b:.4f}")

# --- Direct trajectory fit: minimize simulation error ---
def simulate_trial(params, t, V_trial, w0):
    a, c = params
    w_sim = np.zeros_like(t)
    w_sim[0] = w0
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        wdot_sim = a * V_trial - c * w_sim[j-1]
        w_sim[j] = w_sim[j-1] + wdot_sim * dt
    return w_sim

def objective_traj(params):
    total_error = 0.0
    for trial_id in df_transient["trial"].unique():
        trial_df = df_transient[df_transient["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V_trial = trial_df["V"].iloc[0]
        w_sim = simulate_trial(params, t, V_trial, w_actual[0])
        total_error += np.sum((w_actual - w_sim)**2)
    return total_error

res_traj = minimize(objective_traj, x0=[a_fit2, c_fit2],
                    bounds=[(5000, 30000), (30, 150)],
                    method='L-BFGS-B')
a_traj, c_traj = res_traj.x

print("\n=== Direct trajectory fit: minimize simulation error ===")
print(f"a (Kt/I)       = {a_traj:.2f} rad/s^2/V")
print(f"c (d/I)        = {c_traj:.2f} 1/s")
print(f"Time constant  = {1/c_traj:.4f} s")
Kt_traj = a_traj * I_best
d_traj = c_traj * I_best
print(f"Kt = {Kt_traj:.6e} N*m/V")
print(f"d  = {d_traj:.6e} N*m*s/rad")

# --- Trajectory fit on rising portion only (up to 95% of max) ---
def objective_traj_rising(params):
    total_error = 0.0
    for trial_id in df_transient["trial"].unique():
        trial_df = df_transient[df_transient["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V_trial = trial_df["V"].iloc[0]

        # find index where w reaches 95% of max
        w_max = w_actual.max()
        idx_95 = np.searchsorted(w_actual, 0.95 * w_max)
        if idx_95 < 3:
            idx_95 = len(w_actual)  # use all if too short

        t_rise = t[:idx_95]
        w_rise = w_actual[:idx_95]

        w_sim = simulate_trial(params, t_rise, V_trial, w_rise[0])
        total_error += np.sum((w_rise - w_sim)**2)
    return total_error

res_traj_rise = minimize(objective_traj_rising, x0=[a_traj, c_traj],
                         bounds=[(5000, 40000), (30, 200)],
                         method='L-BFGS-B')
a_traj_r, c_traj_r = res_traj_rise.x

print("\n=== Trajectory fit (rising only, <95% max): ===")
print(f"a (Kt/I)       = {a_traj_r:.2f} rad/s^2/V")
print(f"c (d/I)        = {c_traj_r:.2f} 1/s")
print(f"Time constant  = {1/c_traj_r:.4f} s")
Kt_traj_r = a_traj_r * I_best
d_traj_r = c_traj_r * I_best
print(f"Kt = {Kt_traj_r:.6e} N*m/V")
print(f"d  = {d_traj_r:.6e} N*m*s/rad")

# compute R^2
y_pred = A @ x
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot
print(f"R^2           = {r2:.4f}")

# --- plot predicted vs actual wdot for each trial (linear model) ---
df_transient["wdot_pred"] = wdot_pred2

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df_transient["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df_transient[df_transient["trial"] == trial_id]
    ax = axes[i]
    ax.plot(trial_df["t"], trial_df["wdot"], 'b-', alpha=0.7, label='actual wdot')
    ax.plot(trial_df["t"], trial_df["wdot_pred"], 'r--', label='predicted')
    ax.set_title(f"Trial {trial_id}, V={trial_df['V'].iloc[0]:.1f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("wdot (rad/s^2)")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("sysid_wdot_fit.png", dpi=150)
plt.close()
print("Saved sysid_wdot_fit.png")

# --- also plot vel predicted by integrating the model ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df_transient["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df_transient[df_transient["trial"] == trial_id].copy()
    ax = axes[i]

    # simulate models
    t_sim = trial_df["t"].values
    V_trial = trial_df["V"].iloc[0]
    w_sim = np.zeros_like(t_sim)  # linear model
    w_sim_traj = np.zeros_like(t_sim)  # trajectory fit
    w_sim_rise = np.zeros_like(t_sim)  # rising-only fit
    w_sim[0] = w_sim_traj[0] = w_sim_rise[0] = trial_df["vel_filt"].iloc[0]

    for j in range(1, len(t_sim)):
        dt = t_sim[j] - t_sim[j-1]
        # linear model
        wdot_sim = a_fit2 * V_trial - c_fit2 * w_sim[j-1]
        w_sim[j] = w_sim[j-1] + wdot_sim * dt
        # trajectory fit
        wdot_traj = a_traj * V_trial - c_traj * w_sim_traj[j-1]
        w_sim_traj[j] = w_sim_traj[j-1] + wdot_traj * dt
        # rising-only fit
        wdot_rise = a_traj_r * V_trial - c_traj_r * w_sim_rise[j-1]
        w_sim_rise[j] = w_sim_rise[j-1] + wdot_rise * dt

    ax.plot(trial_df["t"], trial_df["vel_filt"], 'b-', label='actual', linewidth=2)
    ax.plot(t_sim, w_sim, 'r--', label='wdot fit', alpha=0.7)
    ax.plot(t_sim, w_sim_traj, 'g-', label='traj fit')
    ax.plot(t_sim, w_sim_rise, 'm:', label='rising fit', linewidth=2)
    ax.set_title(f"Trial {trial_id}, V={V_trial:.1f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("sysid_vel_sim.png", dpi=150)
plt.close()
print("Saved sysid_vel_sim.png")
