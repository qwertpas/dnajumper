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
    d["V"] = d["set_volts"].iloc[0]
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

df["t"] = df["time_ms"] * 1e-3
df = df.sort_values(["trial", "t"]).reset_index(drop=True)

# --- filter velocity per trial ---
window_length = 7
polyorder = 2

filtered_trials = []
for trial_id in df["trial"].unique():
    trial_df = df[df["trial"] == trial_id].copy()
    trial_df = trial_df.sort_values("t").reset_index(drop=True)

    vel = trial_df["vel"].values
    t = trial_df["t"].values

    wl = min(window_length, len(vel) - 1)
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        wl = 3

    vel_filt = savgol_filter(vel, wl, polyorder)
    trial_df["vel_filt"] = vel_filt

    # Trim to rising portion only (up to 95% of max)
    w_max = vel_filt.max()
    idx_95 = np.searchsorted(vel_filt, 0.95 * w_max)
    if idx_95 < 5:
        idx_95 = len(vel_filt)

    trial_df = trial_df.iloc[:idx_95].copy()

    # Reset time to start at 0
    trial_df["t"] = trial_df["t"] - trial_df["t"].iloc[0]

    filtered_trials.append(trial_df)

df = pd.concat(filtered_trials, ignore_index=True)

# Exclude low voltage trials (too noisy)
df = df[df["V"] >= 1.0].reset_index(drop=True)

print(f"Using {len(df)} data points (rising portion only, V >= 1.0)")

# --- Plot trimmed data ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    ax = axes[i]
    ax.plot(trial_df["t"] * 1000, trial_df["vel"], 'b.', alpha=0.5, markersize=3, label='raw')
    ax.plot(trial_df["t"] * 1000, trial_df["vel_filt"], 'r-', linewidth=2, label='filtered')
    ax.set_title(f"Trial {trial_id}, V={trial_df['V'].iloc[0]:.1f}")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("sysid2_data.png", dpi=150)
plt.close()
print("Saved sysid2_data.png")

# --- Simulation helper ---
def simulate(params, t, V, w0, model='linear'):
    """Simulate motor response"""
    w = np.zeros_like(t)
    w[0] = w0

    if model == 'linear':
        # wdot = a*V - c*w
        a, c = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * w[j-1]
            w[j] = w[j-1] + wdot * dt

    elif model == 'boost':
        # wdot = a*V - c*w + boost*V*exp(-w/w_scale)
        a, c, boost, w_scale = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * w[j-1] + boost * V * np.exp(-w[j-1] / w_scale)
            w[j] = w[j-1] + wdot * dt

    elif model == 'quadratic':
        # wdot = a*V - c*w - q*w^2/V  (quadratic drag)
        a, c, q = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * w[j-1] - q * w[j-1]**2 / V
            w[j] = w[j-1] + wdot * dt

    elif model == 'sqrt':
        # wdot = a*V - c*sqrt(w)*sign(w)  (sqrt damping)
        a, c = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * np.sqrt(abs(w[j-1])) * np.sign(w[j-1])
            w[j] = w[j-1] + wdot * dt

    elif model == 'two_phase':
        # Two time constants: fast initial, slower later
        a, c1, c2, w_trans = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            # Blend between c1 (at w=0) and c2 (at high w)
            blend = 1.0 / (1.0 + np.exp(-(w[j-1] - w_trans) / (w_trans * 0.2 + 1)))
            c_eff = c1 * (1 - blend) + c2 * blend
            wdot = a * V - c_eff * w[j-1]
            w[j] = w[j-1] + wdot * dt

    elif model == 'quad_linear':
        # Quadratic + linear damping
        a, c, q = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * w[j-1] - q * w[j-1]**2 / V
            w[j] = w[j-1] + wdot * dt

    elif model == 'power':
        # Power-law damping: wdot = a*V - c*w^p
        a, c, p = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * (abs(w[j-1])**p) * np.sign(w[j-1])
            w[j] = w[j-1] + wdot * dt

    elif model == 'coulomb':
        # Coulomb + viscous: wdot = a*V - c*w - f0*sign(w)
        a, c, f0 = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wdot = a * V - c * w[j-1] - f0 * np.sign(w[j-1] + 0.001)
            w[j] = w[j-1] + wdot * dt

    elif model == 'stribeck':
        # Stribeck friction: higher friction at low speed
        a, c, f0, w_strib = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            # Friction decreases as speed increases
            friction = f0 * np.exp(-abs(w[j-1]) / w_strib)
            wdot = a * V - c * w[j-1] - friction * np.sign(w[j-1] + 0.001)
            w[j] = w[j-1] + wdot * dt

    elif model == 'inertia_vary':
        # Varying effective inertia (lower at start)
        a_base, a_boost, c, w_scale = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            # Higher effective torque/inertia ratio at low speed
            a_eff = a_base + a_boost * np.exp(-w[j-1] / w_scale)
            wdot = a_eff * V - c * w[j-1]
            w[j] = w[j-1] + wdot * dt

    elif model == 'sigmoid_torque':
        # Torque ramps up with speed (like overcoming stiction)
        # wdot = a*V*(1 - exp(-w/w_rise)) - c*w
        a, c, w_rise = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            # Torque effectiveness increases as motor gets going
            torque_eff = 1.0 - np.exp(-w[j-1] / w_rise)
            # But need initial kick to start
            torque_eff = max(torque_eff, 0.1)
            wdot = a * V * torque_eff - c * w[j-1]
            w[j] = w[j-1] + wdot * dt

    elif model == 'second_order':
        # Second order system: acceleration has its own dynamics
        # wddot = k*(a*V - c*w - wdot)
        # This creates S-curve naturally
        a, c, k = params
        wdot_state = 0.0
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            wddot = k * (a * V - c * w[j-1] - wdot_state)
            wdot_state = wdot_state + wddot * dt
            w[j] = w[j-1] + wdot_state * dt

    elif model == 'stiction':
        # Stiction model: higher friction at very low speed
        # wdot = a*V - c*w - f_static*exp(-w/w_stick)
        a, c, f_static, w_stick = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            stiction = f_static * np.exp(-abs(w[j-1]) / w_stick)
            wdot = a * V - c * w[j-1] - stiction
            wdot = max(wdot, 0)  # can't decelerate below zero from rest
            w[j] = w[j-1] + wdot * dt

    elif model == 'delayed':
        # Delayed torque buildup
        # tau_dot = k*(tau_target - tau), wdot = tau - c*w
        a, c, k = params
        tau = 0.0
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            tau_target = a * V
            tau = tau + k * (tau_target - tau) * dt
            wdot = tau - c * w[j-1]
            w[j] = w[j-1] + wdot * dt

    elif model == 'scurve':
        # Explicit S-curve: combines slow start with quadratic drag
        # wdot = a*V*(w/(w+w_knee)) - q*w^2/V
        a, w_knee, q = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            # Torque ramps up as w increases (overcomes stiction)
            torque_factor = (w[j-1] + 1) / (w[j-1] + w_knee + 1)
            wdot = a * V * torque_factor - q * w[j-1]**2 / V
            w[j] = w[j-1] + wdot * dt

    elif model == 'scurve2':
        # S-curve with initial boost that fades
        # wdot = (a + b/(1+w/w0))*V - c*w - q*w^2/V
        a, b, w0, c, q = params
        for j in range(1, len(t)):
            dt = t[j] - t[j-1]
            boost = b / (1 + w[j-1] / w0)
            wdot = (a + boost) * V - c * w[j-1] - q * w[j-1]**2 / V
            w[j] = w[j-1] + wdot * dt

    return w

# --- Fit different models ---
# Calculate total variance for R²
all_w = df["vel_filt"].values
ss_tot = np.sum((all_w - np.mean(all_w))**2)

def fit_model(model, x0, bounds):
    def objective(params):
        total_error = 0.0
        for trial_id in df["trial"].unique():
            trial_df = df[df["trial"] == trial_id]
            t = trial_df["t"].values
            w_actual = trial_df["vel_filt"].values
            V = trial_df["V"].iloc[0]
            w_sim = simulate(params, t, V, w_actual[0], model=model)
            total_error += np.sum((w_actual - w_sim)**2)
        return total_error

    res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
    r2 = 1 - res.fun / ss_tot
    return res.x, res.fun, r2

# Model 1: Linear (baseline)
print("\n=== Model 1: Linear (wdot = a*V - c*w) ===")
params_lin, err_lin, r2_lin = fit_model('linear', x0=[15000, 70], bounds=[(5000, 40000), (20, 200)])
a_lin, c_lin = params_lin
print(f"a = {a_lin:.1f} rad/s^2/V")
print(f"c = {c_lin:.1f} 1/s")
print(f"tau = {1000/c_lin:.2f} ms")
print(f"R² = {r2_lin:.4f}")

# Model 2: Startup boost
print("\n=== Model 2: Boost (wdot = a*V - c*w + boost*V*exp(-w/w_scale)) ===")
params_boost, err_boost, r2_boost = fit_model('boost',
    x0=[a_lin, c_lin, 2000, 50],
    bounds=[(5000, 40000), (20, 200), (0, 20000), (10, 500)])
a_b, c_b, boost, w_scale = params_boost
print(f"a = {a_b:.1f} rad/s^2/V")
print(f"c = {c_b:.1f} 1/s")
print(f"boost = {boost:.1f} rad/s^2/V")
print(f"w_scale = {w_scale:.1f} rad/s")
print(f"tau = {1000/c_b:.2f} ms")
print(f"R² = {r2_boost:.4f}")

# Model 3: Quadratic drag
print("\n=== Model 3: Quadratic drag (wdot = a*V - c*w - q*w^2/V) ===")
params_quad, err_quad, r2_quad = fit_model('quadratic',
    x0=[a_lin, c_lin, 0.01],
    bounds=[(5000, 40000), (0, 200), (0, 1)])
a_q, c_q, q = params_quad
print(f"a = {a_q:.1f} rad/s^2/V")
print(f"c = {c_q:.1f} 1/s")
print(f"q = {q:.4f}")
print(f"R² = {r2_quad:.4f}")

# Model 4: Sqrt damping
print("\n=== Model 4: Sqrt damping (wdot = a*V - c*sqrt(w)) ===")
params_sqrt, err_sqrt, r2_sqrt = fit_model('sqrt',
    x0=[a_lin, 500],
    bounds=[(5000, 40000), (100, 5000)])
a_s, c_s = params_sqrt
print(f"a = {a_s:.1f} rad/s^2/V")
print(f"c = {c_s:.1f} rad^0.5/s")
print(f"R² = {r2_sqrt:.4f}")

# Model 5: Two-phase (use higher c1 for faster initial response)
print("\n=== Model 5: Two-phase ===")
params_2ph, err_2ph, r2_2ph = fit_model('two_phase',
    x0=[a_lin, 150, 50, 200],
    bounds=[(5000, 40000), (50, 300), (20, 150), (50, 500)])
a_2, c1, c2, w_trans = params_2ph
print(f"a = {a_2:.1f} rad/s^2/V")
print(f"c1 (initial) = {c1:.1f} 1/s  ->  tau1 = {1000/c1:.2f} ms")
print(f"c2 (later) = {c2:.1f} 1/s  ->  tau2 = {1000/c2:.2f} ms")
print(f"w_trans = {w_trans:.1f} rad/s")
print(f"R² = {r2_2ph:.4f}")

# Model 6: Second-order system (creates S-curve)
print("\n=== Model 6: Second-order (wddot = k*(a*V - c*w - wdot)) ===")
params_2nd, err_2nd, r2_2nd = fit_model('second_order',
    x0=[a_lin, c_lin, 200],
    bounds=[(5000, 40000), (20, 200), (50, 1000)])
a_2nd, c_2nd, k_2nd = params_2nd
print(f"a = {a_2nd:.1f} rad/s^2/V")
print(f"c = {c_2nd:.1f} 1/s")
print(f"k = {k_2nd:.1f} 1/s")
print(f"R² = {r2_2nd:.4f}")

# Model 7: Stiction model
print("\n=== Model 7: Stiction (wdot = a*V - c*w - f_static*exp(-w/w_stick)) ===")
params_stick, err_stick, r2_stick = fit_model('stiction',
    x0=[a_lin, c_lin, 5000, 20],
    bounds=[(5000, 40000), (20, 200), (0, 20000), (5, 200)])
a_st, c_st, f_st, w_st = params_stick
print(f"a = {a_st:.1f} rad/s^2/V")
print(f"c = {c_st:.1f} 1/s")
print(f"f_static = {f_st:.1f} rad/s^2")
print(f"w_stick = {w_st:.1f} rad/s")
print(f"R² = {r2_stick:.4f}")

# Model 8: Delayed torque
print("\n=== Model 8: Delayed torque (tau_dot = k*(a*V - tau), wdot = tau - c*w) ===")
params_delay, err_delay, r2_delay = fit_model('delayed',
    x0=[a_lin, c_lin, 500],
    bounds=[(5000, 40000), (20, 200), (100, 2000)])
a_del, c_del, k_del = params_delay
print(f"a = {a_del:.1f} rad/s^2/V")
print(f"c = {c_del:.1f} 1/s")
print(f"k = {k_del:.1f} 1/s (torque time constant = {1000/k_del:.1f} ms)")
print(f"R² = {r2_delay:.4f}")

# Model 9: S-curve model
print("\n=== Model 9: S-curve (wdot = a*V*(w/(w+w_knee)) - q*w²/V) ===")
params_scurve, err_scurve, r2_scurve = fit_model('scurve',
    x0=[20000, 50, 0.3],
    bounds=[(5000, 50000), (5, 300), (0.01, 1.0)])
a_sc, w_knee, q_sc = params_scurve
print(f"a = {a_sc:.1f} rad/s^2/V")
print(f"w_knee = {w_knee:.1f} rad/s")
print(f"q = {q_sc:.4f}")
print(f"R² = {r2_scurve:.4f}")

# --- Find best model ---
r2_scores = {
    'linear': r2_lin,
    'boost': r2_boost,
    'quadratic': r2_quad,
    'two_phase': r2_2ph,
    'second_order': r2_2nd,
    'stiction': r2_stick,
    'delayed': r2_delay,
    'scurve': r2_scurve,
}

# Print ranking
print("\n=== Model Ranking by R² ===")
for model, r2 in sorted(r2_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model:15s}: R² = {r2:.4f}")

best_model = max(r2_scores, key=r2_scores.get)
print(f"\n=== Best model: {best_model} (R² = {r2_scores[best_model]:.4f}) ===")

# --- Plot top models ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    t = trial_df["t"].values
    w_actual = trial_df["vel_filt"].values
    V = trial_df["V"].iloc[0]

    ax = axes[i]
    ax.plot(t * 1000, w_actual, 'b-', label='actual', linewidth=2.5)
    ax.plot(t * 1000, simulate(params_lin, t, V, w_actual[0], 'linear'), 'r--', label='linear', alpha=0.6)
    ax.plot(t * 1000, simulate(params_quad, t, V, w_actual[0], 'quadratic'), 'm:', label='quad', linewidth=1.5)
    ax.plot(t * 1000, simulate(params_2nd, t, V, w_actual[0], 'second_order'), 'g-', label='2nd order', linewidth=1.5)
    ax.plot(t * 1000, simulate(params_delay, t, V, w_actual[0], 'delayed'), 'c-.', label='delayed', linewidth=1.5)
    ax.plot(t * 1000, simulate(params_scurve, t, V, w_actual[0], 'scurve'), 'orange', label='scurve', linewidth=1.5)

    ax.set_title(f"V={V:.1f}")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=5)

plt.tight_layout()
plt.savefig("sysid2_models.png", dpi=150)
plt.close()
print("Saved sysid2_models.png")

# --- Plot best model only ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

all_params = {
    'linear': params_lin,
    'boost': params_boost,
    'quadratic': params_quad,
    'two_phase': params_2ph,
    'second_order': params_2nd,
    'stiction': params_stick,
    'delayed': params_delay,
    'scurve': params_scurve,
}
best_params = all_params[best_model]

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    t = trial_df["t"].values
    w_actual = trial_df["vel_filt"].values
    V = trial_df["V"].iloc[0]

    ax = axes[i]
    ax.plot(t * 1000, w_actual, 'b-', label='actual', linewidth=2)
    ax.plot(t * 1000, simulate(best_params, t, V, w_actual[0], best_model), 'r--', label=best_model, linewidth=2)

    ax.set_title(f"V={V:.1f}")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sysid2_best.png", dpi=150)
plt.close()
print("Saved sysid2_best.png")

# --- Per-voltage fitting to see parameter variation ---
print("\n=== Per-voltage second_order fits ===")
print(f"{'V':>4} {'a':>10} {'c':>8} {'k':>8} {'tau_e':>8} {'tau_m':>8} {'R²':>8}")

per_V_params = {}
for V_val in sorted(df["V"].unique()):
    trial_data = df[df["V"] == V_val]

    def objective_V(params):
        total_error = 0.0
        for trial_id in trial_data["trial"].unique():
            trial_df = trial_data[trial_data["trial"] == trial_id]
            t = trial_df["t"].values
            w_actual = trial_df["vel_filt"].values
            w_sim = simulate(params, t, V_val, w_actual[0], 'second_order')
            total_error += np.sum((w_actual - w_sim)**2)
        return total_error

    res = minimize(objective_V, x0=[a_2nd, c_2nd, k_2nd],
                   bounds=[(5000, 50000), (20, 300), (50, 2000)],
                   method='L-BFGS-B')

    a_v, c_v, k_v = res.x
    per_V_params[V_val] = res.x

    # Calculate R² for this voltage
    ss_res = res.fun
    ss_tot = np.sum((trial_data["vel_filt"].values - trial_data["vel_filt"].mean())**2)
    r2_v = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"{V_val:>4.1f} {a_v:>10.1f} {c_v:>8.1f} {k_v:>8.1f} {1000/k_v:>8.2f} {1000/c_v:>8.2f} {r2_v:>8.4f}")

print("\nObservation: Does k (electrical time constant) vary with voltage?")

# --- Try model with voltage-dependent electrical time constant ---
print("\n=== Model: Second-order with voltage-dependent tau_e ===")

def simulate_2nd_v_dep(params, t, V, w0):
    """Second order with k that depends on V: k = k0 + k1*V"""
    a, c, k0, k1 = params
    w = np.zeros_like(t)
    w[0] = w0
    wdot_state = 0.0
    k_eff = k0 + k1 * V
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        wddot = k_eff * (a * V - c * w[j-1] - wdot_state)
        wdot_state = wdot_state + wddot * dt
        w[j] = w[j-1] + wdot_state * dt
    return w

def objective_2nd_vdep(params):
    total_error = 0.0
    for trial_id in df["trial"].unique():
        trial_df = df[df["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V = trial_df["V"].iloc[0]
        w_sim = simulate_2nd_v_dep(params, t, V, w_actual[0])
        total_error += np.sum((w_actual - w_sim)**2)
    return total_error

res_vdep = minimize(objective_2nd_vdep, x0=[a_2nd, c_2nd, 200, 50],
                    bounds=[(5000, 50000), (20, 300), (50, 1000), (-100, 200)],
                    method='L-BFGS-B')
a_vd, c_vd, k0_vd, k1_vd = res_vdep.x
r2_vdep = 1 - res_vdep.fun / ss_tot

print(f"a = {a_vd:.1f} rad/s²/V")
print(f"c = {c_vd:.1f} 1/s")
print(f"k0 = {k0_vd:.1f} 1/s")
print(f"k1 = {k1_vd:.1f} 1/s/V")
print(f"k(V=1) = {k0_vd + k1_vd:.1f} → tau_e = {1000/(k0_vd + k1_vd):.2f} ms")
print(f"k(V=5) = {k0_vd + 5*k1_vd:.1f} → tau_e = {1000/(k0_vd + 5*k1_vd):.2f} ms")
print(f"R² = {r2_vdep:.4f}")

# --- Try model with stiction + inductance ---
print("\n=== Model: Second-order + stiction ===")

def simulate_2nd_stiction(params, t, V, w0):
    """Second order with stiction at low speed"""
    a, c, k, f_stick, w_stick = params
    w = np.zeros_like(t)
    w[0] = w0
    wdot_state = 0.0
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        stiction = f_stick * np.exp(-abs(w[j-1]) / w_stick)
        wddot = k * (a * V - c * w[j-1] - wdot_state - stiction)
        wdot_state = wdot_state + wddot * dt
        w[j] = w[j-1] + wdot_state * dt
    return w

def objective_2nd_stiction(params):
    total_error = 0.0
    for trial_id in df["trial"].unique():
        trial_df = df[df["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V = trial_df["V"].iloc[0]
        w_sim = simulate_2nd_stiction(params, t, V, w_actual[0])
        total_error += np.sum((w_actual - w_sim)**2)
    return total_error

res_stick2 = minimize(objective_2nd_stiction, x0=[a_2nd, c_2nd, k_2nd, 1000, 30],
                      bounds=[(5000, 50000), (20, 300), (50, 1000), (0, 10000), (5, 200)],
                      method='L-BFGS-B')
a_s2, c_s2, k_s2, f_s2, w_s2 = res_stick2.x
r2_stick2 = 1 - res_stick2.fun / ss_tot

print(f"a = {a_s2:.1f} rad/s²/V")
print(f"c = {c_s2:.1f} 1/s")
print(f"k = {k_s2:.1f} 1/s → tau_e = {1000/k_s2:.2f} ms")
print(f"f_stick = {f_s2:.1f} rad/s²")
print(f"w_stick = {w_s2:.1f} rad/s")
print(f"R² = {r2_stick2:.4f}")

# --- Plot comparison ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    t = trial_df["t"].values
    w_actual = trial_df["vel_filt"].values
    V = trial_df["V"].iloc[0]

    ax = axes[i]
    ax.plot(t * 1000, w_actual, 'b-', label='actual', linewidth=2.5)
    ax.plot(t * 1000, simulate(params_2nd, t, V, w_actual[0], 'second_order'), 'r--', label='2nd order', alpha=0.8)
    ax.plot(t * 1000, simulate_2nd_v_dep(res_vdep.x, t, V, w_actual[0]), 'g-', label='V-dep k', linewidth=1.5)
    ax.plot(t * 1000, simulate_2nd_stiction(res_stick2.x, t, V, w_actual[0]), 'm:', label='2nd+stiction', linewidth=2)

    # Per-voltage fit
    if V in per_V_params:
        ax.plot(t * 1000, simulate(per_V_params[V], t, V, w_actual[0], 'second_order'), 'c-.', label='per-V fit', alpha=0.8)

    ax.set_title(f"V={V:.1f}")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=5)

plt.tight_layout()
plt.savefig("sysid2_detailed.png", dpi=150)
plt.close()
print("\nSaved sysid2_detailed.png")

# --- Analyze parameter scaling with V ---
V_vals = sorted(per_V_params.keys())
a_vals = [per_V_params[v][0] for v in V_vals]
c_vals = [per_V_params[v][1] for v in V_vals]
k_vals = [per_V_params[v][2] for v in V_vals]

print("\n=== Parameter scaling analysis ===")
print("Looking for: a(V), c(V), k(V)")
print(f"a*V product: {[f'{a*v:.0f}' for a, v in zip(a_vals, V_vals)]}")
print(f"c*V product: {[f'{c*v:.1f}' for c, v in zip(c_vals, V_vals)]}")

# Try model: a_eff = a0 + a1/V, c_eff = c0 + c1/V
print("\n=== Model: Parameters scale with 1/V ===")

def simulate_2nd_scaled(params, t, V, w0):
    """Second order with a = a0+a1/V, c = c0+c1/V"""
    a0, a1, c0, c1, k = params
    a_eff = a0 + a1 / V
    c_eff = c0 + c1 / V
    w = np.zeros_like(t)
    w[0] = w0
    wdot_state = 0.0
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        wddot = k * (a_eff * V - c_eff * w[j-1] - wdot_state)
        wdot_state = wdot_state + wddot * dt
        w[j] = w[j-1] + wdot_state * dt
    return w

def objective_scaled(params):
    total_error = 0.0
    for trial_id in df["trial"].unique():
        trial_df = df[df["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V = trial_df["V"].iloc[0]
        w_sim = simulate_2nd_scaled(params, t, V, w_actual[0])
        total_error += np.sum((w_actual - w_sim)**2)
    return total_error

res_scaled = minimize(objective_scaled, x0=[15000, 5000, 50, 50, 300],
                      bounds=[(5000, 40000), (0, 50000), (0, 200), (0, 300), (100, 1000)],
                      method='L-BFGS-B')
a0_sc, a1_sc, c0_sc, c1_sc, k_sc = res_scaled.x
r2_scaled = 1 - res_scaled.fun / ss_tot

print(f"a(V) = {a0_sc:.1f} + {a1_sc:.1f}/V")
print(f"c(V) = {c0_sc:.1f} + {c1_sc:.1f}/V")
print(f"k = {k_sc:.1f} 1/s → tau_e = {1000/k_sc:.2f} ms")
print(f"At V=1: a={a0_sc + a1_sc:.1f}, c={c0_sc + c1_sc:.1f}, tau_m={1000/(c0_sc + c1_sc):.2f}ms")
print(f"At V=5: a={a0_sc + a1_sc/5:.1f}, c={c0_sc + c1_sc/5:.1f}, tau_m={1000/(c0_sc + c1_sc/5):.2f}ms")
print(f"R² = {r2_scaled:.4f}")

# --- Model with time delay ---
print("\n=== Model: Second-order + time delay ===")

def simulate_2nd_delay(params, t, V, w0):
    """Second order with pure time delay"""
    a, c, k, t_delay = params
    w = np.zeros_like(t)
    w[0] = w0
    wdot_state = 0.0
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        # Voltage is zero until t > t_delay
        V_eff = V if t[j] > t_delay else 0.0
        wddot = k * (a * V_eff - c * w[j-1] - wdot_state)
        wdot_state = wdot_state + wddot * dt
        w[j] = w[j-1] + wdot_state * dt
    return w

def objective_delay(params):
    total_error = 0.0
    for trial_id in df["trial"].unique():
        trial_df = df[df["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V = trial_df["V"].iloc[0]
        w_sim = simulate_2nd_delay(params, t, V, w_actual[0])
        total_error += np.sum((w_actual - w_sim)**2)
    return total_error

res_delay2 = minimize(objective_delay, x0=[a_2nd, c_2nd, k_2nd, 0.001],
                      bounds=[(5000, 50000), (20, 300), (100, 1000), (0, 0.01)],
                      method='L-BFGS-B')
a_d2, c_d2, k_d2, t_delay = res_delay2.x
r2_delay2 = 1 - res_delay2.fun / ss_tot

print(f"a = {a_d2:.1f} rad/s²/V")
print(f"c = {c_d2:.1f} 1/s → tau_m = {1000/c_d2:.2f} ms")
print(f"k = {k_d2:.1f} 1/s → tau_e = {1000/k_d2:.2f} ms")
print(f"t_delay = {t_delay*1000:.2f} ms")
print(f"R² = {r2_delay2:.4f}")

# --- Model with delay + scaled params ---
print("\n=== Model: Scaled params + time delay ===")

def simulate_scaled_delay(params, t, V, w0):
    """Scaled params with time delay"""
    a0, a1, c0, c1, k, t_delay = params
    a_eff = a0 + a1 / V
    c_eff = c0 + c1 / V
    w = np.zeros_like(t)
    w[0] = w0
    wdot_state = 0.0
    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        V_eff = V if t[j] > t_delay else 0.0
        wddot = k * (a_eff * V_eff - c_eff * w[j-1] - wdot_state)
        wdot_state = wdot_state + wddot * dt
        w[j] = w[j-1] + wdot_state * dt
    return w

def objective_scaled_delay(params):
    total_error = 0.0
    for trial_id in df["trial"].unique():
        trial_df = df[df["trial"] == trial_id]
        t = trial_df["t"].values
        w_actual = trial_df["vel_filt"].values
        V = trial_df["V"].iloc[0]
        w_sim = simulate_scaled_delay(params, t, V, w_actual[0])
        total_error += np.sum((w_actual - w_sim)**2)
    return total_error

res_sd = minimize(objective_scaled_delay, x0=[a0_sc, a1_sc, c0_sc, c1_sc, k_sc, 0.002],
                  bounds=[(5000, 40000), (0, 50000), (0, 200), (0, 300), (100, 1000), (0, 0.01)],
                  method='L-BFGS-B')
a0_sd, a1_sd, c0_sd, c1_sd, k_sd, t_d_sd = res_sd.x
r2_sd = 1 - res_sd.fun / ss_tot

print(f"a(V) = {a0_sd:.1f} + {a1_sd:.1f}/V")
print(f"c(V) = {c0_sd:.1f} + {c1_sd:.1f}/V")
print(f"k = {k_sd:.1f} 1/s → tau_e = {1000/k_sd:.2f} ms")
print(f"t_delay = {t_d_sd*1000:.2f} ms")
print(f"R² = {r2_sd:.4f}")

# --- Final comparison plot ---
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, trial_id in enumerate(df["trial"].unique()):
    if i >= len(axes):
        break
    trial_df = df[df["trial"] == trial_id]
    t = trial_df["t"].values
    w_actual = trial_df["vel_filt"].values
    V = trial_df["V"].iloc[0]

    ax = axes[i]
    ax.plot(t * 1000, w_actual, 'b-', label='actual', linewidth=2.5)
    ax.plot(t * 1000, simulate(params_2nd, t, V, w_actual[0], 'second_order'), 'r--', label='2nd order', alpha=0.6)
    ax.plot(t * 1000, simulate_2nd_delay(res_delay2.x, t, V, w_actual[0]), 'g-', label='2nd+delay', linewidth=1.5)
    ax.plot(t * 1000, simulate_scaled_delay(res_sd.x, t, V, w_actual[0]), 'm-', label='scaled+delay', linewidth=2)

    if V in per_V_params:
        ax.plot(t * 1000, simulate(per_V_params[V], t, V, w_actual[0], 'second_order'), 'c:', label='per-V', linewidth=1.5, alpha=0.8)

    ax.set_title(f"V={V:.1f}")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("vel (rad/s)")
    ax.legend(fontsize=5)

plt.tight_layout()
plt.savefig("sysid2_final.png", dpi=150)
plt.close()
print("\nSaved sysid2_final.png")

# --- Final Summary ---
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Best model: {best_model}")
print(f"R² = {r2_scores[best_model]:.4f}")
print()

if best_model == 'quadratic':
    print("Model: wdot = a*V - q*w²/V")
    print(f"  a = {a_q:.1f} rad/s²/V  (torque/inertia)")
    print(f"  q = {q:.4f}  (quadratic drag coefficient)")
    w_ss_per_V = np.sqrt(a_q / q)
    print(f"  Predicted free speed: {w_ss_per_V:.1f} rad/s per volt")
elif best_model == 'linear':
    print("Model: wdot = a*V - c*w")
    print(f"  a = {a_lin:.1f} rad/s²/V")
    print(f"  c = {c_lin:.1f} 1/s")
    print(f"  τ = {1000/c_lin:.2f} ms")
elif best_model == 'second_order':
    print("Model: wddot = k*(a*V - c*w - wdot)")
    print("  (Second-order system - like inductance causing torque lag)")
    print(f"  a = {a_2nd:.1f} rad/s²/V")
    print(f"  c = {c_2nd:.1f} 1/s")
    print(f"  k = {k_2nd:.1f} 1/s")
    print(f"  Electrical time constant: {1000/k_2nd:.2f} ms")
    print(f"  Mechanical time constant: {1000/c_2nd:.2f} ms")
elif best_model == 'delayed':
    print("Model: tau_dot = k*(a*V - tau), wdot = tau - c*w")
    print("  (Inductance model - torque builds up with time constant)")
    print(f"  a = {a_del:.1f} rad/s²/V")
    print(f"  c = {c_del:.1f} 1/s")
    print(f"  k = {k_del:.1f} 1/s")
    print(f"  Electrical time constant τ_e = L/R: {1000/k_del:.2f} ms")
    print(f"  Mechanical time constant: {1000/c_del:.2f} ms")
elif best_model == 'scurve':
    print("Model: wdot = a*V*(w/(w+w_knee)) - q*w²/V")
    print("  (S-curve: torque ramps up as motor gets going)")
    print(f"  a = {a_sc:.1f} rad/s²/V")
    print(f"  w_knee = {w_knee:.1f} rad/s")
    print(f"  q = {q_sc:.4f}")
elif best_model == 'stiction':
    print("Model: wdot = a*V - c*w - f_static*exp(-w/w_stick)")
    print("  (Stiction: high friction at low speed)")
    print(f"  a = {a_st:.1f} rad/s²/V")
    print(f"  c = {c_st:.1f} 1/s")
    print(f"  f_static = {f_st:.1f} rad/s²")
    print(f"  w_stick = {w_st:.1f} rad/s")

print("="*60)
