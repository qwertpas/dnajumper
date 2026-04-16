import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# -----------------------------
# Motor parameter utilities
# -----------------------------
@dataclass
class MotorParams:
    Kv_rpm_per_V: float = 2135.0          # given
    # tau_stall_per_V: float = 0.09555555556 # Nm per V (datasheet)
    tau_stall_per_V: float = 0.0871198895 # Nm per V (fitted to 1V)
    k_elec: float = 296.0                  # 1/s  (R/L), from your fitted model (optional but helpful)
    # k_elec: float = 700                        # 1.43 ms electrical time constant
    # friction smoothing (for tanh approx of sign)
    w_eps: float = 5.0                     # rad/s

    def Ke(self) -> float:
        # Ke [V/(rad/s)] = 1 / (rad/s per V)
        Kv_rad_per_s_per_V = self.Kv_rpm_per_V * 2*np.pi/60.0
        return 1.0 / Kv_rad_per_s_per_V

    def Kt(self) -> float:
        # SI: Kt == Ke (Nm/A)
        return self.Ke()

    def R(self) -> float:
        # tau_stall = Kt * (V/R) => tau_stall_per_V = Kt/R
        return self.Kt() / self.tau_stall_per_V

    def L(self) -> float:
        # k_elec = R/L => L = R/k_elec
        return self.R() / self.k_elec


# -----------------------------
# Data loading
# -----------------------------
def load_logs(pattern="motor_logs/*prbs*.csv"):
    dfs = []
    for k, fn in enumerate(sorted(glob.glob(pattern))):
        d = pd.read_csv(fn)
        d["trial"] = k
        d["file"] = fn
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError(f"No files matched: {pattern}")
    df = pd.concat(dfs, ignore_index=True)
    # time in seconds
    if "time_us" in df.columns:
        df["t"] = df["time_us"].astype(float) * 1e-6
    elif "time_ms" in df.columns:
        df["t"] = df["time_ms"].astype(float) * 1e-3
    else:
        raise ValueError("Need time_us or time_ms column.")
    df = df.sort_values(["trial", "t"]).reset_index(drop=True)
    return df


# -----------------------------
# Voltage mapping (edit as needed)
# -----------------------------
def compute_V_in(g: pd.DataFrame, v_cmd_col="set_volts"):
    """
    Returns motor terminal voltage time series.
    Edit this if set_volts is not an actual voltage at the motor.
    """
    Vcmd = g[v_cmd_col].to_numpy(dtype=float)

    # Option A (most likely in your logs): set_volts already in volts at motor
    V_in = Vcmd

    # Option B (if set_volts is a command in volts referenced to ±7 V full scale
    # and your actual terminal is scaled by battery):
    # if "vbat" in g.columns:
    #     V_in = Vcmd * (g["vbat"].to_numpy(dtype=float) / 7.0)

    return V_in


# -----------------------------
# Model + simulation
# -----------------------------
def motor_ode(t, x, t_grid, V_grid, p, mp: MotorParams, include_tau0=True):
    """
    x = [i, w]
    i_dot = (V - R i - Ke w)/L
    w_dot = (Kt i - b w - tau_c*tanh(w/w_eps) - tau0)/I
    """
    i, w = x
    I, b, tau_c = p[:3]
    tau0 = p[3] if include_tau0 else 0.0

    # interpolate input voltage onto solver time
    # V = np.interp(t, t_grid, V_grid)

    # # Zero-order hold instead of linear interpolation
    idx = np.searchsorted(t_grid, t, side='right') - 1
    idx = np.clip(idx, 0, len(V_grid) - 1)
    V = V_grid[idx]

    R = mp.R()
    L = mp.L()
    Ke = mp.Ke()
    Kt = mp.Kt()

    i_dot = (V - R*i - Ke*w) / L
    tau_fric = b*w + tau_c*np.tanh(w / mp.w_eps) + tau0
    w_dot = (Kt*i - tau_fric) / I
    return [i_dot, w_dot]


def simulate_trial(t, V_in, w_meas, p, mp: MotorParams, include_tau0=True):
    """
    Simulate current+speed states. Initialize:
      w0 from data, i0 from quasi-steady electrical relation at t0.
    """
    t0 = float(t[0])
    tf = float(t[-1])

    # initial conditions
    w0 = float(w_meas[0])
    R = mp.R()
    Ke = mp.Ke()

    # quasi-steady i0 ~ (V - Ke*w)/R
    i0 = (float(V_in[0]) - Ke*w0) / R
    x0 = [i0, w0]

    sol = solve_ivp(
        fun=lambda tt, xx: motor_ode(tt, xx, t, V_in, p, mp, include_tau0=include_tau0),
        t_span=(t0, tf),
        y0=x0,
        t_eval=t,
        method="RK45",
        max_step=0.002,   # keep stable; adjust if needed
        rtol=1e-6,
        atol=1e-8
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    i_sim = sol.y[0]
    w_sim = sol.y[1]
    return i_sim, w_sim


# -----------------------------
# Least squares fit across trials
# -----------------------------
def fit_inertia(df: pd.DataFrame, mp: MotorParams, include_tau0=True,
                downsample=1, robust=True):
    """
    Fits parameters: I, b, tau_c, (optional tau0)
    """
    trials = [g for _, g in df.groupby("trial")]

    # Build residual vector across all trials
    def residuals(p):
        # enforce positivity of I (softly): if negative, return huge residual
        if p[0] <= 0:
            return np.ones(1000) * 1e6

        res_all = []
        for g in trials:
            t = g["t"].to_numpy(dtype=float)[::downsample]
            w = g["vel"].to_numpy(dtype=float)[::downsample]
            V_in = compute_V_in(g)[::downsample]

            # simulate
            _, w_sim = simulate_trial(t, V_in, w, p, mp, include_tau0=include_tau0)

            # residual in rad/s
            res = (w_sim - w)
            res_all.append(res)

        return np.concatenate(res_all)

    # ---- Initial guess ----
    # quick guess for I from early accel at moderate V: I ~ (tau_stall_per_V*|V|)/|wdot|
    # Here we use rough numbers; you'll still refine via least squares.
    I0 = 2e-6      # N*m*s^2/rad (guess; adjust if you know order)
    b0 = 1e-5
    tc0 = 1e-3
    tau00 = 0.0
    p0 = np.array([I0, b0, tc0, tau00] if include_tau0 else [I0, b0, tc0], dtype=float)

    # bounds (keep them wide but physical)
    lb = [1e-9, 0.0, 0.0, -0.2] if include_tau0 else [1e-9, 0.0, 0.0]
    ub = [1e-2, 1e-1, 1e-1,  0.2] if include_tau0 else [1e-2, 1e-1, 1e-1]

    loss = "soft_l1" if robust else "linear"
    f_scale = 20.0  # rad/s scale for robust loss; tune if needed

    out = least_squares(
        residuals, p0, bounds=(lb, ub),
        loss=loss, f_scale=f_scale,
        max_nfev=50, verbose=2
    )
    return out


# -----------------------------
# Run on your logs
# -----------------------------
if __name__ == "__main__":
    mp = MotorParams(
        # Kv_rpm_per_V=2135.0,
        # tau_stall_per_V=0.09555555556,  # your estimate
        # k_elec=296.0                    # your estimate (R/L)
    )

    df = load_logs("motor_logs/*prbs*.csv")  # change pattern as needed

    df = df[df["time_ms"] < 9000.0]

    # Optional: only keep times after startup if you have stiction at t~0
    # df = df[df["t"] > (df.groupby("trial")["t"].transform("min") + 0.005)]

    fit = fit_inertia(df, mp, include_tau0=True, downsample=1, robust=True)

    p = fit.x
    if len(p) == 4:
        I_hat, b_hat, tc_hat, tau0_hat = p
        print("\nFitted:")
        print(f"I      = {I_hat:.6e}  N*m*s^2/rad")
        print(f"b      = {b_hat:.6e}  N*m*s/rad")
        print(f"tau_c  = {tc_hat:.6e}  N*m")
        print(f"tau0   = {tau0_hat:.6e}  N*m")
    else:
        I_hat, b_hat, tc_hat = p
        print("\nFitted:")
        print(f"I      = {I_hat:.6e}  N*m*s^2/rad")
        print(f"b      = {b_hat:.6e}  N*m*s/rad")
        print(f"tau_c  = {tc_hat:.6e}  N*m")

    # Print inferred electrical params for sanity
    print("\nElectrical params from Kv + tau_stall_per_V + k_elec:")
    print(f"Ke = {mp.Ke():.6e} V*s/rad   (Kv={mp.Kv_rpm_per_V} rpm/V)")
    print(f"Kt = {mp.Kt():.6e} N*m/A")
    print(f"R  = {mp.R():.6e} ohm")
    print(f"L  = {mp.L():.6e} H")
