from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
from scipy.integrate import solve_ivp


STRAND_COLORS = ["red", "green", "blue", "orange"]
motor_stop_angle = 28.0

n = 37  # bottom motor plus n - 1 small disks
m = 0.0001  # small disk mass
I = 1.45e-9  # small disk inertia
k = 100_000.0  # spring stiffness
c = 1.0  # spring damping
l0 = 0.0045  # spring rest length
R = 0.005  # spring attach radius

M0 = 0.100  # motor mass
I0 = 4e-6  # motor inertia


@dataclass(frozen=True)
class Params:
    n: int = n
    m: float = m
    I: float = I
    k: float = k
    c: float = c
    l0: float = l0
    R: float = R
    M0: float = M0
    I0: float = I0
    spring_count: int = 4
    motor_stop_angle: float = motor_stop_angle
    max_time: float = 0.08
    samples: int = 900


@dataclass(frozen=True)
class Result:
    params: Params
    t: np.ndarray
    h: np.ndarray
    theta: np.ndarray
    h_dot: np.ndarray
    theta_dot: np.ndarray
    solve: object


def get_motor_torque(theta, theta_dot, t):
    rpm = theta_dot * 60 / (2 * np.pi)
    y1 = ((0.2 - 0.2375) / 6110) * rpm + 0.2375
    y2 = ((0.0 - 0.2000) / (14600 - 6110)) * (rpm - 6110) + 0.2
    return min(y1, y2)


def unpack(y, p):
    h = y[: p.n]
    theta = y[p.n : 2 * p.n]
    h_dot = y[2 * p.n : 3 * p.n]
    theta_dot = y[3 * p.n :]
    return h, theta, h_dot, theta_dot


def spring_loads(h, theta, h_dot, theta_dot, p):
    top_h = p.n * p.l0
    upper_h = np.r_[h[1:], top_h]
    upper_theta = np.r_[theta[1:], 0.0]
    upper_h_dot = np.r_[h_dot[1:], 0.0]
    upper_theta_dot = np.r_[theta_dot[1:], 0.0]

    dh = upper_h - h
    dtheta = upper_theta - theta
    dh_dot = upper_h_dot - h_dot
    dtheta_dot = upper_theta_dot - theta_dot
    spring_len = np.sqrt(np.maximum(dh * dh + 2 * p.R * p.R * (1 - np.cos(dtheta)), 1e-14))
    spring_len_dot = (dh * dh_dot + p.R * p.R * np.sin(dtheta) * dtheta_dot) / spring_len

    load = p.spring_count * (p.k * (spring_len - p.l0) + p.c * spring_len_dot) / spring_len
    vertical = load * dh
    twist = load * p.R * p.R * np.sin(dtheta)

    force = vertical.copy()
    torque = twist.copy()
    force[1:] -= vertical[:-1]
    torque[1:] -= twist[:-1]
    return force, torque


def dynamics(t, y, p):
    h, theta, h_dot, theta_dot = unpack(y, p)
    force, torque = spring_loads(h, theta, h_dot, theta_dot, p)
    torque[0] += get_motor_torque(theta[0], theta_dot[0], t)

    mass = np.full(p.n, p.m)
    inertia = np.full(p.n, p.I)
    mass[0] = p.M0
    inertia[0] = p.I0

    return np.r_[h_dot, theta_dot, force / mass, torque / inertia]


def simulate(p=Params()):
    h0 = np.arange(p.n) * p.l0
    theta0 = np.zeros(p.n)
    y0 = np.r_[h0, theta0, np.zeros(p.n), np.zeros(p.n)]

    def stop_at_target(t, y):
        return y[p.n] - p.motor_stop_angle

    stop_at_target.terminal = True
    stop_at_target.direction = 1

    solve = solve_ivp(
        lambda t, y: dynamics(t, y, p),
        (0.0, p.max_time),
        y0,
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
        dense_output=True,
        events=stop_at_target,
    )
    if not solve.success:
        raise RuntimeError(solve.message)
    if len(solve.t_events[0]) == 0:
        raise RuntimeError("motor disk did not reach the stop angle")

    t_end = solve.t_events[0][0]
    t = np.linspace(0.0, t_end, p.samples)
    y = solve.sol(t)
    h, theta, h_dot, theta_dot = unpack(y, p)
    return Result(p, t, h, theta, h_dot, theta_dot, solve)


def save_motion_plots(result, path):
    import matplotlib.pyplot as plt

    p = result.params
    time_ms = result.t * 1000
    colors = plt.cm.viridis(np.linspace(0, 1, p.n))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for i, color in enumerate(colors):
        lw = 2.4 if i == 0 else 0.9
        label = "motor" if i == 0 else None
        axes[0].plot(time_ms, result.h[i] * 1000, color=color, lw=lw, label=label)
        axes[1].plot(time_ms, result.theta[i], color=color, lw=lw)
        axes[2].plot(time_ms, result.h_dot[i], color=color, lw=lw)
        axes[3].plot(time_ms, result.theta_dot[i], color=color, lw=lw)

    axes[1].axhline(p.motor_stop_angle, color="0.25", ls="--", lw=1)
    axes[0].set_ylabel("height (mm)")
    axes[1].set_ylabel("angle (rad)")
    axes[2].set_ylabel("linear velocity (m/s)")
    axes[3].set_ylabel("angular velocity (rad/s)")
    axes[2].set_xlabel("time (ms)")
    axes[3].set_xlabel("time (ms)")
    axes[0].set_title("Disk heights")
    axes[1].set_title("Disk angles")
    axes[2].set_title("Disk linear velocities")
    axes[3].set_title("Disk angular velocities")
    for ax in axes:
        ax.grid(True, color="0.9")
    axes[0].legend(loc="upper left")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, p.n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.015)
    cbar.set_label("disk index, 0 = motor")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def get_loads(result):
    p = result.params
    force = np.zeros_like(result.h)
    torque = np.zeros_like(result.theta)
    for i, t in enumerate(result.t):
        f, tau = spring_loads(
            result.h[:, i],
            result.theta[:, i],
            result.h_dot[:, i],
            result.theta_dot[:, i],
            p,
        )
        tau[0] += get_motor_torque(result.theta[0, i], result.theta_dot[0, i], t)
        force[:, i] = f
        torque[:, i] = tau
    return force, torque


def save_load_plots(result, path):
    import matplotlib.pyplot as plt

    p = result.params
    time_ms = result.t * 1000
    colors = plt.cm.viridis(np.linspace(0, 1, p.n))
    force, torque = get_loads(result)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for i, color in enumerate(colors):
        lw = 2.4 if i == 0 else 0.9
        label = "motor" if i == 0 else None
        axes[0].plot(time_ms, force[i], color=color, lw=lw, label=label)
        axes[1].plot(time_ms, torque[i], color=color, lw=lw)

    axes[0].set_title("Net disk forces")
    axes[1].set_title("Net disk torques")
    axes[0].set_ylabel("force (N)")
    axes[1].set_ylabel("torque (N m)")
    axes[1].set_xlabel("time (ms)")
    for ax in axes:
        ax.grid(True, color="0.9")
    axes[0].legend(loc="upper left")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, p.n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.015)
    cbar.set_label("disk index, 0 = motor")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def disk_points(radius, z, theta, count=60):
    angle = np.linspace(0, 2 * np.pi, count)
    return radius * np.cos(angle), radius * np.sin(angle), np.full_like(angle, z)


def spring_points(a, b, coils=7, radius=0.00028, count=45):
    a = np.asarray(a)
    b = np.asarray(b)
    axis = b - a
    length = np.linalg.norm(axis)
    if length < 1e-12:
        return np.repeat(a[None, :], count, axis=0)

    e = axis / length
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(e, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(e, ref)
    u /= np.linalg.norm(u)
    v = np.cross(e, u)

    s = np.linspace(0, 1, count)
    wiggle = radius * (np.sin(2 * np.pi * coils * s)[:, None] * u + np.cos(2 * np.pi * coils * s)[:, None] * v)
    return a + s[:, None] * axis + wiggle


def draw_frame(ax, result, frame):
    p = result.params
    h = result.h[:, frame]
    theta = result.theta[:, frame]
    top_h = p.n * p.l0
    motor_radius = 2.4 * p.R
    view_r = 3.0 * motor_radius

    ax.clear()
    ax.set_xlim(-view_r, view_r)
    ax.set_ylim(-view_r, view_r)
    ax.set_zlim(-0.01, top_h + 0.01)
    ax.set_box_aspect((2 * view_r, 2 * view_r, top_h + 0.02))
    ax.view_init(elev=18, azim=-52)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("h (m)")
    ax.set_title(f"t = {result.t[frame] * 1000:.1f} ms")

    phi = np.linspace(0, 2 * np.pi, 40)
    for i in range(p.n):
        radius = motor_radius if i == 0 else p.R
        color = "tab:blue" if i == 0 else "0.35"
        x, y, z = disk_points(radius, h[i], theta[i])
        ax.plot(x, y, z, color=color, lw=1.5)
        ax.plot([0, radius * np.cos(theta[i])], [0, radius * np.sin(theta[i])], [h[i], h[i]], color=color, lw=1.0)

    ax.plot(motor_radius * np.cos(phi), motor_radius * np.sin(phi), np.full_like(phi, top_h), color="0.6", lw=2)

    anchors = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for i in range(p.n):
        lower_h = h[i]
        lower_theta = theta[i]
        upper_h = h[i + 1] if i + 1 < p.n else top_h
        upper_theta = theta[i + 1] if i + 1 < p.n else 0.0
        for angle, color in zip(anchors, STRAND_COLORS):
            a = [p.R * np.cos(angle + lower_theta), p.R * np.sin(angle + lower_theta), lower_h]
            b = [p.R * np.cos(angle + upper_theta), p.R * np.sin(angle + upper_theta), upper_h]
            pts = spring_points(a, b)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=0.8, alpha=0.8)


def save_3d_preview(result, path):
    import matplotlib.pyplot as plt

    frames = [0, len(result.t) // 2, len(result.t) - 1]
    fig = plt.figure(figsize=(14, 5))
    for i, frame in enumerate(frames, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        draw_frame(ax, result, frame)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_interactive_plot(result):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.12)
    state = {"frame": 0}
    draw_frame(ax, result, state["frame"])

    slider_ax = fig.add_axes([0.17, 0.04, 0.68, 0.035])
    slider = Slider(
        slider_ax,
        "frame",
        0,
        len(result.t) - 1,
        valinit=0,
        valstep=1,
    )

    def update(value):
        frame = int(np.clip(round(value), 0, len(result.t) - 1))
        state["frame"] = frame
        draw_frame(ax, result, frame)
        fig.canvas.draw_idle()

    def step(event):
        if event.key not in ("left", "right"):
            return
        direction = 1 if event.key == "right" else -1
        speed = 10
        slider.set_val(np.clip(state["frame"] + direction*speed, 0, len(result.t) - 1))

    slider.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", step)
    fig.slider = slider
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="open the interactive 3D slider")
    parser.add_argument("--k", type=float, default=k, help="spring stiffness")
    parser.add_argument("--c", type=float, default=c, help="spring damping")
    args = parser.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    p = Params(k=args.k, c=args.c)
    result = simulate(p)

    out_dir = Path(__file__).resolve().parent
    motion_path = out_dir / "springchain_height_angle.png"
    load_path = out_dir / "springchain_force_torque.png"
    preview_path = out_dir / "springchain_3d_preview.png"
    save_motion_plots(result, motion_path)
    save_load_plots(result, load_path)
    save_3d_preview(result, preview_path)

    gaps = np.diff(np.r_[result.h[:, -1], p.n * p.l0])
    print(f"reached target in {result.t[-1] * 1000:.2f} ms")
    print(f"motor angle: {result.theta[0, -1]:.2f} rad")
    print(f"bottom height: {result.h[0, -1] * 1000:.2f} mm")
    print(f"minimum final gap: {gaps.min() * 1000:.2f} mm")
    print(f"saved {motion_path}")
    print(f"saved {load_path}")
    print(f"saved {preview_path}")

    if args.show:
        import matplotlib.pyplot as plt

        make_interactive_plot(result)
        plt.show()


if __name__ == "__main__":
    main()
