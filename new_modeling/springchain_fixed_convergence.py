from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import springchain as spring
from springchain_fixed import sample_fixed, simulate_fixed


K_VALUES = np.array([5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8])
COMPARE_SAMPLES = 600


def sample_spring(result, t):
    y = result.solve.sol(t)
    return spring.unpack(y, result.params)


def rms_by_disk(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def compare_one(k, fixed):
    p = spring.Params(k=k, samples=300)
    result = spring.simulate(p)
    t_end = min(result.t[-1], fixed.t[-1])
    t = np.linspace(0.0, t_end, COMPARE_SAMPLES)

    _, _, spring_h_dot, spring_theta_dot = sample_spring(result, t)
    _, _, fixed_h_dot, fixed_theta_dot = sample_fixed(fixed, t)

    h_dot_rms = rms_by_disk(spring_h_dot, fixed_h_dot)
    theta_dot_rms = rms_by_disk(spring_theta_dot, fixed_theta_dot)
    return {
        "k": k,
        "result": result,
        "t_end": t_end,
        "spring_end": result.t[-1],
        "fixed_end": fixed.t[-1],
        "h_dot_global": np.sqrt(np.mean((spring_h_dot - fixed_h_dot) ** 2)),
        "theta_dot_global": np.sqrt(np.mean((spring_theta_dot - fixed_theta_dot) ** 2)),
        "h_dot_max_disk": h_dot_rms.max(),
        "theta_dot_max_disk": theta_dot_rms.max(),
    }


def save_error_plot(rows, path):
    k = rows[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].loglog(k, rows[:, 1], "o-", label="global RMS")
    axes[0].loglog(k, rows[:, 2], "s--", label="worst disk RMS")
    axes[1].loglog(k, rows[:, 3], "o-", label="global RMS")
    axes[1].loglog(k, rows[:, 4], "s--", label="worst disk RMS")

    axes[0].set_ylabel("vertical velocity error (m/s)")
    axes[1].set_ylabel("angular velocity error (rad/s)")
    axes[1].set_xlabel("spring stiffness k (N/m)")
    axes[0].set_title("Spring model convergence to fixed-length model")
    for ax in axes:
        ax.grid(True, which="both", color="0.9")
        ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_plot_linear(rows, path):
    k = rows[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].semilogx(k, rows[:, 1], "o-", label="global RMS")
    axes[0].semilogx(k, rows[:, 2], "s--", label="worst disk RMS")
    axes[1].semilogx(k, rows[:, 3], "o-", label="global RMS")
    axes[1].semilogx(k, rows[:, 4], "s--", label="worst disk RMS")

    axes[0].set_ylabel("vertical velocity error (m/s)")
    axes[1].set_ylabel("angular velocity error (rad/s)")
    axes[1].set_xlabel("spring stiffness k (N/m)")
    axes[0].set_title("Spring model convergence to fixed-length model")
    for ax in axes:
        ax.grid(True, which="both", color="0.9")
        ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overlay_plot(fixed, spring_result, path):
    n = fixed.params.n
    t_end = min(fixed.t[-1], spring_result.t[-1])
    t = np.linspace(0.0, t_end, COMPARE_SAMPLES)
    _, _, spring_h_dot, spring_theta_dot = sample_spring(spring_result, t)
    _, _, fixed_h_dot, fixed_theta_dot = sample_fixed(fixed, t)

    colors = plt.cm.viridis(np.linspace(0, 1, n))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for i, color in enumerate(colors):
        lw = 1.6 if i == 0 else 0.8
        axes[0].plot(t * 1000, fixed_h_dot[i], color=color, lw=lw)
        axes[0].plot(t * 1000, spring_h_dot[i], color=color, lw=lw, ls="--")
        axes[1].plot(t * 1000, fixed_theta_dot[i], color=color, lw=lw)
        axes[1].plot(t * 1000, spring_theta_dot[i], color=color, lw=lw, ls="--")

    axes[0].set_ylabel("vertical velocity (m/s)")
    axes[1].set_ylabel("angular velocity (rad/s)")
    axes[1].set_xlabel("time (ms)")
    axes[0].set_title(f"Fixed-length vs spring model at k={spring_result.params.k:.0e}")
    for ax in axes:
        ax.grid(True, color="0.9")
    axes[0].plot([], [], color="black", lw=1.5, label="fixed")
    axes[0].plot([], [], color="black", lw=1.5, ls="--", label="spring")
    axes[0].legend(loc="upper left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(exist_ok=True)

    fixed = simulate_fixed(spring.Params(samples=300))
    comparisons = []
    for k in K_VALUES:
        item = compare_one(k, fixed)
        comparisons.append(item)
        print(
            f"k={k:.1e} hdot_global={item['h_dot_global']:.5f} "
            f"hdot_max={item['h_dot_max_disk']:.5f} "
            f"thetadot_global={item['theta_dot_global']:.5f} "
            f"thetadot_max={item['theta_dot_max_disk']:.5f}",
            flush=True,
        )

    rows = np.array(
        [
            [
                item["k"],
                item["h_dot_global"],
                item["h_dot_max_disk"],
                item["theta_dot_global"],
                item["theta_dot_max_disk"],
                item["spring_end"],
                item["fixed_end"],
            ]
            for item in comparisons
        ]
    )

    csv_path = out_dir / "fixed_length_convergence.csv"
    error_path = out_dir / "fixed_length_convergence_errors.png"
    linear_error_path = out_dir / "fixed_length_convergence_errors_linear.png"
    overlay_path = out_dir / "fixed_length_convergence_overlay.png"
    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header="k,h_dot_global,h_dot_max_disk,theta_dot_global,theta_dot_max_disk,spring_end,fixed_end",
        comments="",
    )
    save_error_plot(rows, error_path)
    save_error_plot_linear(rows, linear_error_path)
    save_overlay_plot(fixed, comparisons[-1]["result"], overlay_path)
    print(f"fixed end time: {fixed.t[-1] * 1000:.3f} ms")
    print(f"saved {csv_path}")
    print(f"saved {error_path}")
    print(f"saved {linear_error_path}")
    print(f"saved {overlay_path}")


if __name__ == "__main__":
    main()
