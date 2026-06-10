from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from springchain import I, m, Params, simulate
from springchain_fixed import simulate_fixed


K_MIN = 10_000.0
K_MAX = 500_000.0
K_SAMPLES = 50
RADII = tuple(mm / 1000 for mm in range(3, 12))
REFERENCE_R = 0.005
REFERENCE_M = m
REFERENCE_I = I


def final_com_velocity(result):
    p = result.params
    mass = np.full(p.n, p.m)
    mass[0] = p.M0
    return (mass @ result.h_dot[:, -1]) / mass.sum()


def run_case(case):
    radius, k = case
    scale = radius / REFERENCE_R
    p = Params(
        k=k,
        R=radius,
        m=REFERENCE_M * scale**2,
        I=REFERENCE_I * scale**4,
        samples=200,
    )
    result = simulate(p)
    row = [
        radius,
        k,
        p.m,
        p.I,
        final_com_velocity(result),
        result.t[-1],
        result.theta[0, -1],
        result.h[0, -1],
    ]
    return row


def run_sweep():
    k_values = np.geomspace(K_MIN, K_MAX, K_SAMPLES)
    cases = [(radius, k) for radius in RADII for k in k_values]
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_case, cases))

    rows = np.array(rows)
    for row in rows:
        print(
            f"R={row[0] * 1000:.0f}mm k={row[1]:8.0f} "
            f"m={row[2] * 1000:.3f}g I={row[3]:.3e} "
            f"vCOM={row[4]:.3f}m/s t={row[5] * 1000:.1f}ms",
            flush=True,
        )
    return rows


def fixed_radius_rows():
    rows = []
    for radius in RADII:
        scale = radius / REFERENCE_R
        p = Params(
            R=radius,
            m=REFERENCE_M * scale**2,
            I=REFERENCE_I * scale**4,
            samples=200,
        )
        result = simulate_fixed(p)
        rows.append(
            [
                radius,
                p.m,
                p.I,
                final_com_velocity(result),
                result.t[-1],
                result.theta[0, -1],
                result.h[0, -1],
            ]
        )
    return np.array(rows)


def save_plot(rows, path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    cmap = plt.cm.rainbow
    colors = cmap(np.linspace(0, 1, len(RADII)))
    for radius in RADII:
        data = rows[rows[:, 0] == radius]
        color = colors[RADII.index(radius)]
        ax.scatter(
            data[:, 1],
            data[:, 4],
            s=28,
            color=color,
            label=f"R = {radius * 1000:.0f} mm",
        )

    ax.set_xscale("log")
    ax.set_xlabel("spring stiffness k (N/m)")
    ax.set_ylabel("final COM velocity (m/s)")
    ax.set_title("Final COM velocity vs spring stiffness")
    ax.grid(True, which="both", color="0.9")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sampled_radius_data(rows):
    target_k = np.array([10_000.0, 20_000.0, 50_000.0, 100_000.0, 500_000.0])
    k_values = np.unique(rows[:, 1])
    return np.array([k_values[np.argmin(np.abs(k_values - k))] for k in target_k])


def save_radius_plot(rows, path, fixed_rows=None):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    sample_k = sampled_radius_data(rows)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_k)))

    for k, color in zip(sample_k, colors):
        data = rows[np.isclose(rows[:, 1], k)]
        data = data[np.argsort(data[:, 0])]
        ax.plot(
            data[:, 0] * 1000,
            data[:, 4],
            marker="o",
            lw=1.8,
            color=color,
            label=f"k = {k / 1000:.0f}k N/m",
        )

    if fixed_rows is not None:
        fixed_rows = fixed_rows[np.argsort(fixed_rows[:, 0])]
        ax.plot(
            fixed_rows[:, 0] * 1000,
            fixed_rows[:, 3],
            marker="s",
            ms=5,
            lw=2.4,
            color="red",
            label="fixed length",
        )

    ax.set_xlabel("spring attach radius R (mm)")
    ax.set_ylabel("final COM velocity (m/s)")
    ax.set_title("Final COM velocity vs radius")
    ax.grid(True, color="0.9")
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_radius_plot_all_stiffness(rows, path, fixed_rows=None):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    k_values = np.unique(rows[:, 1])
    norm = matplotlib.colors.LogNorm(k_values.min(), k_values.max())
    cmap = plt.cm.viridis

    for k in k_values:
        data = rows[np.isclose(rows[:, 1], k)]
        data = data[np.argsort(data[:, 0])]
        ax.plot(
            data[:, 0] * 1000,
            data[:, 4],
            marker="o",
            ms=2.7,
            lw=1.0,
            alpha=0.72,
            color=cmap(norm(k)),
        )

    if fixed_rows is not None:
        fixed_rows = fixed_rows[np.argsort(fixed_rows[:, 0])]
        ax.plot(
            fixed_rows[:, 0] * 1000,
            fixed_rows[:, 3],
            marker="s",
            ms=5,
            lw=2.6,
            color="red",
            label="fixed length",
        )
        ax.legend(loc="lower right")

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar_ticks = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([f"{tick // 1000}k" for tick in colorbar_ticks])
    colorbar.set_label("spring stiffness k (N/m)")
    ax.set_xlabel("spring attach radius R (mm)")
    ax.set_ylabel("final COM velocity (m/s)")
    ax.set_title("Final COM velocity vs radius, all stiffnesses")
    ax.grid(True, color="0.9")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(exist_ok=True)
    rows = run_sweep()

    csv_path = out_dir / "stiffness_sweep.csv"
    plot_path = out_dir / "stiffness_sweep_com_velocity.png"
    radius_plot_path = out_dir / "stiffness_sweep_com_velocity_by_radius.png"
    fixed_csv_path = out_dir / "stiffness_sweep_fixed_length_by_radius.csv"
    radius_fixed_plot_path = out_dir / "stiffness_sweep_com_velocity_by_radius_with_fixed.png"
    radius_all_plot_path = out_dir / "stiffness_sweep_com_velocity_by_radius_all_k.png"
    fixed_rows = fixed_radius_rows()
    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header="R,k,m,I,final_com_velocity,end_time,motor_angle,bottom_height",
        comments="",
    )
    np.savetxt(
        fixed_csv_path,
        fixed_rows,
        delimiter=",",
        header="R,m,I,final_com_velocity,end_time,motor_angle,bottom_height",
        comments="",
    )
    save_plot(rows, plot_path)
    save_radius_plot(rows, radius_plot_path)
    save_radius_plot(rows, radius_fixed_plot_path, fixed_rows)
    save_radius_plot_all_stiffness(rows, radius_all_plot_path, fixed_rows)
    print(f"saved {csv_path}")
    print(f"saved {fixed_csv_path}")
    print(f"saved {plot_path}")
    print(f"saved {radius_plot_path}")
    print(f"saved {radius_fixed_plot_path}")
    print(f"saved {radius_all_plot_path}")


if __name__ == "__main__":
    main()
