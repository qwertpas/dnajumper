"""Render the stability boundary, a tilt mode, and the proposed attachment layout."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from .alignment import lowest_mode, tilt_matrix
from .offset_links import points


def draw():
    fig = plt.figure(figsize=(16, 5.4), layout="constrained")
    ax = fig.add_subplot(131)
    counts = np.arange(4, 21)
    critical = []
    for count in counts:
        def value(angle):
            return np.linalg.eigvalsh(tilt_matrix(int(count), angle)[0])[0]
        critical.append(np.rad2deg(brentq(value, .01, np.pi/2 - 1e-5)))
    ax.fill_between(counts, critical, 90, color="#f6dfdc")
    ax.fill_between(counts, 0, critical, color="#e5efe8")
    ax.plot(counts, critical, "o-", color="#333333", markersize=4)
    ax.axhline(45, color="#777777", linestyle="--", linewidth=1)
    ax.scatter([5], [45], marker="x", s=65, color="#333333", zorder=4)
    ax.annotate("Your approximate observation", (5, 45), (5, 22), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(15, 74, "Unstable", color="#9a413a")
    ax.text(15, 17, "Stable", color="#35634a")
    ax.set(xlim=(4, 20), ylim=(0, 90), xlabel="Number of cells",
           ylabel="Twist per cell (degrees)",
           title="Four equal-radius links per cell")
    ax.set_xticks([4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.grid(alpha=.2)

    ax = fig.add_subplot(132, projection="3d")
    cells, angle, scale = 12, np.pi/4, .16
    _, tilts, centers = lowest_mode(cells, angle)
    height = np.sqrt(2*np.cos(angle))
    phi = np.arange(5)*np.pi/2
    outline = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(5)))
    for j in range(cells + 1):
        arm = outline @ Rotation.from_rotvec([0, 0, j*angle]).as_matrix().T
        arm = arm @ Rotation.from_rotvec([*list(scale*tilts[j]), 0]).as_matrix().T
        plate = arm + [*list(scale*centers[j]), j*height]
        ax.plot(*plate.T, color="#21618c", linewidth=1.1)
    ax.plot(scale*centers[:, 0], scale*centers[:, 1], np.arange(cells+1)*height,
            color="#a44236", linewidth=2)
    ax.plot([0, 0], [0, 0], [0, cells*height], "--", color="#999999", linewidth=.8)
    ax.set_title("First unstable tilt mode\n12 square cells, 45° each")
    ax.text2D(.08, .01, "Infinitesimal mode, shown enlarged", transform=ax.transAxes, fontsize=9)
    ax.set_box_aspect((3, 3, cells*height))
    ax.view_init(elev=12, azim=-50)
    ax.set_axis_off()

    ax = fig.add_subplot(133, projection="3d")
    outer, inner = 1., .65
    length, angle = np.sqrt(outer**2 + inner**2), np.pi/5
    bottom, top = points(outer, inner, length, angle)
    circle = np.linspace(0, 2*np.pi, 121)
    for z in (0, top[0, 2]):
        ax.plot(np.cos(circle), np.sin(circle), np.full_like(circle, z), color="#777777")
        ax.plot(inner*np.cos(circle), inner*np.sin(circle), np.full_like(circle, z),
                color="#aaaaaa", linewidth=.8, linestyle="--")
    for i in range(8):
        color = "#21618c" if i % 2 == 0 else "#b86722"
        segment = np.array([bottom[i], top[i]])
        ax.plot(*segment.T, color=color, linewidth=2,
                label=("Outer → inner" if i == 0 else "Inner → outer") if i < 2 else None)
        ax.scatter(*segment.T, color=color, s=14)
    ax.set_title("Proposed eight-link cell\nInner radius = 0.65 × outer radius")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.03), frameon=False, fontsize=9)
    ax.set_box_aspect((2, 2, 1.8))
    ax.view_init(elev=20, azim=-65)
    ax.set_axis_off()
    fig.suptitle("Alignment under twist and axial tension", fontsize=15)
    fig.supxlabel("Large axial stiffness; rigid plates; fixed end centers and normals. Fabric slack and contact omitted.",
                  fontsize=9)
    path = Path(__file__).with_name("alignment.png")
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    draw()
