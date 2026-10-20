"""Concept sketches, not manufacturing patterns, for integrated tensile straps."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np


BLUE, ORANGE, GRAY = "#2876a8", "#c57930", "#c7cdd2"


def dimension(ax, start, end, label, position):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=9,
                                color="#444444", linewidth=.8))
    ax.text(*position, label, ha="center", va="center", fontsize=10)


def draw():
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 5.7), layout="constrained")
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes[0]
    height = 1.3
    for x, color in ((.1, BLUE), (.9, ORANGE)):
        ax.add_patch(Rectangle((x-.055, 0), .11, height, color=color))
        ax.plot([x, x], [0, height], "--", color="white", linewidth=.9)
    for y in (-.19, height):
        ax.add_patch(Rectangle((.045, y), .91, .19, facecolor=GRAY,
                               edgecolor="#66717b", hatch="///", linewidth=.7))
    ax.text(.5, height+.095, "Top clamp zone", ha="center", va="center", fontsize=9)
    ax.text(.5, -.095, "Bottom clamp zone", ha="center", va="center", fontsize=9)
    ax.text(.5, .82, "Cut out\nor slit open\nthe free span", ha="center", va="center", fontsize=10)
    ax.text(.1, .39, "R → r", rotation=90, color="white", ha="center", va="center", fontsize=9)
    ax.text(.9, .39, "r → R", rotation=90, color="white", ha="center", va="center", fontsize=9)
    dimension(ax, (1.1, 0), (1.1, height), "L", (1.21, height/2))
    dimension(ax, (.1, -.31), (.9, -.31), "d", (.5, -.40))
    ax.set_xlim(-.18, 1.42)
    ax.set_ylim(-.56, 1.64)
    ax.set_title("A. Three identical window straps", fontsize=12, pad=14)

    ax = axes[1]
    for i in range(6):
        x = .12 + i*.22
        ax.add_patch(Rectangle((x-.035, 0), .07, height,
                               color=BLUE if i % 2 == 0 else ORANGE))
    for y in (-.16, height):
        ax.add_patch(Rectangle((.04, y), 1.26, .16, facecolor=GRAY,
                               edgecolor="#66717b", hatch="///", linewidth=.7))
    ax.text(.67, height+.08, "Common clamped carrier", ha="center", va="center", fontsize=9)
    ax.text(.67, -.08, "Common clamped carrier", ha="center", va="center", fontsize=9)
    ax.text(.67, -.36, "Six narrow lanes; one textile part\nRadial tails and registration still need design",
            ha="center", va="center", fontsize=9)
    ax.set_xlim(-.1, 1.5)
    ax.set_ylim(-.56, 1.64)
    ax.set_title("B. One integrated web / sleeve", fontsize=12, pad=14)

    ax = axes[2]
    ratio = .65
    angle = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(angle), np.sin(angle), color="#66717b", linewidth=1.2)
    ax.plot(ratio*np.cos(angle), ratio*np.sin(angle), "--", color="#a6afb6", linewidth=1)
    phi = np.arange(6)*np.pi/3
    radii = np.where(np.arange(6) % 2 == 0, 1., ratio)
    points = radii[:, None]*np.column_stack((np.cos(phi), np.sin(phi)))
    for i in range(0, 6, 2):
        pair = points[i:i+2]
        ax.plot(*pair.T, color=GRAY, linewidth=15, solid_capstyle="butt", zorder=1)
    ax.scatter(*points[::2].T, color=BLUE, s=50, zorder=3)
    ax.scatter(*points[1::2].T, color=ORANGE, s=50, zorder=3)
    ax.text(1.12, 0, "R", va="center", fontsize=10)
    ax.text(-.12, .57, "r", fontsize=10, color="#66717b")
    ax.text(0, -1.23, "Paired roots register as one bridge\nSwap inner / outer roots on the next plate",
            ha="center", va="center", fontsize=9)
    ax.set_xlim(-1.3, 1.4)
    ax.set_ylim(-1.55, 1.3)
    ax.set_title("Plate registration · top view", fontsize=12, pad=14)

    fig.supxlabel("Gray: material fixed to a rigid plate. Colored lanes: independent tensile paths. Schematic only; finite-width folding is not proved.",
                  fontsize=9)
    folder = Path(__file__).parent
    fig.savefig(folder / "strap_concepts.png", dpi=180)
    fig.savefig(folder / "strap_concepts.svg")
    plt.close(fig)


if __name__ == "__main__":
    draw()
