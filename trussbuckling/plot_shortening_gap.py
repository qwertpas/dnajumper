"""Compare the requested max expression with a physical 2 mm gap limit."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = ("#2374AB", "#D2692E", "#26876D", "#8455A5")
GAP = 2.0  # mm


def draw(ax, radius):
    for n, color in zip((3, 4, 5, 6), COLORS):
        minimum = 2 * np.cos(np.pi / n)
        peak = np.sqrt(minimum**2 + (GAP / radius)**2)
        ratio = np.sort(np.append(
            minimum + (4 - minimum) * np.linspace(0, 1, 1001)**2, peak
        ))
        length = radius * ratio
        geometry = 1 - np.sqrt(np.maximum(0, 1 - (minimum / ratio)**2))
        gap_limit = 1 - GAP / length
        allowed = np.minimum(geometry, gap_limit)
        requested = np.maximum(geometry, gap_limit)
        ax.plot(ratio, 100 * requested, color=color, linewidth=1.7,
                linestyle=(0, (5, 3)), alpha=.65)
        ax.plot(ratio, 100 * allowed, color=color, linewidth=2.5)
        ax.plot(peak, 100 * (1 - GAP / (radius * peak)), "o",
                color=color, markersize=5)
        # Independent check: the capped result must leave at least GAP mm.
        np.testing.assert_allclose(length * (1 - allowed),
                                   np.maximum(length * (1 - geometry), GAP), atol=1e-12)
        assert np.all(length * (1 - allowed) >= GAP - 1e-12)

    ax.set_title(fr"$R={radius}$ mm", fontsize=15, pad=12)
    ax.set_xlim(.8, 4)
    ax.set_ylim(0, 105)
    ax.set_xticks((1, 1.5, 2, 2.5, 3, 3.5, 4))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel(r"Reference aspect ratio $L/R$", fontsize=11)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def finish(fig, axes, radii, output, name):
    for ax, radius in zip(axes, radii):
        draw(ax, radius)
    axes[0].set_ylabel("Final shortening (%)", fontsize=12)
    fig.suptitle(r"Final shortening at $\gamma=90^\circ$, with a 2 mm plate gap", fontsize=17, y=.98)
    handles = [Line2D([], [], color=color, linewidth=2.5, label=fr"$n={n}$")
               for n, color in zip((3, 4, 5, 6), COLORS)]
    handles += [
        Line2D([], [], color="#444444", linewidth=2.5, label="Minimum gap enforced (min)"),
        Line2D([], [], color="#777777", linewidth=1.7, linestyle="--", label="Requested expression (max)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=6 if len(axes) > 1 else 3, frameon=False, fontsize=10)
    fig.text(.075, .075,
             "Solid curves stop at the 2 mm gap or the zero-strain endpoint, whichever comes first; dots mark their peaks.\n"
             "Dashed curves use the requested max expression. Stops limited by the gap can retain bar strain.",
             fontsize=10, color="#4b5563", linespacing=1.6)
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


def main():
    output = Path(__file__).parent / "plots"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.78, bottom=.23, wspace=.13)
    finish(fig, axes, (5, 7, 9), output, "shortening_gap2")
    for radius in (5, 7, 9):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.subplots_adjust(left=.1, right=.97, top=.73, bottom=.23)
        finish(fig, [ax], [radius], output, f"shortening_gap2_r{radius}")


if __name__ == "__main__":
    main()
