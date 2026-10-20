"""Shortening before link intersection or a 2 mm gap, gamma = 90 degrees.

For the equal-radius Hunt truss, the long links first meet on the axis at
theta = pi - 2*pi/n. At that twist the free-height energy minimum is
h_cross = R*sqrt(max(0, (L/R)**2 - 4*cos(pi/n)**2)).
Before then, links of each family have distinct points at every interior
height. Cross-family intersections require theta=pi-pi/n, which is later.
Stop earlier if the plate gap reaches 2 mm. Members are straight, have zero
thickness, and can carry compression; this does not model loose cord shapes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    output = Path(__file__).parent / "plots"
    output.mkdir(exist_ok=True)
    colors = ("#2374AB", "#D2692E", "#26876D", "#8455A5")
    radii = (5, 7, 9)
    styles = ("-", "--", ":")
    markers = ("o", "s", "^")
    gap = 2.0
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=.1, right=.97, top=.85, bottom=.23)

    for n, color in zip((3, 4, 5, 6), colors):
        width = 2 * np.cos(np.pi / n)
        for radius, style, marker in zip(radii, styles, markers):
            minimum = gap / radius
            peak = np.sqrt(width**2 + minimum**2)
            ratios = np.sort(np.r_[np.linspace(minimum, 4, 1601), peak])
            # h=0 is the constrained energy minimum if crossing would otherwise
            # require imaginary height. The positive gap is reached first then.
            at_crossing = np.sqrt(np.maximum(0, ratios**2 - width**2))
            final_height = np.maximum(minimum, at_crossing)  # h/R
            shortening = 100 * (1 - final_height / ratios)
            ax.plot(ratios, shortening, color=color, linestyle=style, linewidth=2)
            ax.plot(peak, 100 * (1 - minimum / peak), marker=marker,
                    color=color, markersize=5)

    ax.set_title(r"Shortening before first crossing or a 2 mm plate gap ($\gamma=90^\circ$)",
                 fontsize=15, loc="left", pad=18)
    ax.set_xlabel(r"Reference aspect ratio $L/R$", fontsize=12)
    ax.set_ylabel("Available shortening (%)", fontsize=12)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 4.01, .5))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    counts = [Line2D([], [], color=color, linewidth=2, label=fr"$n={n}$")
              for n, color in zip((3, 4, 5, 6), colors)]
    sizes = [Line2D([], [], color="#454545", linestyle=style, marker=marker,
                   linewidth=2, label=fr"$R={radius}$ mm")
             for radius, style, marker in zip(radii, styles, markers)]
    legend = ax.legend(handles=counts, loc="upper right", frameon=False, fontsize=11)
    ax.add_artist(legend)
    ax.legend(handles=sizes, loc="center right", bbox_to_anchor=(1, .58),
              frameon=False, fontsize=11)
    fig.text(.1, .075,
             "Axially free truss path; residual strain allowed. Markers show maxima; curves merge when crossing limits travel.\n"
             "Ideal straight links with zero thickness. Finite cord diameter or slack can cause earlier contact.",
             fontsize=9.5, color="#4b5563", linespacing=1.6)
    for suffix in ("png", "svg"):
        path = output / f"shortening_before_crossing.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
