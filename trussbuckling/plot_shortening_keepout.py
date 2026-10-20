"""Free-height truss shortening with a cylindrical keepout around the axis.

For gamma=90 degrees, the longest links have angular span theta+2*pi/n.
Their closest distance to the axis is R*cos((theta+2*pi/n)/2), at midheight.
The first keepout contact is therefore theta=2*acos(keepout/R)-2*pi/n.
This precedes link crossing. A negative angle means the reference geometry
already intersects the keepout and is excluded, not assigned zero travel.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import brentq

from .model import Parameters, state


def main():
    output = Path(__file__).parent / "plots"
    output.mkdir(exist_ok=True)
    colors = ("#2374AB", "#D2692E", "#26876D", "#8455A5")
    radii, styles, markers = (5, 7, 9), ("-", "--", ":"), ("o", "s", "^")
    gap, keepout = 2.0, 1.5  # mm; keepout is a radius, not a diameter
    xmax = 2.0
    optima = []
    gap_ends = {radius: gap / radius for radius in radii}
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.subplots_adjust(left=.09, right=.74, top=.85, bottom=.23)

    for n, color in zip((3, 4, 5, 6), colors):
        for radius, style, marker in zip(radii, styles, markers):
            twist = 2 * np.arccos(keepout / radius) - 2 * np.pi / n
            if twist < 0:
                print(f"Excluded R={radius}, n={n}: initial clearance {radius*np.cos(np.pi/n):.3f} mm")
                continue

            def parameters(ratio):
                return Parameters(radius=radius, length=radius * ratio,
                                  creases=n, gamma=np.pi / 2)

            # Exact transition between gap-limited and keepout-limited motion:
            # free axial force must vanish at the keepout twist and h=gap.
            peak = brentq(lambda ratio: state(parameters(ratio), twist, height=gap).force,
                          gap / radius, 4)
            ratios = np.sort(np.r_[np.linspace(gap / radius, xmax, 1001), peak])
            heights = np.array([state(parameters(ratio), twist).height for ratio in ratios])
            final_heights = np.maximum(gap, heights)
            shortening = 100 * (1 - final_heights / (radius * ratios))
            assert np.all(final_heights >= gap)
            after_peak = ratios >= peak
            ax.plot(ratios[after_peak], shortening[after_peak],
                    color=color, linestyle=style, linewidth=2)
            gap_ends[radius] = max(gap_ends[radius], peak)
            ax.plot(peak, 100 * (1 - gap / (radius * peak)), marker=marker,
                    color=color, markersize=5, zorder=4)
            optima.append((radius, n, peak, radius * peak, 100 * (1 - gap / (radius * peak))))
            print(f"R={radius}, n={n}: keepout at {np.degrees(twist):.3f} deg; "
                  f"peak {100*(1-gap/(radius*peak)):.3f}% at L/R={peak:.5f}")

    # The gap-limited portions are identical for all n at a given radius.
    for radius, style in zip(radii, styles):
        ratios = np.linspace(gap / radius, gap_ends[radius], 501)
        ax.plot(ratios, 100 * (1 - gap / (radius * ratios)),
                color="#606060", linestyle=style, linewidth=2, zorder=1)

    ax.set_title(r"1.5 mm radius keepout and 2 mm plate gap ($\gamma=90^\circ$)",
                 fontsize=15, loc="left", pad=18)
    ax.set_xlabel(r"Reference aspect ratio $L/R$", fontsize=12)
    ax.set_ylabel("Available shortening (%)", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, xmax + .01, .25))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    counts = [Line2D([], [], color=color, linewidth=2, label=fr"$n={n}$")
              for n, color in zip((3, 4, 5, 6), colors)]
    counts.append(Line2D([], [], color="#606060", linewidth=2, label="2 mm plate-gap limit"))
    sizes = [Line2D([], [], color="#454545", linestyle=style, marker=marker,
                   linewidth=2, label=fr"$R={radius}$ mm")
             for radius, style, marker in zip(radii, styles, markers)]
    legend = ax.legend(handles=counts, loc="upper left", bbox_to_anchor=(1.02, 1),
                       frameon=False, fontsize=11)
    ax.add_artist(legend)
    ax.legend(handles=sizes, loc="upper left", bbox_to_anchor=(1.02, .48),
              frameon=False, fontsize=11)
    fig.text(.1, .075,
             "Axially free straight-link model; residual strain allowed. Keepout applies to zero-thickness link centre lines.\n"
             "Gray: plate-gap constraint. Colored: centre-keepout constraint. Markers show peak shortening.",
             fontsize=9.5, color="#4b5563", linespacing=1.6)
    for suffix in ("png", "svg"):
        path = output / f"shortening_keepout1p5.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)
    print("\n| R (mm) | n | Optimal L/R | Initial L (mm) | Shortening (%) |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for radius, n, ratio, length, shortening in sorted(optima):
        print(f"| {radius} | {n} | {ratio:.4f} | {length:.3f} | {shortening:.2f} |")


if __name__ == "__main__":
    main()
