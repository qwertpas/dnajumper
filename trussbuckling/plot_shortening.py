"""Plot shortening to the second zero-strain state for gamma = 90 degrees."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path(__file__).parent / "plots"
    output.mkdir(exist_ok=True)
    colors = ("#2374AB", "#D2692E", "#26876D", "#8455A5")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=.11, right=.97, bottom=.19, top=.88)

    for n, color in zip((3, 4, 5, 6), colors):
        minimum = 2 * np.cos(np.pi / n)
        ratio = minimum + (4 - minimum) * np.linspace(0, 1, 801)**2
        # Only plot the real domain; a negative radicand is not a flat state.
        height_ratio = np.sqrt(np.maximum(0, 1 - (minimum / ratio)**2))
        shortening = 100 * (1 - height_ratio)
        ax.plot(ratio, shortening, color=color, linewidth=2.4,
                label=fr"$n={n}$   ($L/R\geq {minimum:.3f}$)")
        ax.plot(minimum, 100, "o", color=color, markersize=6)
        ax.vlines(minimum, 0, 100, color=color, alpha=.24, linewidth=1,
                  linestyles="dashed")

    ax.set_title(r"Final shortening at $\gamma=90^\circ$", loc="left",
                 fontsize=17, pad=20)
    ax.set_xlabel(r"Reference aspect ratio $L/R$", fontsize=12)
    ax.set_ylabel(r"Final shortening $100(1-h_f/L)$ (%)", fontsize=12)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 105)
    ax.set_xticks(np.arange(0, 4.01, .5))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11,
              title="Second zero-strain state", title_fontsize=11)
    fig.text(.11, .065,
             "100% = flat. Each curve starts at its smallest feasible aspect ratio.\n"
             "Below that threshold, the second zero-strain state does not exist.",
             fontsize=10, color="#4b5563", linespacing=1.5)
    for suffix in ("png", "svg"):
        path = output / f"shortening_gamma90.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
