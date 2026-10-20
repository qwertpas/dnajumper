"""Plot the saved five-cord designs and export their normalized anchor coordinates."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .five_cords import Design, make_design, stiffness


def main():
    folder = Path(__file__).parent
    data = json.loads((folder/"five_designs.json").read_text())
    saved = {item["id"]:item for item in data["designs"]}
    models = {key:make_design(np.radians(item["angles_degrees"]), item["bottom_radii"], item["product"])
              for key,item in saved.items()}
    a = np.tile([1., .6], 3)
    reference = Design(np.arange(6)*np.pi/3, a, .6/a, np.ones(6)/6, .6)
    angles = np.linspace(0, 75, 151)
    curves = {key:np.array([stiffness(design, np.radians(x)) for x in angles])
              for key,design in models.items()}
    reference_curve = np.array([stiffness(reference, np.radians(x)) for x in angles])
    colors = {"clearance":"#2166ac", "stiff":"#8c7b9b", "balanced":"#bd6b23",
              "regular":"#388360", "free":"#a65a64"}
    labels = {"clearance":"Three radii, more clearance", "stiff":"Three radii, closer cords",
              "balanced":"Balanced cord loading", "regular":"Regular pentagon", "free":"Unrestricted numerical best"}
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=(1, 1.1))
    circle = np.linspace(0, 2*np.pi, 200)
    for column, key in enumerate(("clearance", "balanced", "regular")):
        axis = fig.add_subplot(grid[0, 2*column:2*column+2])
        design = models[key]
        for radius in (1., .6):
            axis.plot(radius*np.cos(circle), radius*np.sin(circle), color="#b4b9bf",
                      linestyle="-" if radius == 1 else "--", linewidth=.8)
        for index, (angle, lower, upper) in enumerate(zip(design.angles, design.bottom, design.top)):
            unit = np.array([np.cos(angle), np.sin(angle)])
            segment = np.array([lower*unit, upper*unit])
            axis.plot(*segment.T, color="#bcc3ca", linewidth=1.5)
            axis.scatter(*(lower*unit), color="#2166ac", s=50, zorder=3)
            axis.scatter(*(upper*unit), facecolors="none", edgecolors="#c27326", s=85, zorder=4, linewidths=1.6)
            axis.text(*(1.12*unit), str(index+1), ha="center", va="center", fontsize=10)
        axis.scatter([], [], color="#2166ac", s=35, label="Bottom anchors")
        axis.scatter([], [], facecolors="none", edgecolors="#c27326", s=65, label="Top anchors at zero twist")
        axis.set(xlim=(-1.24, 1.24), ylim=(-1.24, 1.24), aspect="equal")
        axis.set_title(labels[key], color=colors[key], fontsize=12)
        axis.set_axis_off()
        if column == 0:
            axis.legend(loc="lower center", bbox_to_anchor=(.5, -.08), frameon=False, fontsize=9)
        else:
            gap = saved[key]["minimum_cord_centerline_gap_over_radius"]
            axis.text(.5, -.02, f"Minimum cord gap: {gap:.4f} R", transform=axis.transAxes,
                      ha="center", fontsize=9, color="#555555")
    axis = fig.add_subplot(grid[1, :4])
    axis.plot(angles, reference_curve, "--", color="#222222", linewidth=1.6, label="Six-cord reference")
    for key in ("free", "stiff", "clearance", "balanced", "regular"):
        axis.plot(angles, curves[key], color=colors[key], label=labels[key],
                  linewidth=2 if key == "clearance" else 1.2,
                  linestyle=":" if key in ("free", "stiff") else "-")
    axis.set(xlabel="Twist per cell (degrees)", ylabel="Weakest alignment stiffness / (F R)",
             xlim=(0, 75), ylim=(0, 13))
    axis.grid(alpha=.18)
    axis.legend(fontsize=8.5, frameon=False, ncol=2)
    axis.set_title("12 cells; height and internal twist allowed to relax", loc="left", fontsize=11)
    axis = fig.add_subplot(grid[1, 4:])
    order = ("free", "stiff", "clearance", "balanced", "regular")
    scores = [100*saved[key]["worst_alignment_stiffness"]/reference_curve.min() for key in order]
    positions = np.arange(5)
    axis.barh(positions, scores, color=[colors[key] for key in order], height=.6)
    axis.set_yticks(positions, ["Irregular best", "Three radii", "More clearance", "Balanced", "Pentagon"])
    axis.invert_yaxis()
    axis.set(xlim=(0, 105), xlabel="Worst stiffness / six-cord reference (%)")
    for row, score in enumerate(scores):
        axis.text(score+1.5, row, f"{score:.1f}%", va="center", fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Same total axial rigidity ΣEA", fontsize=11)
    fig.suptitle("Five-cord designs · radius range 0.6 R to R", fontsize=16)
    fig.supxlabel("H₀/R = √1.2; ΣEA/F = 6000; equal EA per cord; local prestressed model, 0–75° stroke.", fontsize=10)
    fig.savefig(folder/"five_designs.png", dpi=180)
    fig.savefig(folder/"five_designs.svg")
    plt.close(fig)
    with (folder/"five_anchors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["design", "cord", "angle_degrees", "bottom_radius_over_R", "top_radius_over_R",
                         "bottom_x_over_R", "bottom_y_over_R", "top_x_over_R", "top_y_over_R", "cord_length_over_R"])
        for key, design in models.items():
            for index, angle in enumerate(design.angles):
                a, b = design.bottom[index], design.top[index]
                writer.writerow([key, index+1, np.degrees(angle), a, b, a*np.cos(angle), a*np.sin(angle),
                                 b*np.cos(angle), b*np.sin(angle), np.sqrt(1.2+(a-b)**2)])
    print(folder/"five_designs.png")


if __name__ == "__main__":
    main()
