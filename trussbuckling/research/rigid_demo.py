"""Visualize exact rigid folding. Run with -m trussbuckling.research.rigid_demo."""

import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .rigid_prism import cell_height, fold_limit, ideal_aspect, mesh, panel_lengths


def draw(ax, radius, sides, fraction, aspect=None):
    camera = (ax.elev, ax.azim)
    ax.clear()
    vertices, faces, cuts, theta = mesh(radius, sides, fraction, aspect)
    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors="#adc7db",
                                       edgecolors="#526a7c", alpha=.65, linewidths=.6))
    ax.add_collection3d(Line3DCollection(vertices[cuts], colors="#b6382e", linewidths=1.5))
    for ring in (vertices[:sides], vertices[sides:2*sides]):
        closed = np.vstack((ring, ring[0]))
        ax.plot(*closed.T, color="#152d40", linewidth=2)
    maximum = cell_height(radius, sides, aspect)
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(0, maximum),
           xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    ax.set_box_aspect((2*radius, 2*radius, maximum))
    ax.view_init(*camera)
    ax.set_title(f"h = {maximum*fraction:.2f} mm · twist {np.degrees(theta):.2f}°", fontsize=10)
    return vertices, faces, cuts, theta


def pattern(ax, sides, side, height):
    ax.clear()
    for k in range(sides):
        x = k * side
        ax.plot([x, x+side], [height, 0], color="#b6382e", linewidth=1.8)
        ax.plot([x, x+side], [0, height], "--", color="#266a98", linewidth=1.2)
        ax.plot([x, x], [0, height], "--", color="#266a98", linewidth=1.2)
        ax.text(x+side/2, -.16*side, str(k+1), ha="center", fontsize=9)
    ax.plot([0, sides*side, sides*side, 0, 0], [0, 0, height, height, 0], color="#152d40")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Faces {side:.2f} × {height:.2f} mm\nRed: full slit · Blue dashed: hinge", fontsize=10)


def snapshot(folder, radius, sides, aspect=None):
    folder.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 7), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=(3, 1))
    minimum, _ = fold_limit(sides, aspect)
    for k, position in enumerate((1., .7, .3, 0.)):
        ax = fig.add_subplot(grid[0, k], projection="3d")
        ax.view_init(25, -60)
        draw(ax, radius, sides, minimum + (1-minimum)*position, aspect)
        ax.set_axis_off()
    pattern(fig.add_subplot(grid[1, :]), sides, 2*radius*np.sin(np.pi/sides), cell_height(radius, sides, aspect))
    fig.suptitle(f"Exact rigid fold with diagonal slits · fixed outer radius {radius:g} mm · {sides} sides", fontsize=14)
    name = "triangular" if sides == 3 else "rigid"
    if aspect is not None and aspect != ideal_aspect(sides):
        name = "aspect"
    fig.savefig(folder / f"{name}_folding.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 3), layout="constrained")
    pattern(ax, sides, 2*radius*np.sin(np.pi/sides), cell_height(radius, sides, aspect))
    fig.savefig(folder / ("crease_pattern.svg" if name == "rigid" else f"{name}_pattern.svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=100.)
    parser.add_argument("--sides", type=int, default=4)
    parser.add_argument("--aspect", type=float, help="Initial height / face width; defaults to the ideal ratio.")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    aspect = ideal_aspect(args.sides) if args.aspect is None else args.aspect
    mesh(args.radius, args.sides, 1., aspect)
    if args.snapshot:
        snapshot(Path(__file__).parent, args.radius, args.sides, aspect)
        return
    os.environ["QT_API"] = "pyqt6"
    plt.switch_backend("qtagg")
    if not (3 <= args.sides <= 24 and 10 <= args.radius <= 250 and .1 <= aspect <= 2):
        parser.error("Viewer ranges: sides 3–24, radius 10–250 mm, aspect 0.1–2.")
    minimum, _ = fold_limit(args.sides, aspect)
    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title("Exact rigid prism with slits")
    view = fig.add_axes((.02, .32, .65, .60), projection="3d")
    view.view_init(25, -60)
    net = fig.add_axes((.69, .68, .29, .25))
    info = fig.text(.70, .34, "", fontsize=10, va="bottom")
    controls = [
        Slider(fig.add_axes((.20, .24, .68, .03)), "Height / maximum", minimum, 1,
               valinit=max(.7, minimum), valfmt="%.4f"),
        Slider(fig.add_axes((.20, .18, .68, .03)), "Radius (mm)", 10, 250, valinit=args.radius),
        Slider(fig.add_axes((.20, .12, .68, .03)), "Sides", 3, 24, valinit=args.sides, valstep=1),
        Slider(fig.add_axes((.20, .06, .48, .03)), "Aspect H / width", .1, 2,
               valinit=aspect, valfmt="%.5f"),
    ]
    reset = Button(fig.add_axes((.80, .048, .18, .055)), "Reset to ideal")

    def update(_=None):
        fraction, radius, sides = controls[0].val, controls[1].val, int(controls[2].val)
        aspect = controls[3].val
        minimum, reason = fold_limit(sides, aspect)
        controls[0].valmin = minimum
        controls[0].ax.set_xlim(minimum, 1)
        if fraction < minimum:
            fraction = minimum
            controls[0].eventson = False
            controls[0].set_val(fraction)
            controls[0].eventson = True
        vertices, faces, cuts, _ = draw(view, radius, sides, fraction, aspect)
        side = 2 * radius * np.sin(np.pi/sides)
        maximum = cell_height(radius, sides, aspect)
        original, _, _, _ = mesh(radius, sides, 1., aspect)
        expected = panel_lengths(original, faces)
        error = np.max(np.abs(panel_lengths(vertices, faces)/expected - 1))
        pattern(net, sides, side, maximum)
        status = {"flat": "Fully flat-foldable; final contact.",
                  "panel_contact": "Stops when side panels meet.",
                  "end_plane": "Stops at the end-plane boundary.\nCaps would contact here."}[reason]
        info.set_text(f"Ideal aspect: {ideal_aspect(sides):.5f}\n{status}\n"
                      f"Minimum height: {maximum*minimum:.2f} mm\n"
                      f"Maximum height: {maximum:.2f} mm\n"
                      f"Outer radius: {radius:.2f} mm\n"
                      f"Slit length: {np.hypot(side, maximum):.2f} mm\n\n"
                      f"Panel strain: exactly zero\nNumerical residual: {error:.1e}\n"
                      "Ideal hinges; zero thickness.")
        reset.color = ".85" if reason == "flat" else "#d7e8f4"
        reset.ax.set_facecolor(reset.color)
        fig.canvas.draw_idle()

    for control in controls:
        control.on_changed(update)
    reset.on_clicked(lambda _: controls[3].set_val(ideal_aspect(int(controls[2].val))))
    update()
    plt.show()


if __name__ == "__main__":
    main()
