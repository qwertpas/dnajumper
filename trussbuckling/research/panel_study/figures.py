"""Geometry of three continuous textile panels; carrier wrinkles are omitted."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .reciprocal import build, metrics


ROOT = Path(__file__).parent
COLORS = ('#147d92', '#c96538', '#7f62a3')


def draw(ax, angle, cells):
    theta = np.deg2rad(angle)
    h = np.sqrt(1.3*np.cos(theta))
    r = np.linspace(.65, 1, 33)
    t = np.linspace(0, 1, 16)
    for j in range(cells + 1):
        phi = np.linspace(0, 2*np.pi, 121)
        circle = np.column_stack((np.cos(phi), np.sin(phi), np.full(len(phi), j*h)))
        ax.add_collection3d(Poly3DCollection([circle], facecolor='#cad1d6',
                                           alpha=.15, edgecolor='#929ea6', linewidth=.5))
        for k in range(3):
            phi = k*2*np.pi/3 + j*theta
            seam = np.array([[.65*np.cos(phi), .65*np.sin(phi), j*h],
                             [np.cos(phi), np.sin(phi), j*h]])
            ax.plot(*seam.T, color=COLORS[k], lw=3)
    for j in range(cells):
        a, b = (r, .65/r) if j % 2 == 0 else (.65/r, r)
        for k in range(3):
            phi = k*2*np.pi/3 + j*theta
            lower = np.column_stack((a*np.cos(phi), a*np.sin(phi), np.full(len(r), j*h)))
            upper = np.column_stack((b*np.cos(phi+theta), b*np.sin(phi+theta), np.full(len(r), (j+1)*h)))
            surface = lower[:, None, :]*(1-t)[None, :, None] + upper[:, None, :]*t[None, :, None]
            ax.plot_surface(*[surface[:, :, c] for c in range(3)], color=COLORS[k],
                            alpha=.6, linewidth=0, shade=False)
            for index in (0, 8, 16, 24, 32):
                fiber = np.array([lower[index], upper[index]])
                ax.plot(*fiber.T, color=COLORS[k], lw=.75, alpha=.95)
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-.08, cells*h+.12))
    ax.set_box_aspect((2.2, 2.2, max(1., cells*h+.2)))
    ax.view_init(elev=19, azim=-65)
    ax.set_axis_off()


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig = plt.figure(figsize=(14, 6), facecolor='white')
    grid = fig.add_gridspec(2, 4, width_ratios=(1.4, 1, 1, 1), height_ratios=(1, .14))
    ax = fig.add_subplot(grid[:, 0], projection='3d')
    draw(ax, 45, 3)
    ax.set_title('Three continuous straps\n3 cells at 45°', pad=8)
    for i, angle in enumerate((15, 45, 75)):
        ax = fig.add_subplot(grid[0, i+1], projection='3d')
        draw(ax, angle, 1)
        ax.set_title(f'{angle}° per cell\nh/R = {np.sqrt(1.3*np.cos(np.deg2rad(angle))):.3f}', pad=4)
    ax = fig.add_subplot(grid[1, 1:])
    ax.axis('off')
    ax.text(.5, .9, 'Each yarn: radius r → 0.65R²/r → r at successive plates', ha='center', va='center')
    ax.text(.5, .05, 'Ideal straight-yarn surfaces; transverse gathering and clamp thickness omitted',
            ha='center', va='center', fontsize=10, color='#4f5e68')
    fig.subplots_adjust(left=.02, right=.98, bottom=.08, top=.9, wspace=.03, hspace=.02)
    fig.savefig(ROOT/'panel_geometry.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    angles = np.linspace(15, 85, 71)
    for edge_only, label, color in ((False, 'Rigidity spread across panel', COLORS[0]),
                                   (True, 'Same total rigidity in two edges', COLORS[1])):
        values = [metrics(build(np.deg2rad(a), .01, points=16,
                                edge_only=edge_only))['minimum_tilt'] for a in angles]
        ax.plot(angles, values, color=color, label=label, lw=2)
    ax.axhline(0, color='#555555', lw=.8)
    ax.plot([75.6119036, 81.6943908], [0, 0], 'o', color='#333333', ms=4)
    ax.annotate('75.6°', (75.6119036, 0), xytext=(65, -.003),
                arrowprops=dict(arrowstyle='-', color='#555555'))
    ax.annotate('81.7°', (81.6943908, 0), xytext=(77, .003),
                arrowprops=dict(arrowstyle='-', color='#555555'))
    ax.set(xlabel='Twist per cell (degrees)',
           ylabel='Smallest transverse eigenvalue\nnormalized by EA per strap × R',
           title='12 cells · F = 10 N · EA per strap = 1000 N',
           xlim=(15, 85), ylim=(-.012, .006))
    ax.text(18, .0003, 'Positive: locally stable', fontsize=10)
    ax.text(18, -.0012, 'Negative: tilt instability', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', frameon=False, fontsize=10)
    fig.savefig(ROOT/'panel_stability.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
