# Radius selection studies

Start with the [motor sweep](motor_sweep/README.md) for the latest design recommendation: 100 g and 15 W is more tolerant of uncertain shaft inertia than the lighter alternatives. The mass sweeps answer a different question: how the best radius changes with body mass for a fixed motor curve.

| Study | Current source | Main evidence |
| --- | --- | --- |
| Equal power motors and inertia uncertainty | [Motor sweep guide](motor_sweep/README.md) | [Motor curves and heights](motor_sweep/results/motor_curves_and_heights.png), [inertia sensitivity](motor_sweep/results/inertia_sensitivity.png) |
| Mass sweep with a 0.21 m design floor | [Notebook](mass_sweep/radius_experiment.ipynb) | [Energy comparison](mass_sweep/results/radius_energy_comparison.png), [motor comparison](mass_sweep/results/radius_motor_comparison.png) |
| Mass sweep at two inertias with a 0.15 m floor | [Notebook](inertia_comparison/full_stroke_radius_experiment.ipynb), [optimizer](inertia_comparison/optimize.py) | [Combined figure](inertia_comparison/results/full_stroke_radius_final.png), [unrounded optimization](inertia_comparison/results/optimization.json) |
| Original length and mass exploration | [Notebook](length_sweep/simpleradiusvoltage2.ipynb) | [Energy](length_sweep/results/energy_J.png), [body power](length_sweep/results/avg_power_W.png), [motor power](length_sweep/results/avg_motor_power_W.png) |

The first mass sweep comes from **Add an L sweep to the plot**; the two-inertia comparison comes from **Add an L sweep to the plot (2)**. The equal-power study comes from **Design a radius separation study**. The original length notebook is retained as useful supporting work, with its latest saved parameters: 6 V linear motor, lengths 154 and 180 mm, fixed 49 mm dead length, radii 5/7/9 mm, and masses 20–220 g. It uses a different geometry constraint from the later studies.

The later studies use 180 mm total length, dead/total = 49.5/157.5, zero foot mass, and continuous drive through the full stroke. Plotted kinetic energy is body energy only. The models include rotor inertia but omit transmission losses and compliance; their motor targets require calibration. The 7 mm winning region is small and does not establish a separate 10% advantage.

Each study keeps source at the top of its folder and reproducible evidence in `results/`. Run notebook cells from their notebook's folder; they regenerate figures and CSVs there. Python scripts use paths relative to their source. Use Python 3.11 with NumPy, SciPy, pandas, Matplotlib, Numba, and Jupyter (`ipympl` for the original notebook's widget backend).

Superseded 200 mm previews, plot-layout experiments, pulse proposals, old recommendations, redundant plots, and temporary previews were moved to macOS Trash. No retained source or evidence depends on a temporary directory. Earlier unrelated modeling and measured data remain in [new_modeling](../../new_modeling/).
