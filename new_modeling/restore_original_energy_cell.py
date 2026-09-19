from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell


NOTEBOOK = Path(__file__).with_name("simpleradiusvoltage.ipynb")
CELL_ID = "original-energy-sweep"

SOURCE = """# Original 4 V energy sweep from the earlier notebook version.
# Keep these names separate so this cell does not change other model sweeps.
from pathlib import Path

ORIGINAL_ENERGY_VOLTAGE = 4.0
ORIGINAL_ENERGY_INERTIA_KG_M2 = 1.82396e-5
ORIGINAL_ENERGY_FOOT_MASS_KG = 0.0
original_energy_masses_kg = np.linspace(0.035, 0.200, 15)
original_energy_radii_m = [0.005, 0.007, 0.009]
original_energy_colors = dict(zip(
    original_energy_radii_m,
    plt.cm.rainbow(np.linspace(0.08, 0.9, len(original_energy_radii_m))),
))
original_energy_rows = []

for radius_m in original_energy_radii_m:
    for body_mass_kg in original_energy_masses_kg:
        total_mass_kg = body_mass_kg + ORIGINAL_ENERGY_FOOT_MASS_KG
        jump_height_m = simulate_jump_height(
            voltage=ORIGINAL_ENERGY_VOLTAGE,
            radius_m=radius_m,
            inertia=ORIGINAL_ENERGY_INERTIA_KG_M2,
            torque_cap_nm=TORQUE_CAP_NM,
            body_mass=body_mass_kg,
            total_mass_kg=total_mass_kg,
        )
        takeoff_velocity_m_s = np.sqrt(2 * params['g'] * jump_height_m)
        kinetic_energy_j = 0.5 * total_mass_kg * takeoff_velocity_m_s**2
        original_energy_rows.append({
            'radius_mm': 1000 * radius_m,
            'body_mass_kg': body_mass_kg,
            'jump_height_m': jump_height_m,
            'takeoff_velocity_m_s': takeoff_velocity_m_s,
            'kinetic_energy_j': kinetic_energy_j,
        })

original_energy_results = pd.DataFrame(original_energy_rows)
fig_original_energy, ax_original_energy = plt.subplots(figsize=(8.5, 5.5))
for radius_m in original_energy_radii_m:
    radius_data = original_energy_results[np.isclose(
        original_energy_results['radius_mm'], 1000 * radius_m
    )]
    ax_original_energy.plot(
        radius_data['body_mass_kg'],
        radius_data['kinetic_energy_j'],
        'o-',
        color=original_energy_colors[radius_m],
        linewidth=2,
        label=f'r={1000 * radius_m:.0f} mm',
    )

ax_original_energy.set_xlabel('body mass (kg)')
ax_original_energy.set_ylabel('energy J')
ax_original_energy.set_title(
    f'Jump height vs body mass at {ORIGINAL_ENERGY_VOLTAGE:.0f} V\\n'
    f'foot mass fixed at {1000 * ORIGINAL_ENERGY_FOOT_MASS_KG:.0f} g; '
    f'body mass {1000 * original_energy_masses_kg[-1]:.0f} g'
)
ax_original_energy.grid(alpha=0.3)
ax_original_energy.legend()
fig_original_energy.tight_layout()

original_energy_plot_path = Path('simple_1d_energy_vs_body_mass_4v.png')
fig_original_energy.savefig(original_energy_plot_path, dpi=180)
original_energy_plot_path
"""


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = new_code_cell(SOURCE, id=CELL_ID)
    existing = next(
        (index for index, item in enumerate(notebook.cells) if item.get("id") == CELL_ID),
        None,
    )
    if existing is None:
        insert_at = max(len(notebook.cells) - 1, 0)
        notebook.cells.insert(insert_at, cell)
    else:
        notebook.cells[existing] = cell
    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
