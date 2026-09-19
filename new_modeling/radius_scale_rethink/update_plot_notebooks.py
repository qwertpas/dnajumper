from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[2]
SHARED_NOTEBOOK = ROOT / "data" / "plot_tracking_results.ipynb"
R7_NOTEBOOK = ROOT / "data" / "8-24-26_r7" / "vids_and_motor" / "plot_tracking_results.ipynb"


def find_root_cell():
    return new_code_cell(
        """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

root = next(
    path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / 'new_modeling/radius_scale_rethink/results/camera_scale.csv').exists()
)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
results_dir = root / 'new_modeling/radius_scale_rethink/results'
calibration = pd.read_csv(results_dir / 'camera_scale.csv').set_index('radius_mm')
calibration"""
    )


def filter_cell():
    return new_code_cell(
        """def filter_track(track, window, order):
    time = track['capture_time_s'].to_numpy()
    position = track['raw_position_m'].to_numpy()
    max_window = len(position) if len(position) % 2 else len(position) - 1
    use_window = min(window if window % 2 else window + 1, max_window)
    if use_window <= order:
        raise ValueError(f'Filter window {use_window} must exceed order {order}')
    dt = np.median(np.diff(time))
    filtered_position = savgol_filter(position, use_window, order)
    filtered_velocity = savgol_filter(position, use_window, order, deriv=1, delta=dt)
    return time, position, filtered_position, filtered_velocity
"""
    )


def update_shared():
    notebook = nbformat.read(SHARED_NOTEBOOK, as_version=4)
    notebook.cells = [
        new_markdown_cell(
            """# Camera-calibrated tracking results

Maximum filtered velocity and kinetic energy versus payload mass for all pulley radii. Positions are calibrated independently for each camera setup from the fixture's known 46 mm height. The original tracking CSVs are not modified."""
        ),
        find_root_cell(),
        new_markdown_cell(
            """## Controls

Change these values and rerun the notebook. `TRANSMISSION_SPEED_SQUARED_FACTOR` is the fraction of transmission mass contributing to kinetic energy at the tracked carrier speed. The spring-chain uniform-twist estimate is approximately 1/3."""
        ),
        new_code_cell(
            """FILTER_WINDOW = 9
FILTER_ORDER = 2
CARRIER_MASS_G = 10.0
TRANSMISSION_MASS_G = {5: 7.5, 7: 7.5, 9: 7.5, 11: 7.5}
CHAIN_ELEMENTS = 37
TRANSMISSION_SPEED_SQUARED_FACTOR = (2 * CHAIN_ELEMENTS - 1) / (6 * CHAIN_ELEMENTS)
RADIUS_COLORS = {5: 'C0', 7: 'C1', 9: 'C2', 11: 'C3'}

TRANSMISSION_SPEED_SQUARED_FACTOR"""
        ),
        filter_cell(),
        new_code_cell(
            """tracking_root = results_dir / 'tracking_csv'
folders = {radius: tracking_root / f'r{radius}' for radius in (5, 7, 9, 11)}
rows = []
tracks = {}

for radius, folder in folders.items():
    for path in sorted(folder.glob(f'r{radius}_*g_tracking.csv')):
        payload_mass_g = float(path.stem.split('_')[1][:-1])
        track = pd.read_csv(path)
        time, raw_position, filtered_position, filtered_velocity = filter_track(
            track, FILTER_WINDOW, FILTER_ORDER
        )
        effective_mass_kg = (
            payload_mass_g
            + CARRIER_MASS_G
            + TRANSMISSION_SPEED_SQUARED_FACTOR * TRANSMISSION_MASS_G[radius]
        ) / 1000
        full_mass_kg = (
            payload_mass_g + CARRIER_MASS_G + TRANSMISSION_MASS_G[radius]
        ) / 1000
        max_velocity = float(filtered_velocity.max())
        rows.append({
            'radius_mm': radius,
            'payload_mass_g': payload_mass_g,
            'max_velocity_m_s': max_velocity,
            'effective_kinetic_mass_kg': effective_mass_kg,
            'max_kinetic_energy_j': 0.5 * effective_mass_kg * max_velocity**2,
            'full_mass_energy_j': 0.5 * full_mass_kg * max_velocity**2,
            'pixels_per_meter': calibration.loc[radius, 'pixels_per_meter'],
        })
        tracks[(radius, payload_mass_g)] = (time, raw_position, filtered_position, filtered_velocity)

summary = pd.DataFrame(rows).sort_values(['radius_mm', 'payload_mass_g']).reset_index(drop=True)
summary"""
        ),
        new_markdown_cell("## Maximum filtered velocity versus payload mass"),
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(9, 5.5))
for radius, group in summary.groupby('radius_mm'):
    ax.plot(group['payload_mass_g'], group['max_velocity_m_s'], 'o-',
            color=RADIUS_COLORS[radius], label=f'r{radius}')
ax.set(xlabel='Payload mass (g)', ylabel='Maximum filtered velocity (m/s)',
       title='Maximum velocity vs payload mass (46 mm camera calibration)')
ax.grid(alpha=0.3)
ax.legend(title='Pulley radius (mm)')
fig.tight_layout()
velocity_plot = results_dir / 'corrected_velocity_vs_mass.png'
fig.savefig(velocity_plot, dpi=180)
plt.show()
velocity_plot"""
        ),
        new_markdown_cell("## Maximum kinetic energy versus payload mass"),
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(9, 5.5))
for radius, group in summary.groupby('radius_mm'):
    ax.plot(group['payload_mass_g'], group['max_kinetic_energy_j'], 'o-',
            color=RADIUS_COLORS[radius], label=f'r{radius}')
ax.set(xlabel='Payload mass (g)', ylabel='Maximum kinetic energy (J)',
       title='Maximum kinetic energy vs payload mass (46 mm camera calibration)')
ax.grid(alpha=0.3)
ax.legend(title='Pulley radius (mm)')
fig.tight_layout()
ke_plot = results_dir / 'corrected_ke_vs_mass.png'
fig.savefig(ke_plot, dpi=180)
ke_limits = ax.get_ylim()
plt.show()
ke_plot"""
        ),
        new_markdown_cell(
            """## Model kinetic energy versus payload mass

This uses the isolated-motor 4 V torque-speed curve, spring-chain effective mass and inertia, gravity, and one common fitted downstream loss torque for every radius. Colors and axis limits match the measured-data plot above."""
        ),
        new_code_cell(
            """from new_modeling.radius_scale_rethink.analyze import model_velocity

loss_fit = pd.read_csv(results_dir / 'transmission_loss_fit.csv')
common_loss_nm = float(loss_fit.loc[loss_fit['scope'] == 'common', 'friction_nm'].iloc[0])
model_rows = []
for radius in (5, 7, 9, 11):
    payload_mass_g = summary.loc[summary['radius_mm'] == radius, 'payload_mass_g'].to_numpy()
    velocity = model_velocity(payload_mass_g, radius, common_loss_nm)
    effective_mass_kg = (
        payload_mass_g + CARRIER_MASS_G
        + TRANSMISSION_SPEED_SQUARED_FACTOR * TRANSMISSION_MASS_G[radius]
    ) / 1000
    for mass, speed, kinetic_mass, energy in zip(
        payload_mass_g, velocity, effective_mass_kg, 0.5 * effective_mass_kg * velocity**2
    ):
        model_rows.append({
            'radius_mm': radius,
            'payload_mass_g': mass,
            'model_velocity_m_s': speed,
            'effective_kinetic_mass_kg': kinetic_mass,
            'model_max_kinetic_energy_j': energy,
            'common_loss_nm': common_loss_nm,
        })

model_summary = pd.DataFrame(model_rows)
model_summary.to_csv(results_dir / 'model_ke_vs_mass.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
for radius, group in model_summary.groupby('radius_mm'):
    ax.plot(group['payload_mass_g'], group['model_max_kinetic_energy_j'], 'o-',
            color=RADIUS_COLORS[radius], label=f'r{radius}')
ax.set(xlabel='Payload mass (g)', ylabel='Maximum kinetic energy (J)',
       title=f'Fixed-motor TSA model: kinetic energy vs payload mass at 4 V\\n'
             f'common downstream loss = {common_loss_nm:.3f} N m')
ax.set_ylim(ke_limits)
ax.grid(alpha=0.3)
ax.legend(title='Pulley radius (mm)')
fig.tight_layout()
model_ke_plot = results_dir / 'model_ke_vs_mass.png'
fig.savefig(model_ke_plot, dpi=180)
plt.show()
model_ke_plot"""
        ),
        new_markdown_cell(
            """## Original lumped 1D model, no friction

This reproduces the original simple model: the payload, carrier, and entire transmission are one translating mass; rotational inertia is fixed at $5\\times10^{-5}$ kg m²; gravity and the isolated-motor 4 V torque-speed curve remain; friction is zero."""
        ),
        new_code_cell(
            """from new_modeling.fit_tracking_model import model_velocity as simple_model_velocity

SIMPLE_INERTIA_KG_M2 = 5e-5
SIMPLE_TORQUE_CAP_NM = 0.260
simple_rows = []
for radius in (5, 7, 9, 11):
    payload_mass_g = summary.loc[summary['radius_mm'] == radius, 'payload_mass_g'].to_numpy()
    moving_mass_kg = (
        payload_mass_g + CARRIER_MASS_G + TRANSMISSION_MASS_G[radius]
    ) / 1000
    velocity = simple_model_velocity(
        moving_mass_kg, radius, SIMPLE_INERTIA_KG_M2, SIMPLE_TORQUE_CAP_NM
    )
    for mass, speed, moving_mass, energy in zip(
        payload_mass_g, velocity, moving_mass_kg, 0.5 * moving_mass_kg * velocity**2
    ):
        simple_rows.append({
            'radius_mm': radius,
            'payload_mass_g': mass,
            'model_velocity_m_s': speed,
            'moving_mass_kg': moving_mass,
            'model_max_kinetic_energy_j': energy,
            'inertia_kg_m2': SIMPLE_INERTIA_KG_M2,
            'torque_cap_nm': SIMPLE_TORQUE_CAP_NM,
            'friction_nm': 0.0,
        })

simple_summary = pd.DataFrame(simple_rows)
simple_summary.to_csv(results_dir / 'simple_1d_no_friction_ke_vs_mass.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
for radius, group in simple_summary.groupby('radius_mm'):
    ax.plot(group['payload_mass_g'], group['model_max_kinetic_energy_j'], 'o-',
            color=RADIUS_COLORS[radius], label=f'r{radius}')
ax.set(xlabel='Payload mass (g)', ylabel='Maximum kinetic energy (J)',
       title='Original lumped 1D model: kinetic energy vs payload mass at 4 V\\n'
             'no friction; I = 5e-5 kg m²')
ax.set_ylim(0.95 * simple_summary['model_max_kinetic_energy_j'].min(),
            1.04 * simple_summary['model_max_kinetic_energy_j'].max())
ax.grid(alpha=0.3)
ax.legend(title='Pulley radius (mm)')
fig.tight_layout()
simple_ke_plot = results_dir / 'simple_1d_no_friction_ke_vs_mass.png'
fig.savefig(simple_ke_plot, dpi=180)
plt.show()
simple_ke_plot"""
        ),
        new_markdown_cell("## All filtered velocity traces"),
        new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False, sharey=True)
for ax, radius in zip(axes.flat, (5, 7, 9, 11)):
    for (track_radius, mass), (time, _, _, velocity) in tracks.items():
        if track_radius == radius:
            ax.plot(time - time[0], velocity, alpha=0.75, label=f'{mass:g} g')
    ax.set_title(f'r{radius}')
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=0.25)
axes[0, 0].set_ylabel('Filtered velocity (m/s)')
axes[1, 0].set_ylabel('Filtered velocity (m/s)')
axes[0, 1].legend(ncol=2, fontsize=7, title='Payload', bbox_to_anchor=(1.02, 1), loc='upper left')
fig.suptitle('All calibrated filtered velocity traces')
fig.tight_layout()
plt.show()"""
        ),
        new_markdown_cell("## Best measured radius at each mass"),
        new_code_cell(
            """winner_rows = summary.loc[summary.groupby('payload_mass_g')['max_velocity_m_s'].idxmax()]
winner_rows[['payload_mass_g', 'radius_mm', 'max_velocity_m_s', 'max_kinetic_energy_j']].reset_index(drop=True)"""
        ),
    ]
    nbformat.write(notebook, SHARED_NOTEBOOK)


def update_r7():
    notebook = nbformat.read(R7_NOTEBOOK, as_version=4)
    notebook.cells = [
        new_markdown_cell(
            """# r7 camera-calibrated tracking results

Filter inspection and mass trends for r7. Positions use the 46 mm fixture calibration; original tracking CSVs remain unchanged."""
        ),
        find_root_cell(),
        new_markdown_cell("## Controls"),
        new_code_cell(
            """FILTER_WINDOW = 9
FILTER_ORDER = 2
COMPARE_MASS_G = 100
CARRIER_MASS_G = 10.0
TRANSMISSION_MASS_G = 7.5
CHAIN_ELEMENTS = 37
TRANSMISSION_SPEED_SQUARED_FACTOR = (2 * CHAIN_ELEMENTS - 1) / (6 * CHAIN_ELEMENTS)"""
        ),
        filter_cell(),
        new_code_cell(
            """data_dir = results_dir / 'tracking_csv/r7'
rows = []
tracks = {}
for path in sorted(data_dir.glob('r7_*g_tracking.csv')):
    payload_mass_g = float(path.stem.split('_')[1][:-1])
    track = pd.read_csv(path)
    time, raw_position, filtered_position, filtered_velocity = filter_track(
        track, FILTER_WINDOW, FILTER_ORDER
    )
    dt = np.median(np.diff(time))
    raw_velocity = np.gradient(raw_position, dt)
    effective_mass_kg = (
        payload_mass_g + CARRIER_MASS_G
        + TRANSMISSION_SPEED_SQUARED_FACTOR * TRANSMISSION_MASS_G
    ) / 1000
    max_velocity = float(filtered_velocity.max())
    rows.append({
        'payload_mass_g': payload_mass_g,
        'average_velocity_m_s': float(filtered_velocity.mean()),
        'max_velocity_m_s': max_velocity,
        'max_kinetic_energy_j': 0.5 * effective_mass_kg * max_velocity**2,
    })
    tracks[payload_mass_g] = (time, raw_position, filtered_position, raw_velocity, filtered_velocity)

summary = pd.DataFrame(rows).sort_values('payload_mass_g').reset_index(drop=True)
summary"""
        ),
        new_markdown_cell("## Filtered versus unfiltered position and velocity"),
        new_code_cell(
            """mass = min(tracks, key=lambda value: abs(value - COMPARE_MASS_G))
time, raw_position, filtered_position, raw_velocity, filtered_velocity = tracks[mass]
relative_time = time - time[0]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(relative_time, raw_position, alpha=0.55, label='Unfiltered')
axes[0].plot(relative_time, filtered_position, linewidth=2, label='Filtered')
axes[0].set_ylabel('Position (m)')
axes[0].legend()
axes[0].grid(alpha=0.25)
axes[1].plot(relative_time, raw_velocity, alpha=0.45, label='Unfiltered')
axes[1].plot(relative_time, filtered_velocity, linewidth=2, label='Filtered')
axes[1].set(xlabel='Time (s)', ylabel='Velocity (m/s)')
axes[1].legend()
axes[1].grid(alpha=0.25)
fig.suptitle(f'r7, {mass:g} g: filter comparison')
fig.tight_layout()
plt.show()"""
        ),
        new_markdown_cell("## Velocity and kinetic-energy trends"),
        new_code_cell(
            """fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(summary['payload_mass_g'], summary['average_velocity_m_s'], 'o-')
axes[0].set(xlabel='Payload mass (g)', ylabel='Average velocity (m/s)', title='Average velocity')
axes[1].plot(summary['payload_mass_g'], summary['max_velocity_m_s'], 'o-')
axes[1].set(xlabel='Payload mass (g)', ylabel='Maximum velocity (m/s)', title='Maximum velocity')
axes[2].plot(summary['payload_mass_g'], summary['max_kinetic_energy_j'], 'o-')
axes[2].set(xlabel='Payload mass (g)', ylabel='Maximum kinetic energy (J)', title='Maximum kinetic energy')
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle('r7 trends (46 mm camera calibration)')
fig.tight_layout()
r7_plot = results_dir / 'corrected_r7_mass_trends.png'
fig.savefig(r7_plot, dpi=180)
plt.show()
r7_plot"""
        ),
        new_markdown_cell("## All filtered velocity traces"),
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(10, 5.5))
for mass, (time, _, _, _, velocity) in tracks.items():
    ax.plot(time - time[0], velocity, label=f'{mass:g} g')
ax.set(xlabel='Time (s)', ylabel='Filtered velocity (m/s)', title='r7 filtered velocity traces')
ax.grid(alpha=0.25)
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
plt.show()"""
        ),
    ]
    nbformat.write(notebook, R7_NOTEBOOK)


if __name__ == "__main__":
    update_shared()
    update_r7()
    print(SHARED_NOTEBOOK)
    print(R7_NOTEBOOK)
