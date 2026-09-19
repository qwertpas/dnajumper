#!/usr/bin/env python3
"""Fit the TSA model to the four tracked payload sweeps."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
RADII_MM = (5, 7, 9, 11)
PAYLOAD_MASSES_G = np.arange(20.0, 221.0, 10.0)
TRACKING_FOLDERS = {
    5: REPO_ROOT / "data/8-25-26_r5/tracking_csv",
    7: REPO_ROOT / "data/8-24-26_r7/vids_and_motor/tracking_csv",
    9: REPO_ROOT / "data/8-27-26_r9/tracking_csv",
    11: REPO_ROOT / "data/8-27-26_r11/tracking_csv",
}

# Experiment mass convention. Change individual transmission values when weighed.
CARRIER_MASS_G = 10.0
TRANSMISSION_MASS_G = {radius: 7.5 for radius in RADII_MM}

# Constants copied from simpleradiusvoltage.ipynb.
LENGTH_M = 0.1595
STROKE_M = 0.110
GRAVITY_M_S2 = 9.81
VOLTAGE_V = 4.0
VOLTAGE_POINTS = np.arange(1.0, 11.0)
FREE_SPEED_RPM = np.array(
    [
        2049.7002,
        4224.7279,
        6390.3895,
        8368.1081,
        10005.7876,
        12022.9659,
        14866.4361,
        17137.8066,
        19872.9528,
        21643.4236,
    ]
)
SLOPE_0 = -4.70391607e-5
SLOPE_RATE = 4.52823176e-6
SLOPE_KNEE_V = 5.352084
FREE_SPEED_4V_RPM = float(np.interp(VOLTAGE_V, VOLTAGE_POINTS, FREE_SPEED_RPM))
TORQUE_SLOPE_4V = SLOPE_0 + SLOPE_RATE * min(VOLTAGE_V, SLOPE_KNEE_V)


def load_tracking_data(
    filter_window: int = 9,
    filter_order: int = 2,
    carrier_mass_g: float = CARRIER_MASS_G,
    transmission_mass_g: dict[int, float] = TRANSMISSION_MASS_G,
) -> pd.DataFrame:
    if filter_window < 3 or filter_window % 2 == 0:
        raise ValueError("filter_window must be an odd integer of at least 3")
    if filter_order < 1 or filter_order >= filter_window:
        raise ValueError("filter_order must be smaller than filter_window")
    if set(transmission_mass_g) != set(RADII_MM):
        raise ValueError(f"transmission masses must be provided for {RADII_MM}")
    if any(not 5.0 <= mass <= 10.0 for mass in transmission_mass_g.values()):
        raise ValueError("each transmission mass must be between 5 and 10 g")

    rows = []
    for radius in RADII_MM:
        pattern = re.compile(rf"^r{radius}_(\d+)g_tracking\.csv$")
        paths = sorted(
            (path for path in TRACKING_FOLDERS[radius].glob("*.csv") if pattern.match(path.name)),
            key=lambda path: int(pattern.match(path.name).group(1)),
        )
        if len(paths) != len(PAYLOAD_MASSES_G):
            raise RuntimeError(f"Expected 21 canonical r{radius} tracking files, found {len(paths)}")

        for path in paths:
            payload_mass_g = int(pattern.match(path.name).group(1))
            data = pd.read_csv(path)
            time = data["capture_time_s"].to_numpy()
            position = data["raw_position_m"].to_numpy()
            if filter_window > len(data):
                raise ValueError(f"filter_window={filter_window} exceeds {path.name}'s sample count")
            dt = float(np.median(np.diff(time)))
            velocity = savgol_filter(
                position,
                filter_window,
                filter_order,
                deriv=1,
                delta=dt,
            )
            moving_mass_g = payload_mass_g + carrier_mass_g + transmission_mass_g[radius]
            rows.append(
                {
                    "radius_mm": radius,
                    "payload_mass_g": payload_mass_g,
                    "carrier_mass_g": carrier_mass_g,
                    "transmission_mass_g": transmission_mass_g[radius],
                    "moving_mass_kg": moving_mass_g / 1000.0,
                    "measured_velocity_m_s": float(velocity.max()),
                    "mean_velocity_m_s": float(velocity.mean()),
                    "samples": len(data),
                }
            )
    return pd.DataFrame(rows).sort_values(["radius_mm", "payload_mass_g"]).reset_index(drop=True)


def model_velocity(
    moving_mass_kg: np.ndarray,
    radius_mm: float,
    inertia_kg_m2: float,
    torque_cap_nm: float,
    steps: int = 300,
) -> np.ndarray:
    """Return end-of-stroke speed using u=theta_dot^2 and RK4 integration."""
    mass = np.asarray(moving_mass_kg, dtype=float)
    radius_m = radius_mm / 1000.0
    theta_end = np.sqrt(LENGTH_M**2 - (LENGTH_M - STROKE_M) ** 2) / radius_m
    step = theta_end / steps
    speed_squared = np.zeros_like(mass)

    def derivative(theta: float, value: np.ndarray) -> np.ndarray:
        value = np.maximum(value, 0.0)
        root = np.sqrt(max(LENGTH_M**2 - (radius_m * theta) ** 2, 1e-14))
        dy_dtheta = radius_m**2 * theta / root
        d2y_dtheta2 = radius_m**2 * LENGTH_M**2 / root**3
        rpm = np.sqrt(value) * 60.0 / (2.0 * np.pi)
        torque = np.minimum(TORQUE_SLOPE_4V * (rpm - FREE_SPEED_4V_RPM), torque_cap_nm)
        return 2.0 * (
            -value * mass * dy_dtheta * d2y_dtheta2
            - mass * GRAVITY_M_S2 * dy_dtheta
            + torque
        ) / (inertia_kg_m2 + mass * dy_dtheta**2)

    theta = 0.0
    for _ in range(steps):
        k1 = derivative(theta, speed_squared)
        k2 = derivative(theta + step / 2.0, speed_squared + step * k1 / 2.0)
        k3 = derivative(theta + step / 2.0, speed_squared + step * k2 / 2.0)
        k4 = derivative(theta + step, speed_squared + step * k3)
        speed_squared = np.maximum(
            speed_squared + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
            0.0,
        )
        theta += step

    root = np.sqrt(LENGTH_M**2 - (radius_m * theta_end) ** 2)
    dy_dtheta = radius_m**2 * theta_end / root
    return dy_dtheta * np.sqrt(speed_squared)


def fit_radius(data: pd.DataFrame, train_mask: np.ndarray | None = None) -> np.ndarray:
    mass = data["moving_mass_kg"].to_numpy()
    measured = data["measured_velocity_m_s"].to_numpy()
    radius = float(data["radius_mm"].iloc[0])
    if train_mask is None:
        train_mask = np.ones(len(data), dtype=bool)

    def residual(parameters: np.ndarray) -> np.ndarray:
        inertia = parameters[0] * 1e-5
        predicted = model_velocity(mass, radius, inertia, parameters[1])
        return (predicted - measured)[train_mask]

    result = least_squares(
        residual,
        x0=[2.0, 0.10],
        bounds=([0.05, 0.03], [20.0, 0.50]),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=300,
    )
    if not result.success:
        raise RuntimeError(f"r{radius:g} fit failed: {result.message}")
    return np.array([result.x[0] * 1e-5, result.x[1]])


def fit_global(data: pd.DataFrame) -> np.ndarray:
    def residual(parameters: np.ndarray) -> np.ndarray:
        inertia = parameters[0] * 1e-5
        parts = []
        for radius, radius_data in data.groupby("radius_mm"):
            predicted = model_velocity(
                radius_data["moving_mass_kg"].to_numpy(),
                radius,
                inertia,
                parameters[1],
            )
            parts.append(predicted - radius_data["measured_velocity_m_s"].to_numpy())
        return np.concatenate(parts)

    result = least_squares(
        residual,
        x0=[2.0, 0.12],
        bounds=([0.05, 0.03], [20.0, 0.50]),
        max_nfev=300,
    )
    if not result.success:
        raise RuntimeError(f"Global fit failed: {result.message}")
    return np.array([result.x[0] * 1e-5, result.x[1]])


def fit_shared_cap(data: pd.DataFrame) -> np.ndarray:
    def residual(parameters: np.ndarray) -> np.ndarray:
        parts = []
        for index, (radius, radius_data) in enumerate(data.groupby("radius_mm")):
            predicted = model_velocity(
                radius_data["moving_mass_kg"].to_numpy(),
                radius,
                parameters[index] * 1e-5,
                parameters[-1],
            )
            parts.append(predicted - radius_data["measured_velocity_m_s"].to_numpy())
        return np.concatenate(parts)

    result = least_squares(
        residual,
        x0=[2.0, 2.0, 2.0, 2.0, 0.12],
        bounds=([0.05, 0.05, 0.05, 0.05, 0.03], [20.0, 20.0, 20.0, 20.0, 0.50]),
        max_nfev=500,
    )
    if not result.success:
        raise RuntimeError(f"Shared-cap fit failed: {result.message}")
    return np.r_[result.x[:4] * 1e-5, result.x[-1]]


def predict_with_parameters(data: pd.DataFrame, parameters: pd.DataFrame) -> np.ndarray:
    predicted = np.empty(len(data))
    for radius, radius_data in data.groupby("radius_mm"):
        row = parameters.loc[parameters["radius_mm"] == radius].iloc[0]
        predicted[radius_data.index] = model_velocity(
            radius_data["moving_mass_kg"].to_numpy(),
            radius,
            row["inertia_kg_m2"],
            row["torque_cap_nm"],
            steps=600,
        )
    return predicted


def score(predicted: np.ndarray, measured: np.ndarray, parameter_count: int) -> dict[str, float]:
    error = predicted - measured
    rss = float(np.sum(error**2))
    count = len(error)
    return {
        "parameter_count": parameter_count,
        "rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "max_error_m_s": float(np.max(np.abs(error))),
        "aic": float(count * np.log(rss / count) + 2 * parameter_count),
    }


def fit_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameter_rows = []
    holdout_predictions = np.empty(len(data))
    for radius, radius_data in data.groupby("radius_mm"):
        radius_data = radius_data.sort_values("payload_mass_g")
        holdout = np.arange(len(radius_data)) % 3 == 1
        fitted = fit_radius(radius_data)
        cross_validated = fit_radius(radius_data, ~holdout)
        predicted = model_velocity(
            radius_data["moving_mass_kg"].to_numpy(),
            radius,
            fitted[0],
            fitted[1],
            steps=600,
        )
        cv_predicted = model_velocity(
            radius_data["moving_mass_kg"].to_numpy(),
            radius,
            cross_validated[0],
            cross_validated[1],
            steps=600,
        )
        holdout_predictions[radius_data.index] = cv_predicted
        error = predicted - radius_data["measured_velocity_m_s"].to_numpy()
        parameter_rows.append(
            {
                "radius_mm": int(radius),
                "carrier_mass_g": float(radius_data["carrier_mass_g"].iloc[0]),
                "transmission_mass_g": float(radius_data["transmission_mass_g"].iloc[0]),
                "inertia_kg_m2": fitted[0],
                "torque_cap_nm": fitted[1],
                "rmse_m_s": float(np.sqrt(np.mean(error**2))),
                "max_error_m_s": float(np.max(np.abs(error))),
                "cv_inertia_kg_m2": cross_validated[0],
                "cv_torque_cap_nm": cross_validated[1],
                "cv_test_rmse_m_s": float(np.sqrt(np.mean((cv_predicted[holdout] - radius_data.loc[holdout, "measured_velocity_m_s"]) ** 2))),
            }
        )

    parameters = pd.DataFrame(parameter_rows)
    results = data.copy()
    results["model_velocity_m_s"] = predict_with_parameters(results, parameters)
    results["residual_m_s"] = results["model_velocity_m_s"] - results["measured_velocity_m_s"]
    results["measured_energy_j"] = 0.5 * results["moving_mass_kg"] * results["measured_velocity_m_s"] ** 2
    results["model_energy_j"] = 0.5 * results["moving_mass_kg"] * results["model_velocity_m_s"] ** 2
    results["cv_model_velocity_m_s"] = holdout_predictions
    results["is_cv_holdout"] = results.groupby("radius_mm").cumcount() % 3 == 1

    measured = results["measured_velocity_m_s"].to_numpy()
    baseline = np.concatenate(
        [
            model_velocity(group["moving_mass_kg"].to_numpy(), radius, 5e-5, 0.260)
            for radius, group in results.groupby("radius_mm")
        ]
    )
    global_parameters = fit_global(results)
    global_prediction = np.concatenate(
        [
            model_velocity(group["moving_mass_kg"].to_numpy(), radius, *global_parameters)
            for radius, group in results.groupby("radius_mm")
        ]
    )
    shared_parameters = fit_shared_cap(results)
    shared_prediction = np.concatenate(
        [
            model_velocity(
                group["moving_mass_kg"].to_numpy(),
                radius,
                shared_parameters[index],
                shared_parameters[-1],
            )
            for index, (radius, group) in enumerate(results.groupby("radius_mm"))
        ]
    )
    comparison = pd.DataFrame(
        [
            {"model": "notebook baseline", **score(baseline, measured, 0)},
            {"model": "global inertia + torque cap", **score(global_prediction, measured, 2)},
            {"model": "radius inertia + global torque cap", **score(shared_prediction, measured, 5)},
            {
                "model": "radius inertia + radius torque cap",
                **score(results["model_velocity_m_s"].to_numpy(), measured, 8),
            },
        ]
    )
    return results, parameters, comparison


def plot_fit(results: pd.DataFrame, path: Path) -> None:
    colors = dict(zip(RADII_MM, plt.cm.viridis(np.linspace(0.08, 0.9, len(RADII_MM)))))
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    for radius, data in results.groupby("radius_mm"):
        color = colors[int(radius)]
        axes[0, 0].plot(data["payload_mass_g"], data["measured_velocity_m_s"], "o-", color=color, label=f"r{int(radius)} measured")
        axes[0, 0].plot(data["payload_mass_g"], data["model_velocity_m_s"], "--", color=color, label=f"r{int(radius)} model")
        axes[0, 1].plot(data["payload_mass_g"], data["residual_m_s"], "o-", color=color, label=f"r{int(radius)}")
        axes[1, 0].plot(data["payload_mass_g"], data["measured_energy_j"], "o-", color=color)
        axes[1, 0].plot(data["payload_mass_g"], data["model_energy_j"], "--", color=color)
        axes[1, 1].scatter(data["measured_velocity_m_s"], data["model_velocity_m_s"], color=color, label=f"r{int(radius)}")

    overall_rmse = float(np.sqrt(np.mean(results["residual_m_s"] ** 2)))
    held_out = results[results["is_cv_holdout"]]
    cv_rmse = float(np.sqrt(np.mean((held_out["cv_model_velocity_m_s"] - held_out["measured_velocity_m_s"]) ** 2)))
    axes[0, 0].set(title=f"Velocity fit: RMSE={overall_rmse:.3f} m/s", xlabel="Payload mass (g)", ylabel="Maximum velocity (m/s)")
    axes[0, 0].legend(ncols=2, fontsize=8)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].axhspan(-0.2, 0.2, color="gray", alpha=0.12)
    axes[0, 1].set(title=f"Residuals; held-out RMSE={cv_rmse:.3f} m/s", xlabel="Payload mass (g)", ylabel="Model − measured (m/s)")
    axes[0, 1].legend()
    axes[1, 0].set(title="Kinetic energy with carrier and transmission mass", xlabel="Payload mass (g)", ylabel="Kinetic energy (J)")
    low = min(results["measured_velocity_m_s"].min(), results["model_velocity_m_s"].min())
    high = max(results["measured_velocity_m_s"].max(), results["model_velocity_m_s"].max())
    axes[1, 1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1, 1].set(title="Predicted versus measured", xlabel="Measured velocity (m/s)", ylabel="Model velocity (m/s)")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("Calibrated 4 V TSA model; moving mass = payload + 10 g carrier + transmission")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_fit(
    filter_window: int = 9,
    filter_order: int = 2,
    carrier_mass_g: float = CARRIER_MASS_G,
    transmission_mass_g: dict[int, float] = TRANSMISSION_MASS_G,
    output_dir: Path = REPO_ROOT / "data/model_fit",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_tracking_data(filter_window, filter_order, carrier_mass_g, transmission_mass_g)
    results, parameters, comparison = fit_model(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "tracking_model_fit.csv", index=False)
    parameters.to_csv(output_dir / "tracking_model_parameters.csv", index=False)
    comparison.to_csv(output_dir / "tracking_model_comparison.csv", index=False)
    plot_fit(results, output_dir / "tracking_model_fit.png")
    return results, parameters, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-window", type=int, default=9)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--carrier-g", type=float, default=CARRIER_MASS_G)
    parser.add_argument("--transmission-g", type=float, nargs=4, default=[7.5, 7.5, 7.5, 7.5], metavar=("R5", "R7", "R9", "R11"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/model_fit")
    args = parser.parse_args()
    transmission_mass_g = dict(zip(RADII_MM, args.transmission_g))
    results, parameters, comparison = run_fit(
        args.filter_window,
        args.filter_order,
        args.carrier_g,
        transmission_mass_g,
        args.output_dir,
    )
    held_out = results[results["is_cv_holdout"]]
    cv_rmse = np.sqrt(np.mean((held_out["cv_model_velocity_m_s"] - held_out["measured_velocity_m_s"]) ** 2))
    print(parameters.to_string(index=False))
    print()
    print(comparison.to_string(index=False))
    print(f"\nHeld-out RMSE: {cv_rmse:.4f} m/s")


if __name__ == "__main__":
    main()
