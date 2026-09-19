#!/usr/bin/env python3.11
"""Recalibrate tracking scale from the 46 mm fixture and compare radius models."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
FIXTURE_HEIGHT_M = 0.046
OLD_SCALE = 10680.0
RADII = (5, 7, 9, 11)
FOLDERS = {
    5: ROOT / "data/8-25-26_r5",
    7: ROOT / "data/8-24-26_r7/vids_and_motor",
    9: ROOT / "data/8-27-26_r9",
    11: ROOT / "data/8-27-26_r11",
}
TRACKING = {
    5: FOLDERS[5] / "tracking_csv",
    7: FOLDERS[7] / "tracking_csv",
    9: FOLDERS[9] / "tracking_csv",
    11: FOLDERS[11] / "tracking_csv",
}

# Final smooth 4 V torque-speed model from flywheeljumper motormodel.ipynb.
FREE_SPEED_RPM = 8368.108101319473
SLOPE_PARAMS = np.array([-4.84787835e-5, 5.32297125e-6, 4.84445085, 0.683395924])
TORQUE_CAP_NM = 0.259990

LENGTH_M = 0.1595
STROKE_M = 0.110
ROTOR_INERTIA = 4e-6
CARRIER_MASS_KG = 0.010
TRANSMISSION_MASS_KG = 0.0075
GRAVITY = 9.81
ELEMENTS = 37
MASS_SPEED_FACTOR = (2 * ELEMENTS - 1) / (6 * ELEMENTS)
GRAVITY_FACTOR = 0.5


def trials(folder: Path) -> list[dict[str, str]]:
    with (folder / "trial_manifest.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def fixture_box(path: Path) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None, None

    scene = frame[: int(frame.shape[0] * 0.90)]
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    mask = (gray < 0.65 * np.median(gray)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    boxes = []
    for x, y, width, height, area in stats[1:]:
        touches_right = x + width >= gray.shape[1] - 2
        if touches_right and height > 300 and area > 10000:
            boxes.append((int(x), int(y), int(width), int(height)))
    return frame, max(boxes, key=lambda box: box[3], default=None)


def fixture_height(path: Path) -> float:
    _, box = fixture_box(path)
    return float(box[3]) if box else np.nan


def measure_scales() -> tuple[pd.DataFrame, dict[int, float]]:
    rows = []
    scales = {}
    for radius in RADII:
        heights = np.array(
            [fixture_height(FOLDERS[radius] / row["video"]) for row in trials(FOLDERS[radius])]
        )
        heights = heights[np.isfinite(heights)]
        center = float(np.median(heights))
        good = heights[np.abs(heights - center) <= 0.05 * center]
        center = float(np.median(good))
        scale = center / FIXTURE_HEIGHT_M
        scales[radius] = scale
        rows.append(
            {
                "radius_mm": radius,
                "fixture_height_px": center,
                "fixture_height_std_px": float(np.std(good)),
                "videos_used": len(good),
                "pixels_per_meter": scale,
                "old_to_corrected_factor": OLD_SCALE / scale,
            }
        )
    return pd.DataFrame(rows), scales


def correct_tracking(scales: dict[int, float]) -> pd.DataFrame:
    rows = []
    metric_columns = ["raw_position_m", "raw_velocity_m_s", "position_m", "velocity_m_s"]
    corrected_root = OUT / "tracking_csv"
    for radius in RADII:
        factor = OLD_SCALE / scales[radius]
        output_folder = corrected_root / f"r{radius}"
        output_folder.mkdir(parents=True, exist_ok=True)
        for mass in range(20, 221, 10):
            name = f"r{radius}_{mass}g_tracking.csv"
            data = pd.read_csv(TRACKING[radius] / name)
            original_max = float(data["velocity_m_s"].max())
            data[metric_columns] *= factor
            data.to_csv(output_folder / name, index=False)
            rows.append(
                {
                    "radius_mm": radius,
                    "payload_mass_g": mass,
                    "pixels_per_meter": scales[radius],
                    "original_max_velocity_m_s": original_max,
                    "corrected_max_velocity_m_s": float(data["velocity_m_s"].max()),
                    "corrected_max_raw_velocity_m_s": float(data["raw_velocity_m_s"].max()),
                }
            )
    return pd.DataFrame(rows)


def slope_at_voltage(voltage: float) -> float:
    base, rate, midpoint, width = SLOPE_PARAMS
    soft_voltage = voltage - width * np.logaddexp(0, (voltage - midpoint) / width)
    return base + rate * soft_voltage


SLOPE_4V = slope_at_voltage(4.0)


def model_velocity(
    payload_mass_g: np.ndarray,
    radius_mm: int,
    friction_nm: float = 0.0,
    steps: int = 600,
) -> np.ndarray:
    payload_mass_g = np.asarray(payload_mass_g, dtype=float)
    radius = radius_mm / 1000.0
    bottom_mass = payload_mass_g / 1000.0 + CARRIER_MASS_KG
    kinetic_mass = bottom_mass + MASS_SPEED_FACTOR * TRANSMISSION_MASS_KG
    gravity_mass = bottom_mass + GRAVITY_FACTOR * TRANSMISSION_MASS_KG
    inertia = (
        ROTOR_INERTIA
        + 0.5 * MASS_SPEED_FACTOR * TRANSMISSION_MASS_KG * radius**2
    )
    theta_end = np.sqrt(LENGTH_M**2 - (LENGTH_M - STROKE_M) ** 2) / radius
    step = theta_end / steps
    speed_squared = np.zeros_like(payload_mass_g)

    def derivative(theta: float, value: np.ndarray) -> np.ndarray:
        value = np.maximum(value, 0.0)
        root = np.sqrt(max(LENGTH_M**2 - (radius * theta) ** 2, 1e-14))
        jacobian = radius**2 * theta / root
        jacobian_rate = radius**2 * LENGTH_M**2 / root**3
        speed = np.sqrt(value)
        rpm = speed * 60.0 / (2.0 * np.pi)
        torque = np.minimum(SLOPE_4V * (rpm - FREE_SPEED_RPM), TORQUE_CAP_NM)
        torque -= friction_nm
        return 2.0 * (
            torque
            - kinetic_mass * value * jacobian * jacobian_rate
            - gravity_mass * GRAVITY * jacobian
        ) / (inertia + kinetic_mass * jacobian**2)

    theta = 0.0
    for _ in range(steps):
        k1 = derivative(theta, speed_squared)
        k2 = derivative(theta + step / 2, speed_squared + step * k1 / 2)
        k3 = derivative(theta + step / 2, speed_squared + step * k2 / 2)
        k4 = derivative(theta + step, speed_squared + step * k3)
        speed_squared = np.maximum(
            speed_squared + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6,
            0.0,
        )
        theta += step

    end_jacobian = (
        radius * np.sqrt(LENGTH_M**2 - (LENGTH_M - STROKE_M) ** 2)
        / (LENGTH_M - STROKE_M)
    )
    return end_jacobian * np.sqrt(speed_squared)


def predictions(data: pd.DataFrame, friction_nm: float) -> np.ndarray:
    result = np.empty(len(data))
    for radius, group in data.groupby("radius_mm"):
        result[group.index] = model_velocity(
            group["payload_mass_g"].to_numpy(), int(radius), friction_nm
        )
    return result


def fit_losses(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    measured = data["corrected_max_velocity_m_s"].to_numpy()
    fit = minimize_scalar(
        lambda value: np.mean((predictions(data, value) - measured) ** 2),
        bounds=(0.0, 0.15),
        method="bounded",
    )
    rows = [
        {
            "scope": "common",
            "friction_nm": fit.x,
            "rmse_m_s": np.sqrt(fit.fun),
        }
    ]
    for radius, group in data.groupby("radius_mm"):
        local_measured = group["corrected_max_velocity_m_s"].to_numpy()
        local = minimize_scalar(
            lambda value: np.mean(
                (
                    model_velocity(group["payload_mass_g"].to_numpy(), int(radius), value)
                    - local_measured
                )
                ** 2
            ),
            bounds=(0.0, 0.15),
            method="bounded",
        )
        rows.append(
            {
                "scope": f"r{int(radius)}",
                "friction_nm": local.x,
                "rmse_m_s": np.sqrt(local.fun),
            }
        )
    return pd.DataFrame(rows), float(fit.x)


def winner_table(data: pd.DataFrame, friction_nm: float) -> pd.DataFrame:
    measured = data.pivot(
        index="payload_mass_g", columns="radius_mm", values="corrected_max_velocity_m_s"
    )
    rows = []
    for mass, values in measured.iterrows():
        no_loss = {radius: model_velocity(np.array([mass]), radius)[0] for radius in RADII}
        with_loss = {
            radius: model_velocity(np.array([mass]), radius, friction_nm)[0]
            for radius in RADII
        }
        row = {
            "payload_mass_g": mass,
            "measured_best_radius_mm": int(values.idxmax()),
            "no_loss_model_best_radius_mm": max(no_loss, key=no_loss.get),
            "common_loss_model_best_radius_mm": max(with_loss, key=with_loss.get),
        }
        row.update({f"r{radius}_velocity_m_s": values[radius] for radius in RADII})
        rows.append(row)
    return pd.DataFrame(rows)


def model_switches(friction_nm: float) -> pd.DataFrame:
    masses = np.arange(1.0, 401.0)
    rows = []
    for label, loss in [("no loss", 0.0), ("common fitted loss", friction_nm)]:
        curves = np.column_stack(
            [model_velocity(masses, radius, loss, steps=400) for radius in RADII]
        )
        best = np.array(RADII)[np.argmax(curves, axis=1)]
        start = 0
        for index in range(1, len(masses) + 1):
            if index == len(masses) or best[index] != best[start]:
                rows.append(
                    {
                        "model": label,
                        "mass_start_g": masses[start],
                        "mass_end_g": masses[index - 1],
                        "best_radius_mm": best[start],
                    }
                )
                start = index
    return pd.DataFrame(rows)


def save_plot(data: pd.DataFrame, friction_nm: float) -> None:
    colors = dict(zip(RADII, plt.cm.viridis(np.linspace(0.08, 0.9, len(RADII)))))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    for radius, group in data.groupby("radius_mm"):
        color = colors[int(radius)]
        axes[0].plot(
            group["payload_mass_g"], group["original_max_velocity_m_s"],
            ":", color=color, alpha=0.55,
        )
        axes[0].plot(
            group["payload_mass_g"], group["corrected_max_velocity_m_s"],
            "o-", color=color, label=f"r{int(radius)}",
        )
        model = model_velocity(group["payload_mass_g"].to_numpy(), int(radius), friction_nm)
        axes[1].plot(
            group["payload_mass_g"], group["corrected_max_velocity_m_s"],
            "o", color=color, label=f"r{int(radius)} measured",
        )
        axes[1].plot(
            group["payload_mass_g"], model, "--", color=color,
            label=f"r{int(radius)} model",
        )

    pivot = data.pivot(
        index="payload_mass_g", columns="radius_mm", values="corrected_max_velocity_m_s"
    )
    for radius in (5, 7, 11):
        axes[2].plot(
            pivot.index, pivot[9] - pivot[radius], "o-",
            color=colors[radius], label=f"r9 - r{radius}",
        )
    axes[2].axhline(0, color="black", linewidth=1)

    axes[0].set_title("Original dotted; 46 mm corrected solid")
    axes[1].set_title(f"Corrected data and fixed-motor model\ncommon loss={friction_nm:.3f} N m")
    axes[2].set_title("Corrected r9 advantage")
    for axis in axes:
        axis.set_xlabel("Payload mass (g)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Maximum velocity (m/s)")
    axes[1].set_ylabel("Maximum velocity (m/s)")
    axes[2].set_ylabel("Velocity difference (m/s)")
    figure.savefig(OUT / "radius_scale_model_comparison.png", dpi=180)
    plt.close(figure)


def save_fixture_check(scales: dict[int, float]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, radius in zip(axes.flat, RADII, strict=True):
        row = next(
            row for row in trials(FOLDERS[radius]) if int(row["mass_g"]) == 100
        )
        frame, box = fixture_box(FOLDERS[radius] / row["video"])
        if frame is None or box is None:
            raise RuntimeError(f"Could not measure r{radius} fixture")
        x, y, width, height = box
        axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axis.add_patch(
            plt.Rectangle(
                (x, y), width, height, fill=False, color="red", linewidth=2
            )
        )
        axis.set_title(
            f"r{radius}: frame {height} px; median {scales[radius]:,.0f} px/m"
        )
        axis.axis("off")
    figure.savefig(OUT / "fixture_calibration_check.png", dpi=160)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scale_data, scales = measure_scales()
    corrected = correct_tracking(scales)
    losses, common_loss = fit_losses(corrected)
    winners = winner_table(corrected, common_loss)
    switches = model_switches(common_loss)
    corrected["common_loss_model_velocity_m_s"] = predictions(corrected, common_loss)
    corrected["common_loss_residual_m_s"] = (
        corrected["common_loss_model_velocity_m_s"]
        - corrected["corrected_max_velocity_m_s"]
    )

    scale_data.to_csv(OUT / "camera_scale.csv", index=False)
    corrected.to_csv(OUT / "corrected_max_velocity.csv", index=False)
    losses.to_csv(OUT / "transmission_loss_fit.csv", index=False)
    winners.to_csv(OUT / "radius_winners.csv", index=False)
    switches.to_csv(OUT / "model_radius_switches.csv", index=False)
    save_plot(corrected, common_loss)
    save_fixture_check(scales)

    print(scale_data.to_string(index=False))
    print()
    print(losses.to_string(index=False))
    print()
    print(switches.to_string(index=False))


if __name__ == "__main__":
    main()
