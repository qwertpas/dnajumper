"""Native viewer for rigid plates connected by two families of tensile cords."""

from __future__ import annotations

import math
import sys

import numpy as np
from PyQt6.QtCore import QSignalBlocker, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .gui import ParameterSlider
from .cable_model import Parameters, deform, evaluate, response, response_warnings, twist_limit


BLUE = "#2573ad"
ORANGE = "#d77a26"
RED = "#c84b55"
GRAY = "#667687"
DEFAULTS = {
    "radius": 100.0, "ratio": .65, "height": math.sqrt(1.3), "family": 3,
    "cells": 12, "twist": 45.0, "force": 10.0, "rigidity": 10000.0,
    "amplitude": 10.0, "offset": .01, "tilt": .01, "direction": 0.0,
}


class CableWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tensile cord cells")
        self.resize(1530, 1000)
        self.setMinimumSize(1120, 780)
        self.controls = {}
        self._result = None
        self._model_key = None
        self._updating = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.refresh)
        self._build()
        self.reset()

    def _build(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f7f8fa; color: #243346; }
            QLabel { background: transparent; }
            QComboBox, QDoubleSpinBox { background: white; border: 1px solid #ccd4de;
                border-radius: 4px; padding: 5px; }
            QSlider::groove:horizontal { height: 4px; background: #d8dfe8; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #487aaf; border-radius: 2px; }
            QSlider::handle:horizontal { background: #2563a6; border: 1px solid #2563a6;
                width: 12px; margin: -5px 0; border-radius: 6px; }
            QDoubleSpinBox:disabled { color: #8d99a9; background: #eef1f5; }
            QSlider::handle:horizontal:disabled { background: #bfc9d5; border-color: #bfc9d5; }
            QPushButton { background: white; border: 1px solid #ccd4de;
                border-radius: 4px; padding: 6px 10px; }
            QPushButton:hover { background: #eaf0f7; }
            QScrollArea { border: 0; }
            QFrame#metric { background: white; border: 1px solid #e0e5eb; border-radius: 6px; }
        """)
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(12)
        self.setCentralWidget(central)
        side = QWidget()
        side.setMinimumWidth(302)
        controls = QVBoxLayout(side)
        controls.setContentsMargins(2, 0, 10, 0)
        controls.setSpacing(10)
        row = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.setObjectName("cordPreset")
        self.preset.addItems(("6 cords", "8 cords", "Equal radii (weak)", "Custom"))
        self.preset.currentIndexChanged.connect(self._preset_changed)
        row.addWidget(self.preset, 1)
        reset = QPushButton("Reset")
        reset.setObjectName("resetParameters")
        reset.clicked.connect(self.reset)
        row.addWidget(reset)
        controls.addLayout(row)
        self._section(controls, "Geometry")
        self._slider(controls, "radius", "Outer radius R", 10, 250, "mm", 1)
        self._slider(controls, "ratio", "Inner / outer radius r/R", .3, 1, decimals=3)
        self._slider(controls, "height", "Untwisted height H₀ / R", .3, 2, decimals=3)
        self._slider(controls, "family", "Cords per family", 3, 8, decimals=0)
        self._slider(controls, "cells", "Stacked cells", 2, 24, decimals=0)
        self._slider(controls, "twist", "Twist per cell θ", 0, 170, "°", 2)
        self.geometry_note = self._note(controls)
        self._section(controls, "Loading")
        self._slider(controls, "force", "Axial tensile load F", .1, 1000, "N", 2, True)
        self._slider(controls, "rigidity", "Axial rigidity EA per cord", 10, 1e6, "N", 1, True)
        self.controls["rigidity"].setToolTip(
            "Young's modulus × cord cross-sectional area. Finite stiffness and existing "
            "tension both enter the local stability calculation."
        )
        self._section(controls, "Alignment")
        self.view = QComboBox()
        self.view.setObjectName("alignmentView")
        self.view.addItems(("Weakest tilt mode", "End misalignment"))
        self.view.currentIndexChanged.connect(self._view_changed)
        controls.addWidget(self.view)
        self._slider(controls, "amplitude", "Mode display amplitude", 0, 30, "% R", 1)
        self._slider(controls, "offset", "Top-center offset / R", 0, 5, "%", 3)
        self._slider(controls, "tilt", "Top-plate tilt", 0, 2, "°", 3)
        self._slider(controls, "direction", "Misalignment direction", 0, 360, "°", 1)
        row = QHBoxLayout()
        self.show_mode = QCheckBox("Show 3D displacement")
        self.show_mode.setObjectName("showDisplacement")
        self.show_mode.setChecked(True)
        self.show_mode.toggled.connect(self.schedule)
        row.addWidget(self.show_mode)
        camera = QPushButton("Reset view")
        camera.setObjectName("resetCamera")
        camera.clicked.connect(self.reset_camera)
        row.addWidget(camera)
        controls.addLayout(row)
        self.model_note = self._note(controls)
        self.model_note.setText(
            "Rigid plates and straight tensile cords. Ideal geometry with a local prestressed stiffness "
            "model; does not trace finite elastic deformation. Cloth, friction, plate thickness "
            "and contact are excluded."
        )
        controls.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(329)
        layout.addWidget(scroll)
        right = QVBoxLayout()
        right.setSpacing(8)
        cards = QHBoxLayout()
        self.metrics = {}
        for title in ("Cell height", "Contraction", "Cord tension", "Motor torque", "Tilt margin"):
            frame = QFrame()
            frame.setObjectName("metric")
            box = QVBoxLayout(frame)
            box.setContentsMargins(10, 8, 10, 8)
            label = QLabel(title)
            label.setStyleSheet("color: #65758a; font-size: 11px;")
            box.addWidget(label)
            value = QLabel("—")
            value.setStyleSheet("font-size: 17px; font-weight: 600;")
            value.setObjectName(title.replace(" ", "") + "Metric")
            box.addWidget(value)
            self.metrics[title] = value
            cards.addWidget(frame)
        right.addLayout(cards)
        self.status = QLabel()
        self.status.setObjectName("stabilityStatus")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(43)
        right.addWidget(self.status)
        self.figure = Figure(figsize=(10, 8), facecolor="#f7f8fa", layout="constrained")
        grid = self.figure.add_gridspec(2, 2, height_ratios=(4.4, 1.4))
        self.cell_axis = self.figure.add_subplot(grid[0, 0], projection="3d")
        self.stack_axis = self.figure.add_subplot(grid[0, 1], projection="3d")
        self.tilt_axis = self.figure.add_subplot(grid[1, :])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setObjectName("cordFigure")
        right.addWidget(self.canvas, 1)
        self.caption = QLabel()
        self.caption.setObjectName("modeCaption")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color: #53647a; font-size: 11px;")
        right.addWidget(self.caption)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        right.addWidget(toolbar)
        layout.addLayout(right, 1)
        self.reset_camera(draw=False)

    def _section(self, box, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 13px; font-weight: 600; padding-top: 6px;")
        box.addWidget(label)

    def _note(self, box):
        label = QLabel()
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 11px; color: #65758a;")
        box.addWidget(label)
        return label

    def _slider(self, box, key, title, low, high, unit="", decimals=2, log=False):
        control = ParameterSlider(title, low, high, DEFAULTS[key], unit, decimals, log)
        control.setObjectName(key + "Control")
        control.number.setObjectName(key + "Value")
        control.slider.setObjectName(key + "Slider")
        control.changed.connect(self.schedule)
        self.controls[key] = control
        box.addWidget(control)

    def reset(self):
        self._updating = True
        for key, value in DEFAULTS.items():
            if key != "twist":
                self.controls[key].set_value(value)
        self.controls["twist"].set_range(0, twist_limit(self.parameters())[0])
        self.controls["twist"].set_value(DEFAULTS["twist"])
        self.preset.setCurrentIndex(0)
        self.view.setCurrentIndex(0)
        self.show_mode.setChecked(True)
        self._updating = False
        self._view_changed()
        self.refresh()

    def _preset_changed(self, index):
        if self._updating or index == 3:
            return
        self.controls["family"].set_value(4 if index == 1 else 3)
        self.controls["ratio"].set_value(1 if index == 2 else .65)
        self.controls["height"].set_value(math.sqrt(2) if index == 2 else math.sqrt(1.3))
        self.controls["cells"].set_value(12)
        self.controls["twist"].set_range(0, twist_limit(self.parameters())[0])
        self.controls["twist"].set_value(45)
        self.schedule()

    def _view_changed(self, *_):
        mode = self.view.currentIndex() == 0
        self.controls["amplitude"].setEnabled(mode)
        for key in ("offset", "tilt", "direction"):
            self.controls[key].setEnabled(not mode)
        self.schedule()

    def schedule(self, *_):
        if not self._updating:
            index = self.preset.currentIndex()
            expected = (4 if index == 1 else 3, 1 if index == 2 else .65,
                        round(math.sqrt(2) if index == 2 else math.sqrt(1.3), 3), 12, 45)
            actual = tuple(self.controls[key].value() for key in ("family", "ratio", "height", "cells", "twist"))
            if index != 3 and actual != expected:
                with QSignalBlocker(self.preset):
                    self.preset.setCurrentIndex(3)
            self.timer.start()

    def reset_camera(self, *_ , draw=True):
        self.cell_axis.view_init(elev=24, azim=-52)
        self.stack_axis.view_init(elev=14, azim=-58)
        if draw:
            self.canvas.draw_idle()

    def parameters(self):
        values = {key: item.value() for key, item in self.controls.items()}
        return Parameters(
            radius=values["radius"], ratio=values["ratio"], height=values["height"],
            twist=values["twist"], per_family=int(values["family"]), cells=int(values["cells"]),
            force=values["force"], rigidity=values["rigidity"],
        )

    def refresh(self):
        self.timer.stop()
        if self._updating:
            return
        p = self.parameters()
        limit, reason = twist_limit(p)
        self.controls["twist"].set_range(0, limit)
        p = self.parameters()
        key = tuple(getattr(p, field) for field in (
            "radius", "ratio", "height", "twist", "per_family", "cells", "force", "rigidity",
        ))
        try:
            if key != self._model_key:
                stack = evaluate(p)
                cell = evaluate(p, cells=1)
                self._result = (cell, stack)
                self._model_key = key
            cell, stack = self._result
        except ValueError as error:
            self.status.setText(f"Invalid settings: {error}  The previous drawing is retained.")
            self.status.setStyleSheet("color: #a33b44; font-weight: 600;")
            return
        self.metrics["Cell height"].setText(f"{stack.height:.2f} mm")
        self.metrics["Contraction"].setText(f"{100*(1-stack.height/(p.radius*p.height)):.1f}%")
        self.metrics["Cord tension"].setText(f"{stack.tension:.3g} N")
        self.metrics["Motor torque"].setText(f"{stack.torque/1000:.3g} N·m")
        self.metrics["Tilt margin"].setText(f"{stack.tilt_smallest:+.3g}")
        self.metrics["Tilt margin"].setToolTip(
            "Smallest tilt stiffness divided by F·R, with translations measured in units of R. "
            "Positive is locally stable; negative is unstable."
        )
        self.geometry_note.setText(
            f"{2*p.per_family} cords per cell · cord length {stack.length:.2f} mm. "
            f"Twist limit {limit:.1f}°: {reason}."
        )
        mode = self.view.currentIndex() == 0
        error = None
        response_notes = []
        if mode:
            scale = self.controls["amplitude"].value() / 100
            cell_motion = cell.mode * scale
            stack_motion = stack.mode * scale
        else:
            offset = p.radius * self.controls["offset"].value() / 100
            tilt = self.controls["tilt"].value()
            direction = self.controls["direction"].value()
            try:
                cell_motion = response(cell, offset, tilt, direction)
                stack_motion = response(stack, offset, tilt, direction)
                response_notes.extend(f"Single cell: {note}" for note in response_warnings(cell, cell_motion))
                response_notes.extend(f"Stack: {note}" for note in response_warnings(stack, stack_motion))
            except ValueError as failure:
                error = str(failure)
                cell_motion = np.zeros_like(cell.mode)
                stack_motion = np.zeros_like(stack.mode)
        show = self.show_mode.isChecked() and error is None
        self._draw_assembly(self.cell_axis, cell, cell_motion if show else None)
        self._draw_assembly(self.stack_axis, stack, stack_motion if show else None)
        self.cell_axis.set_title("Single cell · " + ("free top alignment" if mode else "imposed end error"),
                                 fontsize=12, pad=10)
        self.stack_axis.set_title(f"{p.cells} cells · ends constrained", fontsize=12, pad=10)
        self._draw_tilts(stack_motion, mode, error)
        stable = stack.stable
        label = "Stable" if stable else ("Tilt-unstable" if not stack.tilt_stable else "Axial/twist-unstable")
        tilt_label = "stable" if stack.tilt_stable else "unstable"
        cell_label = "stable" if cell.tilt_stable else "unstable"
        message = (
            f"{label} stack in the local model · stack tilt: {tilt_label} · single-cell tilt: {cell_label}. "
            f"Length-constraint rank: {cell.rank}/5 · {2*p.per_family} cords per cell."
        )
        if error:
            message += f"  {error}"
        if stack.warnings:
            message += "  " + " ".join(stack.warnings)
        if response_notes:
            message += "  " + " ".join(response_notes)
        self.status.setText(message)
        self.status.setStyleSheet(
            f"color: {'#2d6b56' if stable and not error and not response_notes else '#a33b44'}; font-weight: 600;"
        )
        if mode:
            caption = (
                f"Blue: outer → inner cords. Orange: inner → outer cords. Red: each view's weakest "
                f"tilt mode, scaled to {self.controls['amplitude'].value():g}% R maximum anchor displacement. "
                "The red shape is an infinitesimal direction, not a finite folding solution."
            )
        elif error:
            caption = "A stable linear response is unavailable at these settings. Nominal geometry is shown."
        else:
            peak = np.linalg.norm(stack_motion[:, 3:5], axis=1).max()
            gain = peak / max(math.radians(self.controls["tilt"].value()), 1e-12)
            caption = (
                "Red: linear prediction for top-end offset and tilt; no visual amplification. "
                f"Maximum plate tilt {math.degrees(peak):.3g}°; "
                f"center displacement {np.linalg.norm(stack_motion[:, :2], axis=1).max():.3g} mm. "
            )
            if self.controls["offset"].value() == 0 and self.controls["tilt"].value() > 0:
                caption += f"Tilt amplification {gain:.3g}×. "
            if peak > math.radians(5):
                caption += "Large response: the linear approximation is no longer quantitatively reliable. "
        if any("slacken" in note for note in response_notes):
            caption += " Cords slacken: the taut-cord prediction is invalid. "
        caption += f" Local extension ΔL/L (current length): {stack.extension:.3g}%."
        self.caption.setText(caption)
        self.canvas.draw_idle()

    def _draw_assembly(self, axis, report, motion):
        elevation, azimuth = axis.elev, axis.azim
        axis.clear()
        axis.set_facecolor("#f7f8fa")
        axis.view_init(elev=elevation, azim=azimuth)
        points = report.points
        family = np.arange(points.shape[1]) % 2
        lines = np.stack((points[:-1], points[1:]), axis=2).reshape(-1, 2, 3)
        outer_below = (family[None, :] + np.arange(report.cells)[:, None]) % 2 == 0
        colors = np.where(outer_below, BLUE, ORANGE).ravel()
        axis.add_collection3d(Line3DCollection(lines, colors=colors, linewidths=1.8, alpha=.85))
        rings = self._plate_rings(report.centers, report.rotations, report.radius)
        axis.add_collection3d(Poly3DCollection(
            rings, facecolors="#b6c5d6", edgecolors="#708398", alpha=.15, linewidths=.7,
        ))
        axis.scatter(points[..., 0], points[..., 1], points[..., 2], color=GRAY, s=6, alpha=.75)
        extent_points = [points.reshape(-1, 3), rings.reshape(-1, 3)]
        if motion is not None and np.any(motion):
            centers, rotations, displaced = deform(report, motion)
            lines = np.stack((displaced[:-1], displaced[1:]), axis=2).reshape(-1, 2, 3)
            axis.add_collection3d(Line3DCollection(lines, colors=RED, linewidths=1.25, alpha=.75))
            changed_rings = self._plate_rings(centers, rotations, report.radius)
            axis.add_collection3d(Line3DCollection(changed_rings, colors=RED, linewidths=.8, alpha=.8))
            extent_points.extend((displaced.reshape(-1, 3), changed_rings.reshape(-1, 3)))
        extent = np.concatenate(extent_points)
        center = (extent.max(axis=0) + extent.min(axis=0))/2
        span = np.maximum(extent.max(axis=0)-extent.min(axis=0), report.radius*.25)
        span[:2] = max(span[:2])*1.12
        span[2] *= 1.08
        for setter, middle, width in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), center, span):
            setter(middle-width/2, middle+width/2)
        axis.set_box_aspect(span)
        axis.set_xlabel("x (mm)", fontsize=9, labelpad=4)
        axis.set_ylabel("y (mm)", fontsize=9, labelpad=4)
        axis.set_zlabel("z (mm)", fontsize=9, labelpad=18 if report.cells > 1 else 6)
        if report.cells > 1:
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_xticks([])
            axis.set_yticks([])
        axis.tick_params(labelsize=8, pad=1)
        if report.cells == 1:
            axis.locator_params(axis="x", nbins=3)
            axis.locator_params(axis="y", nbins=3)
            axis.set_zlabel("")
        axis.locator_params(axis="z", nbins=5)
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#e1e6ed")
        axis.grid(True, alpha=.2)

    @staticmethod
    def _plate_rings(centers, rotations, radius):
        angle = np.linspace(0, 2*np.pi, 65)
        circle = radius*np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
        return np.einsum("nij,kj->nki", rotations, circle) + centers[:, None, :]

    def _draw_tilts(self, motion, mode, error):
        axis = self.tilt_axis
        axis.clear()
        axis.set_facecolor("white")
        axis.axhline(0, color="#d9e0e8", linewidth=.8)
        if error is None:
            tilt = np.degrees(motion[:, 3:5])
            plates = np.arange(len(motion))
            axis.plot(plates, tilt[:, 0], "o-", color=RED, markersize=3, label="Tilt about x")
            axis.plot(plates, tilt[:, 1], "o-", color=BLUE, markersize=3, label="Tilt about y")
            axis.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
            axis.set_xlim(0, max(1, len(motion)-1))
        else:
            axis.text(.5, .5, "No stable linear response", ha="center", va="center", transform=axis.transAxes, color=RED)
        axis.set_xlabel("Plate index · bottom = 0", fontsize=9)
        axis.set_ylabel("Plate tilt (°)", fontsize=9)
        axis.set_title("Displayed tilt mode" if mode else "Response to end misalignment", loc="left", fontsize=10)
        axis.tick_params(labelsize=9)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Tensile cord cells")
    app.setStyle("Fusion")
    palette = app.style().standardPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#f7f8fa"), (QPalette.ColorRole.WindowText, "#243346"),
        (QPalette.ColorRole.Base, "#ffffff"), (QPalette.ColorRole.Text, "#243346"),
        (QPalette.ColorRole.Button, "#ffffff"), (QPalette.ColorRole.ButtonText, "#243346"),
        (QPalette.ColorRole.Highlight, BLUE), (QPalette.ColorRole.HighlightedText, "#ffffff"),
    ):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    window = CableWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
