"""Native controls and plots for the Hunt two-ring truss."""

from __future__ import annotations

import math
import sys

import numpy as np
from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .model import (
    Parameters, choose_creases, collapse_twist, endpoint_angle, endpoint_strain,
    endpoint_twist, folding_angles, mesh, state, sweep,
)


LENGTHS = (22.9, 35.9, 51.0, 71.2, 113.8, 160.8)
SOURCES = (
    "Folding geometry", "Strain-free end states", "Simple supports", "Clamped supports", "Experiment", "Custom",
)
END_STATES = "Strain-free end states"
TABLE = {
    "Folding geometry": ((23, 21, 19, 17, 14, 12), None),
    "Simple supports": ((22, 20, 18, 16, 13, 11), (59, 63, 67, 70, 75, 77)),
    "Clamped supports": ((23, 21, 19, 17, 14, 12), (56, 61, 65, 69, 73, 76)),
    "Experiment": ((23, 21, 19, 17, 15, 13), (44, 56, 59, 67, 73, 70)),
}
BLUE = "#2563a6"
RED = "#b84c40"
BAR_COLORS = LinearSegmentedColormap.from_list("bar_strain", (BLUE, "#8795a5", RED))


class ParameterSlider(QWidget):
    changed = pyqtSignal(float)

    def __init__(self, name, low, high, value, unit="", decimals=2, log=False):
        super().__init__()
        self.log = log
        self.low, self.high = low, high
        self.steps = int(high - low) if decimals == 0 else 2000
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        self.number = QDoubleSpinBox()
        self.number.setDecimals(decimals)
        self.number.setSuffix(f" {unit}" if unit else "")
        self.number.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.number.setMinimumWidth(102)
        self.number.setMaximumWidth(118)
        self.number.setKeyboardTracking(False)
        row.addWidget(self.number)
        layout.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._slide)
        self.number.valueChanged.connect(self._type)
        self.set_range(low, high)
        self.set_value(value)

    def value(self):
        return self.number.value()

    def _position(self, value):
        low, high = (math.log(self.low), math.log(self.high)) if self.log else (self.low, self.high)
        value = math.log(value) if self.log else value
        return round(self.steps * (value - low) / (high - low))

    def _slide(self, position):
        fraction = position / self.steps
        if self.log:
            value = math.exp(math.log(self.low) + fraction * math.log(self.high / self.low))
        else:
            value = self.low + fraction * (self.high - self.low)
        self.set_value(value)
        self.changed.emit(self.value())

    def _type(self, value):
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._position(value))
        self.changed.emit(value)

    def set_value(self, value):
        value = min(self.high, max(self.low, value))
        with QSignalBlocker(self.number), QSignalBlocker(self.slider):
            self.number.setValue(value)
            self.slider.setValue(self._position(self.number.value()))

    def set_range(self, low, high):
        value = self.value()
        step = 10 ** -self.number.decimals()
        low = math.ceil(low / step - 1e-9) * step
        high = math.floor(high / step + 1e-9) * step
        self.low, self.high = low, max(low + step, high)
        with QSignalBlocker(self.number), QSignalBlocker(self.slider):
            self.number.setRange(self.low, self.high)
            self.number.setSingleStep(1 if self.number.decimals() == 0 else (self.high - self.low) / 100)
            self.slider.setRange(0, self.steps)
        self.set_value(value)


class TrussWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hunt truss buckling")
        self.resize(1560, 1000)
        self.setMinimumSize(1150, 800)
        self.controls = {}
        self._updating = False
        self._curve_key = None
        self._curves = None
        self._geometry_error = None
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
            QDoubleSpinBox:disabled { color: #8d99a9; background: #eef1f5; }
            QSlider::sub-page:horizontal:disabled { background: #bfc9d5; }
            QSlider::handle:horizontal:disabled { background: #bfc9d5; border-color: #bfc9d5; }
            QSlider::handle:horizontal { background: #2563a6; border: 1px solid #2563a6;
                width: 12px; margin: -5px 0; border-radius: 6px; }
            QPushButton { background: white; border: 1px solid #ccd4de;
                border-radius: 4px; padding: 7px 12px; }
            QPushButton:hover { background: #eaf0f7; }
            QScrollArea { border: 0; }
            QFrame#metric { background: white; border: 1px solid #e0e5eb; border-radius: 6px; }
        """)
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(18)
        self.setCentralWidget(central)
        side = QWidget()
        side.setFixedWidth(300)
        controls = QVBoxLayout(side)
        controls.setContentsMargins(2, 0, 10, 0)
        controls.setSpacing(13)
        title = QLabel("Reference geometry")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        controls.addWidget(title)
        self.source = QComboBox()
        self.source.setObjectName("construction")
        self.source.addItems(SOURCES)
        self.source.setToolTip(
            "Folding geometry uses Hunt's analytic construction. Strain-free end states designs a fixed-radius "
            "truss with an unloaded reference and flat state; its intermediate states still strain. "
            "Other paper sources select the nearest L/R row in Table 3."
        )
        self.source.currentTextChanged.connect(self._source_changed)
        controls.addWidget(self.source)
        self.preset = QComboBox()
        self.preset.setObjectName("lengthPreset")
        self.preset.addItem("Custom length")
        for length in LENGTHS:
            self.preset.addItem(f"Yamaki  L/R = {length / 100:g}", length / 100)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        controls.addWidget(self.preset)
        self._slider(controls, "radius", "Radius R", 10, 500, 100, "mm")
        self._slider(controls, "length", "Reference height L", 2, 198, 71.2, "mm")
        self._slider(controls, "creases", "Creases / waves n", 3, 256, 17, decimals=0)
        self.auto_n = QCheckBox("Choose n automatically")
        self.auto_n.setObjectName("autoCreases")
        self.auto_n.setChecked(True)
        self.auto_n.setToolTip("Choose the smallest n from 3–256 meeting the free-path strain limit.")
        self.auto_n.toggled.connect(self._design_changed)
        controls.addWidget(self.auto_n)
        self._slider(controls, "peak_strain", "Peak strain limit", .01, 2, .1, "%", 3, True)
        self._slider(controls, "gamma", "Reference angle γ", 5, 89.9, 70, "°")
        self.geometry_note = QLabel()
        self.geometry_note.setWordWrap(True)
        self.geometry_note.setStyleSheet("color: #65758a; font-size: 11px;")
        controls.addWidget(self.geometry_note)
        self._divider(controls, "Deformation")
        self._slider(controls, "twist", "Added twist φ", 0, 160, 0, "°", 3)
        self.mode = QComboBox()
        self.mode.setObjectName("heightMode")
        self.mode.addItems(("Axially free · minimum bar energy", "Prescribed height"))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        controls.addWidget(self.mode)
        self._slider(controls, "height", "Height h / L", 0, 1.5, 1, decimals=3)
        self._divider(controls, "Bar material")
        self._slider(controls, "modulus", "Young's modulus E", .01, 10, 5.55, "GPa", 3, True)
        self._slider(controls, "area", "Area per inclined bar A", .001, 10, 1, "mm²", 4, True)
        row = QHBoxLayout()
        self.faces = QCheckBox("Show facets")
        self.faces.setToolTip("Facets are a visual guide. The model contains axial bars, with no panel bending energy.")
        self.faces.toggled.connect(self._refresh)
        row.addWidget(self.faces)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset)
        row.addWidget(reset)
        controls.addLayout(row)
        controls.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(320)
        outer.addWidget(scroll)
        right = QVBoxLayout()
        right.setSpacing(10)
        metrics = QHBoxLayout()
        self.metrics = {}
        for name in ("Height", "Max tension", "Max compression", "Torque", "Axial force"):
            frame = QFrame()
            frame.setObjectName("metric")
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 9, 12, 9)
            label = QLabel(name)
            label.setStyleSheet("color: #65758a; font-size: 11px;")
            box.addWidget(label)
            value = QLabel("—")
            value.setStyleSheet("font-size: 19px; font-weight: 600;")
            box.addWidget(value)
            metrics.addWidget(frame)
            self.metrics[name] = value
            if name == "Axial force":
                frame.setToolTip("External holding force dU/dh. Positive force pulls the rings apart (tension).")
        right.addLayout(metrics)
        self.figure = Figure(facecolor="white", layout="constrained")
        self.figure.set_constrained_layout_pads(w_pad=.05, h_pad=.06, wspace=.08, hspace=.10)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setObjectName("trussCanvas")
        self.canvas.setMinimumSize(700, 530)
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1.1, 1))
        self.view = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.torque_plot = self.figure.add_subplot(grid[0, 1])
        self.strain_plot = self.figure.add_subplot(grid[1, 1])
        self.height_plot = self.strain_plot.twinx()
        self.view.view_init(elev=24, azim=-65)
        right.addWidget(self.canvas, 1)
        right.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(42)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right.addWidget(self.status)
        limits = QLabel(
            "Fixed radius: continuous folding requires bar strain; isolated stress-free states can coexist. "
            "Strain-free end states uses a fixed-radius specialization, not Guest's full helical construction. "
            "This axial-bar model excludes member Euler buckling, panel bending, contact and localization. "
            "Shaded plot regions have negative torque slope."
        )
        limits.setWordWrap(True)
        limits.setStyleSheet("color: #65758a; font-size: 11px;")
        limits.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right.addWidget(limits)
        outer.addLayout(right, 1)

    def _slider(self, layout, key, name, low, high, value, unit="", decimals=2, log=False):
        slider = ParameterSlider(name, low, high, value, unit, decimals, log)
        slider.setObjectName(key)
        slider.changed.connect(lambda value, key=key: self._changed(key))
        self.controls[key] = slider
        layout.addWidget(slider)

    @staticmethod
    def _divider(layout, title):
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600; padding-top: 3px;")
        layout.addWidget(label)

    def _values(self):
        return {key: slider.value() for key, slider in self.controls.items()}

    def reset(self):
        self._updating = True
        self.controls["length"].set_range(2, 199)
        self.controls["gamma"].set_range(.01, 89.9)
        self.controls["twist"].set_range(0, 180)
        defaults = dict(radius=100, length=71.2, creases=17, peak_strain=.1, twist=0,
                        height=1, modulus=5.55, area=1)
        for key, value in defaults.items():
            self.controls[key].set_value(value)
        self.source.setCurrentIndex(0)
        self.auto_n.setChecked(True)
        self.preset.setCurrentIndex(4)
        self.mode.setCurrentIndex(0)
        self._updating = False
        self._limits()
        self.controls["height"].setEnabled(False)
        self._refresh()

    def _source_changed(self, source):
        if self._updating:
            return
        if source != "Custom":
            self._apply_source()
        self._limits()
        self._refresh()

    def _design_changed(self):
        if not self._updating:
            self._limits()
            self._refresh()

    def _apply_source(self, index=None):
        values = self._values()
        source = self.source.currentText()
        if source not in TABLE:
            return
        if index is None:
            index = int(np.argmin(np.abs(np.array(LENGTHS) / 100 - values["length"] / values["radius"])))
        counts, angles = TABLE[source]
        self.controls["creases"].set_value(counts[index])
        if angles is not None:
            self.controls["gamma"].set_range(.01, 89.9)
            self.controls["gamma"].set_value(angles[index])

    def _preset_changed(self, index):
        if self._updating or index == 0:
            return
        self._apply_source(index - 1)
        self._limits()
        self.controls["length"].set_value(self.preset.currentData() * self.controls["radius"].value())
        self._apply_source(index - 1)
        self._limits()
        self._refresh()

    def _mode_changed(self):
        if not self._updating:
            self._limits()
            self._refresh()

    def _changed(self, key):
        if self._updating:
            return
        if key == "gamma":
            with QSignalBlocker(self.source):
                self.source.setCurrentText("Custom")
        if key == "length":
            with QSignalBlocker(self.preset):
                self.preset.setCurrentIndex(0)
        if key == "radius" and self.preset.currentIndex():
            self._limits()
            self.controls["length"].set_value(self.preset.currentData() * self.controls["radius"].value())
        self._limits()
        if self.preset.currentIndex() and abs(
            self.controls["length"].value() - self.preset.currentData() * self.controls["radius"].value()
        ) > .011:
            with QSignalBlocker(self.preset):
                self.preset.setCurrentIndex(0)
        self._refresh()

    def _limits(self):
        self._geometry_error = None
        design = self.source.currentText() == END_STATES
        self.auto_n.setVisible(design)
        self.controls["peak_strain"].setVisible(design)
        self.controls["peak_strain"].setEnabled(self.auto_n.isChecked())
        self.controls["creases"].setEnabled(not (design and self.auto_n.isChecked()))
        try:
            self._set_limits(design)
        except (ValueError, ArithmeticError) as error:
            self._geometry_error = str(error)
            self.geometry_note.setText(
                "No design meets these inputs. Adjust R, L or the strain limit, or choose n manually."
            )
        available = self._geometry_error is None
        self.controls["twist"].setEnabled(available)
        self.controls["height"].setEnabled(available and self.mode.currentIndex() == 1)
        self.canvas.setEnabled(available)

    def _set_limits(self, design):
        values = self._values()
        radius, count = values["radius"], int(values["creases"])
        auto = design and self.auto_n.isChecked()
        bound_count = 256 if auto else count
        length_max = (2 * radius * math.cos(math.pi / bound_count) if design else
                      radius * (2 * math.pi / count) / math.tan(math.pi / count))
        self.controls["length"].set_range(.02 * radius, length_max * .999)
        length = self.controls["length"].value()
        if auto:
            count = choose_creases(radius, length, values["peak_strain"] / 100, limit=256)
            self.controls["creases"].set_value(count)
        pitch = 2 * math.pi / count
        gamma_min = math.degrees(math.atan(length / (radius * (math.pi - pitch)))) + .02
        gamma_max = 90 if design or self.source.currentIndex() == 0 else 89.9
        self.controls["gamma"].set_range(.01 if design else gamma_min, gamma_max)
        if self.source.currentIndex() == 0:
            gamma = math.degrees(folding_angles(radius, length, count)[1])
            self.controls["gamma"].set_value(gamma)
        elif design:
            self.controls["gamma"].set_value(math.degrees(endpoint_angle(radius, length, count)))
        p = self.parameters()
        theta0 = p.reference_twist
        self._twist_stop = math.pi - pitch - theta0
        if design:
            self._twist_stop = endpoint_twist(radius, length, count)
        elif self.mode.currentIndex() == 0:
            collapse = collapse_twist(p)
            if collapse is not None:
                self._twist_stop = collapse
        self.controls["twist"].set_range(0, math.degrees(self._twist_stop))
        note = (
            f"{2 * count} inclined bars · pitch {math.degrees(pitch):.2f}°\n"
            f"Reference ring offset θ₀ = {math.degrees(theta0):.2f}°\n"
            + ("Fixed rings; γ encodes θ₀ = L / (R tan γ)." if design else
               "γ is measured from the circumferential direction in the flat pattern.")
        )
        if design:
            short, long = p.rest_lengths
            rim = 2 * radius * math.sin(math.pi / count)
            peak = endpoint_strain(radius, length, count) * 100
            note += (
                f"\nFree-path peak strain {peak:.4g}%; zero strain at both end states."
                f"\nShort / long / rim: {short:.3f} / {long:.3f} / {rim:.3f} mm."
                f"\nFlat added twist: {math.degrees(self._twist_stop):.3f}°."
                "\nR and L do not select a unique n; auto uses the smallest n meeting the limit."
            )
            if self.mode.currentIndex() == 1:
                note += "\nThe strain limit applies to the axially free path, not prescribed height."
        self.geometry_note.setText(note)
        self.controls["length"].setToolTip(
            f"{.02 * radius:.2f}–{length_max * .999:.2f} mm. "
            + ("The flat-endpoint design needs L < 2R cos(π/n)." if design else
               "Upper limit keeps Hunt's folding construction real.")
        )
        self.controls["gamma"].setToolTip(
            ("Derived analytically from R, L and n. Editing switches to Custom geometry." if design else
             f"{gamma_min:.2f}–{gamma_max:g}°. This keeps both inclined families in the selected non-overhanging branch.")
        )
        if design:
            twist_note = "Ends at the analytically exact flat, unloaded end state of the free path."
        elif self.mode.currentIndex() == 0:
            twist_note = "Axially free motion ends at first collapse or the end of the non-wrapping branch."
        else:
            twist_note = "Total ring offset stops before either bar family wraps past 180°."
        self.controls["twist"].setToolTip(
            f"Added twist is 0–{math.degrees(self._twist_stop):.3f}°. {twist_note}"
        )

    def parameters(self):
        values = self._values()
        radius, length, count = values["radius"], values["length"], int(values["creases"])
        source = self.source.currentText()
        if source == "Folding geometry":
            gamma = folding_angles(radius, length, count)[1]
        elif source == END_STATES:
            gamma = endpoint_angle(radius, length, count)
        else:
            gamma = math.radians(values["gamma"])
        return Parameters(
            radius=radius, length=length, creases=count,
            gamma=gamma, modulus=1000 * values["modulus"], area=values["area"],
        )

    def _refresh(self, *_):
        if self._updating:
            return
        if self._geometry_error is not None:
            self._show_error(self._geometry_error)
            return
        try:
            p = self.parameters()
            twist_control = self.controls["twist"]
            twist = (self._twist_stop if twist_control.slider.value() == twist_control.slider.maximum()
                     else math.radians(twist_control.value()))
            height = self.controls["height"].value() * p.length if self.mode.currentIndex() else None
            current = state(p, twist, height)
            key = (p.radius, p.length, p.creases, p.gamma, p.modulus, p.area, height, self._twist_stop)
            if key != self._curve_key:
                twists = np.linspace(0, self._twist_stop, 401)
                self._curves = (twists, sweep(p, twists, height))
                self._curve_key = key
            self._draw_truss(p, twist, current)
            self._draw_curves(twist)
            self._show_state(p, twist, current, height)
            self.canvas.draw_idle()
        except (ValueError, ArithmeticError) as error:
            self._show_error(error)

    def _show_error(self, error):
        self._curve_key = None
        self._curves = None
        for axis in (self.view, self.torque_plot, self.strain_plot, self.height_plot):
            axis.clear()
        for label in self.metrics.values():
            label.setText("—")
        self.controls["twist"].setEnabled(False)
        self.controls["height"].setEnabled(False)
        self.canvas.setEnabled(False)
        self.status.setStyleSheet("color: #a7392f; font-size: 12px;")
        self.status.setText(f"Geometry unavailable: {error}")
        self.canvas.draw_idle()

    def _draw_truss(self, p, twist, current):
        vertices, edges, faces = mesh(p, twist, current.height)
        view = self.view
        camera = (view.elev, view.azim, view.roll)
        view.clear()
        view.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        view.set_title("Two-ring truss", loc="left", fontsize=12, fontweight="semibold", pad=15)
        if self.faces.isChecked():
            view.add_collection3d(Poly3DCollection(
                vertices[faces], facecolor="#8daacb", edgecolor="none", alpha=.12, zsort="average",
            ))
        n = p.creases
        view.add_collection3d(Line3DCollection(vertices[edges[:2 * n]], colors="#283b50", linewidths=1.4))
        strains = np.repeat(np.asarray(current.strain), n)
        limit = max(.001, float(np.max(np.abs(strains))))
        norm = Normalize(-limit, limit)
        bars = Line3DCollection(vertices[edges[2 * n:]], cmap=BAR_COLORS, norm=norm, linewidths=1.5)
        bars.set_array(strains)
        view.add_collection3d(bars)
        view.scatter(*vertices.T, s=7, c="#243346", depthshade=False)
        span = max(2 * p.radius, current.height, p.length * .3)
        middle = current.height / 2
        view.set_xlim(-span * .55, span * .55)
        view.set_ylim(-span * .55, span * .55)
        view.set_zlim(middle - span * .55, middle + span * .55)
        view.set_box_aspect((1, 1, 1))
        view.set_xlabel("x (mm)", labelpad=8, fontsize=9)
        view.set_ylabel("y (mm)", labelpad=8, fontsize=9)
        view.set_zlabel("h (mm)", labelpad=8, fontsize=9)
        view.tick_params(labelsize=8)
        for axis in (view.xaxis, view.yaxis, view.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["color"] = (.8, .84, .89, .45)
        view.text2D(0, .91, "Blue: compression     Red: tension\nDrag to rotate · dimensions in mm",
                    transform=view.transAxes, fontsize=9, color="#65758a")

    @staticmethod
    def _style_plot(axis, title, ylabel):
        axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=10)
        axis.set_xlabel("Added twist φ (°)", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(0, color="#8793a3", linewidth=.8, linestyle=":")

    @staticmethod
    def _shade(axis, x, stiffness):
        axis.fill_between(x, 0, 1, where=np.asarray(stiffness) < 0, color="#e2ac62", alpha=.20,
                          transform=axis.get_xaxis_transform(), linewidth=0)

    @staticmethod
    def _data_limits(axis, values):
        low, high = min(0., float(np.min(values))), max(0., float(np.max(values)))
        pad = .08 * (high - low) if high > low else .05
        axis.set_ylim(low - pad, high + pad)

    def _draw_curves(self, twist):
        twists, curves = self._curves
        x = np.degrees(twists)
        selected = math.degrees(twist)
        torque = self.torque_plot
        torque.clear()
        self._style_plot(torque, "Torque response", "Torque (N m)")
        torque.plot(x, curves["torque"] / 1000, color=BLUE, linewidth=1.8)
        torque.axhline(0, color="#98a3b1", linewidth=.6)
        torque.axvline(selected, color="#243346", linewidth=1, linestyle="--")
        torque.set_xlim(x[0], x[-1])
        self._shade(torque, x, curves["stiffness"])
        self._data_limits(torque, curves["torque"] / 1000)
        strain, height = self.strain_plot, self.height_plot
        strain.clear()
        height.clear()
        self._style_plot(strain, "Bar strain and height", "Axial strain (%)")
        short, = strain.plot(x, curves["strain"][:, 0] * 100, color=BLUE, linewidth=1.5, label="Short bars")
        long, = strain.plot(x, curves["strain"][:, 1] * 100, color=RED, linewidth=1.5, label="Long bars")
        height_line, = height.plot(x, curves["height"] / self.controls["length"].value(), color="#75816b", linestyle=":",
                    linewidth=1.5, label="Height / L")
        height.yaxis.set_label_position("right")
        height.yaxis.tick_right()
        height.set_ylabel("Height / L", color="#65715c", fontsize=9)
        height.tick_params(axis="y", colors="#65715c", labelsize=8)
        height.spines["top"].set_visible(False)
        height.set_ylim(0, max(.05, float(np.max(curves["height"])) / self.controls["length"].value()) * 1.08)
        strain.axhline(0, color="#98a3b1", linewidth=.6)
        strain.axvline(selected, color="#243346", linewidth=1, linestyle="--")
        strain.set_xlim(x[0], x[-1])
        self._shade(strain, x, curves["stiffness"])
        self._data_limits(strain, curves["strain"] * 100)
        lines = (short, long, height_line)
        strain.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper right", frameon=False)

    def _show_state(self, p, twist, current, height):
        tensile = max(0., float(np.max(current.strain)))
        compressive = min(0., float(np.min(current.strain)))
        tensile = 0. if abs(tensile) < 1e-9 else tensile
        compressive = 0. if abs(compressive) < 1e-9 else compressive
        torque = 0. if abs(current.torque) < 1e-6 else current.torque / 1000
        force = 0. if abs(current.force) < 1e-6 else current.force
        values = (
            f"{current.height:.2f} mm", f"{tensile * 100:+.3f}%", f"{compressive * 100:+.3f}%",
            f"{torque:+.4g} N m", f"{force:+.4g} N",
        )
        for label, value in zip(self.metrics.values(), values):
            label.setText(value)
        slope = current.stiffness / 1000
        if current.stiffness < -1e-8:
            stability = "Negative torque slope: unstable under torque control; imposed twist constrains that mode."
        else:
            stability = "Nonnegative torque slope in this symmetric mode; other buckling modes are not assessed."
        notes = [
            f"θ = {math.degrees(p.reference_twist + twist):.2f}° · dT/dφ = {slope:+.4g} N m/rad. {stability}",
        ]
        if height is not None and current.axial_stiffness < -1e-8:
            notes.append("Negative axial stiffness: the height constraint holds that mode.")
        if current.height < p.length * 1e-5:
            notes.append("Collapsed height: contact is omitted.")
        elif height is None and abs(twist - self._twist_stop) < 1e-9:
            notes.append("End of the selected twist branch; no flat equilibrium was reached.")
        if max(tensile, -compressive) > .02:
            notes.append("Strain exceeds 2%: linear elastic bars may be outside their useful material range.")
        self.status.setText(" ".join(notes))
        self.status.setStyleSheet("color: #805218; font-size: 12px;" if current.stiffness < 0 or max(tensile, -compressive) > .02
                                 else "color: #334960; font-size: 12px;")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Hunt truss buckling")
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
    window = TrussWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
