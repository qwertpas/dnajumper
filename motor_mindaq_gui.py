#!/usr/bin/env python3.11
import asyncio
import csv
import importlib.util
import queue
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from bleak import BleakClient, BleakScanner
from pyqtgraph.Qt import QtCore, QtWidgets


BLE_NAME = "dnajumper-v2"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MAGIC = 0xD24B
OLD_MAGIC = 0xD24A
HEADER = struct.Struct("<HHIfffHHBbBB")
SAMPLE = struct.Struct("<HffH")
PLOT_SECONDS = 10.0
LOG_DIR = Path(__file__).parent / "motor_logs"
CAPTURE_GUI = Path(
    "/Users/chris/Kicad/mindaq/mindaq_fw/scripts/capture/capture_gui.py")
STATE_NAMES = ("idle", "moving", "waiting", "homing")
MODE_NAMES = ("voltage", "velocity")


def load_capture_gui():
    spec = importlib.util.spec_from_file_location("mindaq_capture_gui", CAPTURE_GUI)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CAPTURE_GUI}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_gui = load_capture_gui()
capture_gui.DEFAULT_UV_PER_N = 58.47876


class ScalePasteFilter(QtCore.QObject):
    def __init__(self, spin):
        super().__init__(spin)
        self.spin = spin

    def eventFilter(self, watched, event):
        command = (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)
        if (event.type() == QtCore.QEvent.KeyPress
                and event.modifiers() & command
                and event.key() == QtCore.Qt.Key_V):
            text = QtWidgets.QApplication.clipboard().text().strip()
            text = text.replace("µV/N", "").replace("uV/N", "").strip()
            try:
                self.spin.setValue(float(text))
            except ValueError:
                return False
            return True
        if (event.type() == QtCore.QEvent.KeyPress
                and event.modifiers() & command
                and event.key() == QtCore.Qt.Key_C):
            editor = self.spin.lineEdit()
            text = editor.selectedText() or editor.text()
            QtWidgets.QApplication.clipboard().setText(text)
            return True
        return super().eventFilter(watched, event)


def recent_rows(rows):
    if not rows:
        return []
    end = rows[-1][0]
    return [row for row in rows if row[0] >= end - PLOT_SECONDS]


class DataStore:
    def __init__(self, messages):
        self.messages = messages
        self.samples = deque(maxlen=30_000)
        self.recording = False
        self.lock = threading.Lock()
        self.expected_packet = None
        self.next_sample = 0
        self.last_device_us = None
        self.time_epoch = 0
        self.missing_packets = 0
        self.bad_packets = 0
        self.uart_timeouts = 0
        self.queue_drops = 0
        self.motor_id = -1
        self.state = 0
        self.mode = 0
        self.old_protocol_reported = False

    def begin_link(self):
        with self.lock:
            self.samples.clear()
            self.expected_packet = None
            self.next_sample = 0
            self.last_device_us = None
            self.time_epoch = 0
            self.old_protocol_reported = False

    def receive(self, _, value):
        data = bytes(value)
        packet_magic = struct.unpack_from("<H", data)[0] if len(data) >= 2 else 0
        if packet_magic == OLD_MAGIC:
            if not self.old_protocol_reported:
                self.messages.put("ERROR firmware update required")
                self.old_protocol_reported = True
            return
        if packet_magic != MAGIC:
            self.messages.put(data.decode(errors="replace").strip())
            return
        if len(data) < HEADER.size:
            with self.lock:
                self.bad_packets += 1
            return

        fields = HEADER.unpack_from(data)
        (magic, packet_seq, base_us, command, target, rebound_trigger,
         uart_timeouts, queue_drops, count,
         motor_id, state, mode) = fields
        if (magic != MAGIC or count == 0 or
                len(data) != HEADER.size + count * SAMPLE.size):
            with self.lock:
                self.bad_packets += 1
            return

        raw_times = []
        values = []
        for index in range(count):
            offset = HEADER.size + index * SAMPLE.size
            sample_us, angle, velocity, battery_mv = SAMPLE.unpack_from(
                data, offset)
            raw_times.append((base_us + sample_us) & 0xFFFFFFFF)
            values.append((angle, velocity, battery_mv * 0.001))

        with self.lock:
            if self.expected_packet is not None:
                gap = (packet_seq - self.expected_packet) & 0xFFFF
                if gap < 0x8000:
                    self.missing_packets += gap
            rows = []
            for index, raw_us in enumerate(raw_times):
                if (self.last_device_us is not None and
                        raw_us < self.last_device_us and
                        self.last_device_us - raw_us > 0x80000000):
                    self.time_epoch += 1 << 32
                self.last_device_us = raw_us
                angle, velocity, battery = values[index]
                rows.append((
                    (self.time_epoch + raw_us) * 1e-6,
                    self.next_sample,
                    angle,
                    velocity,
                    battery,
                    command,
                    target,
                    rebound_trigger,
                    state,
                    mode,
                ))
                self.next_sample += 1

            self.expected_packet = (packet_seq + 1) & 0xFFFF
            self.uart_timeouts = uart_timeouts
            self.queue_drops = queue_drops
            self.motor_id = motor_id
            self.state = state
            self.mode = mode
            self.samples.extend(rows)

    def begin_recording(self):
        with self.lock:
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False

    def snapshot(self):
        with self.lock:
            return (
                list(self.samples), self.recording, self.missing_packets,
                self.bad_packets, self.uart_timeouts, self.queue_drops,
                self.motor_id, self.state, self.mode,
            )

class BleWorker(threading.Thread):
    def __init__(self, data, messages):
        super().__init__(daemon=True)
        self.data = data
        self.messages = messages
        self.commands = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.connected = threading.Event()

    def send(self, command):
        self.commands.put(command)

    def stop(self):
        self.stop_event.set()

    def run(self):
        asyncio.run(self.run_async())

    async def find_device(self):
        service = SERVICE_UUID.lower()

        def matches(device, advertisement):
            names = {device.name, advertisement.local_name}
            services = {uuid.lower() for uuid in advertisement.service_uuids}
            return BLE_NAME in names or service in services

        return await BleakScanner.find_device_by_filter(matches, timeout=5.0)

    async def run_async(self):
        while not self.stop_event.is_set():
            try:
                self.messages.put("scanning")
                device = await self.find_device()
                if device is None:
                    continue
                async with BleakClient(device, timeout=10.0) as client:
                    self.data.begin_link()
                    await client.start_notify(NOTIFY_UUID, self.data.receive)
                    await client.write_gatt_char(
                        WRITE_UUID, b"STREAM", response=False)
                    self.connected.set()
                    self.messages.put(f"connected {device.address}")

                    while (not self.stop_event.is_set() and
                           client.is_connected):
                        while True:
                            try:
                                command = self.commands.get_nowait()
                            except queue.Empty:
                                break
                            await client.write_gatt_char(
                                WRITE_UUID, command.encode(), response=False)
                        await asyncio.sleep(0.005)

                    if client.is_connected:
                        await client.write_gatt_char(
                            WRITE_UUID, b"STOP", response=False)
                        await client.stop_notify(NOTIFY_UUID)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.messages.put(f"BLE error: {exc}")
            finally:
                self.connected.clear()
            if not self.stop_event.is_set():
                self.messages.put("reconnecting")
                await asyncio.sleep(0.5)


def spin(minimum, maximum, value, decimals=2, step=0.1):
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    return widget


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.messages = queue.SimpleQueue()
        self.data = DataStore(self.messages)
        self.worker = BleWorker(self.data, self.messages)
        self.last_message = "starting"
        self.error = ""
        self.paused = False
        self.paused_rows = None
        self.was_connected = False
        self.rebound_count = 0
        self.last_state = 0
        self.settings = QtCore.QSettings("DNA Jumper", "Motor GUI")

        self.setWindowTitle("DNA Jumper + minDAQ")
        self.resize(2400, 1000)
        self.setMinimumSize(1200, 700)
        outer_layout = QtWidgets.QVBoxLayout(self)

        self.content = QtWidgets.QWidget()
        self.content.setMinimumSize(1400, 780)
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        content_layout.addWidget(self.splitter)
        self.content_scroll = QtWidgets.QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content_scroll.setWidget(self.content)
        outer_layout.addWidget(self.content_scroll, 1)

        motor_page = QtWidgets.QWidget()
        motor_page.setMinimumSize(600, 780)
        layout = QtWidgets.QHBoxLayout(motor_page)
        self.splitter.addWidget(motor_page)
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(220)
        controls = QtWidgets.QVBoxLayout(sidebar)
        self.status = QtWidgets.QLabel("starting")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        self.device_status = QtWidgets.QLabel(
            "Set voltage: — | Vbat: —\nTarget: — | Angle: —")
        self.device_status.setWordWrap(True)
        controls.addWidget(self.device_status)

        form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(("Voltage", "Velocity"))
        self.setpoint = spin(0.1, 300.0, 1.0, 1, 0.1)
        self.target = spin(-188.0, 188.0, 20.0, 1, 0.1)
        form.addRow("Mode", self.mode)
        self.setpoint_label = QtWidgets.QLabel("Set voltage")
        form.addRow(self.setpoint_label, self.setpoint)
        form.addRow("Target (rad)", self.target)
        self.off_state = QtWidgets.QComboBox()
        self.off_state.addItems(("Coast", "Brake"))
        saved_off = self.settings.value("off_state", "COAST")
        self.off_state.setCurrentIndex(
            1 if str(saved_off).upper() == "BRAKE" else 0)
        form.addRow("Off state", self.off_state)
        controls.addLayout(form)

        self.move_button = QtWidgets.QPushButton("Move")
        self.move_button.clicked.connect(self.move)
        controls.addWidget(self.move_button)

        rebound_box = QtWidgets.QGroupBox("Rebound")
        rebound_layout = QtWidgets.QFormLayout(rebound_box)
        count_layout = QtWidgets.QHBoxLayout()
        self.minus_button = QtWidgets.QPushButton("−")
        self.plus_button = QtWidgets.QPushButton("+")
        self.count_label = QtWidgets.QLabel("0")
        self.count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.minus_button.clicked.connect(lambda: self.change_count(-1))
        self.plus_button.clicked.connect(lambda: self.change_count(1))
        count_layout.addWidget(self.minus_button)
        count_layout.addWidget(self.count_label)
        count_layout.addWidget(self.plus_button)
        rebound_layout.addRow("Count", count_layout)
        self.threshold = spin(0.05, 50.0, 2.0, 1, 0.1)
        self.delay = QtWidgets.QSpinBox()
        self.delay.setRange(0, 10_000)
        self.delay.setValue(500)
        rebound_layout.addRow("Threshold (rad)", self.threshold)
        rebound_layout.addRow("Delay (ms)", self.delay)
        controls.addWidget(rebound_box)

        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.setpoint.valueChanged.connect(self.schedule_mode)
        self.off_state.currentIndexChanged.connect(self.off_changed)

        for label, action in (
            ("Home", self.home), ("Zero here", self.zero),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(action)
            controls.addWidget(button)
        self.off_button = QtWidgets.QPushButton(
            self.off_state.currentText().upper())
        self.off_button.clicked.connect(self.stop_motor)
        controls.addWidget(self.off_button)
        save_button = QtWidgets.QPushButton("Save CSV")
        save_button.clicked.connect(self.save_csv)
        controls.addWidget(save_button)
        self.pause_button = QtWidgets.QPushButton("Pause Both")
        self.pause_button.setToolTip(
            "Freeze both GUIs while retaining their pre-pause data")
        self.pause_button.clicked.connect(self.toggle_pause)
        controls.addWidget(self.pause_button)
        self.save_both_button = QtWidgets.QPushButton("Save Both")
        self.save_both_button.setToolTip(
            "Save paused motor data and the current minDAQ capture")
        self.save_both_button.clicked.connect(self.save_both)
        controls.addWidget(self.save_both_button)
        controls.addStretch(1)
        layout.addWidget(sidebar)

        plots = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.angle_plot = pg.PlotWidget(title="Angle and target")
        self.velocity_plot = pg.PlotWidget(title="Velocity")
        self.power_plot = pg.PlotWidget(title="Command and battery")
        self.angle_curve = self.angle_plot.plot(
            pen=pg.mkPen("#2980b9", width=2), name="angle")
        self.target_line = pg.InfiniteLine(
            pos=self.target.value(), angle=0,
            pen=pg.mkPen("#8e44ad", width=1), label="target")
        self.trigger_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#e67e22", width=1, style=QtCore.Qt.DashLine),
            label="rebound trigger")
        self.trigger_line.hide()
        self.angle_plot.addItem(self.target_line)
        self.angle_plot.addItem(self.trigger_line)
        self.target.valueChanged.connect(self.target_line.setValue)
        self.velocity_curve = self.velocity_plot.plot(
            pen=pg.mkPen("#27ae60", width=2))
        self.command_curve = self.power_plot.plot(
            pen=pg.mkPen("#c0392b", width=2), name="command")
        self.battery_curve = self.power_plot.plot(
            pen=pg.mkPen("#f39c12", width=1), name="battery")
        self.angle_plot.addLegend()
        self.power_plot.addLegend()
        self.velocity_plot.setXLink(self.angle_plot)
        self.power_plot.setXLink(self.angle_plot)
        for plot, unit in (
            (self.angle_plot, "rad"), (self.velocity_plot, "rad/s"),
            (self.power_plot, "V or rad/s"),
        ):
            plot.setLabel("left", unit)
            plot.setLabel("bottom", "time", "s")
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot_layout.addWidget(plot)
        layout.addWidget(plots, 1)

        self.mindaq_samples = capture_gui.SampleBuffer()
        self.mindaq_messages = queue.Queue()
        self.mindaq_reader = capture_gui.SerialReader(
            capture_gui.SERIAL_PORT,
            self.mindaq_samples,
            self.mindaq_messages,
        )
        self.mindaq_reader.start()
        self.mindaq = capture_gui.PlotWindow(
            self.mindaq_reader,
            self.mindaq_samples,
            self.mindaq_messages,
        )
        self.mindaq_plot_splitter = self.mindaq.win.findChild(
            QtWidgets.QSplitter)
        self.mindaq_plot_splitter.setOrientation(QtCore.Qt.Vertical)
        self.mindaq_plot_splitter.setSizes((1, 1))
        self.compact_mindaq_controls()
        self.mindaq.win.setMinimumSize(790, 760)
        self.splitter.addWidget(self.mindaq.win)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes((600, 800))

        self.hover_proxies = []
        self.add_hover(self.angle_plot, (
            ("Angle", self.angle_curve, "rad", False),
            ("Target", self.target_line, "rad", True),
            ("Rebound trigger", self.trigger_line, "rad", True),
        ))
        self.add_hover(self.velocity_plot, (
            ("Velocity", self.velocity_curve, "rad/s", False),
        ))
        self.add_hover(self.power_plot, (
            ("Command", self.command_curve, "", False),
            ("Battery", self.battery_curve, "V", False),
        ))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)
        self.mode_timer = QtCore.QTimer(self)
        self.mode_timer.setSingleShot(True)
        self.mode_timer.timeout.connect(self.send_mode)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.request_status)
        self.status_timer.start(1000)
        self.worker.start()

    def add_hover(self, plot, entries):
        view = plot.getViewBox()

        def hover(event):
            position = event[0]
            if not plot.sceneBoundingRect().contains(position):
                QtWidgets.QToolTip.hideText()
                return
            mouse = view.mapSceneToView(position)
            closest = None
            for name, item, unit, horizontal in entries:
                if not item.isVisible():
                    continue
                if horizontal:
                    x = mouse.x()
                    y = float(item.value())
                else:
                    xs, ys = item.xData, item.yData
                    if xs is None or len(xs) == 0:
                        continue
                    index = int(np.searchsorted(xs, mouse.x()))
                    choices = {
                        max(0, min(len(xs) - 1, index)),
                        max(0, min(len(xs) - 1, index - 1)),
                    }
                    index = min(
                        choices, key=lambda i: abs(xs[i] - mouse.x()))
                    x, y = float(xs[index]), float(ys[index])
                point = view.mapViewToScene(QtCore.QPointF(x, y))
                distance = np.hypot(
                    point.x() - position.x(), point.y() - position.y())
                if closest is None or distance < closest[0]:
                    closest = distance, name, unit, x, y

            if closest is None or closest[0] > 12:
                QtWidgets.QToolTip.hideText()
                return
            _, name, unit, x, y = closest
            suffix = f" {unit}" if unit else ""
            viewport_point = plot.mapFromScene(position)
            global_point = plot.viewport().mapToGlobal(viewport_point)
            QtWidgets.QToolTip.showText(
                global_point,
                f"{name}: {y:.3f}{suffix}\nt: {x:.3f} s",
                plot)

        proxy = pg.SignalProxy(
            plot.scene().sigMouseMoved, rateLimit=60, slot=hover)
        self.hover_proxies.append(proxy)

    def compact_mindaq_controls(self):
        plot = self.mindaq
        plot.pause_button.hide()
        plot.refresh_ports_button.setText("Refresh")
        plot.port_box.setFixedWidth(140)
        plot.refresh_ports_button.setFixedWidth(80)
        plot.zero_button.setFixedWidth(60)
        plot.pos_box.setFixedWidth(55)
        plot.neg_box.setFixedWidth(55)
        plot.gain_box.setFixedWidth(65)
        plot.fir_cutoff.setFixedWidth(75)
        plot.trigger_level.setFixedWidth(85)
        plot.trigger_delay.setFixedWidth(85)
        plot.trigger_window_start.setFixedWidth(90)
        plot.trigger_window_end.setFixedWidth(80)
        plot.uv_per_n.setSuffix("")
        plot.uv_per_n.setFixedWidth(120)
        plot.uv_per_n.setReadOnly(False)
        self.scale_paste_filter = ScalePasteFilter(plot.uv_per_n)
        plot.uv_per_n.lineEdit().installEventFilter(self.scale_paste_filter)
        plot.trigger_pin.setFixedWidth(60)
        plot.trigger_width.setFixedWidth(80)
        labels = {
            label.text(): label
            for label in plot.win.findChildren(QtWidgets.QLabel)
        }

        def containing_layout(layout, widget):
            if layout.indexOf(widget) >= 0:
                return layout
            for index in range(layout.count()):
                child = layout.itemAt(index).layout()
                if child is not None:
                    found = containing_layout(child, widget)
                    if found is not None:
                        return found
            return None

        root_layout = plot.win.layout()
        pulse_layout = containing_layout(root_layout, plot.trigger_width)
        window_widgets = (
            labels["Capture window (ms)"],
            plot.trigger_window_start,
            labels["to"],
            plot.trigger_window_end,
        )
        pulse_widgets = (
            labels["Scale"],
            plot.uv_per_n,
        )
        filter_widgets = (
            plot.fir_enable,
            labels["Cutoff (Hz)"],
            plot.fir_cutoff,
        )
        for widget in window_widgets + pulse_widgets + filter_widgets:
            containing_layout(root_layout, widget).removeWidget(widget)

        pulse_index = next(
            index for index in range(root_layout.count())
            if root_layout.itemAt(index).layout() is pulse_layout
        )
        window_layout = QtWidgets.QHBoxLayout()
        window_layout.setSpacing(8)
        for widget in window_widgets:
            window_layout.addWidget(widget)
        window_layout.addStretch(1)
        root_layout.insertLayout(pulse_index, window_layout)

        insert_at = pulse_layout.count() - 1
        for widget in pulse_widgets:
            pulse_layout.insertWidget(insert_at, widget)
            insert_at += 1

        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.setSpacing(8)
        for widget in filter_widgets:
            filter_layout.addWidget(widget)
        filter_layout.addStretch(1)
        root_layout.insertLayout(pulse_index + 2, filter_layout)

        replacements = {
            "Scale": "Scale (uV/N)",
            "Cutoff (Hz)": "Cutoff",
            "Capture window (ms)": "Window (ms)",
            "Pulse width (ms)": "Width (ms)",
        }
        for label in plot.win.findChildren(QtWidgets.QLabel):
            if label.text() in replacements:
                label.setText(replacements[label.text()])

    def schedule_mode(self):
        self.mode_timer.start(150)

    def mode_changed(self):
        self.setpoint_label.setText(
            "Set voltage" if self.mode.currentIndex() == 0
            else "Set velocity")
        self.schedule_mode()

    def send_mode(self):
        if not self.worker.connected.is_set():
            return
        name = self.mode.currentText().upper()
        self.worker.send(f"MODE {name} {self.setpoint.value():.4f}")

    def off_changed(self):
        value = self.off_state.currentText().upper()
        self.off_button.setText(value)
        self.settings.setValue("off_state", value)
        self.settings.sync()
        self.send_off_mode()

    def send_off_mode(self):
        if self.worker.connected.is_set():
            self.worker.send(
                f"OFF_MODE {self.off_state.currentText().upper()}")

    def move(self):
        self.data.begin_recording()
        self.send_mode()
        self.worker.send(
            f"MOVE {self.target.value():.5f} {self.rebound_count} "
            f"{self.threshold.value():.5f} {self.delay.value()}")

    def change_count(self, amount):
        state = self.data.snapshot()[7]
        if self.worker.connected.is_set() and state in (1, 2):
            if amount > 0:
                self.worker.send(
                    f"ADD_REBOUND {self.threshold.value():.5f} "
                    f"{self.delay.value()}")
            elif self.rebound_count > 0:
                self.worker.send("REMOVE_REBOUND")
            return
        self.rebound_count = max(
            0, min(8, self.rebound_count + amount))
        self.count_label.setText(str(self.rebound_count))

    def home(self):
        self.data.begin_recording()
        self.worker.send("HOME")

    def zero(self):
        self.worker.send("ZERO")

    def stop_motor(self):
        self.worker.send("STOP")
        self.data.stop_recording()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            snapshot = self.data.snapshot()
            self.paused_rows = recent_rows(snapshot[0])
            self.draw_rows(self.paused_rows, snapshot[7])
        else:
            self.paused_rows = None
        if self.mindaq.paused != self.paused:
            self.mindaq.toggle_pause()
        self.pause_button.setText(
            "Resume Both" if self.paused else "Pause Both")

    def request_status(self):
        if self.worker.connected.is_set():
            self.worker.send("STATUS")

    def draw_rows(self, rows, state):
        if not rows:
            return
        end = rows[-1][0]
        visible = np.asarray(recent_rows(rows), dtype=float)
        x = visible[:, 0] - end
        self.angle_curve.setData(x, visible[:, 2])
        self.velocity_curve.setData(x, visible[:, 3])
        self.command_curve.setData(x, visible[:, 5])
        self.battery_curve.setData(x, visible[:, 4])
        target = (rows[-1][6] if state == 2
                  else self.target.value())
        trigger = rows[-1][7]
        self.target_line.show()
        trigger_visible = bool(np.isfinite(trigger))
        self.trigger_line.setVisible(trigger_visible)
        self.target_line.setValue(target)
        if trigger_visible:
            self.trigger_line.setValue(trigger)

    def update(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            if message:
                self.last_message = message
            if message.startswith("OK STATUS"):
                fields = dict(
                    part.split("=", 1) for part in message.split()
                    if "=" in part)
                label = ("Set voltage" if fields.get("MODE") == "VOLTAGE"
                         else "Set velocity")
                try:
                    self.device_status.setText(
                        f"{label}: {float(fields['SET']):.1f} | "
                        f"Vbat: {float(fields['VBAT']):.1f} V\n"
                        f"Target: {float(fields['TARGET']):.1f} rad | "
                        f"Angle: {float(fields['ANGLE']):.1f} rad")
                except (KeyError, ValueError):
                    pass
                if self.last_state in (1, 2):
                    try:
                        self.rebound_count = int(fields["REBOUNDS"])
                    except (KeyError, ValueError):
                        pass
            if message.startswith((
                    "OK MOVE", "OK ADD_REBOUND", "OK REMOVE_REBOUND",
                    "EVENT TARGET", "EVENT REBOUND")):
                self.error = ""
                try:
                    self.rebound_count = int(
                        message.rsplit(" ", 1)[-1].split("=")[-1])
                except ValueError:
                    pass
            elif message.startswith((
                    "EVENT DONE", "EVENT HOMED", "EVENT STOPPED")):
                self.error = ""
                self.rebound_count = 0
            elif message.startswith(("ERROR", "BLE error")):
                self.error = message
            elif message.startswith(("OK ", "EVENT ")):
                self.error = ""
            if message.startswith(("EVENT DONE", "EVENT HOMED",
                                   "EVENT STOPPED", "ERROR")):
                self.data.stop_recording()

        (rows, recording, missing, bad, timeouts, drops, motor_id,
         state, mode) = self.data.snapshot()
        if state == 0 and self.last_state in (1, 2):
            self.rebound_count = 0
        self.last_state = state
        self.count_label.setText(str(self.rebound_count))
        connected = self.worker.connected.is_set()
        if connected and not self.was_connected:
            self.send_off_mode()
            self.send_mode()
            self.request_status()
        self.was_connected = connected
        rate = 0.0
        if len(rows) > 1:
            end = rows[-1][0]
            times = np.fromiter((row[0] for row in rows), float)
            rate_start = np.searchsorted(times, end - 1.0)
            if len(rows) - rate_start > 1:
                rate = ((len(rows) - rate_start - 1) /
                        (end - times[rate_start]))
            if not self.paused:
                self.draw_rows(rows, state)

        state_name = STATE_NAMES[state] if state < len(STATE_NAMES) else "?"
        mode_name = MODE_NAMES[mode] if mode < len(MODE_NAMES) else "?"
        self.status.setText(
            f"{'connected' if connected else 'disconnected'} | "
            f"{rate:.0f} Hz | {state_name}/{mode_name}"
            f"{' | recording' if recording else ''}"
            f"{' | plot paused' if self.paused else ''}"
            f"{f' | {self.error}' if self.error else ''}")
        self.move_button.setEnabled(connected)
        self.mode.setEnabled(connected and state == 0)
        self.setpoint.setEnabled(connected and state == 0)
        self.target.setEnabled(connected and state == 0)
        self.minus_button.setEnabled(
            connected and state != 3 and self.rebound_count > 0)
        self.plus_button.setEnabled(
            connected and state != 3 and self.rebound_count < 8)

    def save_csv(self):
        if self.paused_rows is not None:
            rows = self.paused_rows
        else:
            rows = self.data.snapshot()[0]
            rows = recent_rows(rows)
        if not rows:
            return
        directory = Path(self.settings.value("save_dir", str(LOG_DIR)))
        default = directory / f"motor_{datetime.now():%Y%m%d_%H%M%S}.csv"
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save motor log", str(default), "CSV files (*.csv)")
        if not selected:
            return
        path = Path(selected)
        if not path.suffix:
            path = path.with_suffix(".csv")
        self.write_motor_csv(path, rows)
        self.settings.setValue("save_dir", str(path.parent))
        self.settings.sync()
        self.last_message = f"saved {path}"
        self.device_status.setText(f"Saved: {path}")

    @staticmethod
    def write_motor_csv(path, rows):
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow((
                "time_s", "sample", "angle_rad", "velocity_rad_s",
                "battery_v", "command", "target_rad",
                "rebound_trigger_rad", "state", "mode",
            ))
            writer.writerows(rows)

    def write_mindaq_csv(self, path, samples):
        plot = self.mindaq
        trace = plot.trigger_trace
        pos = plot.pos_channel()
        neg = plot.neg_channel()
        latest_sample = samples[-1]
        wall_now_s = time.time()
        adc_header = [
            name
            for index in range(capture_gui.CHANNELS)
            for name in (f"adc_{index}_uv", f"adc_{index}_raw")
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["datetime", "capture_time_s", "voltage_zeroed_uv",
                 "voltage_uv", "device_time_s", "trigger_time_s",
                 "pulse_start_time_s", "pulse_end_time_s",
                 "trigger_threshold_uv", "trigger_cross", "pulse_start",
                 "pulse_active", "trigger_gpio", "trigger_edge", "seq",
                 "pos_channel", "neg_channel"]
                + adc_header
                + ["gain_code", "warn_flags", "clip_flags"]
            )
            for sample in samples:
                value, common_values = plot.common_csv_values(sample, pos, neg)
                capture_time_s = (
                    sample.timestamp_s
                    - trace.pulse_start_seq / capture_gui.SAMPLE_RATE_HZ
                )
                row = [
                    plot.sample_datetime(sample, latest_sample, wall_now_s),
                    f"{capture_time_s:.8f}",
                    f"{value - plot.zero_uv:.3f}",
                    common_values[0],
                    f"{sample.timestamp_s:.8f}",
                ]
                row += (
                    plot.capture_csv_values(sample)
                    + common_values[1:]
                    + [sample.gain_code, sample.warn_flags, sample.clip_flags]
                )
                writer.writerow(row)

    def save_both(self):
        if self.paused_rows is None:
            QtWidgets.QMessageBox.information(
                self, "Save Both", "Pause the motor plot before saving.")
            return
        if not self.paused_rows:
            QtWidgets.QMessageBox.information(
                self, "Save Both", "There is no paused motor data.")
            return
        trace = self.mindaq.trigger_trace
        if trace is None or not trace.samples:
            QtWidgets.QMessageBox.information(
                self, "Save Both", "There is no minDAQ capture to save.")
            return

        directory = Path(self.settings.value("save_dir", str(LOG_DIR)))
        default = directory / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save both logs", str(default), "CSV files (*.csv)")
        if not selected:
            return
        base = Path(selected)
        if base.suffix.lower() == ".csv":
            base = base.with_suffix("")
        motor_path = base.parent / f"{base.name}_motor.csv"
        mindaq_path = base.parent / f"{base.name}_mindaq.csv"
        existing = [path.name for path in (motor_path, mindaq_path)
                    if path.exists()]
        if existing:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace files?",
                "Replace " + " and ".join(existing) + "?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

        base.parent.mkdir(parents=True, exist_ok=True)
        self.write_motor_csv(motor_path, self.paused_rows)
        self.write_mindaq_csv(mindaq_path, trace.samples)
        self.settings.setValue("save_dir", str(base.parent))
        self.settings.sync()
        self.last_message = f"saved {motor_path.name} and {mindaq_path.name}"
        self.device_status.setText(
            f"Saved: {motor_path.name} + {mindaq_path.name}")

    def closeEvent(self, event):
        self.worker.stop()
        self.mindaq_reader.stop()
        self.worker.join(timeout=2.0)
        self.mindaq_reader.join(timeout=1.0)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    window = Window()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
