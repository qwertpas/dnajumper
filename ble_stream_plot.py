#!/usr/bin/env python3.11
import asyncio
import csv
import queue
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from bleak import BleakClient, BleakScanner
from pyqtgraph.Qt import QtCore, QtWidgets


BLE_NAME = "dnajumper-test"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MAGIC = 0xD14A
HEADER = struct.Struct("<HHIIfffHHHBb")
SAMPLE = struct.Struct("<Hff")
PLOT_SECONDS = 10.0
WAVE_SECONDS = 10.0


class DataStore:
    def __init__(self):
        self.samples = deque(maxlen=90000)
        self.recorded = []
        self.recording = False
        self.lock = threading.Lock()
        self.expected_packet = None
        self.expected_sample = None
        self.missing_packets = 0
        self.missing_samples = 0
        self.bad_packets = 0
        self.uart_timeouts = 0
        self.queue_drops = 0
        self.notify_errors = 0
        self.motor_id = -1

    def begin_link(self):
        with self.lock:
            self.expected_packet = None
            self.expected_sample = None

    def receive(self, _, data):
        data = bytes(data)
        if len(data) < HEADER.size:
            with self.lock:
                self.bad_packets += 1
            return

        (magic, packet_seq, first_sample, base_time_us, battery, command,
         target, uart_timeouts, queue_drops, notify_errors, count,
         motor_id) = HEADER.unpack_from(data)
        if (magic != MAGIC or count == 0 or
                len(data) != HEADER.size + count * SAMPLE.size):
            with self.lock:
                self.bad_packets += 1
            return

        rows = []
        for index in range(count):
            offset = HEADER.size + index * SAMPLE.size
            sample_offset_us, angle, velocity = SAMPLE.unpack_from(data, offset)
            rows.append((
                (base_time_us + sample_offset_us) & 0xFFFFFFFF,
                (first_sample + index) & 0xFFFFFFFF,
                angle,
                velocity,
                battery,
                command,
                target,
            ))

        with self.lock:
            if self.expected_packet is not None:
                gap = (packet_seq - self.expected_packet) & 0xFFFF
                if gap < 0x8000:
                    self.missing_packets += gap
            if self.expected_sample is not None:
                gap = (first_sample - self.expected_sample) & 0xFFFFFFFF
                if gap < 0x80000000:
                    self.missing_samples += gap

            self.expected_packet = (packet_seq + 1) & 0xFFFF
            self.expected_sample = (first_sample + count) & 0xFFFFFFFF
            self.uart_timeouts = uart_timeouts
            self.queue_drops = queue_drops
            self.notify_errors = notify_errors
            self.motor_id = motor_id
            self.samples.extend(rows)
            if self.recording:
                self.recorded.extend(rows)

    def begin_recording(self):
        with self.lock:
            self.recorded.clear()
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False

    def clear(self):
        with self.lock:
            self.samples.clear()
            self.recorded.clear()

    def snapshot(self):
        with self.lock:
            return (
                list(self.samples),
                self.missing_samples,
                self.bad_packets,
                self.uart_timeouts,
                self.queue_drops,
                self.notify_errors,
                self.motor_id,
            )

    def recording_snapshot(self):
        with self.lock:
            return list(self.recorded)


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
        service_uuid = SERVICE_UUID.lower()

        def matches(device, advertisement):
            names = {device.name, advertisement.local_name}
            services = {uuid.lower() for uuid in advertisement.service_uuids}
            return BLE_NAME in names or service_uuid in services

        return await BleakScanner.find_device_by_filter(matches, timeout=5.0)

    async def run_async(self):
        while not self.stop_event.is_set():
            try:
                self.messages.put("scanning")
                device = await self.find_device()
                if device is None:
                    await asyncio.sleep(0.5)
                    continue

                async with BleakClient(device, timeout=10.0) as client:
                    self.data.begin_link()
                    await client.start_notify(NOTIFY_UUID, self.data.receive)
                    await client.write_gatt_char(
                        WRITE_UUID, b"START", response=False)
                    self.connected.set()
                    self.messages.put(f"connected {device.address}")

                    while not self.stop_event.is_set() and client.is_connected:
                        while True:
                            try:
                                command = self.commands.get_nowait()
                            except queue.Empty:
                                break
                            await client.write_gatt_char(
                                WRITE_UUID, command.encode(), response=False)
                        await asyncio.sleep(0.01)

                    if client.is_connected:
                        await client.write_gatt_char(
                            WRITE_UUID, b"OFF", response=False)
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


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, worker, data, messages):
        super().__init__()
        self.worker = worker
        self.data = data
        self.messages = messages
        self.run_id = 0
        self.last_message = "starting"

        self.setWindowTitle("DNA Jumper BLE stream test")
        self.resize(1200, 850)
        layout = QtWidgets.QVBoxLayout(self)

        self.status = QtWidgets.QLabel("starting")
        layout.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton(
            "Run ±1 V square wave (10 s)")
        self.run_button.clicked.connect(self.start_wave)
        self.run_button.setEnabled(False)
        buttons.addWidget(self.run_button)

        self.stop_button = QtWidgets.QPushButton("STOP MOTOR")
        self.stop_button.clicked.connect(self.stop_wave)
        buttons.addWidget(self.stop_button)

        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.data.clear)
        buttons.addWidget(clear_button)

        save_button = QtWidgets.QPushButton("Save CSV")
        save_button.clicked.connect(self.save_csv)
        buttons.addWidget(save_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.angle_plot = pg.PlotWidget(title="Angle")
        self.angle_plot.setLabel("left", "rad")
        self.angle_plot.setLabel("bottom", "time", "s")
        self.angle_curve = self.angle_plot.plot(
            pen=pg.mkPen("#2980b9", width=2))
        layout.addWidget(self.angle_plot)

        self.velocity_plot = pg.PlotWidget(title="Velocity")
        self.velocity_plot.setLabel("left", "rad/s")
        self.velocity_plot.setLabel("bottom", "time", "s")
        self.velocity_curve = self.velocity_plot.plot(
            pen=pg.mkPen("#27ae60", width=2))
        layout.addWidget(self.velocity_plot)

        self.voltage_plot = pg.PlotWidget(title="Voltage")
        self.voltage_plot.setLabel("left", "V")
        self.voltage_plot.setLabel("bottom", "time", "s")
        self.voltage_plot.addLegend()
        self.command_curve = self.voltage_plot.plot(
            pen=pg.mkPen("#c0392b", width=2), name="command")
        self.battery_curve = self.voltage_plot.plot(
            pen=pg.mkPen("#f39c12", width=2), name="battery")
        layout.addWidget(self.voltage_plot)

        for plot in (
            self.angle_plot, self.velocity_plot, self.voltage_plot
        ):
            plot.showGrid(x=True, y=True, alpha=0.25)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(33)

    def start_wave(self):
        if not self.worker.connected.is_set():
            return
        self.run_id += 1
        run_id = self.run_id
        self.data.begin_recording()
        self.worker.send("WAVE")
        self.run_button.setEnabled(False)
        self.last_message = "running ±1 V square wave"
        QtCore.QTimer.singleShot(
            int((WAVE_SECONDS + 0.5) * 1000),
            lambda: self.finish_wave(run_id),
        )

    def finish_wave(self, run_id):
        if run_id != self.run_id:
            return
        self.data.stop_recording()
        self.run_button.setEnabled(self.worker.connected.is_set())
        self.last_message = "square wave complete; ready to save"

    def stop_wave(self):
        self.run_id += 1
        self.worker.send("OFF")
        self.data.stop_recording()
        self.run_button.setEnabled(self.worker.connected.is_set())
        self.last_message = "motor stopped"

    def save_csv(self):
        rows = self.data.recording_snapshot()
        if not rows:
            self.last_message = "no recorded samples"
            return

        default = "motor_square_" + datetime.now().strftime(
            "%Y%m%d_%H%M%S") + ".csv"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save motor log", default, "CSV files (*.csv)")
        if not filename:
            return

        first_time = rows[0][0]
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "time_us", "time_s", "sample_seq", "angle_rad",
                "velocity_rad_s", "battery_v", "command_v", "target_rad",
            ])
            for row in rows:
                elapsed = ((row[0] - first_time) & 0xFFFFFFFF) / 1e6
                writer.writerow([row[0], elapsed, *row[1:]])
        self.last_message = f"saved {len(rows)} samples to {filename}"

    def update_plot(self):
        while True:
            try:
                self.last_message = self.messages.get_nowait()
            except queue.Empty:
                break

        (rows, missing, bad, timeouts, drops, notify_errors,
         motor_id) = self.data.snapshot()
        connected = self.worker.connected.is_set()
        if not connected:
            self.run_button.setEnabled(False)
        elif not self.data.recording:
            self.run_button.setEnabled(True)

        rate = 0.0
        if len(rows) >= 2:
            duration = ((rows[-1][0] - rows[0][0]) & 0xFFFFFFFF) / 1e6
            if duration > 0:
                rate = (len(rows) - 1) / duration

            array = np.asarray(rows, dtype=float)
            elapsed = (
                (array[:, 0] - array[-1, 0]) % (2 ** 32)
            ) / 1e6
            elapsed[elapsed > 1000] -= 2 ** 32 / 1e6
            keep = elapsed >= -PLOT_SECONDS
            array = array[keep]
            elapsed = elapsed[keep]
            step = max(1, len(array) // 4000)
            array = array[::step]
            elapsed = elapsed[::step]

            self.angle_curve.setData(elapsed, array[:, 2])
            self.velocity_curve.setData(elapsed, array[:, 3])
            self.battery_curve.setData(elapsed, array[:, 4])
            self.command_curve.setData(elapsed, array[:, 5])

        self.status.setText(
            f"{self.last_message} | connected={connected} motor={motor_id} "
            f"rate={rate:.0f}Hz missing={missing} bad={bad} "
            f"uart_timeouts={timeouts} queue_drops={drops} "
            f"notify_errors={notify_errors}"
        )

    def closeEvent(self, event):
        self.worker.send("OFF")
        self.worker.stop()
        self.worker.join(timeout=3.0)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    data = DataStore()
    messages = queue.Queue()
    worker = BleWorker(data, messages)
    worker.start()
    window = PlotWindow(worker, data, messages)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
