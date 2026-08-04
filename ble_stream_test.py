#!/usr/bin/env python3.11
import argparse
import asyncio
import csv
import statistics
import struct
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner


BLE_NAME = "dnajumper-test"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
MAGIC = 0xD14A
HEADER = struct.Struct("<HHIIfffHHHBb")
SAMPLE = struct.Struct("<Hff")


class Results:
    def __init__(self):
        self.started = time.monotonic()
        self.packets = 0
        self.samples = 0
        self.bad_packets = 0
        self.missing_packets = 0
        self.missing_samples = 0
        self.expected_packet = None
        self.expected_sample = None
        self.first_time_us = None
        self.last_time_us = None
        self.last_sample_time_us = None
        self.periods_us = []
        self.uart_timeouts = 0
        self.queue_drops = 0
        self.notify_errors = 0
        self.motor_id = -1
        self.rows = []

    def receive(self, _, data):
        data = bytes(data)
        if len(data) < HEADER.size:
            self.bad_packets += 1
            return

        (magic, packet_seq, first_sample, base_time_us, battery, command, target,
         uart_timeouts, queue_drops, notify_errors, count,
         motor_id) = HEADER.unpack_from(data)
        expected_size = HEADER.size + count * SAMPLE.size
        if magic != MAGIC or count == 0 or len(data) != expected_size:
            self.bad_packets += 1
            return

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

        for index in range(count):
            offset = HEADER.size + index * SAMPLE.size
            sample_offset_us, angle, velocity = SAMPLE.unpack_from(data, offset)
            time_us = (base_time_us + sample_offset_us) & 0xFFFFFFFF
            if self.first_time_us is None:
                self.first_time_us = time_us
            if self.last_sample_time_us is not None:
                period = (time_us - self.last_sample_time_us) & 0xFFFFFFFF
                if period < 100000:
                    self.periods_us.append(period)
            self.last_sample_time_us = time_us
            self.last_time_us = time_us
            self.rows.append((
                time_us,
                (first_sample + index) & 0xFFFFFFFF,
                angle,
                velocity,
                battery,
                command,
                target,
            ))

        self.packets += 1
        self.samples += count

    def device_rate(self):
        if self.samples < 2 or self.first_time_us == self.last_time_us:
            return 0.0
        elapsed = ((self.last_time_us - self.first_time_us) & 0xFFFFFFFF) / 1e6
        return (self.samples - 1) / elapsed

    def report(self):
        wall = time.monotonic() - self.started
        median = statistics.median(self.periods_us) if self.periods_us else 0
        ordered = sorted(self.periods_us)
        p99 = ordered[int(0.99 * (len(ordered) - 1))] if ordered else 0
        print(
            f"wall={wall:.1f}s samples={self.samples} "
            f"receive={self.samples / wall:.1f}Hz device={self.device_rate():.1f}Hz "
            f"period_median={median:.0f}us period_p99={p99:.0f}us "
            f"missing_samples={self.missing_samples} "
            f"missing_packets={self.missing_packets} bad={self.bad_packets} "
            f"uart_timeouts={self.uart_timeouts} queue_drops={self.queue_drops} "
            f"notify_errors={self.notify_errors} motor={self.motor_id}"
        )


async def find_device(timeout):
    service_uuid = SERVICE_UUID.lower()

    def matches(device, advertisement):
        names = {device.name, advertisement.local_name}
        services = {uuid.lower() for uuid in advertisement.service_uuids}
        return BLE_NAME in names or service_uuid in services

    return await BleakScanner.find_device_by_filter(matches, timeout=timeout)


async def run(args):
    print(f"scanning for {BLE_NAME}")
    device = await find_device(args.scan_timeout)
    if device is None:
        raise SystemExit("benchmark device not found")

    results = Results()
    async with BleakClient(device, timeout=10.0) as client:
        await client.start_notify(NOTIFY_UUID, results.receive)
        await client.write_gatt_char(WRITE_UUID, b"START", response=False)
        if args.wave:
            await client.write_gatt_char(WRITE_UUID, b"WAVE", response=False)
        results.started = time.monotonic()
        print(f"connected {device.address}")

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            await asyncio.sleep(min(1.0, deadline - time.monotonic()))
            results.report()

        await client.write_gatt_char(WRITE_UUID, b"OFF", response=False)
        await asyncio.sleep(0.1)
        await client.write_gatt_char(WRITE_UUID, b"STOP", response=False)
        await client.stop_notify(NOTIFY_UUID)
    print("final")
    results.report()

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        first_time = results.rows[0][0]
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "time_us", "time_s", "sample_seq", "angle_rad",
                "velocity_rad_s", "battery_v", "command_v", "target_rad",
            ])
            for row in results.rows:
                elapsed = ((row[0] - first_time) & 0xFFFFFFFF) / 1e6
                writer.writerow([row[0], elapsed, *row[1:]])
        print(f"saved {len(results.rows)} samples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--scan-timeout", type=float, default=8.0)
    parser.add_argument("--wave", action="store_true")
    parser.add_argument("--csv")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
