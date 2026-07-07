#!/usr/bin/env python3
"""
Motor Control Terminal Interface
Connects to the dnajumper BLE device and sends motor commands.
Handles LOG command with base64 decoding and CSV export.

Usage:
    python3.11 motor_terminal.py

Commands:
    T<angle>  - Set target angle (e.g., T10, T-5.5)
    V<volts>  - Set voltage (e.g., V0.3)
    VOLT<n>   - Set voltage control mode with nV (e.g., VOLT2)
    VEL<n>    - Set velocity control mode with n rad/s (e.g., VEL20)
    R         - Load rebound after current/last target reaches and drops 2 rad after 500ms
    RTH<n>    - Set rebound reversal threshold in rad (e.g., RTH1.5)
    H/HOME    - Home to 0 rad at 5 rad/s, then restore previous mode
    PRBS [amp] [bit_ms] [dur_s] - PRBS excitation for sysid (default: 5V, 20ms, 10s)
    STOP/OFF  - Stop motor
    ZERO/Z    - Zero angle
    STATUS/S  - Get status
    LOG       - Download log data (prompts: Y [filename], N, or C to clear)
    quit/exit - Exit terminal

Chained Commands:
    v<V1>t<T1>v<V2>t<T2>... - Chain voltage/target commands without braking
    t<T>r - Move to target, then load rebound (e.g., t29r)
"""

import argparse
import asyncio
import base64
import csv
import os
import statistics
import struct
from datetime import datetime

from bleak import BleakClient, BleakScanner


BLE_NAME = "dnajumper"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
TIMEOUT = 2.0
LOG_TIMEOUT = 30.0

save_dir = "./motor_logs/"


class BleLink:
    def __init__(self, client):
        self.client = client
        self.queue = asyncio.Queue()
        self.text = ""

    def on_notify(self, sender, data):
        self.queue.put_nowait(bytes(data))

    async def clear(self):
        self.text = ""
        while not self.queue.empty():
            self.queue.get_nowait()

    async def send(self, cmd):
        await self.clear()
        await self.client.write_gatt_char(WRITE_UUID, cmd.encode("utf-8"), response=False)

    async def send_command(self, cmd, timeout=TIMEOUT):
        await self.send(cmd)
        return await self.read_response(timeout)

    async def read_response(self, timeout=TIMEOUT, quiet=0.03):
        text = self.text
        self.text = ""
        wait = quiet if text else timeout
        while True:
            try:
                data = await asyncio.wait_for(self.queue.get(), wait)
            except asyncio.TimeoutError:
                return text if text else None

            text += data.decode("utf-8", errors="replace")
            wait = quiet

    async def read_until(self, marker, timeout=TIMEOUT):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while marker not in self.text:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                data = await asyncio.wait_for(self.queue.get(), remaining)
            except asyncio.TimeoutError:
                return None
            self.text += data.decode("utf-8", errors="replace")

        end = self.text.find(marker) + len(marker)
        result = self.text[:end]
        self.text = self.text[end:]
        return result

    async def receive_until_end(self, timeout=LOG_TIMEOUT):
        marker = "END\n"
        while marker not in self.text:
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout)
            except asyncio.TimeoutError:
                print("Timeout waiting for data")
                break

            self.text += data.decode("utf-8", errors="replace")

        end = self.text.find(marker)
        if end >= 0:
            result = self.text[:end]
            self.text = self.text[end + len(marker):]
            return result

        result = self.text
        self.text = ""
        return result


def decode_log_data(base64_data):
    """
    Decode base64 log data into list of samples.
    Each sample is: angle(float), vel(float), vbat(float), set_volts(float), time_us(uint32)
    """
    try:
        raw_data = base64.b64decode(base64_data)
    except Exception as e:
        print(f"Base64 decode error: {e}")
        return None

    sample_size = 20
    num_samples = len(raw_data) // sample_size

    if len(raw_data) % sample_size != 0:
        print(f"Warning: Data size {len(raw_data)} not divisible by {sample_size}")

    samples = []
    for i in range(num_samples):
        offset = i * sample_size
        angle, vel, vbat, set_volts, time_us = struct.unpack(
            "<ffffI", raw_data[offset : offset + sample_size]
        )
        samples.append(
            {
                "time_us": time_us,
                "time_ms": time_us / 1000.0,
                "angle": angle,
                "vel": vel,
                "vbat": vbat,
                "set_volts": set_volts,
            }
        )

    return samples


def save_csv(samples, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["time_us", "time_ms", "angle", "vel", "vbat", "set_volts"]
        )
        writer.writeheader()
        writer.writerows(samples)
    print(f"Saved {len(samples)} samples to {filename}")


async def handle_log_command(link):
    response = await link.send_command("LOG")
    if response is None:
        print("No response from ESP32")
        return

    print(response, end="")

    if "NO LOG DATA" in response or "Download?" not in response:
        return

    raw_input = input().strip()
    parts = raw_input.split(maxsplit=1)
    confirm = parts[0].upper() if parts else "N"
    custom_filename = parts[1] if len(parts) > 1 else None

    if confirm in ("C", "CLEAR"):
        response = await link.send_command("C")
        if response:
            print(response, end="")
        return

    if confirm not in ("Y", "YES"):
        response = await link.send_command("N")
        if response:
            print(response, end="")
        return

    os.makedirs(save_dir, exist_ok=True)

    await link.send("Y")
    ack = await link.read_until("SENDING...\n")
    if ack:
        print(ack, end="")
    else:
        print("No acknowledgment received")
        return

    print("Receiving data...")
    base64_data = await link.receive_until_end()
    if not base64_data:
        print("No data received")
        return

    print(f"Received {len(base64_data)} characters of base64 data")

    samples = decode_log_data(base64_data)
    if samples is None:
        print("Failed to decode data")
        return

    print(f"Decoded {len(samples)} samples")

    if len(samples) > 1:
        dts = [
            samples[i]["time_ms"] - samples[i - 1]["time_ms"]
            for i in range(1, len(samples))
        ]
        median_period = statistics.median(dts)
        print(
            f"Duration: {samples[-1]['time_ms']:.1f}ms, "
            f"Median period: {median_period:.3f}ms ({1000 / median_period:.0f}Hz)"
        )

    if custom_filename:
        filename = custom_filename if custom_filename.endswith(".csv") else custom_filename + ".csv"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motor_log_{timestamp}.csv"

    save_csv(samples, os.path.join(save_dir, filename))


async def find_device(name, address, scan_timeout):
    if address:
        return await BleakScanner.find_device_by_address(address, timeout=scan_timeout)

    service_uuid = SERVICE_UUID.lower()

    def matches(device, adv):
        names = {device.name, adv.local_name}
        service_uuids = {uuid.lower() for uuid in adv.service_uuids}
        return name in names or service_uuid in service_uuids

    return await BleakScanner.find_device_by_filter(matches, timeout=scan_timeout)


async def run_terminal(args):
    print("=" * 50)
    print("Motor Control Terminal")
    print(f"Scanning for BLE device '{args.name}'")
    print("Commands: T<angle>, V<volts>, VOLT<n>, VEL<n>, R, RTH<n>, H, PRBS, STOP, ZERO, STATUS, LOG")
    print("Chains:   v<V1>t<T1>v<V2>t<T2>... or t<T>r (e.g., v4t5v6t15, t29r)")
    print("Type 'quit' or 'exit' to exit")
    print("=" * 50)

    device = await find_device(args.name, args.address, args.scan_timeout)
    if device is None:
        print("ESP32 BLE device not found")
        return

    async with BleakClient(device, timeout=10.0) as client:
        link = BleLink(client)
        await client.start_notify(NOTIFY_UUID, link.on_notify)
        print(f"Connected to {args.name} ({device.address})")

        print("\nTesting connection...")
        response = await link.send_command("STATUS")
        if response:
            print(f"Connected! {response}", end="")
        else:
            print("Warning: No response from ESP32 BLE device.")

        while True:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not cmd:
                continue

            if cmd.lower() in ("quit", "exit", "q"):
                print("Exiting...")
                break

            if cmd.upper() == "LOG":
                await handle_log_command(link)
                continue

            response = await link.send_command(cmd)
            if response:
                print(response, end="")
            else:
                print("No response (timeout)")

        await client.stop_notify(NOTIFY_UUID)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=BLE_NAME, help="BLE device name")
    parser.add_argument("--address", help="BLE device address")
    parser.add_argument("--scan-timeout", type=float, default=8.0, help="BLE scan timeout in seconds")
    return parser.parse_args()


def main():
    asyncio.run(run_terminal(parse_args()))


if __name__ == "__main__":
    main()
