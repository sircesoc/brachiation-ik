"""Standalone read-only feedback for Robstride RS-02/RS-03 motors over gs_usb CAN.

Polls each motor's mechanical position, velocity, torque, temperature, and
bus voltage in background threads and exposes thread-safe getters. Loads
per-motor zero offsets from positions_log.json (the file the servo GUI
writes), so `joint_deg` is already referenced to the logged zero.

This module never commands motors. Pair it with the servo GUI (or your own
commander) to drive them; this side just observes.

Usage as library:
    reader = MotorReader()
    reader.start()
    state = reader.snapshot()
    print(state[1].joint_deg)   # motor 1, degrees from logged zero
    reader.stop()

Usage from shell:
    sudo python3 motor_reader.py
"""

import argparse
import copy
import json
import math
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Path to the directory containing can_config.py and positions_log.json.
DEFAULT_CONFIG_PATH = Path("/home/orin/robstridedebug")
DEFAULT_LOG_FILE = "positions_log.json"

# ── Protocol constants (mirror servo_rs03_ranged.py) ─────────────────────────
HOST_ID = 0xFE
BITRATE = 1_000_000
POLL_HZ = 20.0

COMM_GET_ID           = 0
COMM_MIT_CONTROL      = 1
COMM_OPERATION_STATUS = 2
COMM_ENABLE           = 3
COMM_DISABLE          = 4
COMM_READ_PARAMETER   = 17
COMM_WRITE_PARAMETER  = 18
COMM_FAULT            = 21

PARAM_RUN_MODE      = 0x7005
PARAM_MECH_POSITION = 0x7019
PARAM_VBUS          = 0x701C
RUN_MODE_MIT        = 0


def _bipolar_u16(x, x_max):
    x = max(-x_max, min(x_max, x))
    return int((x / x_max + 1.0) * 0x7FFF)


def _unipolar_u16(x, x_max):
    x = max(0.0, min(x_max, x))
    return int(x / x_max * 0xFFFF)


def _pack_mit(pos_rad, kp, kd, tor, lim):
    """Pack a Robstride MIT-mode command. Returns (8-byte data, 16-bit
    torque embedded in CAN ID)."""
    data = struct.pack(">HHHH",
        _bipolar_u16(pos_rad, lim["pos_max"]),
        _bipolar_u16(0.0,     lim["vel_max"]),
        _unipolar_u16(kp,     lim["kp_max"]),
        _unipolar_u16(kd,     lim["kd_max"]),
    )
    tor_u16 = _bipolar_u16(tor, lim["tor_max"])
    return data, tor_u16


# ── Frame helpers ────────────────────────────────────────────────────────────
def _parse_rx_id(can_id):
    comm_type = (can_id >> 24) & 0x1F
    src_id    = (can_id >> 8) & 0xFF
    extra     = (can_id >> 8) & 0xFFFF
    return comm_type, extra, src_id


def _parse_param_response(raw):
    if len(raw) < 8:
        return None, None
    return struct.unpack('<H', raw[0:2])[0], struct.unpack('<f', raw[4:8])[0]


def _parse_operation_status(raw, pos_max, vel_max, tq_max):
    if len(raw) < 8:
        return None
    p16, v16, t16, temp16 = struct.unpack(">HHHH", raw)
    return {
        "pos":  (p16 / 0x7FFF - 1.0) * pos_max,
        "vel":  (v16 / 0x7FFF - 1.0) * vel_max,
        "tor":  (t16 / 0x7FFF - 1.0) * tq_max,
        "temp": temp16 * 0.1,
    }


def _decode_fault(raw):
    if len(raw) < 8:
        return []
    fault_val, warn_val = struct.unpack("<LL", raw)
    out = []
    if (fault_val >> 0) & 1: out.append("overtemp")
    if (fault_val >> 1) & 1: out.append("gate-fault")
    if (fault_val >> 2) & 1: out.append("undervoltage")
    if (fault_val >> 3) & 1: out.append("overvoltage")
    if (fault_val >> 7) & 1: out.append("uncalibrated")
    if (warn_val  >> 14) & 1: out.append("stall")
    return out


# ── State dataclass ──────────────────────────────────────────────────────────
@dataclass
class MotorState:
    motor_id: int
    bus_name: str
    model: str = "RS03"
    pos_rad: float = 0.0
    vel_rad_s: float = 0.0
    tor_nm: float = 0.0
    temp_c: float = 0.0
    vbus_v: float = float("nan")
    zero_rad: float = 0.0
    min_from_zero_deg: float = -180.0
    max_from_zero_deg: float = 180.0
    last_update: float = 0.0
    fault: List[str] = field(default_factory=list)

    @property
    def joint_deg(self) -> float:
        """Position in degrees, referenced to the logged zero."""
        return math.degrees(self.pos_rad - self.zero_rad)

    @property
    def joint_rad(self) -> float:
        return self.pos_rad - self.zero_rad

    def is_stale(self, max_age_s: float = 0.5) -> bool:
        if self.last_update == 0.0:
            return True
        return (time.monotonic() - self.last_update) > max_age_s


# ── MotorReader ──────────────────────────────────────────────────────────────
class MotorReader:
    def __init__(self,
                 config_path: Path = DEFAULT_CONFIG_PATH,
                 log_file: str = DEFAULT_LOG_FILE,
                 only_buses: Optional[List[str]] = None):
        self.config_path = Path(config_path)
        self.log_file = log_file
        self.only_buses = only_buses

        self._states: Dict[int, MotorState] = {}
        self._buses: Dict[str, object] = {}
        self._bus_motors: Dict[str, List[int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._limits: Dict[int, dict] = {}
        self._models: Dict[int, str] = {}

        self._load_config()
        self._load_log()

    # ── config / log ───────────────────────────────────────────────────
    def _load_config(self):
        if str(self.config_path) not in sys.path:
            sys.path.insert(0, str(self.config_path))
        from can_config import (CAN_BUSES, MODEL_LIMITS, MOTOR_MODELS,
                                gs_usb_channel_index)
        self._can_config = {
            "CAN_BUSES": CAN_BUSES,
            "MODEL_LIMITS": MODEL_LIMITS,
            "MOTOR_MODELS": MOTOR_MODELS,
            "gs_usb_channel_index": gs_usb_channel_index,
        }
        bus_names = list(CAN_BUSES.keys())
        if self.only_buses is not None:
            bus_names = [b for b in bus_names if b in self.only_buses]
        for bus_name in bus_names:
            mids = list(CAN_BUSES[bus_name]["motors"])
            self._bus_motors[bus_name] = mids
            for mid in mids:
                model = MOTOR_MODELS.get(mid, "RS03")
                self._models[mid] = model
                self._limits[mid] = MODEL_LIMITS[model]
                self._states[mid] = MotorState(motor_id=mid, bus_name=bus_name,
                                               model=model)

    def _load_log(self):
        log_path = self.config_path / self.log_file
        if not log_path.exists():
            return
        try:
            with open(log_path) as f:
                entries = json.load(f)
        except Exception as exc:
            print(f"[motor_reader] failed to read {log_path}: {exc}")
            return

        # Most recent entry that has data for at least one of our motors.
        chosen = None
        for entry in reversed(entries):
            motors = entry.get("motors", {})
            if any(str(mid) in motors for mid in self._states):
                chosen = entry
                break
        if chosen is None:
            return

        for mid, state in self._states.items():
            m = chosen["motors"].get(str(mid))
            if not m or m.get("zero_deg") is None:
                continue
            state.zero_rad = math.radians(m["zero_deg"])
            lo = m.get("min_from_zero_deg")
            hi = m.get("max_from_zero_deg")
            if lo is not None and hi is not None and hi - lo >= 1.0:
                state.min_from_zero_deg = lo
                state.max_from_zero_deg = hi

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self, ping_attempts: int = 2, ping_timeout: float = 0.3):
        import can
        gs_usb_channel_index = self._can_config["gs_usb_channel_index"]
        CAN_BUSES = self._can_config["CAN_BUSES"]

        # Open each bus; skip (with warning) if its adapter isn't present.
        for bus_name in list(self._bus_motors):
            serial = CAN_BUSES[bus_name]["serial"]
            try:
                ch = gs_usb_channel_index(serial)
                bus = can.Bus(interface="gs_usb", channel=ch, bitrate=BITRATE)
            except Exception as exc:
                missing_mids = self._bus_motors.pop(bus_name)
                for mid in missing_mids:
                    self._states.pop(mid, None)
                print(f"[motor_reader] {bus_name} skipped "
                      f"(motors {missing_mids}): {exc}")
                continue
            self._buses[bus_name] = bus

        if not self._buses:
            raise RuntimeError("[motor_reader] no usable CAN buses found")

        # Ping each motor so we know which ones are alive (non-fatal).
        for bus_name, mids in self._bus_motors.items():
            bus = self._buses[bus_name]
            for mid in mids:
                if not self._ping(bus, mid, attempts=ping_attempts,
                                  timeout=ping_timeout):
                    print(f"[motor_reader] motor {mid} on {bus_name}: no reply")

        # Spawn one reader thread per bus.
        for bus_name in self._bus_motors:
            t = threading.Thread(target=self._bus_loop, args=(bus_name,),
                                 daemon=True, name=f"motor_reader-{bus_name}")
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for bus in self._buses.values():
            try:
                bus.shutdown()
            except Exception:
                pass

    # ── public reads ────────────────────────────────────────────────────
    def snapshot(self) -> Dict[int, MotorState]:
        with self._lock:
            return {mid: copy.copy(s) for mid, s in self._states.items()}

    def get(self, motor_id: int) -> MotorState:
        with self._lock:
            return copy.copy(self._states[motor_id])

    @property
    def motor_ids(self) -> List[int]:
        return sorted(self._states.keys())

    # ── command (write side) ────────────────────────────────────────────
    def _bus_for_motor(self, motor_id):
        for bus_name, mids in self._bus_motors.items():
            if motor_id in mids:
                return bus_name
        return None

    def enable(self, motor_id):
        """Enable motor and put it in MIT mode."""
        import can
        bus_name = self._bus_for_motor(motor_id)
        if bus_name is None or bus_name not in self._buses:
            raise RuntimeError(f"motor {motor_id} not on any open bus")
        bus = self._buses[bus_name]
        can_id_en = (COMM_ENABLE << 24) | (HOST_ID << 8) | motor_id
        bus.send(can.Message(arbitration_id=can_id_en,
                             is_extended_id=True, data=bytes(8)))
        time.sleep(0.05)
        can_id_wr = (COMM_WRITE_PARAMETER << 24) | (HOST_ID << 8) | motor_id
        data = (struct.pack("<HH", PARAM_RUN_MODE, 0)
                + struct.pack("<bBH", RUN_MODE_MIT, 0, 0))
        bus.send(can.Message(arbitration_id=can_id_wr,
                             is_extended_id=True, data=data))
        time.sleep(0.05)

    def disable(self, motor_id):
        import can
        bus_name = self._bus_for_motor(motor_id)
        if bus_name is None or bus_name not in self._buses:
            return
        bus = self._buses[bus_name]
        can_id = (COMM_DISABLE << 24) | (HOST_ID << 8) | motor_id
        try:
            bus.send(can.Message(arbitration_id=can_id,
                                 is_extended_id=True, data=bytes(8)))
        except Exception:
            pass

    def disable_all(self):
        for mid in list(self._states):
            try:
                self.disable(mid)
            except Exception:
                pass

    def command(self, motor_id, target_pos_rad, kp=5.0, kd=0.5, tor=0.0):
        """Send one MIT-mode command frame. Caller must have called enable()."""
        import can
        bus_name = self._bus_for_motor(motor_id)
        if bus_name is None or bus_name not in self._buses:
            raise RuntimeError(f"motor {motor_id} not on any open bus")
        bus = self._buses[bus_name]
        lim = self._limits[motor_id]
        data, tor_u16 = _pack_mit(target_pos_rad, kp, kd, tor, lim)
        can_id = (COMM_MIT_CONTROL << 24) | (tor_u16 << 8) | motor_id
        bus.send(can.Message(arbitration_id=can_id,
                             is_extended_id=True, data=data))

    # ── internals ───────────────────────────────────────────────────────
    def _ping(self, bus, motor_id, attempts=2, timeout=0.3) -> bool:
        import can
        for _ in range(attempts):
            can_id = (COMM_GET_ID << 24) | (HOST_ID << 8) | motor_id
            try:
                bus.send(can.Message(arbitration_id=can_id,
                                     is_extended_id=True, data=bytes(8)))
            except Exception:
                return False
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=0.02)
                if msg is None or not msg.is_extended_id:
                    continue
                comm, _, src = _parse_rx_id(msg.arbitration_id)
                if comm == COMM_GET_ID and src == motor_id:
                    return True
        return False

    def _request_pos(self, bus, motor_id):
        import can
        can_id = (COMM_READ_PARAMETER << 24) | (HOST_ID << 8) | motor_id
        bus.send(can.Message(
            arbitration_id=can_id, is_extended_id=True,
            data=struct.pack('<H', PARAM_MECH_POSITION) + bytes(6)))

    def _request_vbus(self, bus, motor_id):
        import can
        can_id = (COMM_READ_PARAMETER << 24) | (HOST_ID << 8) | motor_id
        bus.send(can.Message(
            arbitration_id=can_id, is_extended_id=True,
            data=struct.pack('<H', PARAM_VBUS) + bytes(6)))

    def _bus_loop(self, bus_name: str):
        bus = self._buses[bus_name]
        mids = self._bus_motors[bus_name]
        poll_interval = 1.0 / POLL_HZ
        last_poll = {mid: 0.0  for mid in mids}
        last_vbus = {mid: -1.0 for mid in mids}

        while not self._stop.is_set():
            now = time.monotonic()
            for mid in mids:
                if now - last_poll[mid] >= poll_interval:
                    try:
                        self._request_pos(bus, mid)
                    except Exception:
                        pass
                    last_poll[mid] = now
                if now - last_vbus[mid] >= 2.0:
                    try:
                        self._request_vbus(bus, mid)
                    except Exception:
                        pass
                    last_vbus[mid] = now

            msg = bus.recv(timeout=0.002)
            if msg is None or not msg.is_extended_id:
                continue
            comm, _, src = _parse_rx_id(msg.arbitration_id)
            if src not in self._states:
                continue
            raw = bytes(msg.data)
            lim = self._limits[src]

            if comm == COMM_READ_PARAMETER:
                pid, value = _parse_param_response(raw)
                if pid == PARAM_MECH_POSITION:
                    with self._lock:
                        s = self._states[src]
                        s.pos_rad = value
                        s.last_update = now
                elif pid == PARAM_VBUS:
                    with self._lock:
                        self._states[src].vbus_v = value
            elif comm == COMM_OPERATION_STATUS:
                parsed = _parse_operation_status(
                    raw, lim["pos_max"], lim["vel_max"], lim["tor_max"])
                if parsed:
                    with self._lock:
                        s = self._states[src]
                        s.pos_rad = parsed["pos"]
                        s.vel_rad_s = parsed["vel"]
                        s.tor_nm = parsed["tor"]
                        s.temp_c = parsed["temp"]
                        s.last_update = now
            elif comm == COMM_FAULT:
                faults = _decode_fault(raw)
                with self._lock:
                    self._states[src].fault = faults


# ── Standalone test driver ──────────────────────────────────────────────────
def _main(argv=None):
    parser = argparse.ArgumentParser(description="Motor feedback reader (read-only)")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH),
                        help=f"directory with can_config.py & positions_log.json "
                             f"(default {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE,
                        help=f"log filename inside config-path (default {DEFAULT_LOG_FILE})")
    parser.add_argument("--rate", type=float, default=5.0,
                        help="print rate Hz (default 5)")
    parser.add_argument("--bus", action="append", default=None,
                        help="restrict to specific bus name(s); repeat for multiple")
    args = parser.parse_args(argv)

    reader = MotorReader(config_path=Path(args.config_path),
                         log_file=args.log, only_buses=args.bus)
    print(f"motors: {reader.motor_ids}")
    reader.start()
    period = 1.0 / max(args.rate, 0.1)
    try:
        while True:
            snap = reader.snapshot()
            cols = []
            for mid in reader.motor_ids:
                s = snap[mid]
                stale = "*" if s.is_stale(0.5) else " "
                fault = (" " + ",".join(s.fault)) if s.fault else ""
                cols.append(
                    f"M{mid:>2}{stale}{s.joint_deg:+7.2f}° "
                    f"v={s.vel_rad_s:+5.2f} t={s.tor_nm:+5.2f}Nm "
                    f"T={s.temp_c:4.1f}°C{fault}"
                )
            print("  |  ".join(cols))
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        reader.stop()


if __name__ == "__main__":
    _main()
