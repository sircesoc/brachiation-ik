"""Standalone reader for WIT-Motion IMU (e.g. WT901, JY-901) over USB-serial.

Parses the 11-byte WIT frames (0x55 header) for acceleration (0x51),
angular velocity (0x52), Euler angles (0x53), and magnetometer (0x54).

Usage as a library:
    imu = IMUReader(port="/dev/ttyUSB0", baud=9600)
    imu.start()
    ax, ay, az = imu.acceleration       # m/s^2
    wx, wy, wz = imu.angular_velocity   # rad/s
    r, p, y    = imu.angle_degrees      # deg
    imu.stop()

Usage from the shell:
    python3 imu_reader.py            # auto-detect port, print 10 Hz
    python3 imu_reader.py /dev/ttyUSB0
"""

import math
import struct
import sys
import threading
import time
from glob import glob

import serial

WIT_HEADER = 0x55
ACC_ID = 0x51
GYRO_ID = 0x52
ANGLE_ID = 0x53
MAG_ID = 0x54
FRAME_LEN = 11

ACC_SCALE = 16.0 * 9.8 / 32768.0
GYRO_SCALE = 2000.0 * math.pi / 180.0 / 32768.0
ANGLE_SCALE = 180.0 / 32768.0


def _hex_to_short(raw):
    return list(struct.unpack("hhhh", bytearray(raw)))


def _checksum_ok(frame):
    return (sum(frame[0:10]) & 0xFF) == frame[10]


class IMUReader:
    def __init__(self, port="/dev/ttyUSB0", baud=9600, timeout=0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self.acceleration = (0.0, 0.0, 0.0)
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.angle_degrees = (0.0, 0.0, 0.0)
        self.magnetometer = (0, 0, 0)
        self.last_update = 0.0

        self._ser = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        if not self._ser.is_open:
            self._ser.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None and self._ser.is_open:
            self._ser.close()

    def _loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            while len(buf) >= FRAME_LEN:
                if buf[0] != WIT_HEADER:
                    del buf[0]
                    continue
                frame = bytes(buf[:FRAME_LEN])
                if _checksum_ok(frame):
                    self._handle_frame(frame)
                    del buf[:FRAME_LEN]
                else:
                    # Bad checksum — drop the header byte and resync.
                    del buf[0]

    def _handle_frame(self, frame):
        kind = frame[1]
        values = _hex_to_short(frame[2:10])[:3]
        with self._lock:
            if kind == ACC_ID:
                self.acceleration = tuple(v * ACC_SCALE for v in values)
            elif kind == GYRO_ID:
                self.angular_velocity = tuple(v * GYRO_SCALE for v in values)
            elif kind == ANGLE_ID:
                self.angle_degrees = tuple(v * ANGLE_SCALE for v in values)
            elif kind == MAG_ID:
                self.magnetometer = tuple(values)
            self.last_update = time.monotonic()


def autodetect_port():
    candidates = sorted(glob("/dev/ttyUSB*") + glob("/dev/imu_usb"))
    return candidates[0] if candidates else None


def _main(argv):
    port = argv[1] if len(argv) > 1 else (autodetect_port() or "/dev/ttyUSB0")
    print(f"opening {port} @ 9600 baud")
    imu = IMUReader(port=port)
    imu.start()
    try:
        while True:
            time.sleep(0.1)
            age = time.monotonic() - imu.last_update if imu.last_update else float("inf")
            ax, ay, az = imu.acceleration
            wx, wy, wz = imu.angular_velocity
            r, p, y = imu.angle_degrees
            print(
                f"age={age:5.2f}s  "
                f"acc=({ax:+6.2f},{ay:+6.2f},{az:+6.2f})  "
                f"gyro=({wx:+6.2f},{wy:+6.2f},{wz:+6.2f})  "
                f"rpy=({r:+7.2f},{p:+7.2f},{y:+7.2f})"
            )
    except KeyboardInterrupt:
        pass
    finally:
        imu.stop()


if __name__ == "__main__":
    _main(sys.argv)
