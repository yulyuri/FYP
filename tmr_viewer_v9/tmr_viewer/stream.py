"""Data sources and parsers for the TMR viewer.

No Qt in here on purpose, so the same code can be tested from a terminal and
reused later when the ESP32 firmware replaces the STM32 board.

Stage 1 stream format (what the old STM32 firmware prints on USART3):

    \\r\\n  S d.dddddd  S d.dddddd  ss.sss  \\n\\r

    S         sign digit: '0' = positive, '1' = negative
    d.dddddd  volts, 6 decimals (micro-volt resolution)
    ss.sss    seconds since this recording started (resets to 0 on a new start)

There are no separators between the three fields, so it is parsed by position.
Channel 0 = ADS1256 AIN0-AIN1 (TMR_V- minus TMR_V+), channel 1 = AIN2-AIN3.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

try:  # pyserial is only needed for a real board, not for the demo
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None


@dataclass
class Sample:
    t: float     # seconds since the board started this recording
    ch0: float   # volts
    ch1: float   # volts


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
class StmAsciiParser:
    """Incremental parser: feed() it whatever bytes arrived, get samples back.

    Handles records split across reads, boot-up text (chip ID message) mixed
    into the stream, and garbage (it re-syncs on the next valid record).
    """

    RECORD = re.compile(
        rb"\r\n([01])(\d\.\d{6})([01])(\d\.\d{6})(\d{2}\.\d{3})\n\r"
    )
    MAX_TAIL = 40      # a record is 28 bytes; keep a little more than that
    MAX_BUFFER = 400   # if we hold more than this with no record, it is junk

    def __init__(self) -> None:
        self._buf = b""
        self.bad_bytes = 0      # bytes thrown away as junk
        self.records = 0        # valid records decoded

    @staticmethod
    def _signed(sign: bytes, digits: bytes) -> float:
        value = float(digits)
        return -value if sign == b"1" else value

    def feed(self, data: bytes) -> Tuple[List[Sample], List[str]]:
        """Returns (samples, text_lines). text_lines = non-data text such as
        'Ok, ASD1256 Chip ID = 0x3'."""
        self._buf += data
        samples: List[Sample] = []
        text: List[str] = []

        while True:
            m = self.RECORD.search(self._buf)
            if not m:
                break
            self._collect_text(self._buf[: m.start()], text)
            samples.append(
                Sample(
                    t=float(m.group(5)),
                    ch0=self._signed(m.group(1), m.group(2)),
                    ch1=self._signed(m.group(3), m.group(4)),
                )
            )
            self.records += 1
            self._buf = self._buf[m.end():]

        # No complete record left. Everything before the last CRLF is complete
        # (either a finished text line or the stuff before a record that has
        # not fully arrived yet), so flush it and keep only the possible start.
        i = self._buf.rfind(b"\r\n")
        if i > 0:
            self._collect_text(self._buf[:i], text)
            self._buf = self._buf[i:]
        if len(self._buf) > self.MAX_BUFFER:
            self.bad_bytes += len(self._buf) - self.MAX_TAIL
            self._buf = self._buf[-self.MAX_TAIL:]
        return samples, text

    def _collect_text(self, chunk: bytes, text: List[str]) -> None:
        if not chunk:
            return
        decoded = chunk.decode("ascii", errors="replace")
        for line in decoded.replace("\r", "\n").split("\n"):
            line = line.strip()
            if line:
                text.append(line)


# --------------------------------------------------------------------------
# Sources: anything with read() -> bytes and close()
# --------------------------------------------------------------------------
class SerialSource:
    def __init__(self, port: str, baud: int = 921600) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed (pip install pyserial)")
        self._ser = serial.Serial(port, baud, timeout=0)
        self.name = f"{port} @ {baud}"

    def read(self) -> bytes:
        n = self._ser.in_waiting
        return self._ser.read(n) if n else b""

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass


def available_ports() -> List[str]:
    if list_ports is None:
        return []
    return [p.device for p in sorted(list_ports.comports())]


def format_record(ch0: float, ch1: float, t: float) -> bytes:
    """Encode one sample exactly the way the STM32 firmware prints it."""
    def f(v: float) -> str:
        micro = int(round(abs(v) * 1e6))
        sign = "1" if v < 0 else "0"
        return f"{sign}{micro // 1_000_000}.{(micro % 1_000_000) // 1000:03d}{micro % 1000:03d}"
    return f"\r\n{f(ch0)}{f(ch1)}{t % 100:06.3f}\n\r".encode()


class CsvReplaySource:
    """Plays a CSV saved by this viewer back as if it were the board, so
    filters can be tuned on a real recording without any hardware.
    speed=1 is real time; larger values play faster (used for testing)."""

    RATE_HZ = 50

    def __init__(self, path: str, speed: float = 1.0) -> None:
        import csv
        self.name = f"Replay: {path.replace(chr(92), '/').split('/')[-1]}"
        self._speed = speed
        self._rows = []
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "ch0_V" not in rd.fieldnames:
                raise ValueError("not a CSV saved by this viewer (no ch0_V column)")
            for r in rd:
                self._rows.append((float(r["ch0_V"]), float(r.get("ch1_V", 0) or 0)))
        self._t0 = time.monotonic()
        self._sent = 0
        self.finished = False

    def read(self) -> bytes:
        if self.finished:
            return b""
        due = int((time.monotonic() - self._t0) * self.RATE_HZ * self._speed)
        out = b""
        while self._sent < min(due, len(self._rows)):
            c0, c1 = self._rows[self._sent]
            out += format_record(c0, c1, self._sent / self.RATE_HZ)
            self._sent += 1
        if self._sent >= len(self._rows):
            self.finished = True
        return out

    def close(self) -> None:
        pass


class SimulatedBoard:
    """Pretends to be the STM32 board: same bytes, same 50 Hz rate, same 40 s
    auto-stop, so the GUI can be tried without any hardware."""

    RATE_HZ = 50
    RECORD_SECONDS = 40
    PAUSE_SECONDS = 2.0     # gap before the next 'button press'

    def __init__(self, heart_rate_hz: float = 1.2, clock: str = "normal") -> None:
        """clock: 'normal', 'stuck' (timestamp always 0) or 'slow10' (timestamp
        runs 10x too slow). The last two exist to test the viewer's fallback."""
        self.name = "Demo (simulated board)"
        self._hr = heart_rate_hz
        self._clock = clock
        self._rng = random.Random(1)
        self._t_open = time.monotonic()
        self._rec_start = self._t_open + 0.5
        self._sent = 0
        self._boot_sent = False

    @staticmethod
    def _fmt(volts: float) -> str:
        micro = int(round(abs(volts) * 1e6))
        sign = "1" if volts < 0 else "0"
        return f"{sign}{micro // 1_000_000}.{(micro % 1_000_000) // 1000:03d}{micro % 1000:03d}"

    def _signal(self, t: float) -> Tuple[float, float]:
        phase = (t * self._hr) % 1.0
        pulse = math.exp(-((phase - 0.15) / 0.05) ** 2) + 0.4 * math.exp(-((phase - 0.45) / 0.09) ** 2)
        ch0 = -0.0910 + 0.0009 * pulse + self._rng.gauss(0, 0.00003)
        ch0 += 0.00008 * math.sin(2 * math.pi * 49.7 * t)   # mains pickup, aliased
        ch1 = 0.5000 + 0.0100 * math.sin(2 * math.pi * 0.2 * t) + self._rng.gauss(0, 0.0002)
        return ch0, ch1

    def read(self) -> bytes:
        out = b""
        if not self._boot_sent:
            self._boot_sent = True
            out += b"read id\r\nOk, ASD1256 Chip ID = 0x3\r\n"
        now = time.monotonic()
        if now < self._rec_start:
            return out
        elapsed = now - self._rec_start
        due = int(elapsed * self.RATE_HZ)
        while self._sent < due and self._sent < self.RECORD_SECONDS * self.RATE_HZ:
            t = self._sent / self.RATE_HZ
            c0, c1 = self._signal(t)
            shown_t = {"normal": t, "stuck": 0.0, "slow10": t / 10}[self._clock]
            out += f"\r\n{self._fmt(c0)}{self._fmt(c1)}{shown_t:06.3f}\n\r".encode()
            self._sent += 1
        if self._sent >= self.RECORD_SECONDS * self.RATE_HZ and \
                elapsed > self.RECORD_SECONDS + self.PAUSE_SECONDS:
            self._rec_start = now            # 'press the button' again
            self._sent = 0
        return out

    def close(self) -> None:
        pass
