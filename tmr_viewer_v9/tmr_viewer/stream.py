from __future__ import annotations

import csv
import math
import random
import time

from dataclasses import dataclass
from typing import List, Tuple


try:
    import serial
    import serial.tools.list_ports as list_ports

except ImportError:
    serial = None
    list_ports = None


# ==========================================================
# SAMPLE
# ==========================================================

@dataclass
class Sample:
    t: float
    ch0: float
    ch1: float
    ax: float
    ay: float
    az: float


# ==========================================================
# PARSER
# ==========================================================

class StmAsciiParser:

    def __init__(self) -> None:

        self._buf = b""

        self.bad_bytes = 0
        self.records = 0


    def feed(
        self,
        data: bytes
    ) -> Tuple[List[Sample], List[str]]:

        self._buf += data

        samples = []
        text = []


        while b"\n" in self._buf:

            raw, self._buf = self._buf.split(
                b"\n",
                1
            )

            raw = raw.strip()

            if not raw:
                continue


            line = raw.decode(
                "ascii",
                errors="ignore"
            ).strip()


            # ==============================================
            # ESP32 DATA
            #
            # D,time,ch0,ch1,ax,ay,az
            # ==============================================

            if line.startswith("D,"):

                parts = line.split(",")

                if len(parts) != 7:

                    self.bad_bytes += len(raw)

                    continue


                try:

                    sample = Sample(
                        t=float(parts[1]),
                        ch0=float(parts[2]),
                        ch1=float(parts[3]),
                        ax=float(parts[4]),
                        ay=float(parts[5]),
                        az=float(parts[6])
                    )

                except ValueError:

                    self.bad_bytes += len(raw)

                    continue


                samples.append(sample)

                self.records += 1


            else:

                text.append(line)


        return samples, text


# ==========================================================
# SERIAL SOURCE
# ==========================================================

class SerialSource:

    def __init__(
        self,
        port: str,
        baud: int = 921600
    ) -> None:

        if serial is None:

            raise RuntimeError(
                "pyserial is not installed"
            )


        self._ser = serial.Serial(
            port,
            baud,
            timeout=0
        )


        self.name = (
            f"{port} @ {baud}"
        )


    def read(self) -> bytes:

        n = self._ser.in_waiting

        if n:
            return self._ser.read(n)

        return b""


    def close(self) -> None:

        try:
            self._ser.close()

        except Exception:
            pass


# ==========================================================
# PORTS
# ==========================================================

def available_ports() -> List[str]:

    if list_ports is None:
        return []


    return [
        p.device
        for p in list_ports.comports()
    ]


# ==========================================================
# FORMAT RECORD
# ==========================================================

def format_record(
    ch0: float,
    ch1: float,
    t: float,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 1.0
) -> bytes:

    return (
        f"D,"
        f"{t:.3f},"
        f"{ch0:.6f},"
        f"{ch1:.6f},"
        f"{ax:.6f},"
        f"{ay:.6f},"
        f"{az:.6f}\r\n"
    ).encode("ascii")


# ==========================================================
# CSV REPLAY
# ==========================================================

class CsvReplaySource:

    RATE_HZ = 50


    def __init__(
        self,
        path: str,
        speed: float = 1.0
    ) -> None:

        self.name = (
            "Replay: "
            + path.replace("\\", "/").split("/")[-1]
        )

        self._speed = speed

        self._rows = []


        with open(
            path,
            newline=""
        ) as fh:

            rd = csv.DictReader(fh)


            if (
                not rd.fieldnames
                or
                "ch0_V" not in rd.fieldnames
            ):
                raise ValueError(
                    "CSV has no ch0_V column"
                )


            for row in rd:

                ch0 = float(
                    row.get("ch0_V", 0)
                    or 0
                )

                ch1 = float(
                    row.get("ch1_V", 0)
                    or 0
                )


                ax = float(
                    row.get("ax_g", 0)
                    or 0
                )

                ay = float(
                    row.get("ay_g", 0)
                    or 0
                )

                az = float(
                    row.get("az_g", 1)
                    or 1
                )


                self._rows.append(
                    (
                        ch0,
                        ch1,
                        ax,
                        ay,
                        az
                    )
                )


        self._t0 = time.monotonic()

        self._sent = 0

        self.finished = False


    def read(self) -> bytes:

        if self.finished:
            return b""


        due = int(
            (
                time.monotonic()
                -
                self._t0
            )
            *
            self.RATE_HZ
            *
            self._speed
        )


        out = b""


        while (
            self._sent
            <
            min(
                due,
                len(self._rows)
            )
        ):

            (
                ch0,
                ch1,
                ax,
                ay,
                az
            ) = self._rows[
                self._sent
            ]


            t = (
                self._sent
                /
                self.RATE_HZ
            )


            out += format_record(
                ch0,
                ch1,
                t,
                ax,
                ay,
                az
            )


            self._sent += 1


        if (
            self._sent
            >=
            len(self._rows)
        ):
            self.finished = True


        return out


    def close(self) -> None:
        pass


# ==========================================================
# SIMULATED BOARD
# ==========================================================

class SimulatedBoard:

    RATE_HZ = 50

    RECORD_SECONDS = 40

    PAUSE_SECONDS = 2.0


    def __init__(
        self,
        heart_rate_hz: float = 1.2,
        clock: str = "normal"
    ) -> None:

        self.name = (
            "Demo (simulated board)"
        )

        self._hr = heart_rate_hz

        self._clock = clock

        self._rng = random.Random(1)

        self._rec_start = (
            time.monotonic()
            +
            0.5
        )

        self._sent = 0

        self._boot_sent = False


    def _signal(
        self,
        t: float
    ):

        # ==============================================
        # FAKE TMR PULSE
        # ==============================================

        phase = (
            t * self._hr
        ) % 1.0


        pulse = (
            math.exp(
                -(
                    (
                        phase - 0.15
                    )
                    /
                    0.05
                ) ** 2
            )
            +
            0.4
            *
            math.exp(
                -(
                    (
                        phase - 0.45
                    )
                    /
                    0.09
                ) ** 2
            )
        )


        ch0 = (
            -0.091
            +
            0.0009 * pulse
            +
            self._rng.gauss(
                0,
                0.00003
            )
        )

        ch1 = 0.0


        # ==============================================
        # FAKE MPU XYZ
        # ==============================================

        ax = (
            0.02
            +
            self._rng.gauss(
                0,
                0.003
            )
        )

        ay = (
            -0.02
            +
            self._rng.gauss(
                0,
                0.003
            )
        )

        az = (
            1.01
            +
            self._rng.gauss(
                0,
                0.004
            )
        )


        # Movement example
        if 8 < t < 9:

            ax += (
                0.2
                *
                math.sin(
                    2
                    *
                    math.pi
                    *
                    3
                    *
                    t
                )
            )

            ay += (
                0.15
                *
                math.sin(
                    2
                    *
                    math.pi
                    *
                    2
                    *
                    t
                )
            )


        return (
            ch0,
            ch1,
            ax,
            ay,
            az
        )


    def read(self) -> bytes:

        out = b""


        if not self._boot_sent:

            self._boot_sent = True

            out += (
                b"ADS1115 + MPU6050 XYZ ready\r\n"
            )


        now = time.monotonic()


        if now < self._rec_start:

            return out


        elapsed = (
            now
            -
            self._rec_start
        )


        due = int(
            elapsed
            *
            self.RATE_HZ
        )


        while (
            self._sent < due
            and
            self._sent
            <
            self.RECORD_SECONDS
            *
            self.RATE_HZ
        ):

            t = (
                self._sent
                /
                self.RATE_HZ
            )


            (
                ch0,
                ch1,
                ax,
                ay,
                az
            ) = self._signal(t)


            out += format_record(
                ch0,
                ch1,
                t,
                ax,
                ay,
                az
            )


            self._sent += 1


        if (
            self._sent
            >=
            self.RECORD_SECONDS
            *
            self.RATE_HZ
            and
            elapsed
            >
            self.RECORD_SECONDS
            +
            self.PAUSE_SECONDS
        ):

            self._rec_start = now

            self._sent = 0


        return out


    def close(self) -> None:
        pass