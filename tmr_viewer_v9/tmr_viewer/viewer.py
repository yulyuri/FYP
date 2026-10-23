"""TMR / ADS1256 live viewer (stage 1: old STM32 board).

Run:
    pip install -r requirements.txt
    python viewer.py                       # pick the COM port in the window
    python viewer.py --demo                # no hardware: simulated board
    python viewer.py --replay file.csv     # replay a recording (or use the button)

The board only streams after a rising edge on PB9 (its start trigger).
Three plots: raw Ch0, Ch0 through the live pulse band-pass (with detected beats
marked and a BPM / rhythm-quality readout), and Ch1.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dsp import (PulseFollower, PulseProcessor, RateTracker, average_beat,
                 reconstruct_pulse)
from stream import (CsvReplaySource, SerialSource, SimulatedBoard,
                    StmAsciiParser, Sample, available_ports)

FS = 50.0                      # the board outputs 50 samples/s
RECORD_SECONDS = 40            # default x-axis span
POLL_MS = 20                   # how often we look for new bytes
PLOT_MS = 50                   # redraw rate (20 fps)
NO_DATA_HINT_S = 3.0
UNITS = {"V": 1.0, "mV": 1e3, "µV": 1e6}
BAUDS = ["115200", "230400", "460800", "921600", "1000000", "2000000"]

NAN = float("nan")
P_RAW, P_PULSE, P_CH1, P_AVG, P_RECON = 0, 1, 2, 3, 4
RECON_WIN_S = 12            # the reconstruction looks at 12 s around each point (shown ~6 s late)
RECON_MIN_R2 = 0.20         # hide it when the repeating pulse explains less than this of the wobble
DEFAULT_MIN_PEAK_MV = 0.05      # peaks smaller than this are ignored
DEFAULT_MAX_MOTION_MV = 1.5     # more raw wobble than this over 5 s => do not read a pulse (0 = off)
VIEW_CHOICES = ["Last 5 s", "Last 10 s", "Last 20 s", "Last 60 s", "Whole recording", "Manual (mouse)"]
PLOT_TITLES = ("Ch0  TMR (AIN0 − AIN1), raw",
               "Ch0 pulse band (filtered, beats marked)",
               "Ch1  AIN2 − AIN3")
PLOT_COLORS = ("#1f77b4", "#2a9d3a", "#d95f02")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, record_dir: str) -> None:
        super().__init__()
        self.setWindowTitle("TMR / ADS1256 viewer")
        self.resize(1150, 880)

        # ---- state -------------------------------------------------------
        self.source = None
        self._kind = "serial"
        self.parser: Optional[StmAsciiParser] = None
        self.t: List[float] = []             # board timestamp
        self.tp: List[float] = []            # PC-clock time of each sample
        self.c0: List[float] = []
        self.c1: List[float] = []
        self.yf: List[float] = []            # filtered Ch0 (NaN while settling)
        self.beat_idx: List[int] = []        # sample indices of detected beats
        _mp = DEFAULT_MIN_PEAK_MV * 1e-3
        self.proc = PulseProcessor(fs=FS, min_peak=_mp)
        self.rate = RateTracker(fs=FS, min_rms=0.7 * _mp,
                                max_motion=DEFAULT_MAX_MOTION_MV * 1e-3)
        self.follow = PulseFollower(fs=FS, min_peak=_mp)
        self._avg_t = 0.0
        self.yn: List[float] = []            # clean rate-locked trace (NaN when not locked)
        self.lock_flags: List[bool] = []
        self.recon: List[float] = []         # delayed, model-based clean pulse (NaN = not available)
        self.recon_info = None               # (R2, peak-to-peak V) of the latest fit
        self._pc0: Optional[float] = None
        self._mode_checked = False
        self._auto_pc = False
        self.rec_idx = 0
        self.last_t: Optional[float] = None   # last RAW board time (wraps at 100)
        self._wrap = 0.0                      # +100 s for every wrap seen
        self.total_samples = 0
        self._rate_mark = (time.monotonic(), 0)
        self._started_at = 0.0
        self._dirty = False
        self._finished_logged = False
        self.csv_file = None
        self.csv_writer = None
        self._csv_pending = False
        self.record_dir = record_dir

        self._build_ui()
        self.refresh_ports()

        self.poll_timer = QtCore.QTimer(self, interval=POLL_MS)
        self.poll_timer.timeout.connect(self.poll)
        self.plot_timer = QtCore.QTimer(self, interval=PLOT_MS)
        self.plot_timer.timeout.connect(self.redraw)
        self.tick_timer = QtCore.QTimer(self, interval=1000)
        self.tick_timer.timeout.connect(self.tick)
        self.plot_timer.start()
        self.tick_timer.start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)

        # connection row
        row1 = QtWidgets.QHBoxLayout()
        outer.addLayout(row1)
        row1.addWidget(QtWidgets.QLabel("Port"))
        self.port_box = QtWidgets.QComboBox()
        self.port_box.setMinimumWidth(130)
        row1.addWidget(self.port_box)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        row1.addWidget(self.refresh_btn)
        row1.addWidget(QtWidgets.QLabel("Baud"))
        self.baud_box = QtWidgets.QComboBox()
        self.baud_box.addItems(BAUDS)
        self.baud_box.setCurrentText("921600")
        row1.addWidget(self.baud_box)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_serial)
        row1.addWidget(self.connect_btn)
        self.demo_btn = QtWidgets.QPushButton("Demo")
        self.demo_btn.setToolTip("Simulated board, no hardware needed")
        self.demo_btn.clicked.connect(self.toggle_demo)
        row1.addWidget(self.demo_btn)
        self.replay_btn = QtWidgets.QPushButton("Replay CSV…")
        self.replay_btn.setToolTip(
            "Play a recorded CSV back through the viewer and filter, so you "
            "can tune the pulse band on a real recording")
        self.replay_btn.clicked.connect(self.toggle_replay)
        row1.addWidget(self.replay_btn)
        row1.addStretch(1)

        # view row
        row2 = QtWidgets.QHBoxLayout()
        outer.addLayout(row2)
        row2.addWidget(QtWidgets.QLabel("Units"))
        self.unit_box = QtWidgets.QComboBox()
        self.unit_box.addItems(list(UNITS))
        self.unit_box.setCurrentText("mV")
        self.unit_box.currentIndexChanged.connect(self._mark_dirty)
        row2.addWidget(self.unit_box)
        row2.addWidget(QtWidgets.QLabel("Time axis"))
        self.time_box = QtWidgets.QComboBox()
        self.time_box.addItems(["Auto", "Board timestamp", "PC clock"])
        self.time_box.setToolTip(
            "Auto uses the timestamp the board sends, but switches to the PC "
            "clock if the board's time does not run at real-time speed.")
        self.time_box.currentIndexChanged.connect(self._mark_dirty)
        row2.addWidget(self.time_box)
        self.dc_check = QtWidgets.QCheckBox("Subtract DC (raw plots, view only)")
        self.dc_check.setToolTip(
            "Removes the average of the plotted raw data so a small signal on "
            "a large offset is visible. The CSV always keeps the raw values.")
        self.dc_check.setChecked(True)
        self.dc_check.toggled.connect(self._mark_dirty)
        row2.addWidget(self.dc_check)
        self.invert_check = QtWidgets.QCheckBox("Invert Ch0")
        self.invert_check.setToolTip(
            "The firmware reads AIN0 − AIN1 = TMR_V− minus TMR_V+, so Ch0 is "
            "upside-down relative to V+ − V−. Tick this to flip the view "
            "(beats are then detected on the flipped signal).")
        row2.addWidget(self.invert_check)
        row2.addStretch(1)

        # pulse band + recording row
        row3 = QtWidgets.QHBoxLayout()
        outer.addLayout(row3)
        row3.addWidget(QtWidgets.QLabel("Pulse band (Hz)  low"))
        self.low_spin = QtWidgets.QDoubleSpinBox()
        self.low_spin.setRange(0.2, 3.0); self.low_spin.setSingleStep(0.1)
        self.low_spin.setDecimals(1); self.low_spin.setValue(0.6)
        row3.addWidget(self.low_spin)
        row3.addWidget(QtWidgets.QLabel("high"))
        self.high_spin = QtWidgets.QDoubleSpinBox()
        self.high_spin.setRange(1.0, 12.0); self.high_spin.setSingleStep(0.5)
        self.high_spin.setDecimals(1); self.high_spin.setValue(2.5)
        row3.addWidget(self.high_spin)
        self.low_spin.valueChanged.connect(self._filters_changed)
        self.high_spin.valueChanged.connect(self._filters_changed)
        self.invert_check.toggled.connect(self._filters_changed)
        row3.addSpacing(10)
        row3.addWidget(QtWidgets.QLabel("Ignore peaks below (mV)"))
        self.minpk_spin = QtWidgets.QDoubleSpinBox()
        self.minpk_spin.setRange(0.0, 2.0); self.minpk_spin.setSingleStep(0.01)
        self.minpk_spin.setDecimals(2); self.minpk_spin.setValue(DEFAULT_MIN_PEAK_MV)
        self.minpk_spin.setToolTip("Band-passed bumps smaller than this are never counted as beats.")
        row3.addWidget(self.minpk_spin)
        self.minpk_spin.valueChanged.connect(self._filters_changed)
        row3.addSpacing(10)
        row3.addWidget(QtWidgets.QLabel("Max movement (mV/5 s)"))
        self.motion_spin = QtWidgets.QDoubleSpinBox()
        self.motion_spin.setRange(0.0, 200.0); self.motion_spin.setSingleStep(1.0)
        self.motion_spin.setDecimals(1); self.motion_spin.setValue(DEFAULT_MAX_MOTION_MV)
        self.motion_spin.setSpecialValueText("off")
        self.motion_spin.setToolTip(
            "If the raw top graph wobbles more than this over the last 5 s (after removing "
            "a straight-line drift) no pulse is read. 0 = off. Lower = stricter, "
            "but you get fewer readings.")
        row3.addWidget(self.motion_spin)
        self.motion_spin.valueChanged.connect(self._filters_changed)
        row3.addSpacing(10)
        row3.addWidget(QtWidgets.QLabel("View"))
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems(VIEW_CHOICES)
        self.view_box.setCurrentIndex(1)
        self.view_box.setToolTip("Auto-zoom follows the newest data. Scroll / drag on a plot to zoom "
                                 "yourself (switches to Manual); pick a Last … entry to go back.")
        self.view_box.currentIndexChanged.connect(self._mark_dirty)
        row3.addWidget(self.view_box)
        row3.addSpacing(16)
        self.save_check = QtWidgets.QCheckBox("Save CSV")
        self.save_check.setChecked(True)
        row3.addWidget(self.save_check)
        self.folder_btn = QtWidgets.QPushButton("Folder…")
        self.folder_btn.clicked.connect(self.choose_folder)
        row3.addWidget(self.folder_btn)
        self.folder_label = QtWidgets.QLabel(self.record_dir)
        self.folder_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        row3.addWidget(self.folder_label, 1)
        self.clear_btn = QtWidgets.QPushButton("Clear plot")
        self.clear_btn.clicked.connect(self.clear_plot)
        row3.addWidget(self.clear_btn)

        # plots
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.glw = pg.GraphicsLayoutWidget()
        outer.addWidget(self.glw, 1)
        self.plots: List[pg.PlotItem] = []
        self.curves = []
        for i in range(3):
            p = self.glw.addPlot(row=(0, 1, 3)[i], col=0)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setTitle(PLOT_TITLES[i])
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.setXRange(0, RECORD_SECONDS, padding=0)
            p.enableAutoRange(axis="y")
            p.getViewBox().setAutoVisible(y=True)
            p.getViewBox().sigRangeChangedManually.connect(self._user_zoomed)
            if i == P_CH1:
                p.setLabel("bottom", "time since start (s)")
            else:
                p.getAxis("bottom").setStyle(showValues=False)
            self.plots.append(p)
            self.curves.append(p.plot(pen=pg.mkPen(PLOT_COLORS[i], width=2)))
        self.plots[P_PULSE].setXLink(self.plots[P_RAW])
        self.plots[P_CH1].setXLink(self.plots[P_RAW])
        # 4th plot: the average beat shape
        pa = self.glw.addPlot(row=4, col=0)
        pa.showGrid(x=True, y=True, alpha=0.25)
        pa.setTitle("Average beat shape")
        pa.setLabel("bottom", "time around the beat (s)")
        pa.getAxis("left").enableAutoSIPrefix(False)
        pa.getAxis("bottom").enableAutoSIPrefix(False)
        self.plots.append(pa)
        self.avg_segs = [pa.plot(pen=pg.mkPen("#bbbbbb", width=1)) for _ in range(20)]
        self.avg_curve = pa.plot(pen=pg.mkPen("#d62728", width=3))
        # 5th plot (placed 3rd): delayed model-based reconstruction of the pulse
        pr = self.glw.addPlot(row=2, col=0)
        pr.showGrid(x=True, y=True, alpha=0.25)
        pr.setTitle("Reconstructed pulse  (needs a locked rhythm; shown ~6 s late)")
        pr.getAxis("left").enableAutoSIPrefix(False)
        pr.getAxis("bottom").setStyle(showValues=False)
        pr.enableAutoRange(axis="y")
        pr.getViewBox().setAutoVisible(y=True)
        pr.getViewBox().sigRangeChangedManually.connect(self._user_zoomed)
        pr.setXLink(self.plots[P_RAW])
        self.plots.append(pr)
        self.recon_curve = pr.plot(pen=pg.mkPen("#6a3d9a", width=3))
        for r, f in enumerate((3, 4, 4, 1, 3)):
            self.glw.ci.layout.setRowStretchFactor(r, f)
        self.beat_marks = pg.ScatterPlotItem(
            symbol="t", size=12, brush=pg.mkBrush("#d62728"), pen=None)
        self.plots[P_PULSE].addItem(self.beat_marks)
        # clean trace + beat markers that appear once the heart rate is locked
        self.curve_nar = self.plots[P_PULSE].plot(pen=pg.mkPen("#ff7f0e", width=3))
        self.nar_marks = pg.ScatterPlotItem(
            symbol="t", size=14, brush=pg.mkBrush("#d62728"), pen=None)
        self.plots[P_PULSE].addItem(self.nar_marks)

        # live numbers
        mono = self.font()
        mono.setFamily("Consolas")
        mono.setPointSize(11)
        self.value_label = QtWidgets.QLabel("—")
        self.value_label.setFont(mono)
        outer.addWidget(self.value_label)
        self.pulse_label = QtWidgets.QLabel("Pulse: —")
        pf = self.font()
        pf.setPointSize(12)
        pf.setBold(True)
        self.pulse_label.setFont(pf)
        outer.addWidget(self.pulse_label)

        # log
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(100)
        self.log_box.setMaximumBlockCount(500)
        outer.addWidget(self.log_box)

        self.status_label = QtWidgets.QLabel("Not connected")
        self.statusBar().addWidget(self.status_label, 1)
        self._update_axis_labels()

    # --------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {msg}")

    def _mark_dirty(self, *_args) -> None:
        self._update_axis_labels()
        self._dirty = True

    def _update_axis_labels(self) -> None:
        unit = self.unit_box.currentText()
        suffix = " (DC removed)" if self.dc_check.isChecked() else ""
        self.plots[P_RAW].setLabel("left", f"{unit}{suffix}")
        self.plots[P_PULSE].setLabel("left", f"{unit} (band-pass)")
        self.plots[P_CH1].setLabel("left", f"{unit}{suffix}")

    def refresh_ports(self) -> None:
        current = self.port_box.currentText()
        self.port_box.clear()
        ports = available_ports()
        self.port_box.addItems(ports if ports else ["(no ports found)"])
        if current in ports:
            self.port_box.setCurrentText(current)

    def choose_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder for CSV recordings", self.record_dir)
        if d:
            self.record_dir = d
            self.folder_label.setText(d)

    def clear_plot(self) -> None:
        self.t.clear(); self.c0.clear(); self.c1.clear(); self.tp.clear()
        self.yf.clear(); self.beat_idx.clear(); self.proc.reset(); self.rate.reset(); self.follow.reset()
        self.yn.clear(); self.lock_flags.clear()
        self.recon.clear(); self.recon_info = None
        self._pc0 = None
        self._wrap = 0.0
        self._clear_avg()
        self._mode_checked = False
        self._auto_pc = False
        self.plots[P_RAW].setXRange(0, RECORD_SECONDS, padding=0)
        self._dirty = True

    # ----------------------------------------------------------- pulse filter
    def _sign(self) -> float:
        return -1.0 if self.invert_check.isChecked() else 1.0

    def _feed_pulse(self, ch0: float) -> None:
        # once the rhythm is locked, ignore the smaller second bump of each beat
        self.proc.refractory = int(round(
            (0.65 * 60.0 / self.rate.bpm if self.rate.locked else 0.4) * FS))
        self.rate.push(self._sign() * ch0)
        y, beat = self.proc.push(self._sign() * ch0)
        yn, nbeat = self.follow.push(self._sign() * ch0, self.rate.bpm, self.rate.locked)
        self.yn.append(yn)
        self.lock_flags.append(bool(self.rate.locked))
        self.recon.append(NAN)
        n = len(self.recon)
        fi = int(round(FS))
        W = int(round(RECON_WIN_S * FS))
        if n % fi == 0 and n >= W and self.rate.locked:
            res = reconstruct_pulse(self._sign() * np.asarray(self.c0[n - W:n]), FS,
                                    self.rate.bpm / 60.0)
            if res is not None:
                self.recon_info = (res["r2"], res["pp"])
                if res["r2"] >= RECON_MIN_R2:
                    a = n - W // 2 - fi // 2
                    self.recon[a:a + fi] = [float(v) for v in
                                            res["harm"][W // 2 - fi // 2:W // 2 + fi // 2]]
        self.yf.append(y)
        if beat is not None:
            self.beat_idx.append(beat)

    def _filters_changed(self, *_args) -> None:
        """Band edges or polarity changed: rebuild the filter and re-run it
        over everything recorded so far."""
        low, high = self.low_spin.value(), self.high_spin.value()
        if low >= high:
            self.log("Pulse band: low must be below high, keeping old settings")
        else:
            try:
                self.proc = PulseProcessor(fs=FS, low=low, high=high,
                                           min_peak=self.minpk_spin.value() * 1e-3)
            except ValueError as e:
                self.log(f"Pulse band rejected: {e}")
        mp = self.minpk_spin.value() * 1e-3
        self.proc.min_peak = mp; self.follow.min_peak = mp; self.rate.min_rms = 0.7 * mp
        self.rate.max_motion = self.motion_spin.value() * 1e-3
        self.proc.reset(); self.rate.reset(); self.follow.reset()
        self.yf.clear(); self.beat_idx.clear(); self.yn.clear(); self.lock_flags.clear()
        self.recon.clear(); self.recon_info = None
        for v in self.c0:
            self._feed_pulse(v)
        self._mark_dirty()

    # ---------------------------------------------------------- start / stop
    def toggle_serial(self) -> None:
        if self.source is not None:
            self.stop_source("Disconnected")
            return
        port = self.port_box.currentText()
        if not port or port.startswith("("):
            QtWidgets.QMessageBox.warning(self, "No port", "No serial port selected.")
            return
        try:
            src = SerialSource(port, int(self.baud_box.currentText()))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "Could not open port",
                f"{port}\n\n{e}\n\nIs another program (Keil, a terminal, the old "
                "GUI) still holding the port?")
            return
        self.start_source(src, demo=False)

    def toggle_demo(self) -> None:
        if self.source is not None:
            self.stop_source("Demo stopped")
        else:
            self.start_source(SimulatedBoard(), demo=True)

    def toggle_replay(self) -> None:
        if self.source is not None:
            self.stop_source("Replay stopped")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Replay a recorded CSV", self.record_dir, "CSV files (*.csv)")
        if path:
            self.start_replay(path)

    def start_replay(self, path: str, speed: float = 1.0) -> None:
        try:
            src = CsvReplaySource(path, speed=speed)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Could not open CSV", f"{path}\n\n{e}")
            return
        self.start_source(src, demo=False, kind="replay")

    def start_source(self, src, demo: bool, kind: Optional[str] = None) -> None:
        kind = kind or ("demo" if demo else "serial")
        self._kind = kind
        self.source = src
        self.parser = StmAsciiParser()
        self.clear_plot()
        self.rec_idx = 0
        self.last_t = None
        self.total_samples = 0
        self._finished_logged = False
        self._rate_mark = (time.monotonic(), 0)
        self._started_at = time.monotonic()
        self._close_csv()
        self._csv_pending = kind != "replay"   # never re-save a replay
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(kind == "serial")
        self.demo_btn.setText("Stop demo" if kind == "demo" else "Demo")
        self.demo_btn.setEnabled(kind == "demo")
        self.replay_btn.setText("Stop replay" if kind == "replay" else "Replay CSV…")
        self.replay_btn.setEnabled(kind == "replay")
        self.refresh_btn.setEnabled(False)
        self.log(f"Connected: {src.name}")
        self.poll_timer.start()

    def stop_source(self, why: str) -> None:
        self.poll_timer.stop()
        if self.source is not None:
            self.source.close()
        self.source = None
        self._close_csv()
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.demo_btn.setText("Demo")
        self.demo_btn.setEnabled(True)
        self.replay_btn.setText("Replay CSV…")
        self.replay_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Not connected")
        self.log(why)

    # ------------------------------------------------------------------ CSV
    def _open_csv(self) -> None:
        self._close_csv()
        if not self.save_check.isChecked():
            return
        try:
            os.makedirs(self.record_dir, exist_ok=True)
            path = os.path.join(
                self.record_dir, f"tmr_{datetime.now():%Y%m%d_%H%M%S}.csv")
            self.csv_file = open(path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                ["rec", "t_board_s", "t_pc_s", "ch0_V", "ch1_V"])
            self.log(f"Saving to {path}")
        except OSError as e:
            self.csv_file = self.csv_writer = None
            self.log(f"CSV disabled, could not open file: {e}")

    def _close_csv(self) -> None:
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except OSError:
                pass
        self.csv_file = self.csv_writer = None

    # ----------------------------------------------------------------- data
    def poll(self) -> None:
        if self.source is None or self.parser is None:
            return
        try:
            data = self.source.read()
        except Exception as e:  # noqa: BLE001  (e.g. USB cable pulled)
            self.stop_source(f"Read error, disconnected: {e}")
            return
        if getattr(self.source, "finished", False) and not self._finished_logged:
            self._finished_logged = True
            self.log("Replay finished (press Stop replay to close it)")
        if not data:
            return
        samples, text = self.parser.feed(data)
        for line in text:
            self.log(f"board: {line}")
        for s in samples:
            self._handle_sample(s)

    def _handle_sample(self, s: Sample) -> None:
        if self._kind == "replay":
            # the wire format only carries 2 digits of seconds (wraps at 100 s),
            # so a long recording is timed by sample number instead
            s = Sample(self.total_samples / 50.0, s.ch0, s.ch1)
        if self._csv_pending:
            self._csv_pending = False
            if self.save_check.isChecked():
                self._open_csv()
        if (self.last_t is not None and self.last_t > 99.0 and s.t < 1.0
                and self._kind != "replay"):
            self._wrap += 100.0            # board clock only has 2 digits of seconds
            self.last_t = s.t - 100.0      # so the check below does not see a drop
        if self.last_t is not None and s.t + 0.5 < self.last_t:
            self.rec_idx += 1
            self.clear_plot()
            self.log(f"New recording #{self.rec_idx} started")
        self.last_t = s.t
        now = time.monotonic()
        if self._pc0 is None:
            self._pc0 = now
        t_pc = now - self._pc0
        self.t.append(s.t + self._wrap); self.tp.append(t_pc)
        self.c0.append(s.ch0); self.c1.append(s.ch1)
        self._feed_pulse(s.ch0)
        self.total_samples += 1
        self._dirty = True
        self._check_board_clock(t_pc)
        if self.csv_writer is not None:
            self.csv_writer.writerow(
                [self.rec_idx, f"{s.t + self._wrap:.3f}", f"{t_pc:.3f}",
                 f"{s.ch0:.6f}", f"{s.ch1:.6f}"])

    def _check_board_clock(self, t_pc: float) -> None:
        """After ~1 s of data, compare how far the board's timestamp moved
        with how much real time passed. If they disagree, plot against the PC
        clock instead (in Auto mode)."""
        if self._mode_checked or len(self.t) < 20 or t_pc < 1.0:
            return
        self._mode_checked = True
        ratio = (self.t[-1] - self.t[0]) / t_pc
        if ratio < 0.8 or ratio > 1.25:
            self._auto_pc = True
            self.log(f"Board timestamp ran at {ratio:.2f}x real time "
                     f"(0 = stuck). Auto mode is using the PC clock instead.")

    def _use_pc_clock(self) -> bool:
        mode = self.time_box.currentText()
        if mode == "PC clock":
            return True
        if mode == "Board timestamp":
            return False
        return self._auto_pc

    # ------------------------------------------------------------- redrawing
    def _user_zoomed(self, *_a) -> None:
        if not self.view_box.currentText().startswith("Manual"):
            self.view_box.setCurrentIndex(len(VIEW_CHOICES) - 1)

    def _clear_avg(self) -> None:
        if not hasattr(self, "avg_curve"):
            return
        self.avg_curve.setData([], [])
        for c in self.avg_segs:
            c.setData([], [])
        self.plots[P_AVG].setTitle("Average beat shape  (needs a locked rhythm)")

    def _update_avg(self, scale: float) -> None:
        res = None
        if self.rate.locked and len(self.follow.beats) >= 4:
            res = average_beat(np.asarray(self.yf, dtype=float), self.follow.beats, fs=FS)
        if res is None:
            self._clear_avg()
            return
        tt, mean, segs = res
        u = self.unit_box.currentText()
        for i, c in enumerate(self.avg_segs):
            if i < len(segs):
                c.setData(tt, segs[i] * scale)
            else:
                c.setData([], [])
        self.avg_curve.setData(tt, mean * scale)
        self.plots[P_AVG].setTitle(
            f"Average beat shape  ({len(segs)} beats lined up; "
            f"{(mean.max() - mean.min()) * scale:.3f} {u} peak-to-peak)")

    def _pulse_text(self) -> str:
        if not self.yf:
            return "Pulse: \u2014"
        r = self.rate
        mv = "" if math.isnan(r.motion) else f"   |   movement {r.motion * 1e3:.1f} mV / 5 s"
        if r.locked:
            return (f"Pulse: {r.bpm:.0f} BPM   (locked for {r.lock_age} s, "
                    f"signal strength {r.score:.0f}){mv}")
        return f"Pulse: {r.status}{mv}"

    def redraw(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        scale = UNITS[self.unit_box.currentText()]
        use_pc = self._use_pc_clock()
        t = np.asarray(self.tp if use_pc else self.t)
        self.plots[P_CH1].setLabel(
            "bottom",
            "time since start (s, " + ("PC clock" if use_pc else "board clock") + ")")
        mode = self.view_box.currentText()
        manual = mode.startswith("Manual")
        win_s = float(mode.split()[1]) if mode.startswith("Last") else None
        if t.size and not manual:
            if win_s is not None:
                hi = max(float(t[-1]), win_s)
                self.plots[P_RAW].setXRange(hi - win_s, hi, padding=0)
            else:
                self.plots[P_RAW].setXRange(0, max(float(t[-1]), RECORD_SECONDS), padding=0)

        for idx, raw in ((P_RAW, self.c0), (P_CH1, self.c1)):
            y = np.asarray(raw) * scale
            if idx == P_RAW and self.invert_check.isChecked():
                y = -y
            if self.dc_check.isChecked() and y.size:
                y = y - y.mean()
            self.curves[idx].setData(t, y)

        yf = np.asarray(self.yf) * scale          # polarity already applied
        self.curves[P_PULSE].setData(t, yf)
        # scale the pulse plot to the last ~20 s, not to the biggest magnet
        # move ever seen, otherwise a 0.3 mV pulse looks flat after one bump
        recent = yf[-int((win_s or 20.0) * FS):]
        recent = recent[np.isfinite(recent)]
        if recent.size > 50 and not manual:
            half = float(np.percentile(np.abs(recent), 98)) * 1.6
            if half > 0:
                self.plots[P_PULSE].setYRange(-half, half, padding=0)
        rc = np.asarray(self.recon) * scale
        self.recon_curve.setData(t, rc)
        if self.recon_info is not None:
            u = self.unit_box.currentText()
            self.plots[P_RECON].setTitle(
                f"Reconstructed pulse  (shown ~6 s late; fit explains {self.recon_info[0] * 100:.0f}% "
                f"of the wobble; {self.recon_info[1] * scale:.3f} {u} peak-to-peak)")
        yn = np.asarray(self.yn) * scale
        self.curve_nar.setData(t, yn)
        # grey = candidate peaks while no steady rhythm (probably noise);
        # red = beats of the clean trace, only shown while the rate is locked
        lock = np.asarray(self.lock_flags, dtype=bool)
        bi = np.asarray(self.beat_idx, dtype=int)
        bi = bi[(bi < t.size) & (bi < lock.size)]
        bi = bi[~lock[bi]] if bi.size else bi
        if bi.size:
            self.beat_marks.setData(x=t[bi], y=yf[bi])
        else:
            self.beat_marks.setData(x=[], y=[])
        nb = np.asarray(self.follow.beats, dtype=int)
        nb = nb[nb < t.size]
        if nb.size:
            self.nar_marks.setData(x=t[nb], y=yn[nb])
        else:
            self.nar_marks.setData(x=[], y=[])
        self.beat_marks.setBrush(pg.mkBrush("#9a9a9a"))
        now_m = time.monotonic()
        if now_m - self._avg_t > 0.5:
            self._avg_t = now_m
            self._update_avg(scale)
        self.pulse_label.setText(self._pulse_text())
        r_ = self.rate
        colour = ("#2a7a2a" if r_.locked
                  else "#c0392b" if r_.status.startswith("too much movement") else "")
        self.pulse_label.setStyleSheet(f"color: {colour};" if colour else "")

        if self.c0:
            u = self.unit_box.currentText()
            self.value_label.setText(
                f"board t = {self.t[-1]:6.3f} s   PC t = {self.tp[-1]:6.2f} s    "
                f"Ch0 = {self.c0[-1] * scale:12.4f} {u}    "
                f"Ch1 = {self.c1[-1] * scale:12.4f} {u}    (raw, before DC/invert)")

    def tick(self) -> None:
        """Once a second: status bar and the 'no data' hint."""
        if self.source is None or self.parser is None:
            return
        now = time.monotonic()
        mark_t, mark_n = self._rate_mark
        rate = (self.total_samples - mark_n) / max(now - mark_t, 1e-6)
        self._rate_mark = (now, self.total_samples)
        msg = (f"{self.source.name}   |   recording #{self.rec_idx}   |   "
               f"{rate:4.0f} samples/s   |   {self.total_samples} samples   |   "
               f"junk bytes: {self.parser.bad_bytes}")
        if (self._kind == "serial" and self.total_samples == 0
                and now - self._started_at > NO_DATA_HINT_S):
            msg = ("Connected, no data yet. The board only streams after a "
                   "rising edge on PB9 (its start trigger).")
        self.status_label.setText(msg)
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
            except OSError:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self.poll_timer.stop()
        if self.source is not None:
            self.source.close()
        self._close_csv()
        super().closeEvent(event)


def main() -> int:
    ap = argparse.ArgumentParser(description="TMR / ADS1256 live viewer")
    ap.add_argument("--demo", action="store_true", help="start with simulated board")
    ap.add_argument("--replay", metavar="CSV", help="replay a recorded CSV")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings"),
                    help="folder for CSV files")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args.out)
    win.show()
    if args.demo:
        win.toggle_demo()
    elif args.replay:
        win.start_replay(args.replay, args.speed)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
