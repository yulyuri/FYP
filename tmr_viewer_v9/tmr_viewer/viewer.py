"""TMR + MPU live viewer with motion-artifact rejection."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dsp import (
    PulseFollower,
    PulsePresenceDetector,
    PulseProcessor,
    RateTracker,
    average_beat,
    reconstruct_pulse,
)

from stream import (
    CsvReplaySource,
    SerialSource,
    SimulatedBoard,
    StmAsciiParser,
    Sample,
    available_ports,
)


# ==========================================================
# SETTINGS
# ==========================================================

FS = 50.0

RECORD_SECONDS = 40

POLL_MS = 20
PLOT_MS = 50
NO_DATA_HINT_S = 3.0

UNITS = {
    "V": 1.0,
    "mV": 1e3,
    "µV": 1e6,
}

BAUDS = [
    "115200",
    "230400",
    "460800",
    "921600",
    "1000000",
    "2000000",
]

NAN = float("nan")


# ==========================================================
# MOTION ARTIFACT SETTINGS
# ==========================================================

# Automatically calibrate the motion threshold from the first 5 seconds.
# Keep the sensor / wrist still during this period.
CALIBRATION_S = 5.0

CALIBRATION_SAMPLES = max(
    1,
    int(round(FS * CALIBRATION_S))
)

# Safety floor so an unusually quiet calibration does not make the
# detector hypersensitive to tiny accelerometer noise.
MIN_MOTION_THRESHOLD_G = 0.03

# Robust threshold multiplier:
# robust_sigma = 1.4826 * MAD
# threshold = median + CALIBRATION_SIGMA_MULT * robust_sigma
CALIBRATION_SIGMA_MULT = 4.0

# Smooth the combined XYZ motion score over 0.20 s
MPU_SMOOTH_S = 0.20

MPU_SMOOTH_SAMPLES = max(
    1,
    int(round(FS * MPU_SMOOTH_S))
)

# Recent X/Y/Z baseline. This lets rotation/directional changes
# be detected even when total acceleration stays near 1 g.
MPU_BASELINE_S = 0.50

MPU_BASELINE_SAMPLES = max(
    2,
    int(round(FS * MPU_BASELINE_S))
)

# Continue rejecting TMR for 0.25 s after movement
MOTION_PADDING_S = 0.25

MOTION_PADDING_SAMPLES = max(
    1,
    int(round(FS * MOTION_PADDING_S))
)


# ==========================================================
# PULSE-PRESENCE SETTINGS
# ==========================================================

# The first 5 seconds must have NO BODY PART on the sensor.
# The detector learns the no-body Ch0 level and pulse-band noise floor.
PRESENCE_CONTACT_FLOOR_V = 0.00002   # 0.02 mV minimum DC change
PRESENCE_CONTACT_SIGMA = 2.5
PRESENCE_ENERGY_RATIO = 1.05
PRESENCE_ENERGY_FLOOR_V = 0.000001   # 0.001 mV above no-body RMS


# ==========================================================
# PLOT INDICES
# ==========================================================

P_RAW = 0
P_PULSE = 1
P_CH1 = 2
P_AVG = 3
P_RECON = 4
P_MPU = 5


RECON_WIN_S = 12
RECON_MIN_R2 = 0.20

DEFAULT_MIN_PEAK_MV = 0.05


VIEW_CHOICES = [
    "Last 5 s",
    "Last 10 s",
    "Last 20 s",
    "Last 60 s",
    "Whole recording",
    "Manual (mouse)",
]


PLOT_TITLES = (
    "Ch0 TMR (AIN0 − AIN1), raw",
    "Ch0 pulse band (filtered, beats marked)",
    "Ch1 AIN2 − AIN3",
)


PLOT_COLORS = (
    "#1f77b4",
    "#2a9d3a",
    "#d95f02",
)


# ==========================================================
# MAIN WINDOW
# ==========================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, record_dir: str) -> None:

        super().__init__()

        self.setWindowTitle(
            "TMR / MPU Motion Artifact Viewer"
        )

        self.resize(
            1200,
            950
        )


        # ==================================================
        # DATA
        # ==================================================

        self.source = None
        self._kind = "serial"

        self.parser: Optional[
            StmAsciiParser
        ] = None


        self.t: List[float] = []
        self.tp: List[float] = []

        self.c0: List[float] = []
        self.c1: List[float] = []

        # MPU6050 acceleration axes (g)
        self.ax: List[float] = []
        self.ay: List[float] = []
        self.az: List[float] = []

        # |a| = sqrt(ax^2 + ay^2 + az^2)
        self.accel_mag: List[float] = []

        # Directional change from recent XYZ baseline
        self.axis_motion: List[float] = []

        # Raw combined XYZ motion score.  Kept as self.mpu so the
        # rest of the existing plotting code needs minimal changes.
        self.mpu: List[float] = []

        # Smoothed combined XYZ motion score
        self.mpu_avg: List[float] = []

        # True = motion artifact
        self.motion_flags: List[bool] = []


        # ==================================================
        # MOTION STATE
        # ==================================================

        self._motion_window = deque(
            maxlen=MPU_SMOOTH_SAMPLES
        )

        self._ax_baseline = deque(maxlen=MPU_BASELINE_SAMPLES)
        self._ay_baseline = deque(maxlen=MPU_BASELINE_SAMPLES)
        self._az_baseline = deque(maxlen=MPU_BASELINE_SAMPLES)

        self._motion_hold = 0

        self._motion_active = False

        # First 5 s are used to estimate the resting motion noise.
        # None means calibration is still in progress.
        self._motion_threshold: Optional[float] = None
        self._calibration_scores: List[float] = []

        # TMR no-body / pulse-presence detector.  It uses the same first
        # 5 seconds as calibration, so those first 5 seconds must have
        # NO BODY PART on the sensor and the setup should be kept still.
        self.presence = PulsePresenceDetector(
            fs=FS,
            calibration_s=CALIBRATION_S,
            contact_floor=PRESENCE_CONTACT_FLOOR_V,
            contact_sigma=PRESENCE_CONTACT_SIGMA,
            energy_ratio=PRESENCE_ENERGY_RATIO,
            energy_floor=PRESENCE_ENERGY_FLOOR_V,
        )
        self._pulse_present = False
        self.pulse_presence_flags: List[bool] = []


        # ==================================================
        # PULSE PROCESSING
        # ==================================================

        self.yf: List[float] = []

        self.beat_idx: List[int] = []


        _mp = (
            DEFAULT_MIN_PEAK_MV
            * 1e-3
        )


        self.proc = PulseProcessor(
            fs=FS,
            min_peak=_mp
        )


        # Motion gating is handled exclusively by MPU6050 X/Y/Z.
        # Disable the older TMR-amplitude-based motion gate in RateTracker.
        self.rate = RateTracker(
            fs=FS,
            min_rms=0.7 * _mp,
            max_motion=0.0
        )


        self.follow = PulseFollower(
            fs=FS,
            min_peak=_mp
        )


        self.yn: List[float] = []

        self.lock_flags: List[bool] = []


        self.recon: List[float] = []

        self.recon_info = None


        self._avg_t = 0.0


        # ==================================================
        # TIME / RECORDING
        # ==================================================

        self._pc0: Optional[
            float
        ] = None

        self._mode_checked = False
        self._auto_pc = False

        self.rec_idx = 0

        self.last_t: Optional[
            float
        ] = None

        self._wrap = 0.0

        self.total_samples = 0

        self._rate_mark = (
            time.monotonic(),
            0
        )

        self._started_at = 0.0

        self._dirty = False

        self._finished_logged = False


        # ==================================================
        # CSV
        # ==================================================

        self.csv_file = None
        self.csv_writer = None

        self._csv_pending = False

        self.record_dir = record_dir


        # ==================================================
        # UI
        # ==================================================

        self._build_ui()

        self.refresh_ports()


        # ==================================================
        # TIMERS
        # ==================================================

        self.poll_timer = QtCore.QTimer(
            self,
            interval=POLL_MS
        )

        self.poll_timer.timeout.connect(
            self.poll
        )


        self.plot_timer = QtCore.QTimer(
            self,
            interval=PLOT_MS
        )

        self.plot_timer.timeout.connect(
            self.redraw
        )


        self.tick_timer = QtCore.QTimer(
            self,
            interval=1000
        )

        self.tick_timer.timeout.connect(
            self.tick
        )


        self.plot_timer.start()

        self.tick_timer.start()


    # ======================================================
    # BUILD UI
    # ======================================================

    def _build_ui(self) -> None:

        root = QtWidgets.QWidget()

        self.setCentralWidget(
            root
        )

        outer = QtWidgets.QVBoxLayout(
            root
        )


        # ==================================================
        # CONNECTION ROW
        # ==================================================

        row1 = QtWidgets.QHBoxLayout()

        outer.addLayout(
            row1
        )


        row1.addWidget(
            QtWidgets.QLabel(
                "Port"
            )
        )


        self.port_box = QtWidgets.QComboBox()

        self.port_box.setMinimumWidth(
            130
        )

        row1.addWidget(
            self.port_box
        )


        self.refresh_btn = QtWidgets.QPushButton(
            "Refresh"
        )

        self.refresh_btn.clicked.connect(
            self.refresh_ports
        )

        row1.addWidget(
            self.refresh_btn
        )


        row1.addWidget(
            QtWidgets.QLabel(
                "Baud"
            )
        )


        self.baud_box = QtWidgets.QComboBox()

        self.baud_box.addItems(
            BAUDS
        )

        self.baud_box.setCurrentText(
            "921600"
        )

        row1.addWidget(
            self.baud_box
        )


        self.connect_btn = QtWidgets.QPushButton(
            "Connect"
        )

        self.connect_btn.clicked.connect(
            self.toggle_serial
        )

        row1.addWidget(
            self.connect_btn
        )


        self.demo_btn = QtWidgets.QPushButton(
            "Demo"
        )

        self.demo_btn.clicked.connect(
            self.toggle_demo
        )

        row1.addWidget(
            self.demo_btn
        )


        self.replay_btn = QtWidgets.QPushButton(
            "Replay CSV…"
        )

        self.replay_btn.clicked.connect(
            self.toggle_replay
        )

        row1.addWidget(
            self.replay_btn
        )


        row1.addStretch(
            1
        )


        # ==================================================
        # VIEW ROW
        # ==================================================

        row2 = QtWidgets.QHBoxLayout()

        outer.addLayout(
            row2
        )


        row2.addWidget(
            QtWidgets.QLabel(
                "Units"
            )
        )


        self.unit_box = QtWidgets.QComboBox()

        self.unit_box.addItems(
            list(UNITS)
        )

        self.unit_box.setCurrentText(
            "mV"
        )

        self.unit_box.currentIndexChanged.connect(
            self._mark_dirty
        )

        row2.addWidget(
            self.unit_box
        )


        row2.addWidget(
            QtWidgets.QLabel(
                "Time axis"
            )
        )


        self.time_box = QtWidgets.QComboBox()

        self.time_box.addItems(
            [
                "Auto",
                "Board timestamp",
                "PC clock",
            ]
        )

        self.time_box.currentIndexChanged.connect(
            self._mark_dirty
        )

        row2.addWidget(
            self.time_box
        )


        self.dc_check = QtWidgets.QCheckBox(
            "Subtract DC (raw plots, view only)"
        )

        self.dc_check.setChecked(
            True
        )

        self.dc_check.toggled.connect(
            self._mark_dirty
        )

        row2.addWidget(
            self.dc_check
        )


        self.invert_check = QtWidgets.QCheckBox(
            "Invert Ch0"
        )

        self.invert_check.toggled.connect(
            self._filters_changed
        )

        row2.addWidget(
            self.invert_check
        )


        row2.addStretch(
            1
        )


        # ==================================================
        # FILTER SETTINGS
        # ==================================================

        row3 = QtWidgets.QHBoxLayout()

        outer.addLayout(
            row3
        )


        row3.addWidget(
            QtWidgets.QLabel(
                "Pulse band (Hz) low"
            )
        )


        self.low_spin = QtWidgets.QDoubleSpinBox()

        self.low_spin.setRange(
            0.2,
            3.0
        )

        self.low_spin.setSingleStep(
            0.1
        )

        self.low_spin.setDecimals(
            1
        )

        self.low_spin.setValue(
            0.6
        )

        row3.addWidget(
            self.low_spin
        )


        row3.addWidget(
            QtWidgets.QLabel(
                "high"
            )
        )


        self.high_spin = QtWidgets.QDoubleSpinBox()

        self.high_spin.setRange(
            1.0,
            12.0
        )

        self.high_spin.setSingleStep(
            0.5
        )

        self.high_spin.setDecimals(
            1
        )

        self.high_spin.setValue(
            2.5
        )

        row3.addWidget(
            self.high_spin
        )


        self.low_spin.valueChanged.connect(
            self._filters_changed
        )

        self.high_spin.valueChanged.connect(
            self._filters_changed
        )


        row3.addSpacing(
            10
        )


        row3.addWidget(
            QtWidgets.QLabel(
                "Ignore peaks below (mV)"
            )
        )


        self.minpk_spin = QtWidgets.QDoubleSpinBox()

        self.minpk_spin.setRange(
            0.0,
            2.0
        )

        self.minpk_spin.setSingleStep(
            0.01
        )

        self.minpk_spin.setDecimals(
            2
        )

        self.minpk_spin.setValue(
            DEFAULT_MIN_PEAK_MV
        )

        row3.addWidget(
            self.minpk_spin
        )


        self.minpk_spin.valueChanged.connect(
            self._filters_changed
        )


        row3.addSpacing(
            10
        )


        row3.addWidget(
            QtWidgets.QLabel(
                "View"
            )
        )


        self.view_box = QtWidgets.QComboBox()

        self.view_box.addItems(
            VIEW_CHOICES
        )

        self.view_box.setCurrentIndex(
            1
        )

        self.view_box.currentIndexChanged.connect(
            self._mark_dirty
        )

        row3.addWidget(
            self.view_box
        )


        row3.addSpacing(
            10
        )


        self.save_check = QtWidgets.QCheckBox(
            "Save CSV"
        )

        self.save_check.setChecked(
            True
        )

        row3.addWidget(
            self.save_check
        )


        self.folder_btn = QtWidgets.QPushButton(
            "Folder…"
        )

        self.folder_btn.clicked.connect(
            self.choose_folder
        )

        row3.addWidget(
            self.folder_btn
        )


        self.folder_label = QtWidgets.QLabel(
            self.record_dir
        )

        row3.addWidget(
            self.folder_label,
            1
        )


        self.clear_btn = QtWidgets.QPushButton(
            "Clear plot"
        )

        self.clear_btn.clicked.connect(
            self.clear_plot
        )

        row3.addWidget(
            self.clear_btn
        )


        # ==================================================
        # GRAPH AREA
        # ==================================================

        pg.setConfigOption(
            "background",
            "w"
        )

        pg.setConfigOption(
            "foreground",
            "k"
        )


        self.glw = pg.GraphicsLayoutWidget()

        outer.addWidget(
            self.glw,
            1
        )


        self.plots: List[
            pg.PlotItem
        ] = []

        self.curves = []


        # ==================================================
        # ORIGINAL THREE GRAPHS
        # ==================================================

        for i in range(
            3
        ):

            p = self.glw.addPlot(
                row=(0, 1, 3)[i],
                col=0
            )

            p.showGrid(
                x=True,
                y=True,
                alpha=0.25
            )

            p.setTitle(
                PLOT_TITLES[i]
            )

            p.getAxis(
                "left"
            ).enableAutoSIPrefix(
                False
            )

            p.getAxis(
                "bottom"
            ).enableAutoSIPrefix(
                False
            )

            p.setXRange(
                0,
                RECORD_SECONDS,
                padding=0
            )

            p.enableAutoRange(
                axis="y"
            )

            p.getViewBox().setAutoVisible(
                y=True
            )

            p.getViewBox().sigRangeChangedManually.connect(
                self._user_zoomed
            )


            if i == P_CH1:

                p.setLabel(
                    "bottom",
                    "time since start (s)"
                )

            else:

                p.getAxis(
                    "bottom"
                ).setStyle(
                    showValues=False
                )


            self.plots.append(
                p
            )

            self.curves.append(
                p.plot(
                    pen=pg.mkPen(
                        PLOT_COLORS[i],
                        width=2
                    )
                )
            )


        self.plots[
            P_PULSE
        ].setXLink(
            self.plots[P_RAW]
        )

        self.plots[
            P_CH1
        ].setXLink(
            self.plots[P_RAW]
        )


        # ==================================================
        # AVERAGE BEAT
        # ==================================================

        pa = self.glw.addPlot(
            row=5,
            col=0
        )

        pa.showGrid(
            x=True,
            y=True,
            alpha=0.25
        )

        pa.setTitle(
            "Average beat shape"
        )

        pa.setLabel(
            "bottom",
            "time around the beat (s)"
        )

        pa.getAxis(
            "left"
        ).enableAutoSIPrefix(
            False
        )

        pa.getAxis(
            "bottom"
        ).enableAutoSIPrefix(
            False
        )

        self.plots.append(
            pa
        )


        self.avg_segs = [

            pa.plot(
                pen=pg.mkPen(
                    "#bbbbbb",
                    width=1
                )
            )

            for _ in range(
                20
            )
        ]


        self.avg_curve = pa.plot(
            pen=pg.mkPen(
                "#d62728",
                width=3
            )
        )


        # ==================================================
        # RECONSTRUCTION
        # ==================================================

        pr = self.glw.addPlot(
            row=2,
            col=0
        )

        pr.showGrid(
            x=True,
            y=True,
            alpha=0.25
        )

        pr.setTitle(
            "Reconstructed pulse "
            "(needs a locked rhythm; shown ~6 s late)"
        )

        pr.getAxis(
            "left"
        ).enableAutoSIPrefix(
            False
        )

        pr.getAxis(
            "bottom"
        ).setStyle(
            showValues=False
        )

        pr.enableAutoRange(
            axis="y"
        )

        pr.getViewBox().setAutoVisible(
            y=True
        )

        pr.setXLink(
            self.plots[P_RAW]
        )

        self.plots.append(
            pr
        )


        self.recon_curve = pr.plot(
            pen=pg.mkPen(
                "#6a3d9a",
                width=3
            )
        )


        # ==================================================
        # MPU GRAPH
        # ==================================================

        pm = self.glw.addPlot(
            row=4,
            col=0
        )

        pm.showGrid(
            x=True,
            y=True,
            alpha=0.25
        )

        pm.setTitle(
            "MPU6050 XYZ motion score"
        )

        pm.setLabel(
            "left",
            "motion (g)"
        )

        pm.setLabel(
            "bottom",
            "time since start (s)"
        )

        pm.getAxis(
            "left"
        ).enableAutoSIPrefix(
            False
        )

        pm.getAxis(
            "bottom"
        ).enableAutoSIPrefix(
            False
        )

        pm.setXLink(
            self.plots[P_RAW]
        )

        self.plots.append(
            pm
        )


        # Raw XYZ-derived motion score
        self.mpu_curve = pm.plot(
            pen=pg.mkPen(
                "#9467bd",
                width=1
            )
        )


        # Smoothed XYZ score used for detection
        self.mpu_smooth_curve = pm.plot(
            pen=pg.mkPen(
                "#000000",
                width=2
            )
        )


        # Motion threshold line
        self.mpu_threshold_line = pg.InfiniteLine(
            pos=MIN_MOTION_THRESHOLD_G,
            angle=0,
            pen=pg.mkPen(
                "#cc0000",
                width=2,
                style=QtCore.Qt.DashLine
            )
        )

        # Hide the threshold line until the first 5 s calibration
        # has produced the actual threshold.
        self.mpu_threshold_line.setVisible(False)

        pm.addItem(
            self.mpu_threshold_line
        )


        # Mark samples rejected as motion
        self.motion_marks = pg.ScatterPlotItem(
            symbol="o",
            size=5,
            brush=pg.mkBrush(
                "#cc0000"
            ),
            pen=None
        )

        pm.addItem(
            self.motion_marks
        )


        # ==================================================
        # GRAPH SIZES
        # ==================================================

        row_sizes = (
            3,
            4,
            4,
            2,
            3,
            3,
        )

        for row, factor in enumerate(
            row_sizes
        ):

            self.glw.ci.layout.setRowStretchFactor(
                row,
                factor
            )


        # ==================================================
        # BEAT MARKERS
        # ==================================================

        self.beat_marks = pg.ScatterPlotItem(
            symbol="t",
            size=12,
            brush=pg.mkBrush(
                "#9a9a9a"
            ),
            pen=None
        )

        self.plots[
            P_PULSE
        ].addItem(
            self.beat_marks
        )


        self.curve_nar = self.plots[
            P_PULSE
        ].plot(
            pen=pg.mkPen(
                "#ff7f0e",
                width=3
            )
        )


        self.nar_marks = pg.ScatterPlotItem(
            symbol="t",
            size=14,
            brush=pg.mkBrush(
                "#d62728"
            ),
            pen=None
        )

        self.plots[
            P_PULSE
        ].addItem(
            self.nar_marks
        )


        # ==================================================
        # LIVE VALUES
        # ==================================================

        mono = self.font()

        mono.setFamily(
            "Consolas"
        )

        mono.setPointSize(
            11
        )


        self.value_label = QtWidgets.QLabel(
            "—"
        )

        self.value_label.setFont(
            mono
        )

        outer.addWidget(
            self.value_label
        )


        # NEW motion status
        self.motion_label = QtWidgets.QLabel(
            "Motion: —"
        )

        motion_font = self.font()

        motion_font.setPointSize(
            12
        )

        motion_font.setBold(
            True
        )

        self.motion_label.setFont(
            motion_font
        )

        outer.addWidget(
            self.motion_label
        )


        self.pulse_label = QtWidgets.QLabel(
            "Pulse: —"
        )

        pf = self.font()

        pf.setPointSize(
            12
        )

        pf.setBold(
            True
        )

        self.pulse_label.setFont(
            pf
        )

        outer.addWidget(
            self.pulse_label
        )


        # ==================================================
        # LOG
        # ==================================================

        self.log_box = QtWidgets.QPlainTextEdit()

        self.log_box.setReadOnly(
            True
        )

        self.log_box.setMaximumHeight(
            100
        )

        self.log_box.setMaximumBlockCount(
            500
        )

        outer.addWidget(
            self.log_box
        )


        self.status_label = QtWidgets.QLabel(
            "Not connected"
        )

        self.statusBar().addWidget(
            self.status_label,
            1
        )


        self._update_axis_labels()


    # ======================================================
    # HELPERS
    # ======================================================

    def log(
        self,
        msg: str
    ) -> None:

        self.log_box.appendPlainText(
            f"{datetime.now():%H:%M:%S}  {msg}"
        )


    def _mark_dirty(
        self,
        *_args
    ) -> None:

        self._update_axis_labels()

        self._dirty = True


    def _update_axis_labels(
        self
    ) -> None:

        unit = self.unit_box.currentText()

        suffix = (
            " (DC removed)"
            if self.dc_check.isChecked()
            else ""
        )


        self.plots[
            P_RAW
        ].setLabel(
            "left",
            f"{unit}{suffix}"
        )

        self.plots[
            P_PULSE
        ].setLabel(
            "left",
            f"{unit} (band-pass)"
        )

        self.plots[
            P_CH1
        ].setLabel(
            "left",
            f"{unit}{suffix}"
        )

        self.plots[
            P_MPU
        ].setLabel(
            "left",
            "XYZ motion score (g)"
        )


    def refresh_ports(
        self
    ) -> None:

        current = self.port_box.currentText()

        self.port_box.clear()

        ports = available_ports()

        self.port_box.addItems(
            ports if ports
            else ["(no ports found)"]
        )

        if current in ports:

            self.port_box.setCurrentText(
                current
            )


    def choose_folder(
        self
    ) -> None:

        d = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Folder for CSV recordings",
            self.record_dir
        )

        if d:

            self.record_dir = d

            self.folder_label.setText(
                d
            )


    # ======================================================
    # CLEAR
    # ======================================================

    def clear_plot(
        self
    ) -> None:

        self.t.clear()
        self.tp.clear()

        self.c0.clear()
        self.c1.clear()

        self.ax.clear()
        self.ay.clear()
        self.az.clear()
        self.accel_mag.clear()
        self.axis_motion.clear()
        self.mpu.clear()
        self.mpu_avg.clear()
        self.motion_flags.clear()

        self._motion_window.clear()
        self._ax_baseline.clear()
        self._ay_baseline.clear()
        self._az_baseline.clear()

        self._motion_hold = 0
        self._motion_active = False

        self._motion_threshold = None
        self._calibration_scores.clear()

        self.presence.reset()
        self._pulse_present = False
        self.pulse_presence_flags.clear()

        if hasattr(self, "mpu_threshold_line"):
            self.mpu_threshold_line.setPos(MIN_MOTION_THRESHOLD_G)
            self.mpu_threshold_line.setVisible(False)

        self.yf.clear()
        self.beat_idx.clear()

        self.proc.reset()
        self.rate.reset()
        self.follow.reset()

        self.yn.clear()
        self.lock_flags.clear()

        self.recon.clear()
        self.recon_info = None

        self._pc0 = None

        self._wrap = 0.0

        self._clear_avg()

        self._mode_checked = False
        self._auto_pc = False

        self.plots[
            P_RAW
        ].setXRange(
            0,
            RECORD_SECONDS,
            padding=0
        )

        self._dirty = True


    # ======================================================
    # MOTION DETECTION
    # ======================================================

    def _update_motion(
        self,
        ax: float,
        ay: float,
        az: float
    ) -> bool:

        # Store the actual MPU6050 axes.
        self.ax.append(ax)
        self.ay.append(ay)
        self.az.append(az)

        # Total acceleration magnitude.
        magnitude = float(
            math.sqrt(
                ax * ax
                + ay * ay
                + az * az
            )
        )

        self.accel_mag.append(magnitude)

        # Estimate the recent orientation / gravity baseline.
        # Use the OLD baseline to measure the newest sample first,
        # then append the newest sample.  This prevents the current
        # movement from immediately pulling the baseline toward it.
        if self._ax_baseline:
            base_x = float(np.mean(self._ax_baseline))
            base_y = float(np.mean(self._ay_baseline))
            base_z = float(np.mean(self._az_baseline))
        else:
            base_x = ax
            base_y = ay
            base_z = az

        directional_motion = float(
            math.sqrt(
                (ax - base_x) ** 2
                + (ay - base_y) ** 2
                + (az - base_z) ** 2
            )
        )

        self.axis_motion.append(directional_motion)

        self._ax_baseline.append(ax)
        self._ay_baseline.append(ay)
        self._az_baseline.append(az)

        # Keep the useful old magnitude idea as well.
        magnitude_motion = abs(magnitude - 1.0)

        # Combined score:
        # - magnitude_motion detects translational acceleration
        # - directional_motion detects X/Y/Z orientation changes
        raw_motion = max(
            magnitude_motion,
            directional_motion
        )

        self.mpu.append(raw_motion)

        # Smooth the score over 0.20 s.
        self._motion_window.append(raw_motion)

        avg = float(
            np.mean(
                self._motion_window
            )
        )

        self.mpu_avg.append(avg)

        # ==================================================
        # FIRST 5 SECOND CALIBRATION
        # ==================================================

        if self._motion_threshold is None:

            self._calibration_scores.append(raw_motion)

            # Calibration samples are not labelled as motion artifacts.
            # Pulse processing is paused separately while calibration runs.
            self.motion_flags.append(False)

            if len(self._calibration_scores) >= CALIBRATION_SAMPLES:

                values = np.asarray(
                    self._calibration_scores,
                    dtype=float
                )

                median = float(np.median(values))

                mad = float(
                    np.median(
                        np.abs(values - median)
                    )
                )

                robust_sigma = 1.4826 * mad

                calculated_threshold = (
                    median
                    + CALIBRATION_SIGMA_MULT * robust_sigma
                )

                self._motion_threshold = max(
                    MIN_MOTION_THRESHOLD_G,
                    calculated_threshold
                )

                # Start post-calibration smoothing fresh so the first
                # movement decision is not biased by calibration history.
                self._motion_window.clear()
                self._motion_hold = 0

                if hasattr(self, "mpu_threshold_line"):
                    self.mpu_threshold_line.setPos(
                        self._motion_threshold
                    )
                    self.mpu_threshold_line.setVisible(True)

                self.log(
                    "Motion calibration complete: "
                    f"baseline={median:.4f} g, "
                    f"MAD={mad:.4f} g, "
                    f"threshold={self._motion_threshold:.4f} g"
                )

            return False

        over_threshold = (
            avg > self._motion_threshold
        )

        # If movement occurs, start/restart 0.25 s hold.
        if over_threshold:
            self._motion_hold = (
                MOTION_PADDING_SAMPLES
            )

        elif self._motion_hold > 0:
            self._motion_hold -= 1

        motion_now = (
            over_threshold
            or self._motion_hold > 0
        )

        self.motion_flags.append(
            motion_now
        )

        # ==================================================
        # PRE-PADDING
        #
        # When movement is detected, also reject the previous
        # 0.25 s because the TMR may already have been disturbed.
        # ==================================================

        if over_threshold:

            current = (
                len(self.motion_flags)
                - 1
            )

            start = max(
                0,
                current - MOTION_PADDING_SAMPLES
            )

            for i in range(
                start,
                current + 1
            ):

                self.motion_flags[i] = True

                if i < len(self.yf):
                    self.yf[i] = NAN

                if i < len(self.yn):
                    self.yn[i] = NAN

                if i < len(self.recon):
                    self.recon[i] = NAN

            self.beat_idx = [
                b
                for b in self.beat_idx
                if b < start
            ]

            if hasattr(
                self.follow,
                "beats"
            ):

                self.follow.beats = [
                    b
                    for b in self.follow.beats
                    if b < start
                ]

        return motion_now


    # ======================================================
    # NORMAL PULSE PROCESSING
    # ======================================================

    def _feed_pulse(
        self,
        ch0: float
    ) -> None:

        self.proc.refractory = int(
            round(
                (
                    0.65
                    * 60.0
                    / self.rate.bpm

                    if self.rate.locked

                    else 0.4
                )
                * FS
            )
        )


        self.rate.push(
            self._sign()
            * ch0
        )


        y, beat = self.proc.push(
            self._sign()
            * ch0
        )


        yn, _ = self.follow.push(
            self._sign()
            * ch0,
            self.rate.bpm,
            self.rate.locked
        )


        self.yn.append(
            yn
        )

        self.lock_flags.append(
            bool(
                self.rate.locked
            )
        )

        self.recon.append(
            NAN
        )


        n = len(
            self.recon
        )

        fi = int(
            round(
                FS
            )
        )

        W = int(
            round(
                RECON_WIN_S
                * FS
            )
        )


        # Reconstruction only when entire window is motion-free
        clean_window = False

        if len(
            self.motion_flags
        ) >= W:

            clean_window = not any(
                self.motion_flags[
                    -W:
                ]
            )


        if (
            n % fi == 0
            and
            n >= W
            and
            self.rate.locked
            and
            clean_window
        ):

            res = reconstruct_pulse(
                self._sign()
                * np.asarray(
                    self.c0[
                        n - W:
                        n
                    ]
                ),
                FS,
                self.rate.bpm / 60.0
            )


            if res is not None:

                self.recon_info = (
                    res["r2"],
                    res["pp"]
                )


                if (
                    res["r2"]
                    >= RECON_MIN_R2
                ):

                    a = (
                        n
                        - W // 2
                        - fi // 2
                    )

                    self.recon[
                        a:
                        a + fi
                    ] = [

                        float(v)

                        for v in
                        res["harm"][
                            W // 2 - fi // 2:
                            W // 2 + fi // 2
                        ]
                    ]


        self.yf.append(
            y
        )


        if beat is not None:

            self.beat_idx.append(
                beat
            )


    # ======================================================
    # MOTION-REJECTED SAMPLE
    # ======================================================

    def _feed_motion_artifact(
        self
    ) -> None:

        # Keep arrays aligned with time,
        # but do not process TMR as heartbeat.

        self.yf.append(
            NAN
        )

        self.yn.append(
            NAN
        )

        self.lock_flags.append(
            False
        )

        self.recon.append(
            NAN
        )


    # ======================================================
    # SIGN
    # ======================================================

    def _sign(
        self
    ) -> float:

        return (
            -1.0
            if self.invert_check.isChecked()
            else 1.0
        )


    # ======================================================
    # FILTER CHANGE
    # ======================================================

    def _filters_changed(
        self,
        *_args
    ) -> None:

        low = self.low_spin.value()

        high = self.high_spin.value()


        if low < high:

            try:

                self.proc = PulseProcessor(
                    fs=FS,
                    low=low,
                    high=high,
                    min_peak=(
                        self.minpk_spin.value()
                        * 1e-3
                    )
                )

            except ValueError as e:

                self.log(
                    f"Pulse band rejected: {e}"
                )


        mp = (
            self.minpk_spin.value()
            * 1e-3
        )

        self.proc.min_peak = mp
        self.follow.min_peak = mp

        self.rate.min_rms = (
            0.7 * mp
        )

        # MPU6050 X/Y/Z is the only motion gate.
        self.rate.max_motion = 0.0


        self.proc.reset()
        self.rate.reset()
        self.follow.reset()


        self.yf.clear()
        self.beat_idx.clear()

        self.yn.clear()
        self.lock_flags.clear()

        self.recon.clear()

        self.recon_info = None


        was_motion = False


        for i, value in enumerate(
            self.c0
        ):

            motion = (
                i < len(self.motion_flags)
                and self.motion_flags[i]
            )

            pulse_present = (
                i < len(self.pulse_presence_flags)
                and self.pulse_presence_flags[i]
            )

            blocked = motion or not pulse_present

            if blocked:

                if not was_motion:

                    self.proc.reset()
                    self.rate.reset()
                    self.follow.reset()


                self._feed_motion_artifact()

                was_motion = True


            else:

                if was_motion:

                    self.proc.reset()
                    self.rate.reset()
                    self.follow.reset()


                self._feed_pulse(
                    value
                )

                was_motion = False


        self._motion_active = was_motion

        self._mark_dirty()


    # ======================================================
    # START / STOP
    # ======================================================

    def toggle_serial(
        self
    ) -> None:

        if self.source is not None:

            self.stop_source(
                "Disconnected"
            )

            return


        port = self.port_box.currentText()


        if (
            not port
            or port.startswith("(")
        ):

            QtWidgets.QMessageBox.warning(
                self,
                "No port",
                "No serial port selected."
            )

            return


        try:

            src = SerialSource(
                port,
                int(
                    self.baud_box.currentText()
                )
            )

        except Exception as e:

            QtWidgets.QMessageBox.critical(
                self,
                "Could not open port",
                f"{port}\n\n{e}"
            )

            return


        self.start_source(
            src,
            demo=False
        )


    def toggle_demo(
        self
    ) -> None:

        if self.source is not None:

            self.stop_source(
                "Demo stopped"
            )

        else:

            self.start_source(
                SimulatedBoard(),
                demo=True
            )


    def toggle_replay(
        self
    ) -> None:

        if self.source is not None:

            self.stop_source(
                "Replay stopped"
            )

            return


        path, _ = (
            QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Replay a recorded CSV",
                self.record_dir,
                "CSV files (*.csv)"
            )
        )


        if path:

            self.start_replay(
                path
            )


    def start_replay(
        self,
        path: str,
        speed: float = 1.0
    ) -> None:

        try:

            src = CsvReplaySource(
                path,
                speed=speed
            )

        except Exception as e:

            QtWidgets.QMessageBox.critical(
                self,
                "Could not open CSV",
                f"{path}\n\n{e}"
            )

            return


        self.start_source(
            src,
            demo=False,
            kind="replay"
        )


    def start_source(
        self,
        src,
        demo: bool,
        kind: Optional[str] = None
    ) -> None:

        kind = (
            kind
            or (
                "demo"
                if demo
                else "serial"
            )
        )


        self._kind = kind

        self.source = src

        self.parser = StmAsciiParser()

        self.clear_plot()

        self.rec_idx = 0

        self.last_t = None

        self.total_samples = 0

        self._finished_logged = False

        self._rate_mark = (
            time.monotonic(),
            0
        )

        self._started_at = (
            time.monotonic()
        )

        self._close_csv()

        self._csv_pending = (
            kind != "replay"
        )


        self.connect_btn.setText(
            "Disconnect"
        )

        self.refresh_btn.setEnabled(
            False
        )

        self.log(
            f"Connected: {src.name}"
        )

        self.poll_timer.start()


    def stop_source(
        self,
        why: str
    ) -> None:

        self.poll_timer.stop()


        if self.source is not None:

            self.source.close()


        self.source = None

        self._close_csv()


        self.connect_btn.setText(
            "Connect"
        )

        self.refresh_btn.setEnabled(
            True
        )

        self.status_label.setText(
            "Not connected"
        )

        self.log(
            why
        )


    # ======================================================
    # CSV
    # ======================================================

    def _open_csv(
        self
    ) -> None:

        self._close_csv()


        if not self.save_check.isChecked():

            return


        try:

            os.makedirs(
                self.record_dir,
                exist_ok=True
            )


            path = os.path.join(
                self.record_dir,
                f"tmr_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )


            self.csv_file = open(
                path,
                "w",
                newline=""
            )


            self.csv_writer = csv.writer(
                self.csv_file
            )


            self.csv_writer.writerow(
                [
                    "rec",
                    "t_board_s",
                    "t_pc_s",
                    "ch0_V",
                    "ch1_V",
                    "ax_g",
                    "ay_g",
                    "az_g",
                    "accel_mag_g",
                    "axis_motion_g",
                    "motion_score_g",
                    "motion_avg_g",
                    "motion_threshold_g",
                    "calibrating",
                    "motion_artifact",
                    "pulse_present",
                    "presence_contact_shift_mV",
                    "presence_contact_threshold_mV",
                    "presence_band_rms_mV",
                    "presence_band_threshold_mV",
                ]
            )


            self.log(
                f"Saving to {path}"
            )


        except OSError as e:

            self.csv_file = None
            self.csv_writer = None

            self.log(
                f"CSV disabled: {e}"
            )


    def _close_csv(
        self
    ) -> None:

        if self.csv_file is not None:

            try:

                self.csv_file.close()

            except OSError:

                pass


        self.csv_file = None
        self.csv_writer = None


    # ======================================================
    # SERIAL POLL
    # ======================================================

    def poll(
        self
    ) -> None:

        if (
            self.source is None
            or self.parser is None
        ):

            return


        try:

            data = self.source.read()

        except Exception as e:

            self.stop_source(
                f"Read error: {e}"
            )

            return


        if not data:

            return


        samples, text = self.parser.feed(
            data
        )


        for line in text:

            self.log(
                f"board: {line}"
            )


        for sample in samples:

            self._handle_sample(
                sample
            )


    # ======================================================
    # HANDLE SAMPLE
    # ======================================================

    def _handle_sample(
        self,
        s: Sample
    ) -> None:

        if self._kind == "replay":

            s = Sample(
                self.total_samples / FS,
                s.ch0,
                s.ch1,
                s.ax,
                s.ay,
                s.az
            )


        if self._csv_pending:

            self._csv_pending = False

            if self.save_check.isChecked():

                self._open_csv()


        if (
            self.last_t is not None
            and s.t + 0.5 < self.last_t
        ):

            self.rec_idx += 1

            self.clear_plot()


        self.last_t = s.t


        now = time.monotonic()


        if self._pc0 is None:

            self._pc0 = now


        t_pc = (
            now - self._pc0
        )


        # ==================================================
        # STORE RAW DATA
        # ==================================================

        self.t.append(
            s.t + self._wrap
        )

        self.tp.append(
            t_pc
        )

        self.c0.append(
            s.ch0
        )

        self.c1.append(
            s.ch1
        )


        # ==================================================
        # MOTION ARTIFACT DETECTION FROM MPU X/Y/Z
        # ==================================================

        calibrating = (
            self._motion_threshold is None
        )

        motion_now = self._update_motion(
            s.ax,
            s.ay,
            s.az
        )

        # TMR presence detector always receives Ch0, including during
        # calibration and motion, so it can maintain its own rolling state.
        pulse_present = self.presence.push(s.ch0)
        self.pulse_presence_flags.append(bool(pulse_present))


        # ==================================================
        # HEARTBEAT DETECTION
        # ==================================================

        if calibrating or not self.presence.calibrated:

            # First 5 s: learn BOTH the MPU resting baseline and the TMR
            # no-body baseline.  Do not perform heartbeat analysis yet.
            self._feed_motion_artifact()
            self._motion_active = False
            self._pulse_present = False

        elif motion_now:

            # First sample entering movement
            if not self._motion_active:

                self.proc.reset()
                self.rate.reset()
                self.follow.reset()


            self._motion_active = True
            self._pulse_present = bool(pulse_present)


            # Do not analyse TMR during motion
            self._feed_motion_artifact()


        elif not pulse_present:

            # No convincing body/pulse signal above the learned no-body
            # baseline.  Explicitly block the BPM detector.
            if self._pulse_present or self._motion_active:
                self.proc.reset()
                self.rate.reset()
                self.follow.reset()

            self._motion_active = False
            self._pulse_present = False
            self._feed_motion_artifact()


        else:

            # A clean pulse signal has just appeared, or motion has just ended:
            # start the pulse trackers from a fresh window.
            if (
                self._motion_active
                or not self._pulse_present
            ):

                self.proc.reset()
                self.rate.reset()
                self.follow.reset()


            self._motion_active = False
            self._pulse_present = True


            # Normal TMR heartbeat analysis
            self._feed_pulse(
                s.ch0
            )


        self.total_samples += 1

        self._dirty = True


        self._check_board_clock(
            t_pc
        )


        # ==================================================
        # SAVE
        # ==================================================

        if self.csv_writer is not None:

            self.csv_writer.writerow(
                [
                    self.rec_idx,
                    f"{s.t + self._wrap:.3f}",
                    f"{t_pc:.3f}",
                    f"{s.ch0:.6f}",
                    f"{s.ch1:.6f}",
                    f"{s.ax:.6f}",
                    f"{s.ay:.6f}",
                    f"{s.az:.6f}",
                    f"{self.accel_mag[-1]:.6f}",
                    f"{self.axis_motion[-1]:.6f}",
                    f"{self.mpu[-1]:.6f}",
                    f"{self.mpu_avg[-1]:.6f}",
                    (
                        f"{self._motion_threshold:.6f}"
                        if self._motion_threshold is not None
                        else ""
                    ),
                    int(self._motion_threshold is None),
                    int(
                        self.motion_flags[-1]
                    ),
                    int(bool(pulse_present)),
                    f"{1e3 * self.presence.contact_shift:.6f}",
                    (
                        f"{1e3 * self.presence.contact_threshold:.6f}"
                        if self.presence.calibrated
                        else ""
                    ),
                    f"{1e3 * self.presence.band_rms:.6f}",
                    (
                        f"{1e3 * self.presence.band_threshold:.6f}"
                        if self.presence.calibrated
                        else ""
                    ),
                ]
            )


    # ======================================================
    # CLOCK
    # ======================================================

    def _check_board_clock(
        self,
        t_pc: float
    ) -> None:

        if (
            self._mode_checked
            or len(self.t) < 20
            or t_pc < 1.0
        ):

            return


        self._mode_checked = True


        ratio = (
            self.t[-1]
            - self.t[0]
        ) / t_pc


        if (
            ratio < 0.8
            or ratio > 1.25
        ):

            self._auto_pc = True


    def _use_pc_clock(
        self
    ) -> bool:

        mode = self.time_box.currentText()


        if mode == "PC clock":

            return True


        if mode == "Board timestamp":

            return False


        return self._auto_pc


    # ======================================================
    # AVERAGE BEAT
    # ======================================================

    def _clear_avg(
        self
    ) -> None:

        if not hasattr(
            self,
            "avg_curve"
        ):

            return


        self.avg_curve.setData(
            [],
            []
        )


        for curve in self.avg_segs:

            curve.setData(
                [],
                []
            )


        self.plots[
            P_AVG
        ].setTitle(
            "Average beat shape "
            "(needs a locked rhythm)"
        )


    def _update_avg(
        self,
        scale: float
    ) -> None:

        if self._motion_active or not self._pulse_present:

            self._clear_avg()

            return


        res = None


        if (
            self.rate.locked
            and len(
                self.follow.beats
            ) >= 4
        ):

            res = average_beat(
                np.asarray(
                    self.yf,
                    dtype=float
                ),
                self.follow.beats,
                fs=FS
            )


        if res is None:

            self._clear_avg()

            return


        tt, mean, segs = res


        for i, curve in enumerate(
            self.avg_segs
        ):

            if i < len(
                segs
            ):

                curve.setData(
                    tt,
                    segs[i] * scale
                )

            else:

                curve.setData(
                    [],
                    []
                )


        self.avg_curve.setData(
            tt,
            mean * scale
        )


        u = self.unit_box.currentText()


        self.plots[
            P_AVG
        ].setTitle(

            f"Average beat shape "

            f"({len(segs)} beats lined up; "

            f"{(mean.max() - mean.min()) * scale:.3f} "

            f"{u} peak-to-peak)"
        )


    # ======================================================
    # PULSE TEXT
    # ======================================================

    def _pulse_text(
        self
    ) -> str:

        # Calibration gets priority.
        if self._motion_threshold is None:
            return (
                "Pulse: PAUSED — CALIBRATING MOTION BASELINE"
            )

        # MPU motion artifact gets next priority.
        if (
            self.motion_flags
            and self.motion_flags[-1]
        ):

            return (
                "Pulse: PAUSED — MOTION ARTIFACT"
            )

        if self.presence.calibrated and not self._pulse_present:
            return (
                "Pulse: NO PULSE SIGNAL DETECTED"
            )


        if not self.yf:

            return "Pulse: —"


        r = self.rate


        if r.locked:

            return (
                f"Pulse: {r.bpm:.0f} BPM "
                f"(locked for {r.lock_age} s, "
                f"signal strength {r.score:.0f})"
            )


        return (
            f"Pulse: {r.status}"
        )


    # ======================================================
    # USER ZOOM
    # ======================================================

    def _user_zoomed(
        self,
        *_args
    ) -> None:

        if not self.view_box.currentText().startswith(
            "Manual"
        ):

            self.view_box.setCurrentIndex(
                len(VIEW_CHOICES)
                - 1
            )


    # ======================================================
    # REDRAW
    # ======================================================

    def redraw(
        self
    ) -> None:

        if not self._dirty:

            return


        self._dirty = False


        scale = UNITS[
            self.unit_box.currentText()
        ]


        use_pc = self._use_pc_clock()


        t = np.asarray(
            self.tp
            if use_pc
            else self.t
        )


        mode = self.view_box.currentText()

        manual = mode.startswith(
            "Manual"
        )


        win_s = (

            float(
                mode.split()[1]
            )

            if mode.startswith(
                "Last"
            )

            else None
        )


        # ==================================================
        # X RANGE
        # ==================================================

        if (
            t.size
            and not manual
        ):

            if win_s is not None:

                hi = max(
                    float(
                        t[-1]
                    ),
                    win_s
                )

                self.plots[
                    P_RAW
                ].setXRange(
                    hi - win_s,
                    hi,
                    padding=0
                )

            else:

                self.plots[
                    P_RAW
                ].setXRange(
                    0,
                    max(
                        float(
                            t[-1]
                        ),
                        RECORD_SECONDS
                    ),
                    padding=0
                )


        # ==================================================
        # RAW TMR / CH1
        # ==================================================

        for idx, raw in (

            (
                P_RAW,
                self.c0
            ),

            (
                P_CH1,
                self.c1
            )

        ):

            y = (
                np.asarray(
                    raw
                )
                * scale
            )


            if (
                idx == P_RAW
                and self.invert_check.isChecked()
            ):

                y = -y


            if (
                self.dc_check.isChecked()
                and y.size
            ):

                y = (
                    y - y.mean()
                )


            self.curves[
                idx
            ].setData(
                t,
                y
            )


        # ==================================================
        # MPU
        # ==================================================

        raw_mpu = np.asarray(
            self.mpu
        )

        smooth_mpu = np.asarray(
            self.mpu_avg
        )

        flags = np.asarray(
            self.motion_flags,
            dtype=bool
        )


        self.mpu_curve.setData(
            t,
            raw_mpu
        )


        self.mpu_smooth_curve.setData(
            t,
            smooth_mpu
        )


        if (
            t.size
            and flags.size == t.size
        ):

            self.motion_marks.setData(
                x=t[
                    flags
                ],
                y=smooth_mpu[
                    flags
                ]
            )


        # ==================================================
        # FILTERED TMR
        # ==================================================

        yf = (
            np.asarray(
                self.yf
            )
            * scale
        )


        self.curves[
            P_PULSE
        ].setData(
            t,
            yf
        )


        # ==================================================
        # RECON
        # ==================================================

        rc = (
            np.asarray(
                self.recon
            )
            * scale
        )


        self.recon_curve.setData(
            t,
            rc
        )


        # ==================================================
        # RATE LOCK TRACE
        # ==================================================

        yn = (
            np.asarray(
                self.yn
            )
            * scale
        )


        self.curve_nar.setData(
            t,
            yn
        )


        # ==================================================
        # BEAT MARKERS
        # ==================================================

        lock = np.asarray(
            self.lock_flags,
            dtype=bool
        )


        bi = np.asarray(
            self.beat_idx,
            dtype=int
        )


        bi = bi[
            (bi < t.size)
            &
            (bi < lock.size)
        ]


        # Extra safety:
        # never display a heartbeat marker in motion
        if bi.size and flags.size:

            bi = bi[
                ~flags[
                    bi
                ]
            ]


        if bi.size:

            self.beat_marks.setData(
                x=t[
                    bi
                ],
                y=yf[
                    bi
                ]
            )

        else:

            self.beat_marks.setData(
                x=[],
                y=[]
            )


        nb = np.asarray(
            self.follow.beats,
            dtype=int
        )


        nb = nb[
            nb < t.size
        ]


        if nb.size and flags.size:

            nb = nb[
                ~flags[
                    nb
                ]
            ]


        if nb.size:

            self.nar_marks.setData(
                x=t[
                    nb
                ],
                y=yn[
                    nb
                ]
            )

        else:

            self.nar_marks.setData(
                x=[],
                y=[]
            )


        # ==================================================
        # AVERAGE BEAT
        # ==================================================

        now_m = time.monotonic()


        if (
            now_m
            - self._avg_t
            > 0.5
        ):

            self._avg_t = now_m

            self._update_avg(
                scale
            )


        # ==================================================
        # MOTION STATUS
        # ==================================================

        if self.motion_flags:

            motion_now = (
                self.motion_flags[-1]
            )

            avg_now = (
                self.mpu_avg[-1]
            )

            if self._motion_threshold is None:

                elapsed_cal = min(
                    len(self._calibration_scores) / FS,
                    CALIBRATION_S
                )

                self.motion_label.setText(
                    f"CALIBRATING — NO BODY ON SENSOR, KEEP STILL  |  "
                    f"{elapsed_cal:.1f}/{CALIBRATION_S:.1f} s  |  "
                    f"XYZ motion avg = {avg_now:.3f} g"
                )

                self.motion_label.setStyleSheet(
                    "color: #d97706;"
                )

            elif motion_now:

                self.motion_label.setText(

                    f"MOTION ARTIFACT  |  "

                    f"XYZ motion avg = {avg_now:.3f} g  |  "

                    f"threshold = {self._motion_threshold:.3f} g"
                )

                self.motion_label.setStyleSheet(
                    "color: #c0392b;"
                )

            else:

                self.motion_label.setText(

                    f"Motion: CLEAR  |  "

                    f"XYZ motion avg = {avg_now:.3f} g  |  "

                    f"threshold = {self._motion_threshold:.3f} g"
                )

                self.motion_label.setStyleSheet(
                    "color: #2a7a2a;"
                )


        # ==================================================
        # PULSE STATUS
        # ==================================================

        self.pulse_label.setText(
            self._pulse_text()
        )


        if self._motion_threshold is None:

            self.pulse_label.setStyleSheet(
                "color: #d97706;"
            )

        elif (
            self.motion_flags
            and self.motion_flags[-1]
        ):

            self.pulse_label.setStyleSheet(
                "color: #c0392b;"
            )

        elif self.presence.calibrated and not self._pulse_present:

            self.pulse_label.setStyleSheet(
                "color: #d97706;"
            )

        elif self.rate.locked:

            self.pulse_label.setStyleSheet(
                "color: #2a7a2a;"
            )

        else:

            self.pulse_label.setStyleSheet(
                ""
            )


        # ==================================================
        # LIVE VALUES
        # ==================================================

        if self.c0:

            u = self.unit_box.currentText()


            self.value_label.setText(

                f"board t = {self.t[-1]:6.3f} s    "

                f"Ch0 = "
                f"{self.c0[-1] * scale:10.4f} {u}    "

                f"ax = {self.ax[-1]:+.3f} g    "
                f"ay = {self.ay[-1]:+.3f} g    "
                f"az = {self.az[-1]:+.3f} g    "

                f"|a| = {self.accel_mag[-1]:.3f} g    "
                f"XYZ score = {self.mpu[-1]:.4f} g    "
                f"avg = {self.mpu_avg[-1]:.4f} g    "
                f"contact Δ = {1e3 * self.presence.contact_shift:.3f} mV    "
                f"pulse RMS = {1e3 * self.presence.band_rms:.3f} mV    "
                f"threshold = "
                f"{self._motion_threshold:.4f} g"
                if self._motion_threshold is not None
                else (
                    f"board t = {self.t[-1]:6.3f} s    "
                    f"Ch0 = {self.c0[-1] * scale:10.4f} {u}    "
                    f"ax = {self.ax[-1]:+.3f} g    "
                    f"ay = {self.ay[-1]:+.3f} g    "
                    f"az = {self.az[-1]:+.3f} g    "
                    f"|a| = {self.accel_mag[-1]:.3f} g    "
                    f"XYZ score = {self.mpu[-1]:.4f} g    "
                    f"avg = {self.mpu_avg[-1]:.4f} g    "
                    f"contact Δ = {1e3 * self.presence.contact_shift:.3f} mV    "
                    f"pulse RMS = {1e3 * self.presence.band_rms:.3f} mV    "
                    f"threshold = calibrating"
                )
            )


    # ======================================================
    # STATUS
    # ======================================================

    def tick(
        self
    ) -> None:

        if (
            self.source is None
            or self.parser is None
        ):

            return


        now = time.monotonic()


        mark_t, mark_n = (
            self._rate_mark
        )


        rate = (

            (
                self.total_samples
                - mark_n
            )

            /

            max(
                now - mark_t,
                1e-6
            )
        )


        self._rate_mark = (
            now,
            self.total_samples
        )


        msg = (

            f"{self.source.name}"

            f"   |   {rate:4.0f} samples/s"

            f"   |   {self.total_samples} samples"

            f"   |   junk bytes: "
            f"{self.parser.bad_bytes}"
        )


        self.status_label.setText(
            msg
        )


        if self.csv_file is not None:

            try:

                self.csv_file.flush()

            except OSError:

                pass


    # ======================================================
    # CLOSE
    # ======================================================

    def closeEvent(
        self,
        event
    ) -> None:

        self.poll_timer.stop()


        if self.source is not None:

            self.source.close()


        self._close_csv()


        super().closeEvent(
            event
        )


# ==========================================================
# MAIN
# ==========================================================

def main() -> int:

    ap = argparse.ArgumentParser(
        description=(
            "TMR + MPU live viewer"
        )
    )


    ap.add_argument(
        "--demo",
        action="store_true"
    )


    ap.add_argument(
        "--replay",
        metavar="CSV"
    )


    ap.add_argument(
        "--speed",
        type=float,
        default=1.0
    )


    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            ),
            "recordings"
        )
    )


    args = ap.parse_args()


    app = QtWidgets.QApplication(
        sys.argv
    )


    win = MainWindow(
        args.out
    )


    win.show()


    if args.demo:

        win.toggle_demo()


    elif args.replay:

        win.start_replay(
            args.replay,
            args.speed
        )


    return app.exec()


if __name__ == "__main__":

    sys.exit(
        main()
    )