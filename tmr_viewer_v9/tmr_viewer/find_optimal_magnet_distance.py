"""Compare pulse-band quality across distance-labelled CSV recordings.

Record each distance with the same sensor configuration, duration, subject posture,
and minimal movement. File names should include a distance e.g. 2mm.csv, 4mm.csv.

Install: py -m pip install numpy pandas scipy matplotlib
Run: py find_optimal_magnet_distance.py "C:\\Users\\thoma\\Downloads\\tmr_viewer\\recordings" --list-columns
Then: py find_optimal_magnet_distance.py "...\\recordings" --time TIME_COLUMN --signal SIGNAL_COLUMN

This identifies a reproducible pulse-like signal, not proof of blood-flow sensing.
"""
import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
import matplotlib.pyplot as plt

DISTANCE_RE = re.compile(r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(mm|cm)\b", re.I)


def distance_mm(path):
    match = DISTANCE_RE.search(path.stem.replace('_', ' '))
    if not match:
        return None
    number = float(match.group(1))
    return number * (10 if match.group(2).lower() == 'cm' else 1)


def columns_from_first(files):
    for path in files:
        try:
            data = pd.read_csv(path, nrows=4, sep=None, engine='python')
            if len(data.columns) > 1:
                print(f'Columns in {path.name}: {list(data.columns)}')
        except Exception as exc:
            print(f'Cannot inspect {path.name}: {exc}')


def choose_column(df, explicit, hints):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f'Column {explicit!r} not found; available: {list(df.columns)}')
        return explicit
    for hint in hints:
        for col in df.columns:
            if hint in str(col).lower():
                return col
    return None


def read_signal(path, time_col, signal_col, rate):
    df = pd.read_csv(path, sep=None, engine='python')
    tname = choose_column(df, time_col, ['time', 'timestamp', 'elapsed', 'seconds', ' t(s)'])
    sname = choose_column(df, signal_col, ['ch0', 'ain0', 'tmr', 'raw', 'voltage', 'signal'])
    if sname is None:
        available = [c for c in df.columns if c != tname and pd.to_numeric(df[c], errors='coerce').notna().sum() > 10]
        if len(available) == 1:
            sname = available[0]
        else:
            raise ValueError(f'Cannot identify sensor column among {list(df.columns)}. Set --signal.')
    x = pd.to_numeric(df[sname], errors='coerce').to_numpy(dtype=float)
    if tname is None:
        if rate is None:
            raise ValueError(f'No time column found in {path.name}; provide --rate HZ and --signal COLUMN.')
        t = np.arange(len(x), dtype=float) / rate
    else:
        t = pd.to_numeric(df[tname], errors='coerce').to_numpy(dtype=float)
        if 'ms' in tname.lower() or 'millisecond' in tname.lower():
            t /= 1000
        elif 'us' in tname.lower() or 'microsecond' in tname.lower():
            t /= 1e6
    valid = np.isfinite(t) & np.isfinite(x)
    t, x = t[valid], x[valid]
    if len(t) < 40:
        raise ValueError('Too few numeric samples')
    order = np.argsort(t)
    t, x = t[order], x[order]
    t, indices = np.unique(t, return_index=True)
    x = x[indices]
    dt = np.median(np.diff(t))
    fs = 1. / dt
    if not (5 <= fs <= 50000):
        raise ValueError(f'Implausible sampling frequency {fs:.2f} Hz; verify time units.')
    # Resample to a uniform grid for reliable filtering.
    uniform_t = np.arange(t[0], t[-1], dt)
    if len(uniform_t) > 2_000_000:
        raise ValueError('Recording too large')
    uniform_x = np.interp(uniform_t, t, x)
    return uniform_t, uniform_x, fs


def assess(t, x, fs, low, high):
    if fs < high * 2.5:
        raise ValueError('Sampling rate is too low for pulse band')
    if t[-1] - t[0] < 12:
        raise ValueError('Record at least 12 seconds per distance (prefer 30 seconds).')
    x = x - np.median(x)
    sos = butter(3, [low, high], btype='bandpass', fs=fs, output='sos')
    y = sosfiltfilt(sos, x)
    # Trim filter edge effects (2 seconds each end).
    pad = int(2 * fs)
    y, x = y[pad:-pad], x[pad:-pad]
    if len(y) < fs * 8:
        raise ValueError('Usable recording too short after filtering')
    pulse_rms = np.sqrt(np.mean(y*y))
    # A pulse-like waveform should have a narrow peak within the physiological band.
    freqs, power = welch(y, fs=fs, nperseg=min(len(y), int(fs*8)))
    band = (freqs >= low) & (freqs <= high)
    power_band = power[band]
    if power_band.size < 3 or power_band.sum() <= 0:
        raise ValueError('No usable pulse-band power')
    dominant = freqs[band][np.argmax(power_band)]
    concentration = float(power_band.max() / power_band.sum())
    prominence = max(float(np.std(y) * 0.6), 1e-15)
    peaks, _ = find_peaks(y, distance=int(fs/high), prominence=prominence)
    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals >= 1/high) & (intervals <= 1/low)]
    if len(intervals) >= 4:
        cv = float(np.std(intervals) / np.mean(intervals))
        regularity = max(0., 1. - cv)
    else:
        regularity = 0.
    # Penalize ADC saturation / flat-lining and excessive drift in the raw trace.
    extent = np.ptp(x)
    unique_fraction = len(np.unique(np.round(x, decimals=9))) / len(x)
    flat = extent <= 1e-12 or unique_fraction < 0.002
    drift = float(np.std(x) / max(pulse_rms, 1e-12))
    # Relative reproducibility-focused quality index, not a calibrated SNR.
    quality = float(100 * concentration * regularity / (1 + 0.02 * drift))
    if flat:
        quality = 0.
    return dict(score=quality, dominant_hz=dominant, bpm=dominant*60,
                regularity=regularity, concentration=concentration, pulse_rms=pulse_rms,
                raw_to_pulse_rms=drift, flat=flat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', type=Path, help='Folder containing e.g. 2mm.csv, 4mm.csv')
    ap.add_argument('--time', help='Name of time column; omitted if auto-detected')
    ap.add_argument('--signal', help='Name of raw Ch0/TMR column; omitted if auto-detected')
    ap.add_argument('--rate', type=float, help='Sampling Hz if CSV has no time column')
    ap.add_argument('--low', type=float, default=0.6, help='Pulse band low cutoff in Hz')
    ap.add_argument('--high', type=float, default=2.5, help='Pulse band high cutoff in Hz')
    ap.add_argument('--list-columns', action='store_true', help='Show CSV column names and exit')
    args = ap.parse_args()
    files = sorted(p for p in args.folder.glob('*.csv') if distance_mm(p) is not None)
    if not files:
        sys.exit('No distance-labelled CSVs found. Rename files like 2mm.csv, 4mm.csv, 6mm.csv.')
    if args.list_columns:
        columns_from_first(files)
        return
    rows = []
    for f in files:
        try:
            t, x, fs = read_signal(f, args.time, args.signal, args.rate)
            row = assess(t, x, fs, args.low, args.high)
            row.update(distance_mm=distance_mm(f), file=f.name, sample_hz=fs)
            rows.append(row)
            print(f'{f.name}: quality {row["score"]:.1f}, dominant {row["bpm"]:.0f} bpm')
        except Exception as e:
            print(f'Skipping {f.name}: {e}')
    if not rows:
        sys.exit('Could not process any recordings. Inspect columns with --list-columns.')
    result = pd.DataFrame(rows).sort_values('distance_mm')
    out = args.folder / 'magnet_distance_results.csv'
    result.to_csv(out, index=False)
    # Repeated recordings at the same distance can be aggregated via median.
    agg = result.groupby('distance_mm', as_index=False).agg(score=('score','median'), trials=('score','size'))
    best = agg.loc[agg.score.idxmax()]
    print(f'\nBest measured distance: {best.distance_mm:g} mm (quality index {best.score:.1f}; {int(best.trials)} trial(s))')
    print('This is a relative signal-quality ranking, NOT proof that the signal is a heartbeat.')
    print(f'Results: {out}')
    plt.plot(agg.distance_mm, agg.score, 'o-')
    plt.axvline(best.distance_mm, linestyle='--', label=f'Best measured = {best.distance_mm:g} mm')
    plt.xlabel('Magnet-to-sensor distance (mm)')
    plt.ylabel('Relative pulse-like signal quality (index)')
    plt.title('Magnet distance sweep (compare repeated trials)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot = args.folder / 'magnet_distance_quality.png'
    plt.savefig(plot, dpi=170)
    print(f'Graph: {plot}')
    plt.show()

if __name__ == '__main__':
    main()
