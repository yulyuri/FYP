"""Live pulse filtering and beat detection (no Qt, no hardware).

PulseProcessor.push(x) takes one Ch0 sample (volts) at a time and returns
(filtered_value, beat_index_or_None). Everything is causal, so it behaves the
same live as it would on a replayed recording.

Pipeline
    1. Band-pass (default 0.6-2.5 Hz = 36-150 beats/min). This removes slow
       drift and the narrow ~12.5 Hz line seen in the baseline.
    2. Jump guard: when the magnet is moved in or out, Ch0 changes by hundreds
       of mV in a second. That would ring the filter for many seconds, so a
       sample-to-sample step bigger than `jump` re-centres the filter and the
       output is held back (NaN) for `settle_s`.
    3. Beat detector: local maxima above `thresh_sigma` x the recent signal
       spread, at least `refractory_s` after the previous beat.
    4. Rhythm check: a real resting heartbeat has beat-to-beat spacing that
       varies by roughly 10% or less. Random wobble does not. The result is
       reported as good / fair / poor so noise is not mistaken for a pulse.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

NAN = float("nan")


class PulseProcessor:
    def __init__(self, fs: float = 50.0, low: float = 0.6, high: float = 2.5,
                 order: int = 2, jump: float = 0.010, settle_s: float = 3.0,
                 refractory_s: float = 0.4, amp_window_s: float = 5.0,
                 thresh_sigma: float = 0.5, min_peak: float = 0.0) -> None:
        if not (0 < low < high < fs / 2):
            raise ValueError("need 0 < low < high < fs/2")
        self.fs = fs
        self.low, self.high = low, high
        self.jump = jump
        self.settle = int(round(settle_s * fs))
        self.refractory = int(round(refractory_s * fs))
        self.amp_len = int(round(amp_window_s * fs))
        self.thresh_sigma = thresh_sigma
        self.min_peak = min_peak       # volts: peaks smaller than this are never beats
        self.sos = signal.butter(order, [low, high], "bandpass", fs=fs, output="sos")
        self._zi0 = signal.sosfilt_zi(self.sos)
        self.reset()

    def reset(self) -> None:
        self._zi: Optional[np.ndarray] = None
        self._prev_x: Optional[float] = None
        self._hold = 0
        self._n = -1
        self._buf: Deque[float] = deque(maxlen=self.amp_len)
        self._y1 = NAN      # previous filtered sample
        self._y2 = NAN      # the one before
        self._last_beat = -10 ** 9
        self.beats: List[int] = []     # sample indices of detected beats

    # ------------------------------------------------------------------
    def push(self, x: float) -> Tuple[float, Optional[int]]:
        self._n += 1
        n = self._n
        if self._prev_x is None or abs(x - self._prev_x) > self.jump:
            # first sample, or the magnet just moved: start the filter settled
            # at the new level and wait for it to be trustworthy again
            self._zi = self._zi0 * x
            self._hold = self.settle
            self._buf.clear()
            self._y1 = self._y2 = NAN
        self._prev_x = x

        y_arr, self._zi = signal.sosfilt(self.sos, [x], zi=self._zi)
        y = float(y_arr[0])
        if self._hold > 0:
            self._hold -= 1
            return NAN, None

        beat: Optional[int] = None
        self._buf.append(y)
        if (len(self._buf) >= int(2 * self.fs)
                and not math.isnan(self._y1) and not math.isnan(self._y2)):
            sigma = float(np.std(self._buf))
            peak_idx = n - 1
            if (self._y1 > self._y2 and self._y1 >= y
                    and self._y1 > max(self.thresh_sigma * sigma, self.min_peak)
                    and peak_idx - self._last_beat >= self.refractory):
                beat = peak_idx
                self._last_beat = peak_idx
                self.beats.append(peak_idx)
        self._y2, self._y1 = self._y1, y
        return y, beat

    # ------------------------------------------------------------------
    def rhythm(self, n_intervals: int = 10) -> Optional[Dict[str, float]]:
        """BPM and a regularity verdict from the last few beats, or None if
        there are not enough beats yet."""
        if len(self.beats) < 5:
            return None
        b = np.asarray(self.beats[-(n_intervals + 1):], dtype=float)
        ibi = np.diff(b) / self.fs
        mean = float(ibi.mean())
        cv = float(ibi.std() / mean) if mean > 0 else 1.0
        bpm = 60.0 / float(np.median(ibi))
        plausible = 40.0 <= bpm <= 200.0
        if cv < 0.12 and plausible:
            quality = "good"
        elif cv < 0.25 and plausible:
            quality = "fair"
        else:
            quality = "poor"
        return {"bpm": bpm, "cv": cv, "n": len(ibi), "quality": quality}


class RateTracker:
    """Heart-rate estimate from the repeating rhythm over a sliding window.

    Beat-by-beat peak picking fails on this sensor: each beat is a double hump
    (main pulse + a smaller second bump) and the whole thing is only a few
    hundred microvolts, so single peaks are easily mis-counted. The spectrum of
    a 15 s window is much more robust: a real pulse shows up as a sharp line at
    f0 with a second line at 2*f0. Once per second the best (f0, 2*f0) pair is
    chosen. The result is called LOCKED only when the estimate has stayed put
    (within `tol_bpm`) for `lock_s` seconds, which random wobble does not do.

    Any step bigger than `jump` (magnet moved) restarts the window, so a
    window never contains a magnet move.
    """

    def __init__(self, fs: float = 50.0, win_s: float = 15.0, jump: float = 0.010,
                 lo_bpm: float = 49.0, hi_bpm: float = 150.0,
                 lock_s: int = 8, tol_bpm: float = 6.0, min_score: float = 25.0,
                 min_rms: float = 0.0, max_motion: float = 0.0) -> None:
        self.fs, self.win = fs, int(win_s * fs)
        self.jump, self.lo, self.hi = jump, lo_bpm / 60.0, hi_bpm / 60.0
        self.lock_s, self.tol, self.min_score = lock_s, tol_bpm, min_score
        self.min_rms = min_rms       # volts: windows quieter than this cannot be a pulse
        self.max_motion = max_motion # volts: 0 = no limit; else too much movement => no reading
        self.motion = NAN            # volts: wobble of the raw signal over the last 5 s
        self._sos = signal.butter(2, [0.6, 4.0], "bandpass", fs=fs, output="sos")
        self._nfft = 1 << 14
        self._f = np.fft.rfftfreq(self._nfft, 1.0 / fs)
        self.reset()

    def reset(self) -> None:
        self._buf: List[float] = []
        self._mot: Deque[float] = deque(maxlen=int(5 * self.fs))   # last 5 s, never cut by the gate
        self._gated = False
        self._prev: Optional[float] = None
        self._n = 0
        self.history: List[Tuple[float, float]] = []   # (bpm, score) per second
        self.bpm = NAN
        self.score = 0.0
        self.locked = False
        self.lock_age = 0          # seconds the current lock has held
        self.status = "waiting for a steady signal"

    def push(self, x: float) -> None:
        self._n += 1
        if self._prev is not None and abs(x - self._prev) > self.jump:
            self._buf.clear()
            self.history.clear()
            self.locked, self.bpm, self.lock_age = False, NAN, 0
            self.status = "magnet moved - settling"
            self._mot.clear()
        self._prev = x
        self._buf.append(x)
        self._mot.append(x)
        if self._n % 25 == 0 and len(self._mot) == self._mot.maxlen:
            seg = np.asarray(self._mot)
            tt = np.arange(seg.size)
            res = seg - np.polyval(np.polyfit(tt, seg, 1), tt)   # straight-line drift removed
            self.motion = float(np.ptp(res))
            self._gated = self.max_motion > 0 and self.motion > self.max_motion
        if self._gated:                     # too much movement: do not read a pulse
            self._buf.clear()
            self.history.clear()
            self.locked, self.bpm, self.lock_age = False, NAN, 0
            self.status = "too much movement - hold still (%.1f mV in 5 s)" % (1e3 * self.motion)
            return
        if len(self._buf) > self.win:
            del self._buf[0]
        if self._n % int(self.fs) == 0 and len(self._buf) == self.win:
            self._estimate()
        elif len(self._buf) < self.win and self._n % int(self.fs) == 0 and not self.history:
            left = (self.win - len(self._buf)) / self.fs
            if self.status != "magnet moved - settling" or left < 1:
                self.status = "collecting %.0f s of steady signal" % left
            else:
                self.status = "magnet moved - settling (%.0f s)" % left

    def _estimate(self) -> None:
        fs, f = self.fs, self._f
        y = signal.sosfiltfilt(self._sos, np.asarray(self._buf))
        y = y[int(1.5 * fs):-int(1.0 * fs)]
        P = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y)), self._nfft)) ** 2
        base = float(np.median(P[(f >= 0.6) & (f <= 4.0)])) or 1e-30
        df = f[1]; w = max(int(0.08 / df), 1)
        best_sc, best_f = -1.0, NAN
        for f0 in f[(f >= self.lo) & (f <= self.hi)][::4]:
            i1, i2 = int(round(f0 / df)), int(round(2 * f0 / df))
            sc = (P[i1 - w:i1 + w].max() + P[i2 - w:i2 + w].max()) / base
            if sc > best_sc:
                best_sc, best_f = sc, f0
        if float(np.std(y)) < self.min_rms:     # too small to be a pulse, whatever the spectrum says
            best_sc = 0.0
        if best_f < self.lo + 0.08:      # pinned to the bottom edge = slow wobble, not a pulse
            best_sc = 0.0
        self.history.append((60.0 * best_f, best_sc))
        self.history = self.history[-60:]
        recent = self.history[-self.lock_s:]
        steady = (len(recent) >= self.lock_s
                  and all(s >= self.min_score for _, s in recent)
                  and max(b for b, _ in recent) - min(b for b, _ in recent) <= self.tol)
        if steady:
            self.locked = True
            self.lock_age += 1
            self.bpm = float(np.median([b for b, _ in recent]))
            self.score = float(np.median([s for _, s in recent]))
            self.status = "locked"
        else:
            self.locked, self.lock_age = False, 0
            self.bpm, self.score = NAN, best_sc
            self.status = "no steady rhythm yet (best guess %.0f, drifting)" % (60 * best_f)


class PulseFollower:
    """Clean display trace that follows the locked heart rate.

    A bank of narrow band-pass filters (one every 0.05 Hz from 0.8 to 2.5 Hz,
    each about +-0.3 Hz wide) runs all the time. Once RateTracker is locked,
    the filter centred on the heart rate is shown. That removes hand tremor
    (8-12 Hz), slow drift and the double-hump shape, leaving one smooth swing
    per beat, and the beat markers are simply its peaks. Everything is causal.
    """

    def __init__(self, fs: float = 50.0, f_lo: float = 0.8, f_hi: float = 2.5,
                 step: float = 0.05, half_bw: float = 0.3, jump: float = 0.010,
                 settle_s: float = 2.0, min_peak: float = 0.0) -> None:
        self.fs, self.jump = fs, jump
        self.min_peak = min_peak
        self.settle = int(round(settle_s * fs))
        self.centers = np.arange(f_lo, f_hi + 1e-9, step)
        sos = [signal.butter(2, [max(c - half_bw, 0.3), c + half_bw], "bandpass",
                             fs=fs, output="sos") for c in self.centers]
        self.sos = np.asarray(sos)                         # (K, 2, 6)
        self._zi0 = np.asarray([signal.sosfilt_zi(s) for s in sos])   # (K, 2, 2)
        self.reset()

    def reset(self) -> None:
        self._z = np.zeros_like(self._zi0)
        self._prev: Optional[float] = None
        self._hold = 0
        self._n = -1
        self._idx: Optional[int] = None
        self._y1 = self._y2 = NAN
        self._last_beat = -10 ** 9
        self._buf: Deque[float] = deque(maxlen=int(3 * self.fs))
        self.beats: List[int] = []

    def push(self, x: float, bpm: float, locked: bool) -> Tuple[float, Optional[int]]:
        self._n += 1
        if self._prev is None or abs(x - self._prev) > self.jump:
            self._z = self._zi0 * x
            self._hold = self.settle
            self._y1 = self._y2 = NAN
            self._buf.clear()
        self._prev = x
        v = x
        for k in range(2):                       # two biquads, all K filters at once
            b0, b1, b2 = self.sos[:, k, 0], self.sos[:, k, 1], self.sos[:, k, 2]
            a1, a2 = self.sos[:, k, 4], self.sos[:, k, 5]
            out = b0 * v + self._z[:, k, 0]
            self._z[:, k, 0] = b1 * v - a1 * out + self._z[:, k, 1]
            self._z[:, k, 1] = b2 * v - a2 * out
            v = out
        if self._hold > 0:
            self._hold -= 1
        if not locked or math.isnan(bpm) or self._hold > 0:
            self._idx = None
            self._y1 = self._y2 = NAN
            return NAN, None
        target = int(np.argmin(np.abs(self.centers - bpm / 60.0)))
        if self._idx is None or abs(self.centers[self._idx] - self.centers[target]) > 0.1:
            self._idx = target                   # hysteresis: do not hop between filters
        y = float(v[self._idx])
        self._buf.append(y)
        beat: Optional[int] = None
        if len(self._buf) >= int(1.5 * self.fs) and not math.isnan(self._y2):
            period = self.fs * 60.0 / bpm
            if (self._y1 > self._y2 and self._y1 >= y and self._y1 > max(0.3 * float(np.std(self._buf)), self.min_peak)
                    and (self._n - 1) - self._last_beat >= 0.65 * period):
                beat = self._n - 1
                self._last_beat = beat
                self.beats.append(beat)
        self._y2, self._y1 = self._y1, y
        return y, beat


def average_beat(y, beats, fs: float = 50.0, pre: float = 0.35, post: float = 0.85,
                 max_beats: int = 20, max_shift: float = 0.2):
    """The 'general pattern': cut the pulse-band signal around each recent beat,
    line the pieces up with each other and average them. Random noise averages
    away (about sqrt(N) better), the repeating pulse shape does not.
    Returns (time_axis_s, mean_shape, [aligned pieces]) or None (<4 clean beats)."""
    y = np.asarray(y, dtype=float)
    n_pre, n_post, m = int(pre * fs), int(post * fs), int(max_shift * fs)
    L = n_pre + n_post + 1
    segs = []
    for b in list(beats)[-max_beats:]:
        a, e = b - n_pre - m, b + n_post + m + 1
        if a < 0 or e > len(y):
            continue
        s = y[a:e]
        if np.isnan(s).any():
            continue
        segs.append(s)
    if len(segs) < 4:
        return None
    tmpl = np.mean([s[m:m + L] - s[m:m + L].mean() for s in segs], axis=0)
    aligned = []
    for _ in range(2):                     # align every piece to the average, twice
        aligned = []
        for s in segs:
            best_sc, best_c = None, None
            for sh in range(-m, m + 1):
                c = s[m + sh:m + sh + L]
                c = c - c.mean()
                sc = float(np.dot(c, tmpl))
                if best_sc is None or sc > best_sc:
                    best_sc, best_c = sc, c
            aligned.append(best_c)
        tmpl = np.mean(aligned, axis=0)
    return (np.arange(L) - n_pre) / fs, tmpl, aligned


def reconstruct_pulse(x, fs: float = 50.0, f0: float = 1.1, harmonics: int = 3,
                      poly: int = 10, refine: bool = True):
    """Delayed, model-based clean-up of the pulse.

    Fits   drift (a polynomial)  +  a pulse that repeats at f0 (sine/cosine at
    f0, 2*f0, 3*f0 with free amplitude and phase)   to the window `x` by least
    squares, and returns only the repeating part. Anything that does not repeat
    at the heart rate (drift, tremor, noise, a one-off bump) is left out.
    Because it sees the window on both sides of every point it has no phase lag,
    but the newest second can only be shown after the window has moved on.

    Returns dict(harm=array like x, r2=share of the 0.6-6 Hz wobble that the
    repeating pulse explains (0..1), f0=fitted rate in Hz, pp=peak-to-peak) or None.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < int(4 * fs) or not np.isfinite(f0):
        return None
    t = (np.arange(n) - n / 2.0) / fs
    tn = t / (n / 2.0 / fs)
    P = np.column_stack([tn ** k for k in range(poly + 1)])

    def fit(f):
        cols = []
        for h in range(1, harmonics + 1):
            cols += [np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)]
        H = np.column_stack(cols)
        A = np.column_stack([P, H])
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        res = x - A @ coef
        return float(res @ res), H @ coef[P.shape[1]:]

    best_f, (best_sse, best_h) = f0, fit(f0)
    if refine:
        for f in np.arange(f0 - 0.06, f0 + 0.0601, 0.01):
            if f <= 0.5:
                continue
            sse, h = fit(f)
            if sse < best_sse:
                best_f, best_sse, best_h = float(f), sse, h
    sos = signal.butter(2, [0.6, 6.0], "bandpass", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, x - np.polyval(np.polyfit(tn, x, poly), tn))
    c = slice(n // 4, 3 * n // 4)                      # judge the middle half only
    sst = float(np.var(xb[c]))
    r2 = float(min(1.0, np.var(best_h[c]) / sst)) if sst > 0 else 0.0
    return {"harm": best_h, "r2": r2, "f0": best_f, "pp": float(np.ptp(best_h[c]))}


def process_all(x: np.ndarray, **kw):
    """Run a whole array through a fresh processor (offline / tests).
    Returns (filtered array with NaN while settling, list of beat indices)."""
    p = PulseProcessor(**kw)
    y = np.empty(len(x))
    for i, v in enumerate(x):
        y[i], _ = p.push(float(v))
    return y, list(p.beats), p
