"""DSP helpers for VPC25

This module provides:
 - load_wav / save_wav  : robust audio I/O with resampling and mono handling
 - normalize_rms        : simple loudness normalization (dBFS target)
 - mcadams_warp         : lightweight speaker anonymization via LPC formant warping
 - micro_pitch          : small pitch shift (±1–2 semitones) as a post-process

Notes
-----
* McAdams warping is implemented frame-wise with LPC analysis–synthesis.
  It is purposely conservative (stable filters, Hann OLA).
* Loudness normalization is RMS-based (approx. LUFS). True LUFS would
  require a gating + K-weighting pipeline; RMS is used for speed/simplicity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as sps

try:  # High-quality resampling
    import soxr  # type: ignore
    _HAS_SOXR = True
except Exception:  # pragma: no cover
    _HAS_SOXR = False

_EPS = 1e-12

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_wav(
    path: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
    dtype: str = "float32",
) -> Tuple[np.ndarray, int]:
    """Read an audio file.

    Parameters
    ----------
    path : str or Path
        WAV/FLAC/OGG file path.
    target_sr : int or None
        If set, resample to this sampling rate.
    mono : bool
        If True, convert to mono by averaging channels.
    dtype : str
        Numpy dtype for returned audio (default float32).

    Returns
    -------
    y : np.ndarray, shape (n,)
    sr : int
    """
    y, sr = sf.read(str(path), always_2d=True)
    y = y.astype(np.float32, copy=False)
    if mono:
        y = np.mean(y, axis=1)
    else:
        # keep first channel if multi-channel processing is not implemented
        y = y[:, 0]

    if target_sr is not None and sr != target_sr:
        y = resample(y, sr, target_sr)
        sr = target_sr

    if dtype != "float32":
        y = y.astype(dtype)
    return y, sr


def save_wav(path: str | Path, y: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Write a WAV file with safe dtype conversion.

    Parameters
    ----------
    path : str or Path
    y : array, mono signal in [-1, 1]
    sr : sampling rate
    subtype : str
        e.g., 'PCM_16' (default), 'FLOAT'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sr, subtype=subtype)


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality resampling using soxr if available, else librosa.
    """
    if orig_sr == target_sr:
        return y
    if _HAS_SOXR:
        return soxr.resample(y, orig_sr, target_sr, quality="HQ")
    # fall back to librosa (uses resampy under the hood)
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best")


# -----------------------------------------------------------------------------
# Loudness / normalization
# -----------------------------------------------------------------------------

def rms_dbfs(y: np.ndarray) -> float:
    """Return RMS level in dBFS (full-scale is 1.0)."""
    rms = float(np.sqrt(np.maximum(_EPS, np.mean(y ** 2))))
    return 20.0 * np.log10(rms + _EPS)


def normalize_rms(y: np.ndarray, target_dbfs: float = -23.0, clip_limit: float = 0.99) -> np.ndarray:
    """Simple RMS normalization towards target dBFS.

    Parameters
    ----------
    y : array-like, mono in [-1, 1]
    target_dbfs : float
        Target loudness in dBFS; -23 is a common broadcast-like target.
    clip_limit : float
        Safety ceiling after normalization.
    """
    y = np.asarray(y, dtype=np.float32)
    cur = rms_dbfs(y)
    gain_db = target_dbfs - cur
    gain = 10 ** (gain_db / 20.0)
    out = y * gain
    peak = float(np.max(np.abs(out)))
    if peak > clip_limit:
        out *= (clip_limit / (peak + _EPS))
    return out


# -----------------------------------------------------------------------------
# McAdams formant warping (frame-wise LPC analysis–synthesis)
# -----------------------------------------------------------------------------

def mcadams_warp(
    y: np.ndarray,
    sr: int,
    alpha: float = 0.80,
    lpc_order: int | str = "auto",
    frame_length: float = 0.030,
    hop_length: float = 0.015,
    pre_emph: float = 0.0,
) -> np.ndarray:
    """Apply McAdams formant warping.

    Implementation based on LPC analysis–synthesis per frame. For each frame,
    convert LPC polynomial roots (poles) r = |r| * exp(j*theta) into
    r' = |r| * exp(j*sign(theta) * |theta|**alpha). Then re-synthesize using
    the warped all-pole filter.

    Parameters
    ----------
    y : np.ndarray (mono)
    sr : int
    alpha : float
        McAdams exponent in (0, 1] typically; lower compresses angles more.
    lpc_order : int or 'auto'
    frame_length : float (seconds)
    hop_length : float (seconds)
    pre_emph : float
        Optional pre-emphasis coefficient (e.g., 0.97). Set 0 to disable.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("mcadams_warp expects mono signal (1D array)")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y)

    x = y.copy()
    if pre_emph and 0.0 < pre_emph < 1.0:
        x = sps.lfilter([1.0, -pre_emph], [1.0], x)

    L = int(max(32, round(frame_length * sr)))
    H = int(max(16, round(hop_length * sr)))
    win = np.hanning(L).astype(np.float32)
    n_frames = 1 + int(np.floor(max(0, len(x) - L) / H))

    # Output buffers with OLA normalization
    out = np.zeros_like(x, dtype=np.float32)
    acc_win = np.zeros_like(x, dtype=np.float32)

    # Determine LPC order
    if lpc_order == "auto":
        # A common heuristic: order ≈ 2 + sr(kHz) * 2 (capped for stability)
        order = int(min(24, max(8, 2 + (sr // 1000) * 2)))
    else:
        order = int(lpc_order)

    for i in range(n_frames):
        start = i * H
        frame = x[start : start + L]
        if len(frame) < L:
            frame = np.pad(frame, (0, L - len(frame)))
        fwin = frame * win

        # Skip silent frames
        if np.max(np.abs(fwin)) < 1e-5:
            out[start : start + L] += fwin
            acc_win[start : start + L] += win
            continue

        # LPC analysis (autocorr + Levinson-Durbin)
        a = _lpc_coeffs(fwin, order)  # includes a0 = 1
        if not np.all(np.isfinite(a)):
            a = np.r_[1.0, np.zeros(order, dtype=np.float32)]

        # Residual (prediction error)
        e = sps.lfilter(a, [1.0], fwin)

        # Warp LPC poles
        roots = np.roots(a)
        # Keep roots inside unit circle for stability
        roots = _stabilize_roots(roots)
        # Angle warp
        r_mag = np.abs(roots)
        r_ang = np.angle(roots)
        r_ang_w = np.sign(r_ang) * (np.abs(r_ang) ** float(alpha))
        roots_w = r_mag * np.exp(1j * r_ang_w)
        roots_w = _stabilize_roots(roots_w)
        a_w_c = np.poly(roots_w).astype(np.complex128)
        a_w   = np.real_if_close(a_w_c, tol=1e5).real.astype(np.float32)

        if not np.isfinite(a_w).all() or a_w.size != a.size:
            a_w = a

        # Synthesis from residual through warped all-pole filter
        yhat = sps.lfilter([1.0], a_w, e)

        # Energy match (avoid audible level jumps)
        e_in = np.sqrt(np.maximum(_EPS, np.mean(fwin ** 2)))
        e_out = np.sqrt(np.maximum(_EPS, np.mean(yhat ** 2)))
        yhat *= (e_in / e_out)

        # OLA with window
        out[start : start + L] += yhat * win
        acc_win[start : start + L] += win

    acc_win = np.maximum(acc_win, _EPS)
    y_out = out / acc_win

    # De-emphasis to roughly invert pre-emph
    if pre_emph and 0.0 < pre_emph < 1.0:
        y_out = sps.lfilter([1.0], [1.0, -pre_emph], y_out)

    # Safety
    y_out = np.clip(y_out, -1.0, 1.0)
    return y_out.astype(np.float32)


def _stabilize_roots(roots: np.ndarray, radius: float = 0.999) -> np.ndarray:
    """Project any roots outside unit circle back inside for stability."""
    r = np.asarray(roots)
    mag = np.abs(r)
    idx = mag >= radius
    if np.any(idx):
        r[idx] = (radius / (mag[idx] + _EPS)) * r[idx]
    return r


def _lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients (a0..ap) via autocorrelation + Levinson-Durbin.

    Returns
    -------
    a : np.ndarray, shape (order+1,)
        LPC polynomial coefficients with a[0] == 1.
    """
    x = np.asarray(x, dtype=np.float32)
    if order < 1:
        return np.array([1.0], dtype=np.float32)
    # Autocorrelation up to lag 'order'
    r = _autocorr(x, order)
    a, _err = _levinson(r, order)
    return a.astype(np.float32)


def _autocorr(x: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    r = np.zeros(order + 1, dtype=np.float64)
    for k in range(order + 1):
        r[k] = np.dot(x[: n - k], x[k:])
    return r


def _levinson(r: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    """Levinson-Durbin recursion.

    Parameters
    ----------
    r : autocorrelation, shape (order+1,)
    order : LPC order

    Returns
    -------
    a : LPC polynomial coeffs [1, a1..ap]
    e : final prediction error
    """
    if r[0] <= 0:
        return np.r_[1.0, np.zeros(order)], float(r[0])

    a = np.zeros(order + 1, dtype=np.float64)
    e = float(r[0])
    a[0] = 1.0

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / (e + _EPS)
        # update
        a_prev = a.copy()
        a[1:i] = a_prev[1:i] + k * a_prev[i - 1 : 0 : -1]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= _EPS:
            # early stop, pad zeros
            a[i + 1 :] = 0.0
            break
    return a.astype(np.float64), float(e)


# -----------------------------------------------------------------------------
# Small pitch shift (post-process)
# -----------------------------------------------------------------------------

def micro_pitch(y: np.ndarray, sr: int, semitones: float = 1.0) -> np.ndarray:
    """Apply a small pitch shift using librosa.effects.pitch_shift.

    Keep |semitones| <= ~2 to avoid harming intelligibility.
    """
    if abs(semitones) < 1e-6:
        return np.asarray(y, dtype=np.float32)
    # librosa handles time-stretch internally; we only shift pitch
    y_shift = librosa.effects.pitch_shift(np.asarray(y, dtype=np.float32), sr=sr, n_steps=semitones)
    return np.clip(y_shift, -1.0, 1.0).astype(np.float32)


# -----------------------------------------------------------------------------
# Convenience pipeline helpers
# -----------------------------------------------------------------------------

def enforce_mono_sr(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Ensure mono float32 at target sampling rate."""
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    y = y.astype(np.float32, copy=False)
    if sr != target_sr:
        y = resample(y, sr, target_sr)
        sr = target_sr
    return y, sr


def prepare_input(
    path: str | Path,
    target_sr: int = 16000,
    normalize: bool = True,
    target_dbfs: float = -23.0,
) -> Tuple[np.ndarray, int]:
    """Load → enforce mono+SR → optional RMS normalization.
    Useful as a single entry-point in model.py.
    """
    y, sr = load_wav(path, target_sr=target_sr, mono=True)
    if normalize:
        y = normalize_rms(y, target_dbfs=target_dbfs)
    return y, sr
