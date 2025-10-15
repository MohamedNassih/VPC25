"""Post-processing helpers for VPC25

This module provides light, safe post-processing blocks to run after
anonymization, focused on signal hygiene rather than heavy enhancement.

Functions
---------
- dc_block:     remove DC offset (mean) or apply a gentle high-pass
- vad_trim:     energy-based VAD trimming with padding (or librosa-based)
- soft_limiter: peak-safe soft limiter using tanh waveshaping
- fade_edges:   short fade-in/out to avoid clicks
- apply_postproc: one-stop API driven by a small config dict

Design notes
------------
* All routines operate on mono float32 arrays in [-1, 1].
* LUFS/RMS normalization should be handled by utils.dsp.normalize_rms; this
  file focuses on DC/trim/limiting/fades.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import scipy.signal as sps
import librosa

_EPS = 1e-12

# -----------------------------------------------------------------------------
# DC removal / high-pass
# -----------------------------------------------------------------------------

def dc_block(y: np.ndarray, sr: int, method: str = "mean", fc: float = 30.0) -> np.ndarray:
    """Remove DC offset.

    Parameters
    ----------
    method: 'mean' (simple mean subtraction) or 'highpass' (2nd-order Butter).
    fc:     cutoff for 'highpass' in Hz (default 30 Hz).
    """
    y = np.asarray(y, dtype=np.float32)
    if method == "mean":
        return (y - float(np.mean(y))).astype(np.float32)
    # High-pass 2nd-order Butterworth with filtfilt for zero-phase
    fc = float(fc)
    if fc <= 0 or fc >= sr / 2:
        return (y - float(np.mean(y))).astype(np.float32)
    b, a = sps.butter(2, fc / (0.5 * sr), btype="highpass")
    z = sps.filtfilt(b, a, y).astype(np.float32)
    return z


# -----------------------------------------------------------------------------
# VAD-based trimming (energy threshold) / librosa trim
# -----------------------------------------------------------------------------

def _frame_energy_db(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = max(1, 1 + (len(y) - frame) // hop)
    E = np.empty(n, dtype=np.float32)
    w = np.hanning(frame).astype(np.float32)
    for i in range(n):
        s = i * hop
        seg = y[s : s + frame]
        if seg.size < frame:
            seg = np.pad(seg, (0, frame - seg.size))
        E[i] = 20.0 * np.log10(np.sqrt(np.mean((seg * w) ** 2)) + _EPS)
    return E


def vad_trim(
    y: np.ndarray,
    sr: int,
    method: str = "energy",
    top_db: float = 40.0,
    energy_db_threshold: float = -45.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    pad_ms: float = 100.0,
) -> np.ndarray:
    """Trim leading/trailing silence.

    - method="librosa": uses librosa.effects.trim(top_db)
    - method="energy":  energy-based threshold with padding on both sides
    """
    y = np.asarray(y, dtype=np.float32)
    if method.lower() == "librosa":
        yt, idx = librosa.effects.trim(y, top_db=float(top_db))
        pad = int(round((pad_ms / 1000.0) * sr))
        start = max(0, idx[0] - pad)
        end = min(len(y), idx[1] + pad)
        return y[start:end]

    # Energy-based VAD
    frame = int(round(frame_ms / 1000.0 * sr))
    hop = int(round(hop_ms / 1000.0 * sr))
    frame = max(64, frame)
    hop = max(16, hop)

    Edb = _frame_energy_db(y, frame, hop)
    mask = Edb > float(energy_db_threshold)

    # Find first/last True; add padding
    if not np.any(mask):
        return y  # nothing to trim
    idx = np.where(mask)[0]
    first, last = int(idx[0]), int(idx[-1])

    pad_frames = int(round((pad_ms / 1000.0) * sr / hop))
    first = max(0, first - pad_frames)
    last = min(len(Edb) - 1, last + pad_frames)

    start = first * hop
    end = min(len(y), last * hop + frame)
    return y[start:end]


# -----------------------------------------------------------------------------
# Limiter / soft clipping
# -----------------------------------------------------------------------------

def soft_limiter(y: np.ndarray, ceiling: float = 0.98, strength: float = 10.0) -> np.ndarray:
    """Soft limiter using tanh waveshaping.

    - 'strength' controls curvature (higher -> more compression near 0 dBFS)
    - Output is scaled so that the absolute peak is <= 'ceiling'
    """
    y = np.asarray(y, dtype=np.float32)
    # Waveshaping in [-1, 1]
    z = np.tanh(strength * y) / np.tanh(strength)
    peak = float(np.max(np.abs(z))) + _EPS
    if peak > ceiling:
        z = z * (ceiling / peak)
    return z.astype(np.float32)


# -----------------------------------------------------------------------------
# Fades
# -----------------------------------------------------------------------------

def fade_edges(y: np.ndarray, sr: int, in_ms: float = 5.0, out_ms: float = 5.0) -> np.ndarray:
    """Apply short linear fades at start/end to avoid clicks."""
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n == 0:
        return y
    fi = int(round(in_ms / 1000.0 * sr))
    fo = int(round(out_ms / 1000.0 * sr))
    fi = max(0, min(fi, n // 2))
    fo = max(0, min(fo, n // 2))

    out = y.copy()
    if fi > 0:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        out[n - fo :] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


# -----------------------------------------------------------------------------
# Top-level one-stop
# -----------------------------------------------------------------------------

def _merge_cfg(defaults: Dict[str, Any], user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not user:
        return defaults
    out = dict(defaults)
    for k, v in user.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def apply_postproc(y: np.ndarray, sr: int, cfg: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, int]:
    """Apply DC removal, optional trimming, fade, and soft limiter.

    Example cfg (all optional):
    {
      "dc_block": {"enabled": true, "method": "mean", "fc": 30.0},
      "trim": {"enabled": false, "method": "librosa", "top_db": 40.0, "pad_ms": 100.0,
                "energy_db_threshold": -45.0, "frame_ms": 20.0, "hop_ms": 10.0},
      "fade": {"enabled": false, "in_ms": 5.0, "out_ms": 5.0},
      "limiter": {"enabled": true, "ceiling": 0.98, "strength": 10.0}
    }
    """
    defaults = {
        "dc_block": {"enabled": True, "method": "mean", "fc": 30.0},
        "trim": {
            "enabled": False,
            "method": "librosa",
            "top_db": 40.0,
            "pad_ms": 100.0,
            "energy_db_threshold": -45.0,
            "frame_ms": 20.0,
            "hop_ms": 10.0,
        },
        "fade": {"enabled": False, "in_ms": 5.0, "out_ms": 5.0},
        "limiter": {"enabled": True, "ceiling": 0.98, "strength": 10.0},
    }
    cfg = _merge_cfg(defaults, cfg)

    # DC removal
    if cfg["dc_block"]["enabled"]:
        y = dc_block(y, sr, method=str(cfg["dc_block"]["method"]).lower(), fc=float(cfg["dc_block"]["fc"]))

    # Trimming (optional)
    if cfg["trim"]["enabled"]:
        method = str(cfg["trim"]["method"]).lower()
        if method == "librosa":
            y = vad_trim(
                y,
                sr,
                method="librosa",
                top_db=float(cfg["trim"]["top_db"]),
                pad_ms=float(cfg["trim"]["pad_ms"]),
            )
        else:
            y = vad_trim(
                y,
                sr,
                method="energy",
                energy_db_threshold=float(cfg["trim"]["energy_db_threshold"]),
                frame_ms=float(cfg["trim"]["frame_ms"]),
                hop_ms=float(cfg["trim"]["hop_ms"]),
                pad_ms=float(cfg["trim"]["pad_ms"]),
            )

    # Fade edges (optional)
    if cfg["fade"]["enabled"]:
        y = fade_edges(
            y,
            sr,
            in_ms=float(cfg["fade"]["in_ms"]),
            out_ms=float(cfg["fade"]["out_ms"]),
        )

    # Soft limiter
    if cfg["limiter"]["enabled"]:
        y = soft_limiter(y, ceiling=float(cfg["limiter"]["ceiling"]), strength=float(cfg["limiter"]["strength"]))

    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y, int(sr)
