"""Synthesis / enhancement helpers for VPC25

This module focuses on *post*-enhancement (quality smoothing) for an
anonymized waveform. It purposely avoids training-time code and aims to be
optional in the pipeline. If not enabled, functions return the input.

Design
------
- HiFi-GAN (optional): load a **TorchScript** generator checkpoint and run it
  on a log-Mel spectrogram to synthesize/enhance a waveform. This assumes your
  checkpoint was exported to TorchScript and expects Mel features with the
  parameters you provide.
- Lightweight denoise (optional): applies a Wiener filter as a last-mile
  clean-up when HiFi-GAN is not available.

Notes
-----
* Because models differ (sample rate, Mel config), we expose mel parameters so
  you can match them to your checkpoint. If they mismatch, skip HiFi-GAN.
* If you only need a no-op enhancer, just call `apply_synthesis_enhance` with
  a config where `enabled: false` (it will return the input unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import librosa
import scipy.signal as sps

from utils.dsp import resample

__all__ = [
    "MelParams",
    "compute_mel",
    "HiFiGANEnhancer",
    "enhance_with_hifigan",
    "simple_denoise",
    "apply_synthesis_enhance",
]


# -----------------------------------------------------------------------------
# Mel spectrogram helpers
# -----------------------------------------------------------------------------

@dataclass
class MelParams:
    sr: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = None  # None -> sr/2
    power: float = 1.0            # 1.0 = magnitude; 2.0 = power
    ref_level_db: float = 20.0    # used for log scaling
    min_level_db: float = -100.0
    eps: float = 1e-5


def compute_mel(y: np.ndarray, params: MelParams) -> np.ndarray:
    """Compute a (log) Mel spectrogram expected by many neural vocoders.

    Returns
    -------
    mel : np.ndarray, shape (n_mels, T), float32
    """
    y = np.asarray(y, dtype=np.float32)
    if params.fmax is None:
        fmax = params.sr / 2.0
    else:
        fmax = float(params.fmax)

    # STFT
    S = librosa.stft(
        y,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        win_length=params.win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    S_mag = np.abs(S) ** params.power  # magnitude or power

    # Mel filter bank
    mel_fb = librosa.filters.mel(
        sr=params.sr,
        n_fft=params.n_fft,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=fmax,
        htk=True,
        norm=None,
    )
    mel = mel_fb @ S_mag

    # Log scaling (log10 with floor)
    mel = np.maximum(params.eps, mel)
    mel_db = 20.0 * np.log10(mel)
    mel_db = np.clip(mel_db, params.min_level_db, None)
    # Normalize to ~[0, 1] range (optional depending on your checkpoint)
    mel_norm = (mel_db - params.min_level_db) / (params.ref_level_db - params.min_level_db)
    return mel_norm.astype(np.float32)


# -----------------------------------------------------------------------------
# HiFi-GAN TorchScript enhancer
# -----------------------------------------------------------------------------

class HiFiGANEnhancer:
    """Wrapper around a TorchScript HiFi-GAN generator checkpoint.

    Expect the checkpoint to take a Mel tensor of shape [1, n_mels, T] and
    return a waveform [1, T*hop] (or similar). Exact behavior depends on how
    the model was exported.
    """

    def __init__(self, ckpt_path: str | Path, device: str = "cuda") -> None:
        self.ckpt_path = Path(ckpt_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"HiFi-GAN TorchScript checkpoint not found: {self.ckpt_path}")
        self.model = torch.jit.load(str(self.ckpt_path), map_location=self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """Run inference: mel (n_mels, T) -> waveform (T',) in float32 [-1, 1]."""
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(self.device)  # [1, n_mels, T]
        wav_t = self.model(mel_t)
        wav = wav_t.squeeze().detach().cpu().numpy().astype(np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        return wav


def enhance_with_hifigan(
    y: np.ndarray,
    sr: int,
    enhancer: HiFiGANEnhancer,
    mel_params: MelParams,
    model_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Enhance `y` using a TorchScript HiFi-GAN enhancer.

    If `model_sr` is provided and differs from `sr`, the signal is resampled to
    `model_sr` before Mel computation and re-resampled back to `sr` after.
    """
    y_in = y
    sr_in = sr

    if model_sr is None:
        model_sr = mel_params.sr

    if sr_in != model_sr:
        y_proc = resample(y_in, sr_in, model_sr)
        sr_proc = model_sr
    else:
        y_proc = y_in
        sr_proc = sr_in

    mel = compute_mel(y_proc, MelParams(**{**mel_params.__dict__, "sr": sr_proc}))
    wav_hat = enhancer(mel)  # returns model_sr audio

    if sr_proc != sr_in:
        wav_hat = resample(wav_hat, sr_proc, sr_in)
        sr_out = sr_in
    else:
        sr_out = sr_proc

    return wav_hat.astype(np.float32), int(sr_out)


# -----------------------------------------------------------------------------
# Lightweight denoise (fallback)
# -----------------------------------------------------------------------------

def simple_denoise(y: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply a simple Wiener filter for mild denoising.

    This is *not* a speech enhancer; it just reduces small-scale fizz.
    """
    y = np.asarray(y, dtype=np.float32)
    if kernel_size < 3:
        return y
    # Wiener expects 1D; we clamp for safety
    z = sps.wiener(y, mysize=kernel_size)
    z = np.clip(z, -1.0, 1.0)
    return z.astype(np.float32)


# -----------------------------------------------------------------------------
# Top-level convenience
# -----------------------------------------------------------------------------

def apply_synthesis_enhance(
    y: np.ndarray,
    sr: int,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """Apply optional post-enhancement according to a `synthesis` config block.

    Expected config keys (see parameters/config.yaml):
      enabled: bool
      backend: "hifigan"|"nsf"|"none"  (only "hifigan" is supported here)
      enhance_only: bool
      local_ckpt: str
      device: "cuda"|"cpu"|"auto"
      denoise: bool

      # optional mel params (tune to your checkpoint)
      mel: {
        sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, power,
        ref_level_db, min_level_db
      }
    """
    if not cfg or not cfg.get("enabled", False):
        return y, sr

    backend = str(cfg.get("backend", "none")).lower()
    if backend not in {"hifigan"}:
        # Unsupported backend here -> return input unchanged
        return y, sr

    # Load mel params if provided
    mel_cfg = cfg.get("mel", {})
    mel_params = MelParams(**{k: mel_cfg.get(k, getattr(MelParams, k) if hasattr(MelParams, k) else getattr(MelParams(), k)) for k in MelParams().__dict__.keys()})

    # Load enhancer
    ckpt = cfg.get("local_ckpt")
    device = cfg.get("device", "cuda")
    if not ckpt or not Path(ckpt).exists():
        # If no checkpoint, optionally run simple denoise as a tiny clean-up
        if cfg.get("denoise", False):
            return simple_denoise(y), sr
        return y, sr

    enhancer = HiFiGANEnhancer(ckpt_path=ckpt, device=device)
    wav_hat, sr_out = enhance_with_hifigan(y, sr, enhancer, mel_params, model_sr=mel_params.sr)

    if cfg.get("denoise", False):
        wav_hat = simple_denoise(wav_hat)

    return wav_hat, sr_out
