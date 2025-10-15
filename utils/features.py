"""Feature extraction helpers for VPC25

This module provides:
 - ECAPA / x-vector extraction (SpeechBrain)
 - F0 extraction (WORLD/pyworld: Harvest + StoneMask refinement)
 - (Placeholder) BN/PPG extraction hooks

Design notes
------------
* All functions are light-weight wrappers meant to be called by model.py.
* Device handling is done lazily; pass device="auto" to pick CUDA if available.
* Embeddings are L2-normalized to unit length by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

# --- Windows/CPU: shims to avoid symlinks and enforce soundfile backend ---
import os
os.environ.setdefault("SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY", "copy")
os.environ.setdefault("SPEECHBRAIN_LOCAL_FILE_STRATEGY", "copy")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends(): return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass
# -------------------------------------------------------------------------

from speechbrain.utils.fetching import LocalStrategy
# SpeechBrain ECAPA encoder
from speechbrain.pretrained import EncoderClassifier  # type: ignore

# pyworld for F0
try:
    import pyworld as pw  # type: ignore
    _HAS_PYWORLD = True
except Exception:  # pragma: no cover
    _HAS_PYWORLD = False

# Local I/O util (resampling, normalization)
from utils.dsp import prepare_input

__all__ = [
    "EcapaEncoder",
    "load_ecapa",
    "extract_xvec",
    "extract_f0",
    "extract_bn",
]


# -----------------------------------------------------------------------------
# Speaker embedding (ECAPA)
# -----------------------------------------------------------------------------

@dataclass
class EcapaEncoder:
    """Lazy wrapper around SpeechBrain ECAPA-TDNN encoder.

    Parameters
    ----------
    device : str
        "auto" | "cuda" | "cpu"
    local_ckpt : Optional[str]
        Path to a local checkpoint (folder with hyperparams.yaml) if any.
    hf_repo : Optional[str]
        Hugging Face repo id, defaults to "speechbrain/spkrec-ecapa-voxceleb".
    savedir : str
        Local cache directory.
    """
    device: str = "auto"
    local_ckpt: Optional[str] = None
    hf_repo: Optional[str] = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = ".cache/ecapa"

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_opts = {"device": self.device}
        ckpt = self.local_ckpt
        if ckpt and Path(ckpt).exists():
            self.encoder = EncoderClassifier.from_hparams(
                source=ckpt, run_opts=run_opts, savedir=self.savedir + "_local", local_strategy=LocalStrategy.COPY
            )
        else:
            repo = self.hf_repo or "speechbrain/spkrec-ecapa-voxceleb"
            self.encoder = EncoderClassifier.from_hparams(
                source=repo, run_opts=run_opts, savedir=self.savedir + "_hf", local_strategy=LocalStrategy.COPY_SKIP_CACHE
            )
        self.encoder.eval()

    @torch.inference_mode()
    def embed(self, wav: np.ndarray, sr: int, l2_norm: bool = True) -> np.ndarray:
        """Extract one ECAPA embedding for a mono waveform (float32, [-1,1])."""
        wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        emb = self.encoder.encode_batch(wav_t)  # [1, 1, D]
        x = emb.squeeze().cpu().numpy().astype(np.float32)
        if l2_norm:
            n = np.linalg.norm(x) + 1e-10
            x = x / n
        return x


def load_ecapa(device: str = "auto", local_ckpt: Optional[str] = None, hf_repo: Optional[str] = None) -> EcapaEncoder:
    """Factory for ECAPA encoder."""
    return EcapaEncoder(device=device, local_ckpt=local_ckpt, hf_repo=hf_repo)


def extract_xvec(
    wav: np.ndarray | str | Path,
    sr: Optional[int] = None,
    encoder: Optional[EcapaEncoder] = None,
    device: str = "auto",
    local_ckpt: Optional[str] = None,
    hf_repo: Optional[str] = None,
    target_sr: int = 16000,
    normalize_input: bool = False,
) -> Tuple[np.ndarray, int]:
    """Convenience wrapper: read (if path) → prep → ECAPA embed.

    Returns (xvec, sr_used)
    """
    if isinstance(wav, (str, Path)):
        wav, sr_used = prepare_input(wav, target_sr=target_sr, normalize=normalize_input)
    else:
        assert sr is not None, "When passing raw waveform, provide its sampling rate."
        wav = np.asarray(wav, dtype=np.float32)
        # No resampling/normalization here; caller must ensure proper format
        sr_used = sr

    if encoder is None:
        encoder = load_ecapa(device=device, local_ckpt=local_ckpt, hf_repo=hf_repo)

    xvec = encoder.embed(wav, sr_used)
    return xvec, sr_used


# -----------------------------------------------------------------------------
# Fundamental frequency (F0) via WORLD / pyworld
# -----------------------------------------------------------------------------

def _interp_unvoiced(f0: np.ndarray) -> np.ndarray:
    """Linear interpolation over unvoiced (zeros), edge-padded with nearest value."""
    f0 = np.asarray(f0, dtype=np.float32)
    if f0.size == 0:
        return f0
    idx = np.where(f0 > 0)[0]
    if idx.size == 0:
        return f0  # all unvoiced, keep zeros
    x = np.arange(f0.size)
    f = np.interp(x, idx, f0[idx])
    return f.astype(np.float32)


def extract_f0(
    wav: np.ndarray | str | Path,
    sr: Optional[int] = None,
    method: str = "harvest",
    frame_period_ms: float = 10.0,
    interpolate_unvoiced: bool = True,
    target_sr: int = 16000,
    normalize_input: bool = False,
) -> Dict[str, Any]:
    """Extract F0 track using pyworld (Harvest + StoneMask).

    Returns a dict with keys: {"f0_hz", "times", "vuv", "frame_period"}
    """
    if not _HAS_PYWORLD:
        raise RuntimeError("pyworld is not available. Install 'pyworld' (or 'pyworld-prebuilt' on Windows).")

    if isinstance(wav, (str, Path)):
        y, sr_used = prepare_input(wav, target_sr=target_sr, normalize=normalize_input)
    else:
        assert sr is not None, "When passing raw waveform, provide its sampling rate."
        y = np.asarray(wav, dtype=np.float32)
        sr_used = sr

    fp = float(frame_period_ms)

    if method.lower() == "harvest":
        f0, t = pw.harvest(y, sr_used, frame_period=fp)
    elif method.lower() == "dio":
        f0, t = pw.dio(y, sr_used, frame_period=fp)
    else:
        raise ValueError("method must be 'harvest' or 'dio'")

    # Refine with StoneMask for better accuracy
    f0_refined = pw.stonemask(y, f0, t, sr_used)

    vuv = (f0_refined > 0).astype(np.uint8)
    if interpolate_unvoiced:
        f0_used = _interp_unvoiced(f0_refined)
    else:
        f0_used = f0_refined.astype(np.float32)

    return {
        "f0_hz": f0_used.astype(np.float32),
        "times": t.astype(np.float32),
        "vuv": vuv,
        "frame_period": fp,
        "sr": int(sr_used),
        "method": method.lower(),
    }


# -----------------------------------------------------------------------------
# BN / PPG placeholders (optional, to be implemented if needed)
# -----------------------------------------------------------------------------

def extract_bn(
    wav: np.ndarray | str | Path,
    sr: Optional[int] = None,
    backend: str = "none",
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Placeholder for Bottleneck/PPG extraction.

    Returns (bn_features, frame_times). Implement with Kaldi/ONNX if required.
    """
    if backend in (None, "none"):
        return None, None
    raise NotImplementedError("BN/PPG extraction backend is not implemented in this template.")
