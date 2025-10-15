#!/usr/bin/env python3
"""Build an x-vector pool (NPZ) from WAVs for VPC25.

This script extracts speaker embeddings (x-vectors) using SpeechBrain's
ECAPA-TDNN and writes a compact pool to `parameters/pool/xvector_pool.npz`.

Typical usage
-------------
python scripts/build_xvector_pool.py \
  --enroll-root evaluation_data/Enrollment \
  --out parameters/pool/xvector_pool.npz \
  --per-speaker-avg \
  --max-files-per-speaker 10

You can also add external corpora of WAVs (e.g., LibriTTS/VCTK folders):
  --extra-roots /path/to/libritts/train-clean-100 /path/to/vctk/wav48

The resulting NPZ contains:
  - xvectors : float32 array, shape (M, D)
  - keys     : list[str], one key per row (e.g., "spk0001/utt1.wav" or "spk0001")
  - meta     : dict with config info

Note: This script reads audio with utils.dsp.load_wav/prepare_input, resampling
and mono conversion as needed.
"""
from __future__ import annotations

# --- Windows/CPU: SpeechBrain symlink + torchaudio backend shims ---
import os
os.environ.setdefault("SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY", "copy")
os.environ.setdefault("SPEECHBRAIN_LOCAL_FILE_STRATEGY", "copy")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torchaudio  # must be imported before any `speechbrain` import
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        # Minimal backend list for SpeechBrain's check; soundfile works on CPU/Windows
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass
# -------------------------------------------------------------------

from speechbrain.utils.fetching import LocalStrategy
from speechbrain.pretrained import EncoderClassifier  # redirect -> speechbrain.inference

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from speechbrain.utils.fetching import LocalStrategy

# SpeechBrain ECAPA encoder
from speechbrain.pretrained import EncoderClassifier  # type: ignore

# Local helpers
try:
    from utils.dsp import prepare_input
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please ensure utils/dsp.py is available and importable.") from e

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def find_wavs(root: Path, patterns: Tuple[str, ...] = ("*.wav",)) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(root.rglob(pat))
    # filter files only
    return [p for p in paths if p.is_file()]


def group_by_speaker(enroll_root: Path, wavs: List[Path]) -> Dict[str, List[Path]]:
    """Group Enrollment WAVs by speaker directory name.

    Assumes structure: Enrollment/spkXXXX/utt.wav
    Returns {speaker_id: [wav_paths...]}
    """
    grouped: Dict[str, List[Path]] = {}
    try:
        # relative path to figure out speaker directory right after Enrollment/
        for p in wavs:
            rel = p.relative_to(enroll_root)
            spk = rel.parts[0] if len(rel.parts) >= 2 else rel.stem
            grouped.setdefault(spk, []).append(p)
    except Exception:
        # Fallback: put all files under a single speaker if structure is different
        grouped["spk_unknown"] = wavs
    return grouped


def iter_all_audio(
    enroll_root: Path,
    extra_roots: List[Path],
    patterns: Tuple[str, ...] = ("*.wav",),
) -> Tuple[Dict[str, List[Path]], List[Path]]:
    enroll_wavs = find_wavs(enroll_root, patterns) if enroll_root.exists() else []
    grouped = group_by_speaker(enroll_root, enroll_wavs)
    extra_wavs: List[Path] = []
    for r in extra_roots:
        if r.exists():
            extra_wavs.extend(find_wavs(r, patterns))
    return grouped, extra_wavs


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


# ----------------------------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------------------------

def load_encoder(device: str = "auto", ckpt: str | None = None, hf_repo: str | None = None):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_opts = {"device": device}

    if ckpt and Path(ckpt).exists():
        encoder = EncoderClassifier.from_hparams(source=ckpt, run_opts=run_opts, savedir=str(Path(".cache")/"ecapa_local"), local_strategy=LocalStrategy.COPY)
    else:
        repo = hf_repo or "speechbrain/spkrec-ecapa-voxceleb"
        encoder = EncoderClassifier.from_hparams(source=repo, run_opts=run_opts, savedir=str(Path(".cache")/"ecapa_hf"), local_strategy=LocalStrategy.COPY_SKIP_CACHE)
    encoder.eval()
    return encoder, device


def embed_wav(encoder, wav: np.ndarray, sr: int) -> np.ndarray:
    # SpeechBrain expects torch tensor [batch, time]; auto resampling not provided here
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(wav_t)  # shape [1, 1, D]
    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    return emb


# ----------------------------------------------------------------------------
# Main build routine
# ----------------------------------------------------------------------------

def build_pool(
    enroll_root: Path,
    extra_roots: List[Path],
    out_npz: Path,
    sr: int = 16000,
    per_speaker_avg: bool = True,
    max_files_per_speaker: int | None = 10,
    patterns: Tuple[str, ...] = ("*.wav",),
    device: str = "auto",
    ckpt: str | None = None,
    hf_repo: str | None = "speechbrain/spkrec-ecapa-voxceleb",
    normalize: bool = True,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    grouped, extra_wavs = iter_all_audio(enroll_root, extra_roots, patterns)

    encoder, device = load_encoder(device=device, ckpt=ckpt, hf_repo=hf_repo)

    xlist: List[np.ndarray] = []
    klist: List[str] = []

    # First, Enrollment grouped by speaker
    for spk, files in grouped.items():
        if max_files_per_speaker is not None and len(files) > max_files_per_speaker:
            files = files[: max_files_per_speaker]
        embs = []
        for wav_path in files:
            try:
                y, sr_ = prepare_input(wav_path, target_sr=sr, normalize=False)
                if np.max(np.abs(y)) < 1e-5:
                    continue  # skip quasi-silence
                emb = embed_wav(encoder, y, sr_)
                embs.append(emb)
                if not per_speaker_avg:
                    xlist.append(emb)
                    klist.append(str(Path(spk)/wav_path.name))
            except Exception as e:
                print(f"[WARN] Failed: {wav_path} -> {e}")
        if per_speaker_avg and embs:
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            xlist.append(mean_emb)
            klist.append(spk)

    # Then, extra roots (not grouped; one embedding per file)
    for wav_path in extra_wavs:
        try:
            y, sr_ = prepare_input(wav_path, target_sr=sr, normalize=False)
            if np.max(np.abs(y)) < 1e-5:
                continue
            emb = embed_wav(encoder, y, sr_)
            xlist.append(emb)
            klist.append(str(wav_path))
        except Exception as e:
            print(f"[WARN] Failed extra: {wav_path} -> {e}")

    if not xlist:
        raise RuntimeError("No embeddings extracted. Check your paths and audio files.")

    X = np.stack(xlist, axis=0).astype(np.float32)
    if normalize:
        X = l2_normalize(X, axis=1)

    meta = {
        "sr": sr,
        "device": device,
        "per_speaker_avg": per_speaker_avg,
        "max_files_per_speaker": max_files_per_speaker,
        "num_items": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "encoder": "ECAPA (SpeechBrain)",
        "hf_repo": hf_repo,
        "ckpt": ckpt,
    }

    np.savez_compressed(out_npz, xvectors=X, keys=np.array(klist, dtype=object), meta=np.array([json.dumps(meta)], dtype=object))
    print(f"[OK] Saved pool: {out_npz}  (M={X.shape[0]}, D={X.shape[1]})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build x-vector pool for VPC25")
    p.add_argument("--enroll-root", type=Path, default=Path("evaluation_data/Enrollment"))
    p.add_argument("--extra-roots", type=Path, nargs="*", default=[], help="Optional external WAV roots (LibriTTS/VCTK folders, etc.)")
    p.add_argument("--out", type=Path, default=Path("parameters/pool/xvector_pool.npz"))
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--per-speaker-avg", action="store_true", help="Average embeddings per Enrollment speaker (recommended)")
    p.add_argument("--no-per-speaker-avg", dest="per_speaker_avg", action="store_false")
    p.set_defaults(per_speaker_avg=True)
    p.add_argument("--max-files-per-speaker", type=int, default=10)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]) 
    p.add_argument("--local-ckpt", type=str, default=None, help="Path to a local ECAPA checkpoint (optional)")
    p.add_argument("--hf-repo", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--patterns", type=str, nargs="*", default=["*.wav"], help="Glob patterns for audio files")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.set_defaults(normalize=True)
    return p.parse_args()


def main():
    args = parse_args()
    build_pool(
        enroll_root=args.enroll_root,
        extra_roots=list(args.extra_roots),
        out_npz=args.out,
        sr=args.sr,
        per_speaker_avg=args.per_speaker_avg,
        max_files_per_speaker=args.max_files_per_speaker,
        patterns=tuple(args.patterns),
        device=args.device,
        ckpt=args.local_ckpt,
        hf_repo=args.hf_repo,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
