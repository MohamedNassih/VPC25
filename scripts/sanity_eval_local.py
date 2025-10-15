#!/usr/bin/env python3
"""Sanity eval (local) for VPC25

Quickly anonymize a *small subset* of files to validate that your pipeline
(model.anonymize) runs end-to-end, saves audio, and roughly disrupts speaker
identity (via ECAPA cosine similarity proxy). This is **not** the official
EER/WER evaluation used by the challenge.

Examples
--------
# Minimal run on 2 speakers × 2 files each (Trial split)
python scripts/sanity_eval_local.py \
  --root evaluation_data \
  --split Trial \
  --speakers 2 \
  --files-per-speaker 2 \
  --out evaluation_data/anonymized_local \
  --csv logs/sanity_eval_local.csv

# Add cosine proxy checks (requires SpeechBrain)
python scripts/sanity_eval_local.py --compute-ecapa-sim
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Import your pipeline entry point
from model import anonymize

# Local helpers
from utils.dsp import load_wav, save_wav, rms_dbfs, enforce_mono_sr

# Optional: ECAPA for cosine similarity proxy
try:  # Lazy import to keep script lightweight if SpeechBrain isn't installed
    from utils.features import load_ecapa
    _HAS_ECAPA = True
except Exception:
    _HAS_ECAPA = False


# -----------------------------------------------------------------------------
# Files discovery
# -----------------------------------------------------------------------------

def list_speakers(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def list_wavs(speaker_dir: Path) -> List[Path]:
    return sorted([p for p in speaker_dir.rglob("*.wav") if p.is_file()])


def sample_subset(root: Path, split: str, n_speakers: int, n_files: int) -> Dict[str, List[Path]]:
    split_dir = root / split
    spk_dirs = list_speakers(split_dir)
    chosen: Dict[str, List[Path]] = {}
    for spk_dir in spk_dirs[: max(0, n_speakers)]:
        wavs = list_wavs(spk_dir)
        if not wavs:
            continue
        chosen[spk_dir.name] = wavs[: max(1, n_files)]
    return chosen


# -----------------------------------------------------------------------------
# Cosine similarity proxy (ECAPA embeddings)
# -----------------------------------------------------------------------------

def cosine_sim(x: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> float:
    x = x.astype(np.float32).reshape(-1)
    y = y.astype(np.float32).reshape(-1)
    xn = x / (np.linalg.norm(x) + eps)
    yn = y / (np.linalg.norm(y) + eps)
    return float(np.dot(xn, yn))


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def run_sanity(
    root: Path,
    split: str,
    out_root: Path,
    n_speakers: int,
    n_files: int,
    sr: int,
    compute_ecapa: bool,
    csv_path: Path | None,
) -> None:
    subset = sample_subset(root, split, n_speakers, n_files)
    if not subset:
        print(f"[ERR] No WAVs found under {root}/{split}.")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    # Optional encoder
    if compute_ecapa and not _HAS_ECAPA:
        print("[WARN] --compute-ecapa-sim requested but utils.features.load_ecapa not available.")
        compute_ecapa = False

    encoder = load_ecapa(device="auto") if compute_ecapa else None

    rows: List[dict] = []
    t0 = time.time()
    n_total = 0
    sims: List[float] = []

    for spk, files in subset.items():
        for wav_path in files:
            n_total += 1
            rel = wav_path.relative_to(root / split)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Load source for metrics
            y_src, sr_src = load_wav(wav_path, target_sr=sr, mono=True)
            y_src, sr_src = enforce_mono_sr(y_src, sr_src, target_sr=sr)
            db_src = rms_dbfs(y_src)

            # Anonymize
            t1 = time.time()
            y_anon, sr_anon = anonymize(str(wav_path))
            dur_s = time.time() - t1
            save_wav(out_path, y_anon, sr_anon)
            db_anon = rms_dbfs(y_anon)

            row = {
                "speaker": spk,
                "file": str(rel),
                "src_dbfs": f"{db_src:.2f}",
                "anon_dbfs": f"{db_anon:.2f}",
                "proc_time_s": f"{dur_s:.3f}",
                "out_path": str(out_path),
            }

            # Cosine proxy: similarity(source, anonymized) — lower is better
            if encoder is not None:
                try:
                    x_src = encoder.embed(y_src, sr_src)
                    x_anon = encoder.embed(y_anon, sr_anon)
                    sim = cosine_sim(x_src, x_anon)
                    row["cos_src_anon"] = f"{sim:.4f}"
                    sims.append(sim)
                except Exception as e:
                    row["cos_src_anon"] = f"ERR:{e}"

            rows.append(row)
            print(f"[OK] {rel}  time={dur_s:.3f}s  src={db_src:.1f} dBFS  anon={db_anon:.1f} dBFS")

    t1 = time.time()
    print("\n=== Summary ===")
    print(f"Files processed: {n_total}")
    print(f"Total time: {t1 - t0:.2f}s  |  Avg/file: {(t1 - t0)/max(1,n_total):.2f}s")
    if sims:
        print(f"Mean cosine(source, anon): {np.mean(sims):.4f}  (lower is better)")

    # Save CSV
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else ["speaker", "file", "src_dbfs", "anon_dbfs", "proc_time_s", "out_path", "cos_src_anon"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[OK] Wrote CSV: {csv_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity eval (local) for VPC25")
    p.add_argument("--root", type=Path, default=Path("evaluation_data"), help="Path containing Enrollment/ and Trial/ splits")
    p.add_argument("--split", type=str, default="Trial", choices=["Enrollment", "Trial"], help="Which split to process")
    p.add_argument("--out", type=Path, default=Path("evaluation_data/anonymized_local"))
    p.add_argument("--speakers", type=int, default=2, help="Number of speakers to sample")
    p.add_argument("--files-per-speaker", type=int, default=2, help="Number of files per sampled speaker")
    p.add_argument("--sr", type=int, default=16000, help="Target sampling rate for I/O")
    p.add_argument("--compute-ecapa-sim", dest="compute_ecapa", action="store_true", help="Compute cosine(source,anon) proxy with ECAPA")
    p.add_argument("--csv", type=Path, default=Path("logs/sanity_eval_local.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    run_sanity(
        root=args.root,
        split=args.split,
        out_root=args.out,
        n_speakers=args.speakers,
        n_files=args.files_per_speaker,
        sr=args.sr,
        compute_ecapa=args.compute_ecapa,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
