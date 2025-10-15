#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Optional deps are imported lazily
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Local utils
from utils.dsp import (
    prepare_input,
    normalize_rms,
    mcadams_warp,
    micro_pitch,
)

# Feature + selection (optional)
try:
    from utils.features import extract_xvec, load_ecapa  # type: ignore
except Exception:  # pragma: no cover
    extract_xvec = None  # type: ignore
    load_ecapa = None  # type: ignore

try:
    from utils.selection import (
        load_pool_npz,
        get_consistent_pseudo,
        l2_normalize,
    )  # type: ignore
except Exception:  # pragma: no cover
    load_pool_npz = None  # type: ignore
    get_consistent_pseudo = None  # type: ignore
    l2_normalize = None  # type: ignore

# Optional synth enhancer
try:
    from utils.synth import apply_synthesis_enhance  # type: ignore
except Exception:  # pragma: no cover
    apply_synthesis_enhance = None  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Config & global state
# -----------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, Any] = {
    "audio": {"sr": 16000, "target_dbfs": -23.0, "normalize": True},
    "mcadams": {
        "enabled": True,
        "alpha": 0.82,           # base exponent (0.75–0.9 typical)
        "lpc_order": "auto",
        "frame_length": 0.030,
        "hop_length": 0.015,
        "pre_emph": 0.0,
        "jitter": 0.04           # extra ± range for per-speaker variability
    },
    "post": {
        "pitch_shift": {"enabled": True, "semitones_low": -1.0, "semitones_high": 1.0}
    },
    "selection": {
        "pool_npz": "parameters/pool/xvector_pool.npz",
        "pseudo_map": "parameters/pseudo_map.json",
        "topN": 200,
        "sample_k": 100,
        "rng_seed": 1337,
        "use_pool": True
    },
    "synthesis": {
        "enabled": False,
        "backend": "hifigan",
        "local_ckpt": None,
        "device": "auto",
        "denoise": False,
        "mel": {"sr": 16000}
    },
}


@dataclass
class _State:
    cfg: Dict[str, Any]
    pool_X: Optional[np.ndarray] = None
    pool_meta: Optional[Dict[str, Any]] = None
    mapping_path: Path = Path("parameters/pseudo_map.json")
    encoder: Any = None  # ECAPA encoder (optional)
    ready: bool = False


_STATE = _State(cfg=DEFAULT_CFG.copy())


def _load_yaml_cfg(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return DEFAULT_CFG.copy()
    try:
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # shallow-merge: user overrides defaults
        def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(a)
            for k, v in b.items():
                if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                    out[k] = merge(out[k], v)
                else:
                    out[k] = v
            return out
        return merge(DEFAULT_CFG, user_cfg)
    except Exception as e:
        logger.warning(f"Failed to read config.yaml, using defaults. Error: {e}")
        return DEFAULT_CFG.copy()


def _init_state() -> None:
    if _STATE.ready:
        return
    cfg_path = Path("parameters/config.yaml")
    _STATE.cfg = _load_yaml_cfg(cfg_path)

    # Load pool (optional)
    sel_cfg = _STATE.cfg.get("selection", {})
    pool_path = Path(sel_cfg.get("pool_npz", DEFAULT_CFG["selection"]["pool_npz"]))
    _STATE.mapping_path = Path(sel_cfg.get("pseudo_map", DEFAULT_CFG["selection"]["pseudo_map"]))

    if sel_cfg.get("use_pool", True) and load_pool_npz is not None and pool_path.exists():
        try:
            X, keys, meta = load_pool_npz(pool_path)
            _STATE.pool_X = X.astype(np.float32)
            _STATE.pool_meta = meta
            logger.info(f"Loaded xvector pool: {pool_path} (M={X.shape[0]}, D={X.shape[1]})")
        except Exception as e:
            logger.warning(f"Could not load pool at {pool_path}: {e}")
            _STATE.pool_X = None
            _STATE.pool_meta = None
    else:
        logger.info("No pool loaded (missing file or disabled). Fallback to per-speaker randomization.")
        _STATE.pool_X = None
        _STATE.pool_meta = None

    # Lazy encoder init (only if we use pool or want xvec-driven params)
    if _STATE.pool_X is not None and extract_xvec is not None and load_ecapa is not None:
        try:
            _STATE.encoder = load_ecapa(device="auto")
        except Exception as e:
            logger.warning(f"ECAPA encoder init failed: {e}")
            _STATE.encoder = None

    _STATE.ready = True


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _speaker_key_from_path(path: Path) -> str:
    """Infer a stable speaker key from the evaluation path structure.

    Expected layout: .../(Enrollment|Trial)/<speaker_id>/<file>.wav
    """
    parts = path.parts
    for tag in ("Enrollment", "Trial"):
        if tag in parts:
            i = parts.index(tag)
            if i + 1 < len(parts):
                return f"{tag}:{parts[i+1]}"  # include split in the key to avoid accidental cross-split collision
    # Fallback: parent dir name
    return f"parent:{path.parent.name}"


def _per_speaker_det_params(speaker_key: str, cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Deterministically derive (alpha, pitch_semitones) from speaker_key.

    Used when pool/encoder are unavailable. Hash -> PRNG -> values.
    """
    import hashlib

    h = hashlib.sha1(speaker_key.encode("utf-8")).digest()
    # map bytes to [0,1)
    u1 = int.from_bytes(h[:8], "big") / (2**64)
    u2 = int.from_bytes(h[8:16], "big") / (2**64)

    base_alpha = float(cfg["mcadams"].get("alpha", 0.82))
    jitter = float(cfg["mcadams"].get("jitter", 0.04))
    alpha = np.clip(base_alpha + (u1 * 2 - 1) * jitter, 0.70, 0.95)

    p_low = float(cfg["post"]["pitch_shift"].get("semitones_low", -1.0))
    p_high = float(cfg["post"]["pitch_shift"].get("semitones_high", 1.0))
    pitch = p_low + (p_high - p_low) * u2
    return float(alpha), float(pitch)


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def anonymize(input_audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """Anonymize a WAV file path and return (waveform, sr).

    This function is intentionally stateless w.r.t. inputs but keeps a module
    cache for heavy assets (pool, encoder, config) across calls.
    """
    _init_state()
    cfg = _STATE.cfg

    in_path = Path(input_audio_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Audio file not found: {in_path}")

    # --- Load & prepare ---
    sr_target = int(cfg["audio"].get("sr", 16000))
    y, sr = prepare_input(in_path, target_sr=sr_target, normalize=bool(cfg["audio"].get("normalize", True)))

    # --- Choose transform params (via pool if available, else deterministic) ---
    speaker_key = _speaker_key_from_path(in_path)

    alpha = float(cfg["mcadams"].get("alpha", 0.82))
    pitch_semi = 0.0

    if _STATE.pool_X is not None and get_consistent_pseudo is not None:
        try:
            # Compute source x-vector
            xvec_src, _ = extract_xvec(y, sr, encoder=_STATE.encoder, target_sr=sr_target, normalize_input=False)  # type: ignore
            xvec_src = l2_normalize(xvec_src)  # type: ignore

            # Fetch or create consistent pseudo identity
            sel = cfg.get("selection", {})
            pseudo_vec = get_consistent_pseudo(
                speaker_key=speaker_key,
                xvec_src=xvec_src,
                pool_X=_STATE.pool_X,
                mapping_path=_STATE.mapping_path,
                distance="cosine",
                plda_model=None,
                topN=int(sel.get("topN", 200)),
                sample_k=int(sel.get("sample_k", 100)),
                rng_seed=int(sel.get("rng_seed", 1337)),
                store_alpha=None,
                store_seed=None,
                store_xvector=True,
            )
            # Map cosine to small pitch shift deterministically
            cos_sim = float(np.dot(xvec_src, pseudo_vec))
            # similarity in [-1,1] -> pitch in [low, high]
            pconf = cfg["post"]["pitch_shift"]
            low, high = float(pconf.get("semitones_low", -1.0)), float(pconf.get("semitones_high", 1.0))
            t = 0.5 * (cos_sim + 1.0)
            pitch_semi = low + (high - low) * t

            # Slightly adapt alpha in the opposite direction of similarity
            base_alpha = float(cfg["mcadams"].get("alpha", 0.82))
            jitter = float(cfg["mcadams"].get("jitter", 0.04))
            alpha = np.clip(base_alpha + (0.5 - t) * (2 * jitter), 0.70, 0.95)
        except Exception as e:
            logger.warning(f"Pool/encoder path failed, falling back to deterministic params. Err: {e}")
            alpha, pitch_semi = _per_speaker_det_params(speaker_key, cfg)
    else:
        # No pool available → deterministic per-speaker params
        alpha, pitch_semi = _per_speaker_det_params(speaker_key, cfg)

    # --- Transform ---
    if cfg["mcadams"].get("enabled", True):
        y = mcadams_warp(
            y,
            sr,
            alpha=float(alpha),
            lpc_order=cfg["mcadams"].get("lpc_order", "auto"),
            frame_length=float(cfg["mcadams"].get("frame_length", 0.03)),
            hop_length=float(cfg["mcadams"].get("hop_length", 0.015)),
            pre_emph=float(cfg["mcadams"].get("pre_emph", 0.0)),
        )

    if cfg["post"]["pitch_shift"].get("enabled", True) and abs(pitch_semi) > 1e-3:
        y = micro_pitch(y, sr, semitones=float(pitch_semi))

    # --- Optional synthesis/enhancement ---
    synth_cfg = cfg.get("synthesis", {})
    if apply_synthesis_enhance is not None and synth_cfg.get("enabled", False):
        try:
            y, sr = apply_synthesis_enhance(y, sr, synth_cfg)
        except Exception as e:
            logger.warning(f"Enhancer failed; returning pre-enhanced audio. Err: {e}")

    # --- Loudness normalize & safety ---
    if bool(cfg["audio"].get("normalize", True)):
        y = normalize_rms(y, target_dbfs=float(cfg["audio"].get("target_dbfs", -23.0)))

    y = np.asarray(np.clip(y, -1.0, 1.0), dtype=np.float32)
    return y, int(sr)
