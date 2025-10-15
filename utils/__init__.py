"""VPC25 utils package

Convenience imports for DSP, features, selection, optional synth, and postproc.
These symbols are re-exported here for ergonomic imports in model.py and
scripts (e.g., `from utils import mcadams_warp, extract_xvec`).
"""
from __future__ import annotations

# --- DSP ---
from .dsp import (
    load_wav,
    save_wav,
    resample,
    normalize_rms,
    rms_dbfs,
    mcadams_warp,
    micro_pitch,
    enforce_mono_sr,
    prepare_input,
)

# --- Features ---
from .features import (
    EcapaEncoder,
    load_ecapa,
    extract_xvec,
    extract_f0,
    extract_bn,
)

# --- Selection ---
from .selection import (
    load_pool_npz,
    l2_normalize,
    cosine_distance,
    rank_pool,
    sample_and_mean,
    load_mapping,
    save_mapping,
    get_consistent_pseudo,
)

# --- Optional modules (may be missing depending on setup) ---
__optional_all__ = []  # Collect names added by optional modules

try:
    from .synth import (
        MelParams,
        compute_mel,
        HiFiGANEnhancer,
        enhance_with_hifigan,
        simple_denoise,
        apply_synthesis_enhance,
    )
    __optional_all__ += [
        "MelParams",
        "compute_mel",
        "HiFiGANEnhancer",
        "enhance_with_hifigan",
        "simple_denoise",
        "apply_synthesis_enhance",
    ]
except Exception:  # synth is optional
    pass

try:
    from .postproc import (
        dc_block,
        vad_trim,
        soft_limiter,
        fade_edges,
        apply_postproc,
    )
    __optional_all__ += [
        "dc_block",
        "vad_trim",
        "soft_limiter",
        "fade_edges",
        "apply_postproc",
    ]
except Exception:  # postproc is optional
    pass

__all__ = [
    # dsp
    "load_wav",
    "save_wav",
    "resample",
    "normalize_rms",
    "rms_dbfs",
    "mcadams_warp",
    "micro_pitch",
    "enforce_mono_sr",
    "prepare_input",
    # features
    "EcapaEncoder",
    "load_ecapa",
    "extract_xvec",
    "extract_f0",
    "extract_bn",
    # selection
    "load_pool_npz",
    "l2_normalize",
    "cosine_distance",
    "rank_pool",
    "sample_and_mean",
    "load_mapping",
    "save_mapping",
    "get_consistent_pseudo",
] + __optional_all__
