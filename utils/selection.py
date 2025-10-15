"""Selection helpers for VPC25

This module provides utilities to:
 - Load an x-vector pool saved as NPZ (M x D matrix) and its keys
 - Rank pool entries by distance to a source embedding (cosine by default)
 - Sample K embeddings from the Top-N farthest and average them (pseudo-speaker)
 - Maintain a persistent mapping speaker_key -> pseudo identity in JSON

The default policy follows the plan:
  1) Compute cosine distance to all pool items (assuming L2-normalized vectors)
  2) Take the Top-N *farthest* indices (largest distance)
  3) Random-sample K among them, then average and L2-normalize the result
  4) Persist this pseudo x-vector (and metadata) so that the *same* speaker_key
     always receives the *same* pseudo identity across Enrollment/Trial

PLDA support is left as a placeholder. If you have a PLDA backend, implement
`score_plda` and switch `distance="plda"` in config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import json
import numpy as np

__all__ = [
    "load_pool_npz",
    "l2_normalize",
    "cosine_distance",
    "rank_pool",
    "sample_and_mean",
    "load_mapping",
    "save_mapping",
    "get_consistent_pseudo",
]

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_pool_npz(npz_path: str | Path) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Load a pool NPZ with arrays: xvectors [M,D], keys [M], meta [json]."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Pool NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as Z:
        X = Z["xvectors"].astype(np.float32)
        keys = [str(x) for x in Z["keys"].tolist()]
        meta_json = Z["meta"].tolist()[0] if "meta" in Z.files else "{}"
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = {}
    return X, keys, meta


def load_mapping(mapping_path: str | Path) -> Dict[str, Any]:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        return {}
    try:
        return json.loads(mapping_path.read_text())
    except Exception:
        return {}


def save_mapping(mapping: Dict[str, Any], mapping_path: str | Path) -> None:
    mapping_path = Path(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2))


# -----------------------------------------------------------------------------
# Distances and ranking
# -----------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def cosine_distance(x: np.ndarray, Y: np.ndarray, assume_normed: bool = True, eps: float = 1e-10) -> np.ndarray:
    """Return cosine *distance* (1 - cosine similarity) from x to each row of Y.

    Parameters
    ----------
    x : (D,)
    Y : (M, D)
    assume_normed : if False, will L2-normalize x and Y first
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    Y = np.asarray(Y, dtype=np.float32)
    if not assume_normed:
        x = l2_normalize(x)
        Y = l2_normalize(Y, axis=1)
    sim = Y @ x  # (M,)
    return 1.0 - sim


# Placeholder for PLDA scoring (implement if you have a backend)
# def score_plda(x: np.ndarray, Y: np.ndarray, model: Any) -> np.ndarray:
#     """Return a *distance-like* score: larger = more different.
#     Implement using your PLDA backend (e.g., Kaldi), then map similarity -> distance.
#     """
#     raise NotImplementedError


def rank_pool(
    x_source: np.ndarray,
    pool_X: np.ndarray,
    topN: int = 200,
    distance: str = "cosine",
    plda_model: Optional[object] = None,
    farthest: bool = True,
    assume_normed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank pool by distance; return (indices_sorted, distances_sorted).

    If farthest=True (default), sorts by *descending* distance and returns topN.
    If farthest=False, sorts by ascending distance.
    """
    M = pool_X.shape[0]
    N = int(min(max(1, topN), M))

    if distance == "cosine":
        d = cosine_distance(x_source, pool_X, assume_normed=assume_normed)
    elif distance == "plda":
        # d = score_plda(x_source, pool_X, plda_model)
        raise NotImplementedError("PLDA distance not implemented. Use cosine or provide a backend.")
    else:
        raise ValueError("distance must be 'cosine' or 'plda'")

    if farthest:
        idx_sorted = np.argsort(-d)  # larger distance first
    else:
        idx_sorted = np.argsort(d)   # smaller distance first

    idx_top = idx_sorted[:N]
    d_top = d[idx_top]
    return idx_top, d_top


def sample_and_mean(
    pool_X: np.ndarray,
    top_indices: np.ndarray,
    k: int = 100,
    rng_seed: Optional[int] = None,
    l2norm: bool = True,
) -> np.ndarray:
    """Sample K among top_indices, average them, optionally L2-normalize.

    Parameters
    ----------
    pool_X : (M, D)
    top_indices : (N,)
    k : number of samples to average
    rng_seed : for reproducibility
    l2norm : whether to L2-normalize the resulting pseudo x-vector
    """
    rng = np.random.default_rng(rng_seed)
    N = len(top_indices)
    if N == 0:
        raise ValueError("top_indices is empty")
    kk = int(max(1, min(k, N)))
    sel = rng.choice(top_indices, size=kk, replace=False)
    mean_vec = np.mean(pool_X[sel], axis=0).astype(np.float32)
    if l2norm:
        mean_vec = l2_normalize(mean_vec)
    return mean_vec


# -----------------------------------------------------------------------------
# Persistent pseudo-identity mapping
# -----------------------------------------------------------------------------

def get_consistent_pseudo(
    speaker_key: str,
    xvec_src: np.ndarray,
    pool_X: np.ndarray,
    mapping_path: str | Path,
    *,
    distance: str = "cosine",
    plda_model: Optional[object] = None,
    topN: int = 200,
    sample_k: int = 100,
    rng_seed: Optional[int] = 1337,
    store_alpha: Optional[float] = None,
    store_seed: Optional[int] = None,
    store_xvector: bool = True,
) -> np.ndarray:
    """Return a pseudo x-vector, stable per `speaker_key`, and persist it.

    If `speaker_key` is already present in mapping, the stored pseudo x-vector is
    returned. Otherwise, a new one is created via rank→sample→mean and stored.

    The mapping JSON structure is flexible; we store at least:
      mapping[speaker_key] = {"xvector": [...], "distance": "cosine", "topN": 200, ...}
    """
    mapping = load_mapping(mapping_path)

    if speaker_key in mapping and "xvector" in mapping[speaker_key]:
        x = np.asarray(mapping[speaker_key]["xvector"], dtype=np.float32)
        return l2_normalize(x)

    # Compute a new pseudo identity
    idx_top, d_top = rank_pool(
        x_source=l2_normalize(xvec_src),
        pool_X=pool_X,
        topN=topN,
        distance=distance,
        plda_model=plda_model,
        farthest=True,
        assume_normed=True,
    )
    x_pseudo = sample_and_mean(pool_X, idx_top, k=sample_k, rng_seed=rng_seed, l2norm=True)

    # Store
    entry: Dict[str, Any] = {
        "distance": distance,
        "topN": int(topN),
        "sample_k": int(sample_k),
        "rng_seed": int(rng_seed) if rng_seed is not None else None,
        "xvector": x_pseudo.tolist() if store_xvector else None,
    }
    if store_alpha is not None:
        entry["alpha"] = float(store_alpha)
    if store_seed is not None:
        entry["seed"] = int(store_seed)

    mapping[speaker_key] = entry
    save_mapping(mapping, mapping_path)

    return x_pseudo
