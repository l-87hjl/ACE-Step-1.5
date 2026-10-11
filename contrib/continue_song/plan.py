"""Engine-neutral planning helpers: per-pass lyric plans, beat grids, phrase protection,
and loudness matching at a join. Pure Python/numpy, no model imports."""

import json
from typing import List, Optional, Sequence

import numpy as np


def load_lyrics_plan(path: str) -> List[dict]:
    """Load a per-pass plan: ``{"passes": [{"segment": s, "lyrics": "...", "label": "..."}]}``.

    ``segment`` is the number of new seconds that pass keeps; ``lyrics`` is the lyric
    sheet for that pass's context window *plus* the new material (the model sees both).

    Raises:
        ValueError: when a pass lacks a positive ``segment`` or a ``lyrics`` string.
    """
    passes = json.load(open(path))["passes"]
    for i, p in enumerate(passes):
        if not (isinstance(p.get("segment"), (int, float)) and p["segment"] > 0):
            raise ValueError(f"pass {i + 1}: 'segment' must be a positive number")
        if not isinstance(p.get("lyrics"), str):
            raise ValueError(f"pass {i + 1}: 'lyrics' must be a string")
    return passes


def beat_grid(bpm: float, offset_s: float, t0: float, t1: float) -> np.ndarray:
    """Beat times ``offset + k * 60 / bpm`` that fall inside ``[t0, t1)``."""
    period = 60.0 / bpm
    k0 = int(np.ceil((t0 - offset_s) / period))
    k1 = int(np.floor((t1 - offset_s) / period))
    return np.array([offset_s + k * period for k in range(k0, k1 + 1)
                     if t0 <= offset_s + k * period < t1])


def unsung_times(times: np.ndarray, words: Optional[Sequence[dict]],
                 margin_s: float = 0.12) -> np.ndarray:
    """Drop times that fall inside a sung word (``{"s": start, "e": end}``) +/- margin."""
    if not words or len(times) == 0:
        return times
    keep = [t for t in times
            if not any(w["s"] - margin_s <= t <= w["e"] + margin_s for w in words)]
    return np.array(keep)


def match_gain(reference: np.ndarray, candidate: np.ndarray, max_db: float) -> float:
    """Linear gain bringing ``candidate``'s RMS to ``reference``'s, clipped to +/-max_db.

    Both excerpts must cover the same musical moment (the overlap window). Returns 1.0
    when ``max_db`` <= 0 or either excerpt is silent.
    """
    if max_db <= 0 or len(reference) == 0 or len(candidate) == 0:
        return 1.0
    r = float(np.sqrt(np.mean(np.square(reference))))
    c = float(np.sqrt(np.mean(np.square(candidate))))
    if r < 1e-6 or c < 1e-6:
        return 1.0
    db = float(np.clip(20 * np.log10(r / c), -max_db, max_db))
    return float(10 ** (db / 20))
