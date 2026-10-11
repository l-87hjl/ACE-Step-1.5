"""Pure-numpy splice helpers for joining an original clip to a generated continuation.

Both inputs share one timeline: ACE-Step repaint/outpaint returns audio whose sample
``i`` corresponds to sample ``i`` of the (right-padded) source, so no time alignment
is needed - only a choice of *where* inside the overlap window to cut, and a fade.
"""

from typing import Optional

import numpy as np


def frame_log_energy_bands(x: np.ndarray, hop: int, n_fft: int = 2048,
                           n_bands: int = 32) -> np.ndarray:
    """Return a ``[frames, bands]`` log band-energy matrix for a mono signal.

    Args:
        x: Mono float signal.
        hop: Hop size in samples.
        n_fft: FFT size.
        n_bands: Number of log-spaced frequency bands (upper bound).

    Returns:
        Log band energies, one row per hop.
    """
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    edges = np.unique(np.geomspace(1, spec.shape[1] - 1, n_bands + 1).astype(int))
    bands = np.stack([spec[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    return np.log(bands + 1e-10)


def effective_end_s(x: np.ndarray, sr: int, drop_db: float = 9.0, frame_s: float = 0.1,
                    ref_window_s: float = 10.0) -> float:
    """Return where the clip's sustained body ends, ignoring a trailing fade/silence.

    The reference level is the median short-term RMS over the last ``ref_window_s``
    seconds before the final second; the effective end is the end of the last frame
    within ``drop_db`` of that level. A clip that plays to its last sample returns
    its full length.

    Args:
        x: Audio ``[samples, channels]``.
        sr: Sample rate.
        drop_db: Level drop that counts as "faded out".
        frame_s: RMS frame length in seconds.
        ref_window_s: Window used for the reference level.

    Returns:
        Effective end in seconds.
    """
    hop = max(1, int(frame_s * sr))
    n = len(x) // hop
    if n < 3:
        return len(x) / sr
    rms = np.sqrt((x[: n * hop].mean(1).reshape(n, hop) ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    ref_frames = db[max(0, n - int((ref_window_s + 1) / frame_s)): max(1, n - int(1 / frame_s))]
    ref = float(np.median(ref_frames))
    loud = np.nonzero(db >= ref - drop_db)[0]
    if len(loud) == 0:
        return len(x) / sr
    return min(len(x) / sr, (int(loud[-1]) + 1) * frame_s)


def choose_splice_sample(original: np.ndarray, generated: np.ndarray, sr: int,
                         window_start_s: float, window_end_s: float,
                         beat_times_s: Optional[np.ndarray] = None,
                         hop_s: float = 0.02) -> int:
    """Pick the cut inside ``[window_start_s, window_end_s]`` where the two signals
    are spectrally most similar, restricted to beat positions when any are given.

    Args:
        original: Original audio ``[samples, channels]``.
        generated: Generated audio on the same timeline ``[samples, channels]``.
        sr: Sample rate.
        window_start_s: Earliest allowed cut in seconds.
        window_end_s: Latest allowed cut in seconds (clipped to both lengths).
        beat_times_s: Optional beat positions in seconds.
        hop_s: Analysis hop in seconds.

    Returns:
        Cut position in samples.
    """
    end_s = min(window_end_s, len(original) / sr, len(generated) / sr)
    if end_s - window_start_s < 2 * hop_s:
        return int(round(end_s * sr))
    hop = max(1, int(hop_s * sr))
    a0, a1 = int(window_start_s * sr), int(end_s * sr)
    fa = frame_log_energy_bands(original[a0:a1].mean(1), hop)
    fb = frame_log_energy_bands(generated[a0:a1].mean(1), hop)
    n = min(len(fa), len(fb))
    dist = np.linalg.norm(fa[:n] - fb[:n], axis=1)
    times = window_start_s + np.arange(n) * hop / sr
    if beat_times_s is not None:
        beats = np.asarray(beat_times_s)
        beats = beats[(beats >= window_start_s) & (beats < end_s)]
        if len(beats):
            cand = np.clip(np.round((beats - window_start_s) * sr / hop).astype(int), 0, n - 1)
            return int(round(times[cand[np.argmin(dist[cand])]] * sr))
    return int(round(times[int(np.argmin(dist))] * sr))


def equal_power_splice(original: np.ndarray, generated: np.ndarray, cut: int,
                       fade_samples: int) -> np.ndarray:
    """Return ``original`` up to the fade, an equal-power fade centred on ``cut``,
    then ``generated``. Samples before ``cut - fade_samples // 2`` are bit-exact.

    Args:
        original: Original audio ``[samples, channels]``.
        generated: Generated audio on the same timeline ``[samples, channels]``.
        cut: Splice point in samples.
        fade_samples: Crossfade length in samples (0 = hard cut).

    Returns:
        Assembled audio with ``len(generated)`` samples.
    """
    out = generated.copy()
    half = fade_samples // 2
    f0 = max(0, min(cut - half, len(original)))
    f1 = min(cut + half, len(original), len(generated))
    out[:f0] = original[:f0]
    if f1 > f0:
        t = np.linspace(0.0, 1.0, f1 - f0)[:, None]
        out[f0:f1] = (np.cos(t * np.pi / 2) * original[f0:f1]
                      + np.sin(t * np.pi / 2) * generated[f0:f1])
    return out


def overlap_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean log-band distance between two equal-timeline excerpts (lower = closer)."""
    n = min(len(a), len(b))
    fa = frame_log_energy_bands(a[:n].mean(1), 960)
    fb = frame_log_energy_bands(b[:n].mean(1), 960)
    return float(np.linalg.norm(fa - fb, axis=1).mean())


def join_record(full: np.ndarray, cut_abs: int, fade: int, sr: int) -> dict:
    """Level just before and after the join of an assembled candidate."""
    def rms_db(x):
        return round(float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)), 2)
    return {"rms_db_2s_before": rms_db(full[max(0, cut_abs - fade // 2 - 2 * sr):
                                            max(1, cut_abs - fade // 2)]),
            "rms_db_2s_after": rms_db(full[cut_abs + fade // 2:cut_abs + fade // 2 + 2 * sr])}
