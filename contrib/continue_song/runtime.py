"""Runtime helpers for continue_song: audio loading and unified-memory hygiene."""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48000


def load_48k_stereo(path: str) -> np.ndarray:
    """Load audio as float32 ``[samples, 2]`` at 48 kHz."""
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    return x


def release_cached_memory(mlx_cache_gb: float = None) -> None:
    """Return framework buffer caches to the OS between passes (unified-memory hosts).

    Args:
        mlx_cache_gb: When given, also cap MLX's buffer cache at this many GB.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import mlx.core as mx
        if mlx_cache_gb is not None:
            mx.set_cache_limit(int(mlx_cache_gb * (1 << 30)))
        mx.clear_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass


def write_native_master(src_path: str, full_48k: np.ndarray, switch_s: float,
                        out_path: str) -> dict:
    """Write the result at the source's own sample rate, bit-identical to the source decode.

    Samples before ``switch_s`` (the start of the first crossfade) are copied verbatim
    from ``src_path``; everything after is the 48 kHz result resampled to the source
    rate. Both sides of the switch carry the same original audio, so the switch is
    inaudible; the residual at the switch is reported.

    Returns:
        Dict with the path, sample rate, switch sample and switch residual in dB.
    """
    src, sr = sf.read(src_path, always_2d=True, dtype="float32")
    src = src[:, :2] if src.shape[1] >= 2 else np.repeat(src, 2, axis=1)
    if sr == SR:
        body = full_48k
    else:
        g = gcd(sr, SR)
        body = resample_poly(full_48k, sr // g, SR // g, axis=0).astype(np.float32)
    n = min(int(switch_s * sr), len(src))
    out = np.concatenate([src[:n], body[n:]], axis=0)
    w = slice(max(0, n - int(0.05 * sr)), n)
    resid = float(np.sqrt(np.mean(np.square(body[w] - src[w]))) + 1e-12)
    sf.write(out_path, out, sr, subtype="FLOAT")
    return {"path": out_path, "sample_rate": sr, "bit_identical_samples": n,
            "bit_identical_until_s": round(n / sr, 6),
            "switch_residual_db": round(20 * np.log10(resid), 1)}
