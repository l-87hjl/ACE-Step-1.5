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
