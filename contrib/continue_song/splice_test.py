"""Unit tests for the continue_song splice helpers (numpy only, no models)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splice import choose_splice_sample, effective_end_s, equal_power_splice  # noqa: E402

SR = 8000


def _tone(freq: float, secs: float) -> np.ndarray:
    t = np.arange(int(secs * SR)) / SR
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], 1)


class EqualPowerSpliceTest(unittest.TestCase):
    """equal_power_splice keeps the original exact before the fade."""

    def test_original_prefix_is_bit_exact(self):
        orig, gen = _tone(220, 2.0), _tone(330, 4.0)
        out = equal_power_splice(orig, gen, cut=SR, fade_samples=400)
        np.testing.assert_array_equal(out[: SR - 200], orig[: SR - 200])
        np.testing.assert_array_equal(out[SR + 200:], gen[SR + 200:])
        self.assertEqual(len(out), len(gen))

    def test_hard_cut_when_fade_is_zero(self):
        orig, gen = np.ones((100, 2)), np.zeros((150, 2))
        out = equal_power_splice(orig, gen, cut=60, fade_samples=0)
        self.assertTrue(np.all(out[:60] == 1) and np.all(out[60:] == 0))

    def test_cut_beyond_original_is_clamped(self):
        orig, gen = np.ones((100, 2)), np.zeros((150, 2))
        out = equal_power_splice(orig, gen, cut=140, fade_samples=20)
        self.assertTrue(np.all(out[:100] == 1) and np.all(out[100:] == 0))


class ChooseSpliceTest(unittest.TestCase):
    """choose_splice_sample finds where two same-timeline signals agree."""

    def test_finds_region_where_signals_match(self):
        orig = _tone(220, 3.0)
        gen = _tone(440, 3.0)
        gen[int(2.0 * SR):int(2.4 * SR)] = orig[int(2.0 * SR):int(2.4 * SR)]
        cut = choose_splice_sample(orig, gen, SR, 1.0, 3.0, hop_s=0.02)
        self.assertTrue(1.9 * SR <= cut <= 2.4 * SR, cut / SR)

    def test_restricts_to_beats_in_window(self):
        orig, gen = _tone(220, 3.0), _tone(220, 3.0)
        cut = choose_splice_sample(orig, gen, SR, 1.0, 3.0, beat_times_s=np.array([0.5, 1.5, 2.5]))
        self.assertIn(round(cut / SR, 2), (1.5, 2.5))

    def test_degenerate_window_returns_end(self):
        orig, gen = _tone(220, 2.0), _tone(220, 3.0)
        self.assertEqual(choose_splice_sample(orig, gen, SR, 2.0, 2.0), 2 * SR)


class EffectiveEndTest(unittest.TestCase):
    """effective_end_s ignores a trailing fade but not a full-length clip."""

    def test_trailing_fade_is_excluded(self):
        x = _tone(220, 10.0)
        x[int(8.0 * SR):] *= np.linspace(1, 0, len(x) - int(8.0 * SR))[:, None] ** 4
        self.assertTrue(7.5 <= effective_end_s(x, SR) <= 9.0)

    def test_full_length_clip_returns_length(self):
        self.assertAlmostEqual(effective_end_s(_tone(220, 10.0), SR), 10.0, places=1)


if __name__ == "__main__":
    unittest.main()
