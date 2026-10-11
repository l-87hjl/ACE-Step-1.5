"""Unit tests for plan helpers and the native-rate master (numpy only, no models)."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import beat_grid, load_lyrics_plan, match_gain, unsung_times  # noqa: E402
from runtime import write_native_master  # noqa: E402


class PlanTest(unittest.TestCase):
    """Lyric plans, beat grids, phrase protection and gain matching."""

    def test_beat_grid_is_inside_window_and_on_phase(self):
        b = beat_grid(90.0, 0.13, 10.0, 14.0)
        self.assertTrue(np.all((b >= 10.0) & (b < 14.0)))
        np.testing.assert_allclose(((b - 0.13) / (60 / 90.0)) % 1.0, 0.0, atol=1e-9)

    def test_unsung_times_drops_beats_inside_words(self):
        t = np.array([1.0, 2.0, 3.0])
        out = unsung_times(t, [{"s": 1.9, "e": 2.5}], margin_s=0.05)
        np.testing.assert_array_equal(out, [1.0, 3.0])

    def test_unsung_times_without_words_is_identity(self):
        t = np.array([1.0, 2.0])
        np.testing.assert_array_equal(unsung_times(t, None), t)

    def test_match_gain_matches_and_clips(self):
        ref, cand = np.full(100, 0.5), np.full(100, 0.25)
        self.assertAlmostEqual(match_gain(ref, cand, 10.0), 2.0, places=6)
        self.assertAlmostEqual(20 * np.log10(match_gain(ref, cand, 3.0)), 3.0, places=6)
        self.assertEqual(match_gain(ref, cand, 0.0), 1.0)
        self.assertEqual(match_gain(ref, np.zeros(100), 3.0), 1.0)

    def test_lyrics_plan_validation(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"passes": [{"segment": 40, "lyrics": "[Verse]"}, {"segment": 0}]}, f)
        with self.assertRaises(ValueError):
            load_lyrics_plan(f.name)
        os.unlink(f.name)


class NativeMasterTest(unittest.TestCase):
    """write_native_master keeps the source decode bit-exact before the switch."""

    def test_prefix_bit_identical_at_native_rate(self):
        sr = 44100
        t = np.arange(sr * 2) / sr
        src = np.stack([np.sin(2 * np.pi * 220 * t)] * 2, 1).astype(np.float32) * 0.5
        tmp = tempfile.mkdtemp()
        sp = os.path.join(tmp, "src.wav")
        sf.write(sp, src, sr, subtype="FLOAT")
        t48 = np.arange(48000 * 3) / 48000
        full = np.stack([np.sin(2 * np.pi * 220 * t48)] * 2, 1).astype(np.float32) * 0.5
        info = write_native_master(sp, full, 1.5, os.path.join(tmp, "out.wav"))
        out, osr = sf.read(info["path"], always_2d=True, dtype="float32")
        self.assertEqual(osr, sr)
        n = info["bit_identical_samples"]
        np.testing.assert_array_equal(out[:n], src[:n])
        self.assertLess(info["switch_residual_db"], -40)


if __name__ == "__main__":
    unittest.main()
