"""Lightweight tests for pure helper logic in the experimental analysis module."""
import unittest
import numpy as np

from forensics.adaptive_analysis import assess_audio_quality, window_starts


class ForensicHelperTests(unittest.TestCase):
    def test_window_starts_includes_last_window(self):
        starts = window_starts(n_samples=110, window_samples=40, hop_samples=30)
        self.assertEqual(starts, [0, 30, 60, 70])

    def test_short_input_has_one_window(self):
        self.assertEqual(window_starts(20, 40, 10), [0])

    def test_empty_audio_is_low_quality(self):
        quality = assess_audio_quality(np.array([], dtype=np.float32))
        self.assertEqual(quality["quality_score"], 0.0)

    def test_silence_is_low_quality(self):
        quality = assess_audio_quality(np.zeros(16000, dtype=np.float32))
        self.assertEqual(quality["quality_score"], 0.0)

    def test_clean_audio_quality_is_bounded(self):
        x = (0.2 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
        quality = assess_audio_quality(x)
        self.assertGreaterEqual(quality["quality_score"], 0.0)
        self.assertLessEqual(quality["quality_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
