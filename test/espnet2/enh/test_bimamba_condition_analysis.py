"""Standard-library unittest entry point for the offline condition analysis."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "egs2/whamr/enh_rir/local/analyze_bimamba_condition_gains.py"
spec = importlib.util.spec_from_file_location("condition_analysis", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConditionAnalysisTest(unittest.TestCase):
    def test_uid(self):
        self.assertEqual(module.canonical_uid("050_051_050a050m_1.6347_051c010x_-1.6347_reverb"), "050a050m_1.6347_051c010x_-1.6347.wav")
        with self.assertRaises(ValueError):
            module.canonical_uid("999_051_050a050m_1.6347_051c010x_-1.6347_reverb")

    def test_boundaries(self):
        np.testing.assert_array_equal(module.bin_index([.1, .3, .6, 1.0], [.1, .3, .6, 1.0]), [0, 1, 2, 2])
        with self.assertRaises(ValueError):
            module.bin_index([1.01], [.1, 1.])

    def test_geometry(self):
        row = dict(room_x=5, room_y=6, room_z=3, micL_x=2, micL_y=2, mic_z=1,
                   s1_x=3, s1_y=2, s1_z=1, s2_x=2, s2_y=3, s2_z=1, T60=.5)
        result = module.geometry(row)
        self.assertAlmostEqual(result["volume_m3"], 90)
        self.assertAlmostEqual(result["doa_separation_deg"], 90)
        self.assertAlmostEqual(result["mean_distance_m"], 1)
        self.assertAlmostEqual(result["distance_gap_m"], 0)
        self.assertAlmostEqual(result["source_separation_m"], np.sqrt(2))

    def test_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "score"
            path.write_text("uid 1\nuid 2\n")
            with self.assertRaises(ValueError):
                module.read_scores(path)
            path.write_text("uid nan\n")
            with self.assertRaises(ValueError):
                module.read_scores(path)

    def test_bootstrap_repeatability(self):
        a = module.bootstrap_means([1, 2, 3], np.random.default_rng(0), 100)
        b = module.bootstrap_means([1, 2, 3], np.random.default_rng(0), 100)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(module.bootstrap_means([.2] * 5, np.random.default_rng(0), 100), [.2] * 100)

    def test_bh(self):
        np.testing.assert_allclose(module.bh_adjust([.01, .04, .03]), [.03, .04, .04])


if __name__ == "__main__":
    unittest.main()
