"""CPU-only regression tests for paired, mixture-level subgroup aggregation."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[3]
LOCAL = REPO / "egs2/whamr/enh_rir/local"
sys.path.insert(0, str(LOCAL))
spec = importlib.util.spec_from_file_location("concat_groups", LOCAL / "analyze_concat_condition_groups.py")
groups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groups)


class TestConcatGroups(unittest.TestCase):
    def test_paired_gain_and_mixture_count(self):
        a = {metric: np.array([10., 20., 30.]) for metric in groups.METRICS}
        b = {metric: values + np.array([1., 2., 100.]) for metric, values in a.items()}
        result = groups.summarize(a, b, np.array([True, True, False]), np.random.default_rng(0), 1000)
        self.assertEqual(result["n_mixtures"], 2)
        self.assertEqual(result["baseline_SI_SNR"], 15.)
        self.assertEqual(result["concat_SI_SNR"], 16.5)
        self.assertEqual(result["gain_SI_SNR"], 1.5)
        self.assertGreater(result["gain_ci_low"], 0.)
        repeated = groups.summarize(a, b, np.array([True, True, False]), np.random.default_rng(0), 1000)
        self.assertEqual(result, repeated)

    def test_empty_group_rejected(self):
        a = {metric: np.array([1.]) for metric in groups.METRICS}
        with self.assertRaisesRegex(ValueError, "Empty subgroup"):
            groups.summarize(a, a, np.array([False]), np.random.default_rng(0), 10)

    def test_speaker_average_and_key_mismatch(self):
        with tempfile.TemporaryDirectory(prefix=".concat_groups_test_", dir=REPO) as directory:
            root = Path(directory)
            for metric in groups.METRICS:
                for speaker in (1, 2):
                    (root / f"{metric}_spk{speaker}").write_text(f"a {speaker}\nb {speaker+2}\n")
            values = groups.load_scores(root, ["b", "a"])
            np.testing.assert_allclose(values["SI_SNR"], [3.5, 1.5])
            (root / "SI_SNR_spk2").write_text("a 2\n")
            with self.assertRaisesRegex(ValueError, "Score key mismatch"):
                groups.load_scores(root, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
