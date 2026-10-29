"""Check separation scoring against analytic signal and permutation cases."""

import sys
import unittest
from pathlib import Path

import numpy as np

LOCAL = Path(__file__).resolve().parents[3] / "egs2/whamr/rir_1ch/local"
sys.path.insert(0, str(LOCAL))
import evaluate_mird_separation as MODULE  # noqa: E402
sys.path.pop(0)


class TestMIRDSeparationMetrics(unittest.TestCase):
    def test_analytic_orthogonal_error_ratio(self):
        reference = np.array([1., -1., 1., -1.])
        error = np.array([1., 1., -1., -1.])
        self.assertAlmostEqual(MODULE.si_sdr(reference, reference + error), 0.)
        self.assertAlmostEqual(MODULE.si_sdr(reference, reference + error / 2),
                               10 * np.log10(4))

    def test_gain_polarity_and_dc_do_not_change_si_sdr(self):
        reference = np.array([1., -1., 1., -1.])
        estimate = reference + np.array([1., 1., -1., -1.]) / 2
        expected = MODULE.si_sdr(reference, estimate)
        self.assertAlmostEqual(MODULE.si_sdr(reference * 100 + 3,
                                             estimate * -0.00001 - 0.2), expected)

    def test_best_permutation_and_mixture_baseline(self):
        first = np.array([1., -1., 1., -1.])
        second = np.array([1., 1., -1., -1.])
        result = MODULE.pit_scores([first, second], [second * 0.3, first * -2],
                                   first + second)
        self.assertEqual(result["permutation"], [1, 0])
        self.assertAlmostEqual(result["input_si_sdr_db"], 0.)
        self.assertAlmostEqual(result["si_sdr_db"], 120.)
        self.assertAlmostEqual(result["si_sdri_db"], 120.)

    def test_outputting_mixture_twice_has_zero_improvement(self):
        first = np.array([1., -1., 1., -1.])
        second = np.array([1., 1., -1., -1.]) * 0.25
        mix = first + second
        result = MODULE.pit_scores([first, second], [mix, mix], mix)
        self.assertAlmostEqual(result["si_sdri_db"], 0.)

    def test_no_time_alignment_is_fitted(self):
        reference = np.array([1., 0., -1., 0., 1., 0., -1., 0.])
        self.assertEqual(MODULE.si_sdr(reference, np.roll(reference, 1)), -120.)

    def test_invalid_reference_and_estimate_are_rejected(self):
        for reference, estimate in (([0, 0], [1, -1]), ([1, -1], [1]),
                                    ([1, -1], [float("nan"), 1])):
            with self.subTest(reference=reference, estimate=estimate):
                with self.assertRaises(ValueError):
                    MODULE.si_sdr(reference, estimate)
        self.assertEqual(MODULE.si_sdr([1, -1], [0, 0]), -120.)


if __name__ == "__main__":
    unittest.main()
