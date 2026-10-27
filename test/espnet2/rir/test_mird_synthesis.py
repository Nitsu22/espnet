"""Check the audio conventions of the MIRD two-speaker evaluation generator."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


LOCAL = Path(__file__).resolve().parents[3] / "egs2/whamr/rir_1ch/local"
SPEC = importlib.util.spec_from_file_location(
    "create_whamr_mird_2spk_dump", LOCAL / "create_whamr_mird_2spk_dump.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(LOCAL))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


class TestMIRDSynthesis(unittest.TestCase):
    def test_production_rejects_missing_native_condition(self):
        plan = {"records": MODULE.assignment.assign_pairs()}
        audit = {"all_conditions_valid": False, "archives": [
            dict(rt60_ms=rt, spacing=spacing,
                 valid=(rt, spacing) != (610, "3-3-3-8-3-3-3"))
            for rt in MODULE.assignment.RT60_MS
            for spacing in MODULE.assignment.SPACINGS
        ]}
        with self.assertRaisesRegex(ValueError, "all nine"):
            MODULE.expected_selection(plan, audit, None)
        audit["archives"][-3]["valid"] = True
        audit["all_conditions_valid"] = True
        selected = MODULE.expected_selection(plan, audit, None)
        self.assertEqual(len(selected), 3000)
        self.assertEqual([x["plan_index"] for x in selected], list(range(3000)))

    def test_smoke_uses_unchanged_plan_and_covers_every_available_condition(self):
        records = MODULE.assignment.assign_pairs()
        plan = {"records": records}
        audit = {"all_conditions_valid": False, "archives": [
            dict(rt60_ms=rt, spacing=spacing,
                 valid=(rt, spacing) != (610, "3-3-3-8-3-3-3"))
            for rt in MODULE.assignment.RT60_MS
            for spacing in MODULE.assignment.SPACINGS
        ]}
        selected = MODULE.expected_selection(plan, audit, 1)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({(x["rt60_ms"], x["spacing"]) for x in selected}), 8)
        for item in selected:
            index = item["plan_index"]
            self.assertEqual({k: v for k, v in item.items() if k != "plan_index"},
                             records[index])
            self.assertNotEqual((item["rt60_ms"], item["spacing"]),
                                (610, "3-3-3-8-3-3-3"))

    def test_rir_preserves_delay_sign_and_full_duration(self):
        wave = np.array([0.0, -0.75, 0.25, 0.0, 0.125, 0.0])
        h, peak, divisor = MODULE.process_rir(wave, 16000, 16000)
        expected = (wave / 0.75).astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(h, expected)
        self.assertEqual(h.dtype, np.dtype("float64"))
        self.assertEqual(len(h), len(wave))
        self.assertEqual(peak, 1)
        self.assertEqual(h[peak], -1.0)
        self.assertEqual(divisor, 0.75)

    def test_native_rir_resampling_preserves_time_at_both_rates(self):
        wave = np.zeros(480)
        wave[120] = -0.6
        wave[240] = 0.3
        for rate in (8000, 16000):
            with self.subTest(rate=rate):
                h, peak, divisor = MODULE.process_rir(wave, 48000, rate)
                self.assertEqual(len(h), rate // 100)
                self.assertEqual(peak, rate // 400)
                self.assertEqual(h[peak], -1.0)
                self.assertGreater(divisor, 0.0)
                np.testing.assert_array_equal(
                    h, h.astype(np.float32).astype(np.float64)
                )
                # The second arrival remains at twice the first arrival time.
                self.assertAlmostEqual(h[2 * peak], 0.5, places=6)

    def test_resampling_filters_instead_of_decimating(self):
        wave = np.zeros(480)
        wave[121] = 1.0
        h, peak, divisor = MODULE.process_rir(wave, 48000, 8000)
        self.assertEqual(len(h), 80)
        self.assertEqual(peak, 20)
        self.assertGreater(divisor, 0.0)
        self.assertGreater(np.count_nonzero(np.abs(h) > 1e-7), 5)

    def test_invalid_rirs_are_rejected(self):
        for wave in (np.zeros(12), np.array([]),
                     np.array([0.0, np.nan]), np.array([np.inf, 1.0])):
            with self.subTest(wave=wave):
                with self.assertRaises(ValueError):
                    MODULE.process_rir(wave, 48000, 16000)

    def test_convolution_uses_min_speech_length_and_signed_delayed_direct(self):
        originals = [np.array([0.125, 0.25, -0.125, 0.375, 0.2]),
                     np.array([0.25, -0.125, 0.125])]
        h_pair = [np.array([0.0, -1.0, 0.25]),
                  np.array([0.5, 0.0, 1.0])]
        noise = np.array([99.0, 0.03125, -0.0625, 0.125, 99.0])
        signals, joint_gain, peaks = MODULE.synthesize_signals(
            originals, noise, h_pair, [2.0, 0.5], 0.75, 0.5, 1
        )
        source1 = np.array([0.1875, 0.375, -0.1875])
        source2 = np.array([0.09375, -0.046875, 0.046875])
        expected = {
            "source1": source1,
            "source2": source2,
            "direct1": np.array([0.0, -0.1875, -0.375]),
            "direct2": np.array([0.0, 0.0, 0.09375]),
            "reverb1": np.convolve(source1, h_pair[0])[:3],
            "reverb2": np.convolve(source2, h_pair[1])[:3],
            "noise": np.array([0.015625, -0.03125, 0.0625]),
        }
        expected["mix_clean"] = expected["reverb1"] + expected["reverb2"]
        expected["mix_both"] = expected["mix_clean"] + expected["noise"]
        self.assertEqual(set(signals), set(expected))
        self.assertEqual(joint_gain, 1.0)
        self.assertEqual(list(peaks), [1, 2])
        for name, value in expected.items():
            with self.subTest(signal=name):
                self.assertEqual(len(signals[name]), 3)
                np.testing.assert_allclose(signals[name], value, atol=1e-15)

    def test_wsj_gain_precedes_quantization_and_shared_speech_gain_follows(self):
        originals = [np.array([0.51, 0.49, -0.51, -0.49]) / 32768,
                     np.array([1.51, 2.49, -1.51, -2.49]) / 32768]
        signals, joint_gain, _ = MODULE.synthesize_signals(
            originals, np.zeros(4), [np.ones(1), np.ones(1)],
            [1.7, 0.7], 0.3, 0.0, 0
        )
        self.assertEqual(joint_gain, 1.0)
        np.testing.assert_array_equal(
            signals["source1"], np.array([1, 1, -1, -1]) / 32768 * 0.3
        )
        np.testing.assert_array_equal(
            signals["source2"], np.array([1, 2, -1, -2]) / 32768 * 0.3
        )

    def test_one_attenuation_applies_to_all_targets_and_both_mixtures(self):
        originals = [np.full(4, 0.5), np.full(4, 0.25)]
        signals, gain, _ = MODULE.synthesize_signals(
            originals, np.full(4, 0.25), [np.ones(2), np.ones(2)],
            [1.0, 1.0], 1.0, 1.0, 0
        )
        # The largest unattenuated signal is the noisy mixture, peaking at 1.75.
        self.assertAlmostEqual(gain, 0.99 / 1.75)
        for name, value in (("source1", 0.5), ("source2", 0.25),
                            ("direct1", 0.5), ("direct2", 0.25),
                            ("noise", 0.25)):
            with self.subTest(signal=name):
                np.testing.assert_allclose(signals[name], np.full(4, value * gain))
        np.testing.assert_allclose(signals["reverb1"],
                                   np.array([0.5, 1.0, 1.0, 1.0]) * gain)
        np.testing.assert_allclose(signals["reverb2"],
                                   np.array([0.25, 0.5, 0.5, 0.5]) * gain)
        np.testing.assert_allclose(signals["mix_clean"],
                                   signals["reverb1"] + signals["reverb2"])
        np.testing.assert_allclose(signals["mix_both"],
                                   signals["mix_clean"] + signals["noise"])
        self.assertAlmostEqual(max(np.max(np.abs(x)) for x in signals.values()),
                               0.99)

    def test_attenuation_includes_clean_sources_when_receiver_cancels(self):
        signals, gain, _ = MODULE.synthesize_signals(
            [np.full(3, 0.75), np.full(3, 0.75)], np.zeros(3),
            [np.ones(1), -np.ones(1)], [1.0, 1.0], 2.0, 0.0, 0
        )
        self.assertAlmostEqual(gain, 0.99 / 1.5)
        np.testing.assert_allclose(signals["source1"], np.full(3, 0.99))
        np.testing.assert_allclose(signals["direct2"], np.full(3, -0.99))
        np.testing.assert_allclose(signals["mix_clean"], np.zeros(3))

    def test_direct_arrival_after_observation_window_is_silent(self):
        h = np.array([0.0, 0.0, 0.0, -1.0])
        signals, _, peaks = MODULE.synthesize_signals(
            [np.full(2, 0.125), np.full(2, 0.25)], np.zeros(2),
            [h, h], [1.0, 1.0], 1.0, 0.0, 0
        )
        self.assertEqual(list(peaks), [3, 3])
        for name in ("direct1", "direct2", "reverb1", "reverb2", "mix_clean"):
            np.testing.assert_allclose(signals[name], np.zeros(2), atol=1e-15)

    def test_invalid_noise_window_and_nonfinite_noise_are_rejected(self):
        for noise, start in ((np.zeros(2), 0), (np.zeros(3), 1),
                             (np.zeros(6), -1),
                             (np.array([0.0, np.nan, 0.0]), 0)):
            with self.subTest(noise=noise, start=start):
                with self.assertRaises(ValueError):
                    MODULE.synthesize_signals(
                        [np.zeros(3), np.zeros(3)], noise,
                        [np.ones(1), np.ones(1)], [1.0, 1.0], 1.0, 1.0, start
                    )


if __name__ == "__main__":
    unittest.main()
