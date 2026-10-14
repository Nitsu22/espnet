"""LS reference numerical correctness, measurement basis, and shared PIT."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import linalg, signal

from espnet2.rir.ls_rir import (
    LSRIRConfig,
    LSRIRFitter,
    ordinary_sweep_measurement,
    solve_full_convolution_fir,
)

LOCAL = Path(__file__).resolve().parents[3] / "egs2/whamr/rir_2spk/local"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, LOCAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLSRIR(unittest.TestCase):
    def test_fft_cg_matches_dense_finite_linear_convolution(self):
        rng = np.random.default_rng(4)
        for nx, ny, taps in ((19, 13, 7), (3, 20, 9)):
            for alpha in (1e-6, 0.03):
                x, y = rng.normal(size=nx), rng.normal(size=ny)
                actual, metrics = solve_full_convolution_fir(
                    x, y, taps, alpha, rtol=1e-11
                )
                matrix = linalg.convolution_matrix(x, taps, mode="full")
                frames = max(len(matrix), len(y))
                matrix = np.pad(matrix, ((0, frames - len(matrix)), (0, 0)))
                target = np.pad(y, (0, frames - len(y)))
                expected = linalg.solve(
                    matrix.T @ matrix + alpha * np.dot(x, x) * np.eye(taps),
                    matrix.T @ target,
                    assume_a="pos",
                )
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)
                error = np.linalg.norm(matrix @ actual - target) ** 2
                self.assertAlmostEqual(
                    metrics["relative_response_mse"], error / np.dot(y, y), places=12
                )

    def test_known_filter_recovery_and_common_gain_invariance(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        target = np.array([1, 0, -0.4, 0, 0.2])
        y = signal.convolve(x, target)
        actual, _ = solve_full_convolution_fir(x, y, 12)
        np.testing.assert_allclose(actual, np.pad(target, (0, 7)), atol=2e-6)
        for scale in (0.01, -3):
            rescaled, _ = solve_full_convolution_fir(x * scale, y * scale, 12)
            np.testing.assert_allclose(rescaled, actual, atol=1e-12)

    def test_unconverged_fit_is_not_published(self):
        rng = np.random.default_rng(32)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            solve_full_convolution_fir(
                rng.normal(size=150),
                rng.normal(size=140),
                80,
                rtol=1e-13,
                max_iterations=1,
            )

    def test_measurement_uses_ordinary_sweep_not_direct_sweep(self):
        config = LSRIRConfig(
            sample_rate=8000,
            rir_length=128,
            filter_length=128,
            sweep_duration=0.064,
            regularization=1e-10,
            solver_rtol=1e-10,
        )
        fitter = LSRIRFitter(config)
        g = np.zeros(128)
        g[[0, 20, 50]] = [1, -0.25, 0.15]
        direct = np.zeros(30)
        direct[[12, 13]] = [0.3, 0.18]
        reverb = signal.convolve(direct, g)[:128]
        actual, reference, metrics = fitter.fit(direct, reverb)
        expected, _ = ordinary_sweep_measurement(
            g, fitter.sweep, fitter.inverse, 8000, 128
        )
        np.testing.assert_allclose(actual, g, atol=3e-4)
        np.testing.assert_allclose(reference, expected, atol=3e-4)
        # Applying the direct path again restores clean-reverb and changes shape.
        restored, _ = ordinary_sweep_measurement(
            signal.convolve(direct, g), fitter.sweep, fitter.inverse, 8000, 128
        )
        assert (
            np.max(
                np.abs(
                    reference / np.max(np.abs(reference))
                    - restored / np.max(np.abs(restored))
                )
            )
            > 0.2
        )
        assert metrics["relative_response_mse"] < 1e-7

    def test_reference_identity_matches_existing_pim(self):
        import torch

        from espnet2.rir.rec_rir.feature import RecRIRTransforms
        from espnet2.rir.rec_rir.pim import RecRIRPIM

        pim = RecRIRPIM(8000, 0.064)
        transforms = RecRIRTransforms(
            sr=8000, n_fft=256, hop_len=128, win_type="sqrthann", win_len=256
        )
        ordinary, _ = ordinary_sweep_measurement(
            [1], pim.sinesweep.numpy(), pim.invfilter.numpy(), 8000, 128
        )
        actual = pim.ctf_to_rir(
            torch.ones(129, 1, dtype=torch.complex64),
            transforms,
            torch.device("cpu"),
            128,
        ).numpy()
        np.testing.assert_allclose(actual, ordinary, atol=3e-6, rtol=3e-5)

    def test_frame_aligned_echoes_match_existing_pim(self):
        import torch

        from espnet2.rir.rec_rir.feature import RecRIRTransforms
        from espnet2.rir.rec_rir.pim import RecRIRPIM

        pim = RecRIRPIM(8000, 0.064)
        transform = RecRIRTransforms(
            sr=8000, n_fft=256, hop_len=128, win_type="sqrthann", win_len=256
        )
        g = np.zeros(512)
        g[[0, 128, 256]] = [0.7, 0.2, -0.1]
        expected, _ = ordinary_sweep_measurement(
            g, pim.sinesweep.numpy(), pim.invfilter.numpy(), 8000, 512
        )
        # PIM's sliding STFT window takes taps in reverse causal order.
        ctf = torch.tensor([-0.1, 0.2, 0.7], dtype=torch.complex64).repeat(129, 1)
        actual = pim.ctf_to_rir(ctf, transform, torch.device("cpu"), 512).numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)

    def test_dump_and_swapped_prediction_pit(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        sys.path.insert(0, str(LOCAL))
        self.addCleanup(lambda: sys.path.remove(str(LOCAL)))
        prepare = load_script("prepare_ls_rir_reference")
        scorer = load_script("evaluate_ls_direct_reverb")
        source, output = tmp_path / "source", tmp_path / "references"
        source.mkdir()
        config = LSRIRConfig(
            sample_rate=8000, rir_length=128, filter_length=128, sweep_duration=0.064
        )
        originals = {}
        for kind in ("direct", "ref"):
            for speaker in (1, 2):
                wave = np.zeros((80, 2))
                wave[12, :] = 0.25
                if kind == "ref":
                    wave[12 + speaker * 20, :] = -0.08
                path = source / f"{kind}{speaker}.wav"
                sf.write(path, wave, 8000, subtype="FLOAT")
                (source / f"rir_{kind}{speaker}.scp").write_text(f"sample {path}\n")
                originals[path] = path.read_bytes()
        report = prepare.prepare(source, output, config, workers=2)
        assert report["count"] == 1 and report["full_dataset"]
        assert prepare.validate_dump(output) == report
        assert all(path.read_bytes() == payload for path, payload in originals.items())
        with self.assertRaisesRegex(ValueError, "new evaluation-only"):
            prepare.prepare(source, output, config)
        prediction = tmp_path / "prediction.scp"
        rows = (output / "reference.scp").read_text().split()
        prediction.write_text(f"sample {rows[2]} {rows[1]}\n")
        results = scorer.evaluate(output, [("swapped", prediction)], tmp_path / "score")
        summary = results["models"]["swapped"]["metrics"]
        self.assertEqual(summary["rmse_mean"], 0)
        self.assertAlmostEqual(summary["corr_mean"], 1)
        per_source = (tmp_path / "score/models/00/per_source.csv").read_text()
        assert ",1,0,1," in per_source and ",1,1,0," in per_source
        assert (tmp_path / "score/evaluation_complete.json").is_file()
        json_only = scorer.evaluate(
            output, [("swapped", prediction)], tmp_path / "json_only",
            write_comparison_csv=False,
        )
        self.assertEqual(json_only["models"]["swapped"]["metrics"], summary)
        assert not (tmp_path / "json_only/comparison.csv").exists()
        assert (tmp_path / "json_only/evaluation_complete.json").is_file()
        # A damaged reference must prevent publication/scoring.
        reference = Path(rows[1])
        reference.write_bytes(reference.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            scorer.evaluate(output, [("swapped", prediction)], tmp_path / "bad_score")
        assert not (tmp_path / "bad_score").exists()

    def test_wrong_sample_rate_is_rejected(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        prepare = load_script("prepare_ls_rir_reference")
        source = tmp_path / "source"
        source.mkdir()
        for kind in ("direct", "ref"):
            for speaker in (1, 2):
                path = source / f"{kind}{speaker}.wav"
                sf.write(path, [0.1, 0.2], 16000, subtype="FLOAT")
                (source / f"rir_{kind}{speaker}.scp").write_text(f"sample {path}\n")
        with self.assertRaisesRegex(ValueError, "Sample rate/channel"):
            prepare.prepare(source, tmp_path / "invalid", LSRIRConfig(sample_rate=8000))
        assert not (tmp_path / "invalid").exists()
        assert not list(tmp_path.glob(".invalid.preparing-*"))

    def test_limited_reference_needs_explicit_smoke_mode(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        prepare = load_script("prepare_ls_rir_reference")
        source = root / "source"
        source.mkdir()
        for kind in ("direct", "ref"):
            for speaker in (1, 2):
                path = source / f"{kind}{speaker}.wav"
                sf.write(path, [0.1, 0.2], 8000, subtype="FLOAT")
                (source / f"rir_{kind}{speaker}.scp").write_text(
                    f"first {path}\nsecond {path}\n"
                )
        config = LSRIRConfig(
            sample_rate=8000, rir_length=128, filter_length=128, sweep_duration=0.064
        )
        report = prepare.prepare(source, root / "limited", config, limit=1)
        assert report["count"] == 1 and not report["full_dataset"]
        with self.assertRaisesRegex(ValueError, "Limited/smoke"):
            prepare.validate_dump(root / "limited")
        prepare.validate_dump(root / "limited", allow_limited=True)


if __name__ == "__main__":
    unittest.main()
