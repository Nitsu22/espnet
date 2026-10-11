"""Numerical CTF identification and immutable dump/correspondence checks."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import tempfile
import unittest
from scipy import linalg
import soundfile as sf

from espnet2.rir.oracle_ctf import (
    OracleCTFConfig, OracleCTFFitter, normalize_rir_pair, pack_ctf,
    solve_full_convolution_ctf,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "prepare_oracle", ROOT / "egs2/whamr/enh9_condition/local/prepare_oracle_ctf_dump.py")
prepare_oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_oracle)


def test_toeplitz_solver_matches_dense_full_convolution():
    for regularization in (0.0, 0.003):
        for lengths, taps in (((23, 19), 7), ((3, 2), 8)):
            rng = np.random.default_rng(13)
            x = rng.normal(size=(3, lengths[0])) + 1j * rng.normal(size=(3, lengths[0]))
            y = rng.normal(size=(3, lengths[1])) + 1j * rng.normal(size=(3, lengths[1]))
            actual, _ = solve_full_convolution_ctf(x, y, taps, regularization)
            for frequency in range(len(x)):
                matrix = linalg.convolution_matrix(x[frequency], taps, mode="full")
                frames = max(len(matrix), len(y[frequency]))
                matrix = np.pad(matrix, ((0, frames - len(matrix)), (0, 0)))
                target = np.pad(y[frequency], (0, frames - len(y[frequency])))
                gram = matrix.conj().T @ matrix
                gram += np.eye(taps) * regularization * np.sum(np.abs(x[frequency]) ** 2)
                expected = linalg.solve(gram, matrix.conj().T @ target, assume_a="her")
                np.testing.assert_allclose(actual[frequency], expected, rtol=1e-10, atol=1e-10)


def test_known_filter_causal_order_and_unexcited_bin():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 40)) + 1j * rng.normal(size=(2, 40))
    x[1] = 0
    expected = np.array([1.4 + 0.2j, 0, -0.35j, 0.17])
    y = np.stack([np.convolve(row, expected) for row in x])
    actual, metric = solve_full_convolution_ctf(x, y, len(expected), 0)
    np.testing.assert_allclose(actual[0], expected, atol=1e-12)
    np.testing.assert_array_equal(actual[1], 0)
    assert metric["relative_mse"] < 1e-25
    assert metric["zero_excitation_bins"] == 1
    packed = pack_ctf(np.stack((actual, actual)))
    assert packed.shape == (4, 2, 2, 2)
    assert packed.dtype == np.float32
    np.testing.assert_allclose(packed[0, 0, 0], [1.4, 0.2], atol=1e-7)


def test_rir_pair_shared_scale_preserves_delay_and_tail():
    direct = np.array([0, 0, 2, 0])
    reverb = np.array([0, 0, 2, 1, 0.5])
    normalized = normalize_rir_pair(direct, reverb, 8)
    np.testing.assert_array_equal(normalized[0], [0, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(normalized[1], [0, 0, 1, .5, .25, 0, 0, 0])
    with unittest.TestCase().assertRaisesRegex(ValueError, "tail"):
        normalize_rir_pair(direct, reverb, 4)


def test_physical_sweep_identity_gain_and_full_tail():
    config = OracleCTFConfig(regularization=0)
    fitter = OracleCTFFitter(config)
    direct = np.zeros(33)
    direct[17] = .25
    actual, metric = fitter.fit(direct, direct * 1.8)
    np.testing.assert_allclose(actual[:, 0], 1.8, atol=2e-4)
    np.testing.assert_allclose(actual[:, 1:], 0, atol=2e-4)
    assert metric["relative_mse"] < 1e-11
    spectrum = fitter.response_spectrum(np.pad(direct, (0, config.rir_length - len(direct))))
    full_samples = fitter.sweep.numel() + config.rir_length - 1 + 2 * config.n_fft
    assert spectrum.shape == (129, full_samples // config.hop_length + 1)


def test_quantization_clipping_is_explained_without_gain_or_delay_fit():
    candidate = np.array([-1.2, -.111111, .123456, 1.4])
    baseline = np.clip(np.floor(candidate * 32768), -32768, 32767) / 32768
    metric = prepare_oracle.compare_audio(baseline, candidate, "PCM_16")
    assert metric["clipped_samples"] == 2
    assert metric["maximum_residual"] == 0
    with unittest.TestCase().assertRaisesRegex(ValueError, "not explained"):
        prepare_oracle.compare_audio(baseline * .8, candidate, "PCM_16")


def make_fixture(root):
    baseline, rir_recipe = root / "baseline", root / "rir"
    dataset = "tr_mix_clean_reverb_min_8k"
    folder = baseline / "dump/raw" / dataset
    original = baseline / "data" / dataset
    rir = rir_recipe / "dump/raw/tr_rir_2spk_nf_min_8k"
    for directory in (folder, original, rir):
        directory.mkdir(parents=True)
    uid, rid = "speaker1_speaker2_example_reverb", "example"
    wave = np.sin(np.arange(400) * .13) * .11
    names = {"wav.scp": wave * 2, "spk1.scp": wave, "spk2.scp": wave * .5}
    candidate_names = {"wav.scp": "wav.scp", "spk1.scp": "speech_direct1.scp",
                       "spk2.scp": "speech_direct2.scp"}
    for name, signal in names.items():
        path = baseline / (name + ".wav")
        sf.write(path, signal, 8000, subtype="PCM_16")
        (folder / name).write_text(f"{uid} {path}\n")
        source = rir_recipe / (candidate_names[name] + ".wav")
        sf.write(source, signal, 8000, subtype="FLOAT")
        (rir / candidate_names[name]).write_text(f"{rid} {source}\n")
    for speaker in (1, 2):
        reverb = original / f"speaker{speaker}" / (rid + ".wav")
        reverb.parent.mkdir()
        sf.write(reverb, wave * (2 / speaker), 8000, subtype="FLOAT")
        (original / f"spk{speaker}_reverb.scp").write_text(f"{uid} {reverb}\n")
        (rir / f"speech_reverb{speaker}.scp").write_text(f"{rid} {reverb}\n")
        for kind in ("direct", "ref"):
            response = np.zeros(50)
            response[11] = .2
            if kind == "ref":
                response[19] = .06
            path = rir_recipe / f"rir_{kind}{speaker}.wav"
            sf.write(path, response, 8000, subtype="FLOAT")
            (rir / f"rir_{kind}{speaker}.scp").write_text(f"{rid} {path}\n")
    return baseline, rir_recipe, uid


def test_dump_publication_validation_relocation_and_corruption(tmp_path):
    baseline, rir, uid = make_fixture(tmp_path)
    output = tmp_path / "oracle"
    kwargs = dict(baseline_dump=baseline / "dump", baseline_recipe=baseline,
                  baseline_reverb_data=baseline / "data", rir_dump=rir / "dump",
                  rir_recipe=rir, output=output, limit=1, workers=1,
                  expected_counts={"tr": 1})
    prepare_oracle.prepare(mode="verify", **kwargs)
    assert not output.exists()
    assert output.with_name("oracle.verification.json").is_file()
    report = prepare_oracle.prepare(mode="prepare", **kwargs)
    feature = output / "features/tr" / (uid + ".npy")
    assert np.load(feature).shape == (60, 2, 129, 2)
    assert report["spec"]["tap_order"] == "causal"
    prepare_oracle.validate_prepared_dump(output, require_full=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, "limited/smoke"):
        prepare_oracle.validate_prepared_dump(output)
    # RIR provenance is retained, but original RIRs need not exist for training.
    import shutil
    shutil.rmtree(rir)
    prepare_oracle.validate_prepared_dump(output, require_full=False)
    portable = tmp_path / "portable"
    prepare_oracle.relocate(output, baseline / "dump", baseline, portable)
    prepare_oracle.validate_prepared_dump(portable, require_full=False)
    previous_directory = Path.cwd()
    try:
        os.chdir(tmp_path)
        prepare_oracle.validate_prepared_dump(Path("oracle"), require_full=False)
        prepare_oracle.validate_prepared_dump(Path("portable"), require_full=False)
    finally:
        os.chdir(previous_directory)
    assert json.loads((portable / "preparation.json").read_text())["imported_manifest_sha256"]
    feature.write_bytes(feature.read_bytes() + b"corrupt")
    with unittest.TestCase().assertRaisesRegex(ValueError, "checksum"):
        prepare_oracle.validate_prepared_dump(output, require_full=False)


def test_material_correspondence_error_prevents_features(tmp_path):
    baseline, rir, _ = make_fixture(tmp_path)
    path = rir / "speech_direct1.scp.wav"
    sf.write(path, np.ones(400) * .3, 8000, subtype="FLOAT")
    output = tmp_path / "oracle"
    with unittest.TestCase().assertRaisesRegex(ValueError, "not explained"):
        prepare_oracle.prepare(baseline / "dump", baseline, baseline / "data",
                              rir / "dump", rir, output, "prepare", limit=1,
                              workers=1, expected_counts={"tr": 1})
    assert not output.exists()


if __name__ == "__main__":
    def fixture_test(function):
        def run():
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))
        return run
    cases = [unittest.FunctionTestCase(fixture_test(function) if function.__code__.co_argcount
                                      else function)
             for name, function in list(globals().items()) if name.startswith("test_")]
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases)).wasSuccessful())
