"""Physical reuse, exclusion, and self-contained paired-dump regression tests."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import unittest
from unittest.mock import patch
import tempfile
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "egs2/whamr/rir_2spk/local/prepare_room_path_contrastive_dump.py"
spec = importlib.util.spec_from_file_location("paired_dump_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def check_convolution_crops_after_filtering():
    dry = np.zeros(24)
    dry[3] = 2.0
    rir = np.zeros((8, 2))
    rir[5] = [0.5, 0.25]
    result = generator.convolve_crop(dry, rir, start=6, samples=8)
    assert np.allclose(result[2], [1.0, 0.5])
    assert np.allclose(result.sum(0), [1.0, 0.5])
    assert not np.any(dry[6:14])  # Cropping speech before filtering loses this tail.


def check_selection_excludes_original_and_short_sources():
    pool = [{"path": str(i), "speaker": str(i), "frames": n}
            for i, n in enumerate([100, 100, 100, 80, 100])]
    selected = generator.select_dry(pool, {"0", "1"}, 100, np.random.default_rng(0))
    assert {v["speaker"] for v in selected} == {"2", "4"}
    with unittest.TestCase().assertRaisesRegex(ValueError, "Fewer than two"):
        generator.select_dry(pool, {"0", "1", "4"}, 100, np.random.default_rng(0))


def check_new_source_energy_calibration_and_stereo_shared_gain():
    rng = np.random.default_rng(1)
    samples = 1000
    dry = [rng.normal(size=samples + 200), rng.normal(size=samples + 200)]
    direct = [np.array([[0.2, 0.1]]), np.array([[0.5, 0.25]])]
    reverbs = [h.copy() for h in direct]
    refs = [np.ones((samples, 2)) * 0.03, np.ones((samples, 2)) * 0.01]
    mix_reference = np.ones((samples, 2)) * 0.025
    out, gains = generator.synthesize_view(dry, direct, reverbs, refs, mix_reference, [0, 10])
    assert np.allclose(out[:, 1], out[:, 0] * 0.5)
    np.testing.assert_allclose(np.sqrt(np.mean(out[:, 0] ** 2)), 0.025, rtol=1e-6)
    before_mix_scale = []
    for x, h, s, g in zip(dry, direct, [0, 10], gains["direct_energy_gains"]):
        before_mix_scale.append(np.sqrt(np.mean((generator.convolve_crop(x, h, s, samples)[:, 0] * g) ** 2)))
    np.testing.assert_allclose(before_mix_scale, [0.03, 0.01])


def fixture_sources(root):
    source = root / "source"
    metadata = root / "metadata"
    rng = np.random.default_rng(10)
    for split_index, split in enumerate(generator.SPLITS):
        raw = source / "raw" / generator.split_dir(split)
        raw.mkdir(parents=True)
        rows, streams, lengths = [], {n: [] for n in generator.STREAMS}, []
        for i in range(4):
            uid = "{}_{}".format(split, i)
            dry_paths, dry_signals = [], []
            for k in range(2):
                speaker = "{:03d}".format(split_index * 100 + i * 2 + k)
                path = root / "dry" / speaker / (speaker + "_utterance.wav")
                path.parent.mkdir(parents=True)
                signal = rng.normal(0, 0.03, 640)
                sf.write(path, signal, 16000, subtype="FLOAT")
                dry_paths.append(str(path))
                dry_signals.append(signal)
            rows.append({"utterance_id": uid, "source1": dry_paths[0],
                         "source2": dry_paths[1], "target_t60": 0.4})
            dir_rirs = [np.array([[0.4, 0.3]]), np.array([[0.2, 0.1]])]
            rev_rirs = [np.pad(h, ((0, 50), (0, 0))) for h in dir_rirs]
            for h in rev_rirs:
                h[20] = h[0] * 0.2
            direct = [generator.convolve_crop(x, h, 0, 256) for x, h in zip(dry_signals, dir_rirs)]
            reverb = [generator.convolve_crop(x, h, 0, 256) for x, h in zip(dry_signals, rev_rirs)]
            audios = [reverb[0] + reverb[1], *direct, *reverb, *dir_rirs, *rev_rirs]
            for name, audio in zip(generator.STREAMS, audios):
                path = root / "original_audio" / split / name / (uid + ".wav")
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, 16000, subtype="FLOAT")
                streams[name].append(uid + " " + str(path))
            lengths.append(uid + " 256")
        for name, lines in streams.items():
            (raw / (name + ".scp")).write_text("\n".join(lines) + "\n")
        (raw / "utt2num_samples").write_text("\n".join(lengths) + "\n")
        (raw / "channel").write_text("0\n")
        (raw / "feats_type").write_text("raw\n")
        (metadata / split).mkdir(parents=True)
        (metadata / split / "utterances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return source, metadata


def check_end_to_end_staged_dump_and_integrity(tmp_path):
    source, metadata = fixture_sources(tmp_path)
    output = tmp_path / "stage"
    final = tmp_path / "final"
    argv = [str(SCRIPT), "--source-dump", str(source),
                                      "--metadata-root", str(metadata), "--output-dir", str(output),
                                      "--index-root", str(final), "--limit", "2", "--workers", "2"]
    with patch.object(generator, "verify_dump", wraps=generator.verify_dump) as verify:
        with patch.object(sys, "argv", argv):
            generator.main()
        verify.assert_called_once_with(output, workers=2)
    preparation = json.loads((output / "preparation.json").read_text())
    assert preparation["spec"] == {"sample_rate": "16k", "num_spk": 2, "length": "min", "noise": False}
    assert preparation["counts"] == {"tr": 2, "cv": 2, "tt": 2}
    records = [json.loads(l) for l in (output / "manifests/tr.jsonl").read_text().splitlines()]
    for r in records:
        assert not set(r["original_speakers"]) & {s["speaker"] for s in r["view2"]["sources"]}
        assert len({s["speaker"] for s in r["view2"]["sources"]}) == 2
        for name in generator.STREAMS:
            item = r["streams"][name]
            assert (output / item["path"]).read_bytes() == Path(item["source_path"]).read_bytes()
    # Reconstruct from independent NumPy convolution and the saved gains/crops.
    record = records[0]
    reconstructed = np.zeros((record["num_samples"], 2))
    for position, source_info in enumerate(record["view2"]["sources"], start=1):
        dry, _ = sf.read(source_info["path"])
        rir, _ = sf.read(output / record["streams"]["rir_ref" + str(position)]["path"])
        start = source_info["crop_start"]
        gain = record["view2"]["gains"]["source_total_gains"][position - 1]
        for channel in range(2):
            full = np.convolve(dry, rir[:, channel])
            reconstructed[:, channel] += full[start:start + record["num_samples"]] * gain
    saved, _ = sf.read(output / record["streams"]["speech_mix_view2"]["path"])
    np.testing.assert_allclose(saved, reconstructed, atol=2e-8, rtol=1e-6)
    assert np.max(np.abs(saved)) <= 0.99 + 1e-7
    assert not (output / "raw/tt_rir_2spk_nf_min_16k/speech_mix_view2.scp").exists()
    output.rename(final)
    # Serial and parallel verification return the same complete report.
    # The barrier proves separate records really run concurrently.
    verify_record = generator.verify_record
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    started = []

    def concurrent_record(task):
        with lock:
            started.append(threading.get_ident())
            first_pair = len(started) <= 2
        if first_pair:
            barrier.wait(timeout=10)
        return verify_record(task)

    with patch.object(generator.time, "time", return_value=1234567890.0):
        serial = generator.verify_dump(final)
        assert serial["self_contained"]
        with patch.object(generator, "verify_record", side_effect=concurrent_record):
            assert generator.verify_dump(final, workers=2) == serial
        assert len(set(started[:2])) == 2
        for workers in (3, 4):
            assert generator.verify_dump(final, workers=workers) == serial
    # The published dump verifies even when original/dry sources are unavailable.
    import shutil
    shutil.rmtree(source)
    shutil.rmtree(tmp_path / "original_audio")
    shutil.rmtree(tmp_path / "dry")
    argv = [str(SCRIPT), "--verify-existing", str(final), "--workers", "4"]
    with patch.object(generator, "verify_dump", wraps=generator.verify_dump) as verify:
        with patch.object(sys, "argv", argv):
            generator.main()
        verify.assert_called_once_with(final, workers=4)
    assert json.loads((final / "verification.json").read_text())["counts"]["tr"] == 2
    item = records[0]["streams"]["speech_mix_view2"]
    path = final / item["path"]
    audio, rate = sf.read(path)
    sf.write(path, audio * 0.9, rate, subtype="FLOAT")
    for workers in (1, 2, 3, 4):
        with unittest.TestCase().assertRaisesRegex(ValueError, "checksum"):
            generator.verify_dump(final, workers=workers)
    with patch.object(generator.time, "time", return_value=1234567890.0):
        serial = generator.verify_dump(final, check_hashes=False)
        assert not serial["checked_audio_hashes"]
        assert generator.verify_dump(final, check_hashes=False, workers=4) == serial


def check_reject_cross_split_dry_leakage(tmp_path):
    source, metadata = fixture_sources(tmp_path)
    train = json.loads((metadata / "tr/utterances.jsonl").read_text().splitlines()[0])
    cv = [json.loads(l) for l in (metadata / "cv/utterances.jsonl").read_text().splitlines()]
    cv[0]["source1"] = train["source1"]
    (metadata / "cv/utterances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cv))
    with unittest.TestCase().assertRaisesRegex(ValueError, "leakage"):
        generator.load_sources(source, metadata)


class PairedDumpTest(unittest.TestCase):
    def test_convolution_history(self):
        check_convolution_crops_after_filtering()

    def test_speaker_selection(self):
        check_selection_excludes_original_and_short_sources()

    def test_energy_calibration(self):
        check_new_source_energy_calibration_and_stereo_shared_gain()

    def test_staged_dump(self):
        base = ROOT / "egs2/whamr/rir_2spk/exp/paired_dump_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            check_end_to_end_staged_dump_and_integrity(Path(tmp))

    def test_split_leakage(self):
        base = ROOT / "egs2/whamr/rir_2spk/exp/paired_dump_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            check_reject_cross_split_dry_leakage(Path(tmp))

    def test_verification_worker_bounds(self):
        for workers in (-1, 0, 5):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "workers must be between 1 and 4"):
                    generator.verify_dump(ROOT / "unused_dump", workers=workers)
                argv = [str(SCRIPT), "--verify-existing", str(ROOT / "unused_dump"),
                        "--workers", str(workers)]
                with patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as error:
                        generator.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
