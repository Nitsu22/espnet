"""CPU-only synthetic checks; no licensed data, models or GPU is required."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "egs2/whamr/enh_rir/local/prepare_nf8k_same_speaker_test.py"
SPEC = importlib.util.spec_from_file_location("same_speaker_data", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ("output_filename", "s1_path", "s2_path"))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    raw, physical, baseline, csv_dir = [root / n for n in ("raw", "physical", "baseline", "csv")]
    for folder in (raw, physical, baseline, csv_dir):
        folder.mkdir()
    speech = {}
    for speaker in ("050", "051"):
        for index in range(3):
            relative = f"wsj0/si_dt_05/{speaker}/{speaker}a000{index}.wav"
            path = raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            t = np.arange(4096) / 16000
            audio = .03 * np.sin(2 * np.pi * (230 + 53 * index + int(speaker)) * t)
            audio += .004 * np.sin(2 * np.pi * 79 * t)
            sf.write(path, audio, 16000, subtype="PCM_16")
            speech[(speaker, index)] = relative
    rows, metadata = [], []
    scps = {name: [] for name in module.NAMES}
    for a, b in ((0, 0), (1, 1), (2, 2), (0, 2)):
        source1, source2 = speech[("050", a)], speech[("051", b)]
        filename = f"{Path(source1).stem}_0_{Path(source2).stem}_0.wav"
        uid = f"050_051_{Path(filename).stem}_reverb"
        rows.append(dict(output_filename=filename, s1_path=source1, s2_path=source2))
        direct, reverb = [], []
        for k, source in ((1, source1), (2, source2)):
            for condition in ("anechoic", "reverb"):
                h = np.zeros(32)
                h[2 * k] = .75
                if condition == "reverb":
                    h[12 + k] = .15
                p = physical / f"wav16k/min/tt/rir{k}_{condition}" / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, h, 16000, subtype="FLOAT")
                source_audio = module.read_left(raw / source, 16000)
                output = module.convolve_8k(source_audio, module.read_left(p, 16000), 512)
                (direct if condition == "anechoic" else reverb).append(output)
        for name, audio in zip(module.NAMES, [sum(reverb), *direct]):
            path = baseline / f"{uid}_{name}.wav"
            sf.write(path, audio, 8000, subtype="PCM_16")
            scps[name].append(f"{uid} {path}\n")
        metadata.append(dict(utterance_id=Path(filename).stem, source1=str(raw / source1),
                             source2=str(raw / source2), s1_spatial_scaling=1.,
                             s2_spatial_scaling=1., target_t60=.4))
    for name, lines in scps.items():
        (baseline / name).write_text("".join(lines))
    meta_path = physical / "wav8k/min/tt/utterances.jsonl"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text("".join(json.dumps(row) + "\n" for row in metadata))
    write_csv(csv_dir / "mix_2_spk_filenames_tt.csv", rows)
    other = dict(output_filename="999a0000_0_998a0000_0.wav",
                 s1_path="wsj0/si_tr_s/999/999a0000.wav",
                 s2_path="wsj0/si_tr_s/998/998a0000.wav")
    for split in ("tr", "cv"):
        write_csv(csv_dir / f"mix_2_spk_filenames_{split}.csv", [other])
    return dict(baseline_test=baseline, baseline_recipe=root, raw_root=raw,
                physical_root=physical, csv_dir=csv_dir, expected_count=4, workers=2)


class SameSpeakerDataTest(unittest.TestCase):
    def test_balanced_feasible_and_deterministic(self):
        feasible = [[True, False], [False, True], [True, True], [True, True]]
        a = module.balanced_anchors(feasible, np.random.default_rng(0))
        b = module.balanced_anchors(feasible, np.random.default_rng(0))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(list(a[:2]), [1, 2])
        self.assertEqual(int(np.sum(a == 1)), 2)
        with self.assertRaises(ValueError):
            module.balanced_anchors([[False, False]], np.random.default_rng(0))
        with self.assertRaises(ValueError):
            module.balanced_anchors([[True, False]] * 4, np.random.default_rng(0))

    def test_split_leakage(self):
        one = dict(x=dict(s1_path="wsj/050/050x.wav", s2_path="wsj/051/051y.wav"))
        other = dict(y=dict(s1_path="wsj/050/050z.wav", s2_path="wsj/052/052y.wav"))
        with self.assertRaises(ValueError):
            module.check_split_disjoint(one, other)

    def test_no_padding_and_power(self):
        with self.assertRaises(ValueError):
            module.convolve_8k(np.ones(10), np.ones(2), 100)
        direct, reverb, gain = module.match_power(np.ones(10) * 2, np.ones(10) * 3, 1.)
        np.testing.assert_allclose(direct, 1)
        np.testing.assert_allclose(reverb, 1.5)
        self.assertEqual(gain, .5)
        with self.assertRaises(ValueError):
            module.match_power(np.zeros(10), np.zeros(10), 1.)

    def test_clipping_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with self.assertRaises(ValueError):
                module.write_pcm(Path(directory) / "no.wav", np.array([0., 1.]))
            self.assertFalse((Path(directory) / "no.wav").exists())

    def test_common_peak_attenuation_preserves_ratio(self):
        audios = [np.array([1.4, -.3]), np.array([.7, -.2]), np.array([.4, .1])]
        with self.assertRaises(ValueError):
            module.apply_peak_policy(audios, "error")
        scaled, gain, peak = module.apply_peak_policy(audios, "common-attenuate")
        self.assertAlmostEqual(peak, 1.4)
        self.assertAlmostEqual(gain, .99 / 1.4)
        self.assertAlmostEqual(max(np.max(np.abs(x)) for x in scaled), .99)
        for original, result in zip(audios, scaled):
            np.testing.assert_allclose(result, original * gain)
        original_ratio = np.mean(audios[1] ** 2) / np.mean(audios[2] ** 2)
        saved_ratio = np.mean(scaled[1] ** 2) / np.mean(scaled[2] ** 2)
        self.assertAlmostEqual(original_ratio, saved_ratio)
        for original, result in zip(audios, scaled):
            self.assertAlmostEqual(np.mean(result ** 2), np.mean(original ** 2) * gain ** 2)
        unchanged, gain, peak = module.apply_peak_policy([np.array([.4])], "common-attenuate")
        self.assertEqual(gain, 1.)
        np.testing.assert_array_equal(unchanged[0], [.4])

    def test_full_fixture_reproducible_and_test_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            settings = fixture(root)
            destination = Path("/gs/bs/tga-shinoda/nitsu/test-only-destination")
            for name in ("out1", "out2"):
                report = module.prepare(**settings, output=root / name, seed=0,
                                        destination_root=destination)
                self.assertEqual(report["anchor_slot1_count"], 2)
                self.assertEqual(report["anchor_slot2_count"], 2)
                self.assertLessEqual(report["worst_power_error_db"], .01)
                out = root / name
                self.assertEqual([p.name for p in (out / "raw").iterdir()], [module.DATASET])
                self.assertEqual((out / "raw" / module.DATASET / "channel").read_text(), "0\n")
                with (out / "audit.tsv").open() as stream:
                    rows = list(csv.DictReader(stream, delimiter="\t"))
                for row in rows:
                    self.assertEqual(Path(row["source1"]).parent.name, Path(row["source2"]).parent.name)
                    self.assertNotEqual(row["source1"], row["source2"])
                    slot = row["anchor_slot"]
                    self.assertEqual(row[f"source{slot}"], row[f"original_source{slot}"])
                    self.assertLess(float(row["control_mix_max_error"]), 5e-5)
                for name_scp in module.NAMES:
                    mapping = module.read_scp(out / "raw" / module.DATASET / name_scp)
                    self.assertEqual(set(mapping), set(module.read_scp(settings["baseline_test"] / name_scp)))
                    self.assertTrue(all(str(destination) in p for p in mapping.values()))
                for p in (out / "raw" / module.DATASET / "data").rglob("*.wav"):
                    info = sf.info(p)
                    self.assertEqual((info.samplerate, info.frames, info.channels, info.subtype),
                                     (8000, 512, 1, "PCM_16"))
            files = [p.relative_to(root / "out1") for p in (root / "out1").rglob("*") if p.is_file()]
            for relative in files:
                self.assertEqual((root / "out1" / relative).read_bytes(),
                                 (root / "out2" / relative).read_bytes(), str(relative))
            with self.assertRaises(FileExistsError):
                module.prepare(**settings, output=root / "out1")

    def test_failed_identity_control_cleans_temporary(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            settings = fixture(root)
            index = module.read_scp(settings["baseline_test"] / "wav.scp")
            path = Path(next(iter(index.values())))
            sf.write(path, np.zeros(512), 8000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "identity control failed"):
                module.prepare(**settings, output=root / "failed")
            self.assertFalse((root / "failed").exists())
            self.assertEqual(list(root.glob(".failed.*")), [])


if __name__ == "__main__":
    unittest.main()
