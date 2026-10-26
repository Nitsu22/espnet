"""Synthetic CPU tests for conditioning-audio controls; no models or GPU."""

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh_rir/local"
sys.path.insert(0, str(LOCAL))
spec = importlib.util.spec_from_file_location("same_fixture", Path(__file__).with_name("test_same_speaker_test_data.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
import prepare_nf8k_ctf_controls as controls


def audit(folder):
    with (folder / "audit.tsv").open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class TestCTFControlData(unittest.TestCase):
    def test_two_conditions_and_numeric_rir_assignment(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_controls_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = fixtures.fixture(root)
            out = root / "controls"
            result = controls.prepare(**settings, output=out)
            self.assertEqual(result["scene_count"], 4)
            self.assertEqual(result["generated_wav_count"], 24)
            self.assertFalse(result["model_inference_run"])
            self.assertFalse(result["ctf_estimation_run"])
            self.assertFalse(result["scoring_run"])
            self.assertEqual(result["original_identity_verified_count"], 4)
            self.assertLessEqual(result["worst_control_max_error"], 5e-5)
            self.assertLessEqual(result["worst_power_ratio_error_db"], .01)
            records = audit(out)
            self.assertEqual(len(records), 8)
            expected_keys = set(controls.read_scp(settings["baseline_test"] / "wav.scp"))
            for record in records:
                condition = record["condition"]
                sources = [Path(record[f"source{k}"]) for k in (1, 2)]
                original = {record[f"original_source{k}"] for k in (1, 2)}
                if condition == "same_rir_other_utt":
                    self.assertFalse({str(p) for p in sources} & original)
                    self.assertNotEqual(sources[0].parent.name, sources[1].parent.name)
                    self.assertEqual([record[f"source{k}_rir_slot"] for k in (1, 2)], ["1", "2"])
                else:
                    self.assertEqual([str(p) for p in sources], [record[f"original_source{k}"] for k in (1, 2)])
                    self.assertEqual([record[f"source{k}_rir_slot"] for k in (1, 2)], ["2", "1"])
                folder = out / "raw" / record["dataset"]
                for name in controls.NAMES:
                    self.assertEqual(set(controls.read_scp(folder / name)), expected_keys)
                direct, reverb = [], []
                for k, source in enumerate(sources, 1):
                    raw = controls.read_left(source, 16000)
                    slot = record[f"source{k}_rir_slot"]
                    gain = float(record[f"applied_gain{k}"])
                    for kind, destination in (("anechoic", direct), ("reverb", reverb)):
                        h = controls.read_left(record[f"rir{slot}_{kind}"], 16000)
                        destination.append(controls.convolve_8k(raw, h, int(record["frames_8k"])) * gain)
                for signal, expected in zip(("wav", "spk1", "spk2"), [sum(reverb), *direct]):
                    p = folder / "data" / signal / f'{record["scene_uid"]}.wav'
                    audio = controls.read_left(p, 8000)
                    self.assertLess(np.max(np.abs(audio-expected)), 3.1e-5)
                    self.assertEqual(sf.info(p).subtype, "PCM_16")
            with self.assertRaises(FileExistsError):
                controls.prepare(**settings, output=out)

    def test_seed_reproducibility_and_pilot_selection(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_controls_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = fixtures.fixture(root)
            destination = root / "final_destination"
            for name in ("first", "second", "pilot"):
                controls.prepare(**settings, output=root / name, seed=0,
                                 destination_root=destination, limit=2 if name == "pilot" else None)
            self.assertEqual((root / "first/audit.tsv").read_bytes(), (root / "second/audit.tsv").read_bytes())
            full = {(r["condition"], r["scene_uid"]): r for r in audit(root / "first")}
            for row in audit(root / "pilot"):
                self.assertEqual(row, full[row["condition"], row["scene_uid"]])
            for condition, dataset in controls.CONDITIONS.items():
                for path in (root / "first/raw" / dataset / "data").rglob("*.wav"):
                    relative = path.relative_to(root / "first")
                    self.assertEqual(path.read_bytes(), (root / "second" / relative).read_bytes())

    def test_failed_identity_does_not_publish_dump(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_controls_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = fixtures.fixture(root)
            values = controls.read_scp(settings["baseline_test"] / "wav.scp")
            sf.write(next(iter(values.values())), np.zeros(512), 8000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "identity control failed"):
                controls.prepare(**settings, output=root / "failed")
            self.assertFalse((root / "failed").exists())
            self.assertEqual(list(root.glob(".failed.*")), [])

    def test_missing_eligible_donors_rejected(self):
        one = Path("/test/050/050a.wav")
        two = Path("/test/051/051a.wav")
        with self.assertRaisesRegex(ValueError, "No two different-speaker"):
            controls.choose_donors([dict(uid="target", original_sources=[one, two],
                                        raw_frames_16k=[4096, 4096], frames_8k=512)], 0)


if __name__ == "__main__":
    unittest.main()
