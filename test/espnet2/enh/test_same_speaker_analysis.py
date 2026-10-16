"""Small, dependency-light tests for matched-scene stress-test summaries."""

import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh_rir/local"
spec = importlib.util.spec_from_file_location("same_speaker_analysis", LOCAL / "analyze_same_speaker_test.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(LOCAL))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class SameSpeakerAnalysisTest(unittest.TestCase):
    def test_paired_contrast(self):
        original, same = {}, {}
        for metric in module.METRICS:
            original["baseline", metric] = np.array([10., 20., 30.])
            original["bimamba", metric] = np.array([11., 22., 33.])
            same["baseline", metric] = np.array([2., 4., 6.])
            same["bimamba", metric] = np.array([5., 8., 11.])
        out = module.summarize(original, same, np.ones(3, bool), np.random.default_rng(0), 100)
        self.assertEqual(out["gain_change_db"], 2.)
        self.assertEqual(out["gain_change_ci_low"], 2.)
        self.assertEqual(out["gain_change_ci_high"], 2.)
        self.assertEqual(out["same_speaker_gain_SI_SNR"], 4.)

    def test_empty_group(self):
        with self.assertRaises(ValueError):
            module.summarize({}, {}, np.zeros(2, bool), np.random.default_rng(0), 10)

    def test_load_pair_keys_and_speaker_average(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            for metric in module.METRICS:
                (root / f"{metric}_spk1").write_text("a 1\nb 2\n")
                (root / f"{metric}_spk2").write_text("a 3\nb 4\n")
            data = module.load_pair(root, root, ["a", "b"])
            np.testing.assert_array_equal(data["baseline", "SI_SNR"], [2, 3])
            (root / "STOI_spk2").write_text("a 3\nc 4\n")
            with self.assertRaises(ValueError):
                module.load_pair(root, root, ["a", "b"])

    def test_generation_audit_required_and_verified(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            data = root / "raw" / "tt_mix_clean_reverb_min_8k"
            data.mkdir(parents=True)
            row = dict(scene_uid="a", source1="/raw/050/a.wav", source2="/raw/050/b.wav",
                       actual_speaker="050", anchor_slot="1", original_source1="/raw/050/a.wav")
            with (root / "audit.tsv").open("w") as stream:
                writer = csv.DictWriter(stream, list(row), delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            for name in ("wav.scp", "spk1.scp", "spk2.scp"):
                (data / name).write_text("a /audio/a.wav\n")
            paths = [root / "audit.tsv", *data.glob("*.scp")]
            manifest = dict(scene_count=1, original_scene_ids_retained=True, train_valid_created=False,
                            test_train_valid_utterances_and_speakers_disjoint=True,
                            sample_rate=8000, subtype="PCM_16", output_hashes={
                                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in paths})
            (root / "preparation.json").write_text(json.dumps(manifest))
            self.assertEqual(len(module.validate_generation(data, ["a"])), 2)
            (data / "wav.scp").write_text("a /different/a.wav\n")
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                module.validate_generation(data, ["a"])


if __name__ == "__main__":
    unittest.main()
