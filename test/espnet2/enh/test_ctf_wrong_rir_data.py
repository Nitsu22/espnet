"""CPU-only checks for original speech conditioned on another scene's RIRs."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh_rir/local"
sys.path.insert(0, str(LOCAL))
SPEC = importlib.util.spec_from_file_location(
    "wrong_rir_fixtures", Path(__file__).with_name("test_same_speaker_test_data.py"))
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
import prepare_nf8k_ctf_controls as controls


CONDITION = "same_utt_other_rir"
DATASET = controls.CONDITIONS[CONDITION]


def audit(folder):
    with (folder / "audit.tsv").open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def unique_rir_fixture(root):
    """Replace the old fixture's repeated RIRs and rebuild its original audio."""
    settings = fixtures.fixture(root)
    baseline = {name: controls.read_scp(settings["baseline_test"] / name)
                for name in controls.NAMES}
    rows = controls.read_filelist(settings["csv_dir"] / "mix_2_spk_filenames_tt.csv")
    metadata_path = settings["physical_root"] / "wav8k/min/tt/utterances.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    metadata_by_id = {row["utterance_id"]: row for row in metadata}
    for index, uid in enumerate(sorted(baseline["wav.scp"])):
        filename = controls.canonical_filename(uid)
        row = rows[filename]
        direct, reverb = [], []
        for slot in (1, 2):
            source = controls.read_left(settings["raw_root"] / row[f"s{slot}_path"], 16000)
            for kind, destination in (("anechoic", direct), ("reverb", reverb)):
                h = np.zeros(48)
                h[2 * slot + index] = .7 + .025 * index
                if kind == "reverb":
                    h[18 + 2 * slot + index] = .12 + .02 * index
                path = settings["physical_root"] / f"wav16k/min/tt/rir{slot}_{kind}" / filename
                sf.write(path, h, 16000, subtype="FLOAT")
                destination.append(controls.convolve_8k(source, controls.read_left(path, 16000), 512))
        for name, audio in zip(controls.NAMES, [sum(reverb), *direct]):
            sf.write(baseline[name][uid], audio, 8000, subtype="PCM_16")
        metadata_by_id[Path(filename).stem]["target_t60"] = .2 + .1 * index
    metadata_path.write_text("".join(json.dumps(row) + "\n" for row in metadata))
    return settings


def fixture(root):
    """Public fixture with unique paired RIRs and matching original audio."""
    return unique_rir_fixture(root)


class TestCTFWrongRIRData(unittest.TestCase):
    def test_original_speech_and_coherent_deranged_rir_numeric(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_wrong_rir_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = unique_rir_fixture(root)
            out = root / "out"
            with patch.object(controls, "choose_donors", side_effect=AssertionError("speech donors forbidden")):
                result = controls.prepare(**settings, output=out, conditions=[CONDITION])
            self.assertEqual(set(result["condition_definitions"]), {CONDITION})
            self.assertEqual(result["condition_count"], 1)
            self.assertEqual(result["scene_count"], 4)
            self.assertEqual(result["generated_wav_count"], 12)
            self.assertEqual(result["original_identity_verified_count"], 4)
            self.assertEqual(result["rir_donor_fixed_point_count"], 0)
            self.assertEqual(result["rir_donor_unique_count"], 4)
            self.assertFalse(result["ctf_estimation_run"])
            self.assertFalse(result["model_inference_run"])
            self.assertFalse(result["scoring_run"])
            self.assertEqual([path.name for path in (out / "raw").iterdir()], [DATASET])
            records = audit(out)
            by_uid = {row["scene_uid"]: row for row in records}
            self.assertEqual({row["donor_rir_uid"] for row in records}, set(by_uid))
            for row in records:
                donor_uid = row["donor_rir_uid"]
                self.assertNotEqual(donor_uid, row["scene_uid"])
                self.assertEqual(row["source_t60"], by_uid[donor_uid]["target_t60"])
                self.assertEqual(row["source1_rir_slot"], "1")
                self.assertEqual(row["source2_rir_slot"], "2")
                direct, reverb = [], []
                target_reverb_hashes, donor_reverb_hashes = [], []
                for slot in (1, 2):
                    self.assertEqual(row[f"source{slot}"], row[f"original_source{slot}"])
                    self.assertEqual(row[f"source{slot}_origin_scene_uids"], row["scene_uid"])
                    self.assertEqual(row[f"source{slot}_sha256"], row[f"original_source{slot}_sha256"])
                    raw = controls.read_left(row[f"source{slot}"], 16000)
                    for kind, destination in (("anechoic", direct), ("reverb", reverb)):
                        key = f"rir{slot}_{kind}"
                        self.assertEqual(Path(row[key]).name, controls.canonical_filename(donor_uid))
                        self.assertEqual(row[f"{key}_sha256"], by_uid[donor_uid][f"original_{key}_sha256"])
                        self.assertEqual(controls.sha256(Path(row[key])), row[f"{key}_sha256"])
                        expected = controls.convolve_8k(raw, controls.read_left(row[key], 16000), 512)
                        destination.append(expected * float(row[f"applied_gain{slot}"]))
                    target_reverb_hashes.append(row[f"original_rir{slot}_reverb_sha256"])
                    donor_reverb_hashes.append(row[f"rir{slot}_reverb_sha256"])
                self.assertNotEqual(sorted(target_reverb_hashes), sorted(donor_reverb_hashes))
                for signal, expected in zip(("wav", "spk1", "spk2"), [sum(reverb), *direct]):
                    path = out / "raw" / DATASET / "data" / signal / f'{row["scene_uid"]}.wav'
                    self.assertEqual((sf.info(path).frames, sf.info(path).subtype), (512, "PCM_16"))
                    self.assertLess(np.max(np.abs(controls.read_left(path, 8000) - expected)), 3.1e-5)
                self.assertLessEqual(abs(float(row["power_ratio_error_db"])), .01)

    def test_reproducibility_pilot_and_condition_selection(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_wrong_rir_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = unique_rir_fixture(root)
            destination = root / "destination"
            for name in ("first", "second", "pilot", "all"):
                conditions = tuple(controls.CONDITIONS) if name == "all" else [CONDITION]
                controls.prepare(**settings, output=root / name, destination_root=destination,
                                 conditions=conditions, limit=2 if name == "pilot" else None)
            self.assertEqual((root / "first/audit.tsv").read_bytes(), (root / "second/audit.tsv").read_bytes())
            full = {row["scene_uid"]: row for row in audit(root / "first")}
            for row in audit(root / "pilot"):
                self.assertEqual(row, full[row["scene_uid"]])
            for row in audit(root / "all"):
                if row["condition"] == CONDITION:
                    self.assertEqual(row, full[row["scene_uid"]])
            for path in (root / "first/raw" / DATASET / "data").rglob("*.wav"):
                self.assertEqual(path.read_bytes(), (root / "second" / path.relative_to(root / "first")).read_bytes())

    def test_hash_identical_rir_set_rejected_without_publishing(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_wrong_rir_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = fixtures.fixture(root)  # Deliberately identical RIRs in all four scenes.
            with self.assertRaisesRegex(ValueError, "RIR set equals target RIR set by hashes"):
                controls.prepare(**settings, output=root / "out", conditions=[CONDITION])
            self.assertFalse((root / "out").exists())
            self.assertEqual(list(root.glob(".out.*")), [])

    def test_sattolo_single_cycle_seed_and_input_order(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_wrong_rir_test_", dir=ROOT) as directory:
            root = Path(directory)
            settings = fixture(root)
            scenes = []
            for index, uid in enumerate(sorted(controls.read_scp(settings["baseline_test"] / "wav.scp"))):
                filename = controls.canonical_filename(uid)
                paths = {f"rir{slot}_{kind}": settings["physical_root"] / f"wav16k/min/tt/rir{slot}_{kind}" / filename
                         for slot in (1, 2) for kind in ("anechoic", "reverb")}
                scenes.append(dict(uid=uid, rir_paths=paths, metadata=dict(target_t60=.2 + .1 * index)))
            maps = []
            for seed in (0, 1, 0):
                controls.choose_rir_donors(list(reversed(scenes)) if len(maps) == 2 else scenes, seed)
                mapping = {scene["uid"]: scene["donor_rir_uid"] for scene in scenes}
                self.assertEqual(set(mapping), set(mapping.values()))
                self.assertTrue(all(target != donor for target, donor in mapping.items()))
                start = next(iter(mapping))
                visited, current = set(), start
                for _ in range(len(mapping)):
                    self.assertNotIn(current, visited)
                    visited.add(current)
                    current = mapping[current]
                self.assertEqual(current, start)
                self.assertEqual(visited, set(mapping))
                maps.append(mapping)
            self.assertEqual(maps[0], maps[2])
            self.assertNotEqual(maps[0], maps[1])

    def test_order_insensitive_reverb_pair_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_wrong_rir_test_", dir=ROOT) as directory:
            root = Path(directory)
            paths = []
            for index in range(6):
                path = root / f"rir{index}.wav"
                sf.write(path, np.array([0., .1 + .1 * index, 0.]), 16000, subtype="FLOAT")
                paths.append(path)
            first = dict(rir1_anechoic=paths[0], rir2_anechoic=paths[1],
                         rir1_reverb=paths[2], rir2_reverb=paths[3])
            second = dict(rir1_anechoic=paths[4], rir2_anechoic=paths[5],
                          rir1_reverb=paths[3], rir2_reverb=paths[2])
            scenes = [dict(uid="a", rir_paths=first, metadata=dict(target_t60=.2)),
                      dict(uid="b", rir_paths=second, metadata=dict(target_t60=.3))]
            with self.assertRaisesRegex(ValueError, "order-insensitive reverb pairs"):
                controls.choose_rir_donors(scenes, 0)

    def test_invalid_conditions_and_small_derangement(self):
        with self.assertRaisesRegex(ValueError, "at least two unique"):
            controls.choose_rir_donors([], 0)
        for conditions in ([], [CONDITION, CONDITION], ["unknown"]):
            with self.assertRaisesRegex(ValueError, "nonempty, unique subset"):
                controls.prepare(None, None, None, None, None, "/not-used", conditions=conditions)


if __name__ == "__main__":
    unittest.main()
