"""MIRD receiver coverage and two-speaker geometry sampling invariants."""

import importlib.util
import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np


LOCAL = Path(__file__).resolve().parents[3] / "egs2/whamr/rir_1ch/local"
SPEC = importlib.util.spec_from_file_location(
    "mird_pair_assignment", LOCAL / "mird_pair_assignment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestMIRDPairAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = MODULE.assign_pairs(seed=20261027)

    def test_production_coverage_and_source_roles(self):
        records = self.records
        self.assertEqual(len(records), 3000)
        grouped = defaultdict(list)
        for record in records:
            key = record["rt60_ms"], record["spacing"], record["microphone"]
            first, second = record["sources"]
            self.assertNotEqual(first, second)
            self.assertTrue(0 <= first < 26 and 0 <= second < 26)
            grouped[key].append((first, second))
        self.assertEqual(
            set(grouped),
            set(product((160, 360, 610), MODULE.SPACINGS, range(1, 9))),
        )
        for key in ("rt60_ms", "spacing", "microphone"):
            expected = 375 if key == "microphone" else 1000
            self.assertEqual(set(Counter(r[key] for r in records).values()), {expected})
        all_pairs = set()
        for key, pairs in grouped.items():
            with self.subTest(receiver=key):
                self.assertIn(len(pairs), (41, 42))
                unordered = {tuple(sorted(pair)) for pair in pairs}
                self.assertEqual(len(unordered), len(pairs))
                all_pairs.update(unordered)
                for side in (0, 1):
                    counts = Counter(pair[side] for pair in pairs)
                    self.assertEqual(set(counts), set(range(26)))
                    self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(all_pairs, set(combinations(range(26), 2)))
        report = MODULE.audit_assignment(records, require_production=True)
        self.assertEqual(report["used_receiver_conditions"], 72)
        self.assertEqual(report["unique_recorded_unordered_conditions"], 3000)
        self.assertEqual(report["source_pair_types_covered"], 325)

    def test_reproducibility_and_seed_variation(self):
        self.assertEqual(self.records, MODULE.assign_pairs(seed=20261027))
        other = MODULE.assign_pairs(seed=11)
        self.assertNotEqual(self.records, other)
        MODULE.audit_assignment(other, require_production=True)

    def test_utterance_order_is_shuffled(self):
        changes = sum(a["rt60_ms"] != b["rt60_ms"]
                      for a, b in zip(self.records, self.records[1:]))
        self.assertGreater(changes, 1700)

    def test_same_position_and_invalid_receiver_are_rejected(self):
        for changes in ({"sources": [3, 3]}, {"sources": [-1, 3]},
                        {"sources": [0, 26]}, {"microphone": 9},
                        {"rt60_ms": 999}, {"spacing": "unknown"}):
            with self.subTest(changes=changes):
                record = dict(self.records[0], **changes)
                with self.assertRaises(ValueError):
                    MODULE.audit_assignment([record])

    def test_repeated_pair_is_detected_independently(self):
        records = [dict(record) for record in self.records]
        first = records[0]
        key = tuple(first[k] for k in ("rt60_ms", "spacing", "microphone"))
        partner = next(i for i, record in enumerate(records[1:], 1)
                       if tuple(record[k] for k in ("rt60_ms", "spacing", "microphone")) == key)
        records[partner]["sources"] = list(first["sources"])
        with self.assertRaisesRegex(ValueError, "distinct source pairs"):
            MODULE.audit_assignment(records, require_production=True)

    def test_empty_and_small_assignments(self):
        self.assertEqual(MODULE.assign_pairs(0), [])
        for count in (1, 9, 71, 73):
            with self.subTest(count=count):
                records = MODULE.assign_pairs(count, seed=0)
                self.assertEqual(len(records), count)
                self.assertEqual(MODULE.audit_assignment(records)["mixtures"], count)

    def test_invalid_counts(self):
        for count in (-1, 3.5, True, "3000"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                MODULE.assign_pairs(count)

    def test_plan_preserves_source_indexes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filenames, scaling = root / "filenames.csv", root / "scaling.npz"
            ids = [f"sample_{i:04d}.wav" for i in range(3000)]
            with filenames.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["output_filename", "s1_path", "s2_path"])
                writer.writeheader()
                writer.writerows(dict(output_filename=uid, s1_path="speaker_a/first.wav",
                                      s2_path="speaker_b/second.wav") for uid in ids)
            np.savez(scaling, utterance_id=np.asarray(ids))
            original = {path: path.read_bytes() for path in (filenames, scaling)}
            output = root / "plan"
            audit = MODULE.prepare_plan(filenames, scaling, output)
            payload = json.loads((output / "assignment_plan.json").read_text())
            self.assertEqual(audit["mixtures"], 3000)
            self.assertEqual(payload["status"], "awaiting_measured_rir_data")
            self.assertEqual([r["filename"] for r in payload["records"]], ids)
            self.assertEqual(len(payload["source_grid"]), 26)
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(payload["source_index_sha256"][str(path.resolve())],
                                 hashlib.sha256(content).hexdigest())
            saved = (output / "assignment_plan.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "new assignment-plan"):
                MODULE.prepare_plan(filenames, scaling, output)
            self.assertEqual((output / "assignment_plan.json").read_bytes(), saved)

    def test_incomplete_whamr_indexes_do_not_publish_a_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filenames, scaling = root / "filenames.csv", root / "scaling.npz"
            filenames.write_text("output_filename,s1_path,s2_path\nexample.wav,a/x.wav,b/y.wav\n")
            np.savez(scaling, utterance_id=np.asarray(["example.wav"]))
            with self.assertRaisesRegex(ValueError, "3000-mixture"):
                MODULE.prepare_plan(filenames, scaling, root / "invalid")
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
