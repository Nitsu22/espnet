"""Sampling invariants for all-microphone BUT two-speaker mixtures."""

import importlib.util
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "egs2/whamr/rir_1ch/local/but_pair_assignment.py"
)
SPEC = importlib.util.spec_from_file_location("but_pair_assignment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
assign_all_microphone_pairs = MODULE.assign_all_microphone_pairs


def make_groups(source_counts, microphones=31):
    groups, lookup = {}, {}
    index = 0
    for room_number, sources in enumerate(source_counts):
        for microphone in range(microphones):
            key = f"room{room_number:02d}", "MicID01", f"{microphone + 1:02d}"
            groups[key] = list(range(index, index + sources))
            for source, rir_index in enumerate(groups[key]):
                lookup[rir_index] = key, source
            index += sources
    return groups, lookup


def inspect_assignment(groups, lookup, assignment):
    rooms, microphones, by_receiver = Counter(), Counter(), defaultdict(list)
    for first, second in assignment:
        assert first != second
        first_key, first_source = lookup[first]
        second_key, second_source = lookup[second]
        assert first_key == second_key
        rooms[first_key[0]] += 1
        microphones[first_key[2]] += 1
        by_receiver[first_key].append((first_source, second_source))
    for key, pairs in by_receiver.items():
        sources = len(groups[key])
        pair_counts = Counter(tuple(sorted(pair)) for pair in pairs)
        complete = [
            pair_counts[first, second]
            for first in range(sources)
            for second in range(first + 1, sources)
        ]
        assert max(complete) - min(complete) <= 1
        assert len(pair_counts) == min(len(pairs), sources * (sources - 1) // 2)
        for side in (0, 1):
            counts = np.bincount([pair[side] for pair in pairs], minlength=sources)
            assert np.ptp(counts) <= 1
    return rooms, microphones, by_receiver


class TestBUTPairAssignment(unittest.TestCase):
    def test_complete_pair_cycles_and_speaker_marginals(self):
        for sources in (2, 3, 4, 5, 6, 10):
            groups, lookup = make_groups([sources], microphones=1)
            complete = sources * (sources - 1) // 2
            counts = {
                0, 1, sources - 1, sources, 10, 11, complete, complete + 1,
                2 * complete - 1, 2 * complete, 3 * complete + 2,
            }
            for count in sorted(counts):
                with self.subTest(sources=sources, count=count):
                    assignment = assign_all_microphone_pairs(groups, count, seed=73)
                    self.assertEqual(assignment.shape, (count, 2))
                    self.assertTrue(np.issubdtype(assignment.dtype, np.integer))
                    inspect_assignment(groups, lookup, assignment)

    def test_actual_nine_room_inventory(self):
        groups, lookup = make_groups([4, 5, 10, 6, 2, 6, 5, 10, 3])
        assignment = assign_all_microphone_pairs(groups, 3000, seed=2026)
        rooms, microphones, by_receiver = inspect_assignment(groups, lookup, assignment)
        self.assertEqual(assignment.shape, (3000, 2))
        self.assertEqual(sorted(rooms.values()), [333] * 6 + [334] * 3)
        self.assertEqual(len(by_receiver), 9 * 31)
        self.assertEqual({len(pairs) for pairs in by_receiver.values()}, {10, 11})
        self.assertEqual(sorted(microphones.values()), [96] * 7 + [97] * 24)
        # Concrete quality threshold for the production geometry; arbitrary
        # inventories and mixture counts have no exact room-pair guarantee.
        for room in rooms:
            keys = [key for key in groups if key[0] == room]
            sources = len(groups[keys[0]])
            counts = Counter(
                tuple(sorted(pair)) for key in keys for pair in by_receiver[key]
            )
            values = [
                counts[first, second]
                for first in range(sources)
                for second in range(first + 1, sources)
            ]
            self.assertLessEqual(max(values) - min(values), 2)
        # Mixture order must not consist of receiver-sized or room-sized blocks.
        room_runs = sum(
            lookup[int(first)][0][0] != lookup[int(second)][0][0]
            for first, second in zip(assignment[:-1, 0], assignment[1:, 0])
        )
        self.assertGreater(room_runs, 2000)

    def test_seeded_assignment_is_independent_of_input_order(self):
        groups, _ = make_groups([4, 5, 10], microphones=3)
        expected = assign_all_microphone_pairs(groups, 100, seed=44)
        permuted = {
            key: list(reversed(values)) for key, values in reversed(groups.items())
        }
        np.testing.assert_array_equal(
            assign_all_microphone_pairs(permuted, 100, seed=44), expected
        )
        self.assertFalse(
            np.array_equal(assign_all_microphone_pairs(groups, 100, seed=45), expected)
        )

    def test_sparse_count_and_nonuniform_receiver_sets(self):
        groups, lookup = make_groups([3, 5], microphones=3)
        groups.pop(("room01", "MicID01", "03"))
        for count in (1, 3, 7, 27):
            with self.subTest(count=count):
                assignment = assign_all_microphone_pairs(groups, count, seed=2)
                rooms, _, by_receiver = inspect_assignment(
                    groups, lookup, assignment
                )
                room_counts = [rooms[room] for room in ("room00", "room01")]
                self.assertLessEqual(max(room_counts) - min(room_counts), 1)
                for room in ("room00", "room01"):
                    quotas = [
                        len(by_receiver[key]) for key in groups if key[0] == room
                    ]
                    self.assertLessEqual(max(quotas) - min(quotas), 1)

    def test_invalid_inventory_or_count(self):
        cases = [
            ({}, 1),
            ({("r", "s", "m"): [0]}, 1),
            ({("r", "s", "m"): [0, 0]}, 1),
            ({("r", "s", "m"): [-1, 2]}, 1),
            ({("r", "s", "m"): [0, 1.5]}, 1),
            ({("r", "s", "m"): [0, 1], ("r", "s", "n"): [1, 2]}, 1),
            ({"invalid": [0, 1]}, 1),
            ({("r", "s", "m"): [0, 1]}, -1),
            ({("r", "s", "m"): [0, 1]}, 1.5),
        ]
        for groups, count in cases:
            with self.subTest(groups=groups, count=count):
                with self.assertRaises(ValueError):
                    assign_all_microphone_pairs(groups, count, seed=0)

    def test_empty_inventory_for_zero_mixtures(self):
        self.assertEqual(assign_all_microphone_pairs({}, 0, seed=0).shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
