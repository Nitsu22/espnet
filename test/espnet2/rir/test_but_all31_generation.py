"""Measured-RIR selection, geometry and published-assignment safeguards."""

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

LOCAL = (
    Path(__file__).resolve().parents[3] / "egs2/whamr/rir_1ch/local"
)
sys.path.insert(0, str(LOCAL))
try:
    import create_whamr_but_2spk_dump as generation
finally:
    sys.path.pop(0)

SOURCE_COUNTS = (4, 5, 10, 6, 2, 6, 5, 10, 3)
ALL_MICROPHONES = tuple(f"{number:02d}" for number in range(1, 32))


def write_rir(root, room, source, microphone, session="20200103_S", meta_id=None):
    folder = (
        root / room / "MicID01" / f"SpkID{source:02d}_{session}" / microphone
    )
    (folder / "RIR").mkdir(parents=True)
    channel = int(microphone) if meta_id is None else meta_id
    metadata = {"$EnvMicID": channel}
    for axis, receiver, speaker in zip(
        ("Depth", "Width", "Height"),
        (channel / 10, 2.0, 1.5),
        (float(source), 3.0, 1.8),
    ):
        metadata[f"$EnvMic{channel}{axis}"] = receiver
        metadata[f"$EnvSpk1{axis}"] = speaker
    (folder / "mic_meta.txt").write_text(
        "".join(f"{key}\t{value}\n" for key, value in metadata.items())
    )
    wave = np.zeros(16, dtype=np.float32)
    wave[1] = 0.5
    wave[5] = (source + channel) / 1000
    path = folder / "RIR" / "IR_sweep.v00.wav"
    sf.write(path, wave, 16000, subtype="FLOAT")
    return path


def make_rir_tree(root, source_counts, microphones, missing=None, wrong_channel=None):
    root.mkdir()
    (root / "extraction_complete.json").write_text("{}\n")
    for room_number, sources in enumerate(source_counts):
        room = f"room{room_number:02d}"
        for source in range(1, sources + 1):
            for microphone in microphones:
                key = (room_number, source, microphone)
                if key != missing:
                    write_rir(
                        root, room, source, microphone,
                        meta_id=31 if key == wrong_channel else None,
                    )


def load_inventory(root, microphones=None):
    with tempfile.TemporaryDirectory(prefix="but-rir-output-") as temporary:
        output = Path(temporary)
        kwargs = {} if microphones is None else {"microphones": microphones}
        rirs = generation.common.load_rirs(root, output, **kwargs)
        inventory = json.loads((output / "rir_inventory.json").read_text())
        for item in inventory:
            assert Path(item["original_path"]).read_bytes() == Path(
                item["source_path"]
            ).read_bytes()
        return rirs, inventory


class TestRIRSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # tempfile respects the task-owned TMPDIR used to run these tests.
        temporary = tempfile.TemporaryDirectory(prefix="but-rir-selection-")
        cls.addClassCleanup(temporary.cleanup)
        cls.rir_tree = Path(temporary.name) / "input"
        make_rir_tree(cls.rir_tree, SOURCE_COUNTS, ALL_MICROPHONES)
        for microphone in ALL_MICROPHONES:
            # Earlier speech session must lose to a dedicated RIR session;
            # among dedicated sessions the earlier one must win.
            write_rir(cls.rir_tree, "room00", 1, microphone, "20191201_T")
            write_rir(cls.rir_tree, "room00", 1, microphone, "20200104_S")
        variant = (
            cls.rir_tree
            / "room00/MicID01/SpkID01_20200103_S/01/RIR/IR_sweep.v01.wav"
        )
        sf.write(variant, np.ones(16), 16000, subtype="FLOAT")

    def test_default_keeps_microphone01_and_dedicated_session_selection(self):
        rirs, inventory = load_inventory(self.rir_tree)
        self.assertEqual(len(rirs), 51)
        self.assertEqual(len(inventory), 51)
        self.assertEqual({item["microphone"] for item in inventory}, {"01"})
        self.assertEqual(Counter(item["room"] for item in inventory), {
            f"room{number:02d}": count
            for number, count in enumerate(SOURCE_COUNTS)
        })
        selected = next(
            item for item in inventory
            if item["room"] == "room00"
            and item["source_position"] == "SpkID01"
        )
        self.assertEqual(selected["session"], "SpkID01_20200103_S")
        self.assertTrue(all(
            item["source_path"].endswith(".v00.wav") for item in inventory
        ))
        # Existing evaluations preserve their speaker-1 assignments.
        assignment = generation.assign_pairs(rirs, inventory, 3000, seed=20260909)
        np.testing.assert_array_equal(
            assignment[:, 0], generation.common.assign_rirs(rirs, 3000, 20260909)
        )

    def test_all31_selects_complete_position_grid_and_one_common_session(self):
        rirs, inventory = load_inventory(self.rir_tree, ALL_MICROPHONES)
        self.assertEqual(len(rirs), 51 * 31)
        self.assertEqual(len(inventory), 51 * 31)
        coverage, sessions = defaultdict(set), defaultdict(set)
        for item in inventory:
            key = item["room"], item["source_position"]
            coverage[key].add(item["microphone"])
            sessions[key].add(item["session"])
        self.assertEqual(len(coverage), 51)
        self.assertTrue(all(
            channels == set(ALL_MICROPHONES) for channels in coverage.values()
        ))
        self.assertTrue(all(len(selected) == 1 for selected in sessions.values()))
        self.assertEqual(
            sessions["room00", "SpkID01"], {"SpkID01_20200103_S"}
        )
        self.assertTrue(all(
            wave.shape == (16,) and rate == 16000 for _, wave, rate, _ in rirs
        ))

    def test_missing_channel_cannot_silently_reduce_the_receiver_grid(self):
        with tempfile.TemporaryDirectory(prefix="but-missing-channel-") as temporary:
            root = Path(temporary) / "input"
            output = Path(temporary) / "output"
            output.mkdir()
            make_rir_tree(root, [2] * 9, ("01", "02"), missing=(8, 2, "02"))
            with self.assertRaisesRegex(ValueError, "coverage"):
                generation.common.load_rirs(root, output, microphones=("01", "02"))

    def test_path_channel_must_agree_with_metadata(self):
        with tempfile.TemporaryDirectory(prefix="but-channel-metadata-") as temporary:
            root = Path(temporary) / "input"
            output = Path(temporary) / "output"
            output.mkdir()
            make_rir_tree(root, [2] * 9, ("01",), wrong_channel=(0, 1, "01"))
            with self.assertRaisesRegex(ValueError, "ID/metadata mismatch"):
                generation.common.load_rirs(root, output)


def geometry_inventory(microphone="02"):
    items, rirs = [], []
    for source in range(2):
        name = f"source{source}"
        item = dict(
            name=name, room="room", microphone_setup="MicID01",
            microphone=microphone, source_position=f"SpkID{source:02d}",
            metadata={},
        )
        for axis, receiver, speaker in zip(
            ("Depth", "Width", "Height"),
            (2.642571540585832 + source * 4.44e-16, 1.0, 1.5),
            (float(source), 3.0, 1.8),
        ):
            item["metadata"][f"$EnvMic{int(microphone)}{axis}"] = str(receiver)
            item["metadata"][f"$EnvSpk1{axis}"] = str(speaker)
        items.append(item)
        rirs.append((name, np.ones(16), 16000, f"room/{name}"))
    return rirs, items


class TestMeasuredGeometry(unittest.TestCase):
    def test_reads_selected_microphone_and_tolerates_decimal_roundoff(self):
        rirs, inventory = geometry_inventory()
        # Deliberately inconsistent mic-1 coordinates catch a mic-1-only lookup.
        for source, item in enumerate(inventory):
            for axis in ("Depth", "Width", "Height"):
                item["metadata"][f"$EnvMic1{axis}"] = str(100 * source)
        assignment = generation.assign_pairs(rirs, inventory, 10, seed=9)
        self.assertEqual(assignment.shape, (10, 2))
        self.assertTrue(np.all(assignment[:, 0] != assignment[:, 1]))

    def test_rejects_a_receiver_that_moves_between_sources(self):
        rirs, inventory = geometry_inventory()
        inventory[1]["metadata"]["$EnvMic2Depth"] = "2.64258"
        with self.assertRaisesRegex(ValueError, "fixed microphone"):
            generation.assign_pairs(rirs, inventory, 10, seed=9)

    def test_rejects_two_source_ids_at_the_same_physical_position(self):
        rirs, inventory = geometry_inventory()
        inventory[1]["metadata"]["$EnvSpk1Depth"] = "0.0"
        with self.assertRaisesRegex(ValueError, "Repeated physical source"):
            generation.assign_pairs(rirs, inventory, 10, seed=9)


def independent_assignment():
    # A manually constructed four-source cycle covers distinct pairs and gives
    # every source exactly one occurrence in each speaker role per receiver.
    # No production allocation code is used to construct this audit fixture.
    inventory, records, names = [], [], {}
    for room in range(9):
        for channel in range(1, 32):
            for source in range(4):
                name = f"room{room:02d}_mic{channel:02d}_source{source}"
                names[room, channel, source] = name
                inventory.append(dict(
                    name=name, room=f"room{room:02d}",
                    microphone_setup="MicID01", microphone=f"{channel:02d}",
                    source_position=f"SpkID{source:02d}",
                ))
            for source in range(4):
                records.append(dict(but_rirs=[
                    names[room, channel, source],
                    names[room, channel, (source + 1) % 4],
                ]))
    return records, inventory, names


class TestIndependentAssignmentAudit(unittest.TestCase):
    def test_accepts_balanced_complete_receiver_coverage(self):
        records, inventory, _ = independent_assignment()
        audit = generation.audit_assignment(records, inventory, True, True)
        self.assertEqual(audit["mixtures"], 9 * 31 * 4)
        self.assertEqual(audit["receivers_used"], 279)
        self.assertEqual(audit["receivers_available"], 279)
        self.assertEqual(set(audit["room_counts"].values()), {31 * 4})
        self.assertEqual(set(audit["microphone_counts"].values()), {9 * 4})
        self.assertEqual(audit["unique_unordered_receiver_pairs"], 9 * 31 * 4)

    def test_rejects_corrupted_published_assignments(self):
        original, inventory, names = independent_assignment()
        cases = [
            ("different_receiver", "share a receiver"),
            ("same_source", "repeats a source"),
            ("duplicate_pair", "Repeated source pair"),
            ("speaker_role", "speaker/source roles"),
            ("missing_receiver", "cover all 279"),
        ]
        for corruption, message in cases:
            with self.subTest(corruption=corruption):
                records = copy.deepcopy(original)
                if corruption == "different_receiver":
                    records[0]["but_rirs"][1] = names[0, 2, 1]
                elif corruption == "same_source":
                    records[0]["but_rirs"][1] = records[0]["but_rirs"][0]
                elif corruption == "duplicate_pair":
                    records[1]["but_rirs"] = records[0]["but_rirs"][::-1]
                elif corruption == "speaker_role":
                    records[0]["but_rirs"].reverse()
                else:
                    records = records[4:]
                with self.assertRaisesRegex(ValueError, message):
                    generation.audit_assignment(records, inventory, True, True)
