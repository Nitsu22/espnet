"""Reject mislabeled or incomplete measured MIRD source data."""

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat


LOCAL = Path(__file__).resolve().parents[3] / "egs2/whamr/rir_1ch/local"
sys.path.insert(0, str(LOCAL))
import download_mird as DOWNLOADER  # noqa: E402
import mird_rir as MODULE  # noqa: E402
sys.path.pop(0)


class TestMIRDMeasuredData(unittest.TestCase):
    @staticmethod
    def mat_bytes(**changes):
        rir = np.zeros((480000, 8), dtype=np.float32)
        # Distinct signed arrivals expose channel ordering and accidental trim.
        rir[777 + np.arange(8), np.arange(8)] = np.arange(1, 9) * -0.1
        meta = dict(type="impulse_response", room="Acoustic_Lab_Bar-Ilan_University",
                    reverberation=0.61, mic_spacing=[3, 3, 3, 8, 3, 3, 3],
                    distance=2, azimuth=270,
                    mic_position="mic1@90deg, mic8@270deg")
        meta.update(changes.pop("metadata", {}))
        sim = dict(fs=48000, in_ch=np.arange(1, 9))
        sim.update(changes.pop("simulation", {}))
        mat = dict(metapar=meta, simpar=sim, impulse_response=rir)
        mat.update(changes)
        stream = io.BytesIO()
        savemat(stream, mat, do_compression=True)
        return stream.getvalue()

    def test_native_timing_polarity_and_channels_are_preserved(self):
        rir, record = MODULE.inspect_response(self.mat_bytes(), 610,
                                              "3-3-3-8-3-3-3", 2, 270)
        self.assertEqual(rir.shape, (480000, 8))
        self.assertEqual(record["peak_samples"], list(range(777, 785)))
        np.testing.assert_allclose(rir[777 + np.arange(8), np.arange(8)],
                                   np.arange(1, 9) * -0.1)
        self.assertEqual(record["native_azimuth_deg"], 270)

    def test_mat_metadata_overrides_untrusted_outer_filename(self):
        for fields, expected in (
            ({"metadata": {"reverberation": 0.36}}, "reverberation"),
            ({"metadata": {"mic_spacing": [8] * 7}}, "spacing"),
            ({"metadata": {"azimuth": 90}}, "azimuth"),
            ({"metadata": {"distance": 1}}, "distance"),
            ({"simulation": {"fs": 16000}}, "sample_rate"),
            ({"simulation": {"in_ch": np.arange(8, 0, -1)}}, "channel_order"),
        ):
            with self.subTest(fields=fields), self.assertRaisesRegex(
                ValueError, expected
            ):
                MODULE.inspect_response(self.mat_bytes(**fields), 610,
                                        "3-3-3-8-3-3-3", 2, 270)

    def test_nonfinite_silent_and_incorrect_length_are_rejected(self):
        for signal in (np.full((480000, 8), np.nan, dtype=np.float32),
                       np.zeros((480000, 8), dtype=np.float32),
                       np.ones((160000, 8), dtype=np.float32)):
            with self.subTest(shape=signal.shape), self.assertRaises(ValueError):
                MODULE.inspect_response(self.mat_bytes(impulse_response=signal),
                                        610, "3-3-3-8-3-3-3", 2, 270)

    def test_official_nested_wrong_condition_is_rejected_and_fingerprinted(self):
        import hashlib

        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("unrelated.mat", b"wrong condition")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outer.zip"
            wrong_name = MODULE.archive_stem(360, "8-8-8-8-8-8-8") + ".zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(wrong_name, inner.getvalue())
            report = MODULE.inspect_archive(path, 610, "3-3-3-8-3-3-3")
        self.assertFalse(report["valid"])
        self.assertEqual(len(report["missing_members"]), 26)
        self.assertEqual(report["nested_archives"], [dict(
            name=wrong_name, sha256=hashlib.sha256(inner.getvalue()).hexdigest())])

    def test_duplicate_members_cannot_stand_in_for_missing_source_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.zip"
            name = MODULE.member_name(610, "3-3-3-8-3-3-3", 1, 0)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(name, b"placeholder")
                with self.assertWarns(UserWarning):
                    archive.writestr(name, b"placeholder")
            report = MODULE.inspect_archive(path, 610, "3-3-3-8-3-3-3")
        self.assertFalse(report["valid"])

    def test_one_bad_session_blocks_the_entire_production_dataset(self):
        calls = []

        def inspect(path, rt, spacing):
            calls.append((rt, spacing))
            good = (rt, spacing) != (610, "3-3-3-8-3-3-3")
            return dict(valid=good, native_recordings=26 if good else 0)

        with patch.object(MODULE, "inspect_archive", side_effect=inspect):
            report = MODULE.audit_dataset("unused")
        self.assertEqual(len(calls), 9)
        self.assertFalse(report["all_conditions_valid"])
        self.assertEqual(report["status"], "blocked_missing_native_rir")
        self.assertEqual(report["native_recordings"], 208)

    def test_existing_crc_valid_downloads_do_not_bypass_native_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = []
            for rt in MODULE.RT60_MS:
                for spacing in MODULE.SPACINGS:
                    name = MODULE.archive_stem(rt, spacing) + ".zip"
                    inventory.append(["public_id", name, "application/zip"])
            inventory.append(["tools_id", "SP_Package_V2.zip", "application/zip"])
            for _, name, _ in inventory:
                with zipfile.ZipFile(root / name, "w") as archive:
                    archive.writestr("wrong_condition.txt", "not measured RIR data")
            originals = {name: (root / name).read_bytes() for _, name, _ in inventory}
            index = root / "inventory.json"
            index.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(ValueError, "Native MIRD conditions"):
                DOWNLOADER.download(index, root)
            audit = json.loads((root / "rir_audit.json").read_text())
            self.assertFalse(audit["all_conditions_valid"])
            transport = json.loads((root / "download_transport.json").read_text())
            self.assertEqual(transport["status"], "transport_integrity_only")
            self.assertEqual(len(transport["files"]), 10)
            self.assertFalse((root / "download_complete.json").exists())
            for name, content in originals.items():
                self.assertEqual((root / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
