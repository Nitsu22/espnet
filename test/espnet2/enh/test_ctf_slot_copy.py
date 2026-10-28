"""Exact speaker-axis CTF duplication, no waveform or weight changes."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "egs2/whamr/enh_rir/local"))
from ctf_audio_intervention import ctf_slot_override
import evaluate_ctf_audio_controls as evaluate

fixture_spec = importlib.util.spec_from_file_location(
    "copy_fixture", Path(__file__).with_name("test_ctf_audio_intervention.py"))
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor(2.))
        self.invalid = None

    def forward(self, wave, lengths):
        self.last_wave = wave.clone()
        value = torch.arange(60 * 2 * 129 * 2, dtype=wave.dtype).reshape(1, 60, 2, 129, 2)
        self.last_output = value * self.weight
        if self.invalid == "shape": return self.last_output[:, :, :, :-1]
        if self.invalid == "nan": return self.last_output * float("nan")
        return self.last_output


class Separator:
    segmenting, num_spk, device, dtype = False, 2, "cpu", "float32"

    def __init__(self):
        self.enh_model = nn.Module()
        self.enh_model.frozen_ctf = Predictor()
        self.fail = False

    def __call__(self, mixture, fs):
        assert fs == 8000
        wave = torch.as_tensor(mixture)
        wave = wave / torch.std(wave, dim=1, keepdim=True)
        self.encoder_wave = wave.clone()
        self.ctf = self.enh_model.frozen_ctf(wave, torch.tensor([wave.shape[1]]))
        if self.fail: raise RuntimeError("intentional failure")
        return [(wave + self.ctf[:, :, k].mean() * 1e-6).numpy() for k in (0, 1)]


class SlotCopyTest(unittest.TestCase):
    def setUp(self):
        self.separate = Separator()
        self.wave = np.linspace(-.4, .4, 512, dtype=np.float32)[None]

    def test_identity_and_copy_preserve_all_other_axes_and_inputs(self):
        separate = self.separate
        baseline = separate(self.wave, 8000)
        encoder = separate.encoder_wave.clone()
        native = separate.ctf.clone()
        weights = {k: v.clone() for k, v in separate.enh_model.state_dict().items()}
        identity, identity_record = ctf_slot_override(separate, self.wave, (0, 1))
        for a, b in zip(baseline, identity): np.testing.assert_array_equal(a, b)
        self.assertEqual(identity_record["source_ctf_sha256"], identity_record["ctf_sha256"])
        changed, record = ctf_slot_override(separate, self.wave)
        torch.testing.assert_close(separate.ctf[:, :, 0], native[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(separate.ctf[:, :, 1], native[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(separate.enh_model.frozen_ctf.last_output, native, rtol=0, atol=0)
        torch.testing.assert_close(separate.encoder_wave, encoder, rtol=0, atol=0)
        torch.testing.assert_close(separate.enh_model.frozen_ctf.last_wave, encoder, rtol=0, atol=0)
        for key, value in weights.items():
            torch.testing.assert_close(separate.enh_model.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(record["slot_indices"], [0, 0])
        self.assertNotEqual(record["source_slot_sha256"][0], record["source_slot_sha256"][1])
        self.assertEqual(record["used_slot_sha256"], [record["source_slot_sha256"][0]] * 2)
        np.testing.assert_array_equal(changed[0], baseline[0])
        self.assertFalse(np.array_equal(changed[1], baseline[1]))
        for a, b in zip(separate(self.wave, 8000), baseline): np.testing.assert_array_equal(a, b)

    def test_cleanup_on_inference_and_predictor_validation_failure(self):
        self.separate.fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            ctf_slot_override(self.separate, self.wave)
        self.assertFalse(self.separate.enh_model.frozen_ctf._forward_hooks)
        self.separate.fail = False
        for mode in ("shape", "nan"):
            self.separate.enh_model.frozen_ctf.invalid = mode
            with self.assertRaisesRegex(ValueError, "Invalid estimated CTF"):
                ctf_slot_override(self.separate, self.wave)
            self.assertFalse(self.separate.enh_model.frozen_ctf._forward_hooks)

    def test_invalid_slots_and_waveforms(self):
        for slots in ((0,), (0, 2), (-1, 0), (True, 0)):
            with self.assertRaises(ValueError): ctf_slot_override(self.separate, self.wave, slots)
        for wave in (self.wave[0], np.full_like(self.wave, np.nan)):
            with self.assertRaises(ValueError): ctf_slot_override(self.separate, wave)
        self.separate.segmenting = True
        with self.assertRaises(ValueError): ctf_slot_override(self.separate, self.wave)

    def test_dispatch_requires_original_estimator_waveform(self):
        waves, record = evaluate.run_condition(self.separate, self.wave, self.wave.copy(), "ctf1_ctf1")
        self.assertEqual(record["slot_indices"], [0, 0])
        self.assertEqual(len(waves), 2)
        with self.assertRaisesRegex(ValueError, "original mixture"):
            evaluate.run_condition(self.separate, self.wave, -self.wave, "ctf1_ctf1")


class CopyPreparationTest(unittest.TestCase):
    def test_original_indexes_refs_scores_and_summary_unchanged(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_copy_test_", dir=ROOT) as directory:
            root = Path(directory)
            old, settings = fixtures.prepared_fixture(root)
            evaluate.prepare(old)
            args = argparse.Namespace(output=root / "copy_evaluation", source_evaluation=old.output,
                                      expected_count=4)
            evaluate.prepare_ctf_copy(args)
            spec = evaluate.load_spec(args.output)
            self.assertEqual(spec["conditions"], {"ctf1_ctf1": "tt_ctf1_ctf1_8k"})
            self.assertEqual(spec["ctf_slot_indices"], {"ctf1_ctf1": [0, 0]})
            self.assertEqual(spec["score_dirs"]["baseline"], str(old.baseline_score))
            originals = set()
            for name in evaluate.NAMES:
                values = evaluate.read_index(settings["baseline_test"] / name)
                originals.update(values.values())
                self.assertEqual(evaluate.read_index(args.output / "inputs/original" / name), values)
            self.assertEqual(set(spec["audio_hashes"]), originals)
            self.assertEqual(evaluate.read_index(args.output / "inputs/ctf1_ctf1.scp"),
                             evaluate.read_index(settings["baseline_test"] / "wav.scp"))
            target = args.output / "ctf1_ctf1/score"
            target.mkdir(parents=True)
            for metric in evaluate.METRICS:
                for k in (1, 2):
                    evaluate.save_index(target / f"{metric}_spk{k}", {uid: 16. for uid in spec["keys"]})
            evaluate.summarize(args.output, spec)
            with (args.output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["ctf1_ctf1_delta_matched"]), -2.)
            with self.assertRaises(FileExistsError): evaluate.prepare_ctf_copy(args)
            args.output = root / "mismatched_count"
            args.expected_count = 5
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                evaluate.prepare_ctf_copy(args)
            self.assertFalse(args.output.exists())

    def test_changed_original_audio_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_copy_test_", dir=ROOT) as directory:
            root = Path(directory)
            old, settings = fixtures.prepared_fixture(root)
            evaluate.prepare(old)
            path = next(iter(evaluate.read_index(settings["baseline_test"] / "wav.scp").values()))
            with Path(path).open("ab") as stream: stream.write(b"corrupt")
            args = argparse.Namespace(output=root / "copy_evaluation", source_evaluation=old.output,
                                      expected_count=4)
            with self.assertRaisesRegex(ValueError, "changed"): evaluate.prepare_ctf_copy(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
