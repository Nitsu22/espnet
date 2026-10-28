"""CPU tests: intervention routes audio only to CTF, never the separator."""

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "egs2/whamr/enh_rir/local"
sys.path.insert(0, str(LOCAL))
from ctf_audio_intervention import conditioning_override
import evaluate_ctf_audio_controls as evaluate
import prepare_nf8k_ctf_controls as generate

fixture_spec = importlib.util.spec_from_file_location(
    "controls_fixture", Path(__file__).with_name("test_same_speaker_test_data.py"))
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)
wrong_spec = importlib.util.spec_from_file_location(
    "wrong_rir_eval_fixture", Path(__file__).with_name("test_ctf_wrong_rir_data.py"))
wrong_fixtures = importlib.util.module_from_spec(wrong_spec)
wrong_spec.loader.exec_module(wrong_fixtures)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor(2.))
        self.last_wave = None

    def forward(self, wave, lengths):
        self.last_wave = wave.clone()
        return (wave.mean() * self.weight).expand(1, 60, 2, 129, 2).clone()


class Separator:
    segmenting = False
    num_spk = 2
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.enh_model = nn.Module()
        self.enh_model.frozen_ctf = Predictor()
        self.enh_model.normalize_variance = True
        self.fail = False

    def __call__(self, mixture, fs):
        assert fs == 8000
        wave = torch.as_tensor(mixture)
        wave = wave / torch.std(wave, dim=1, keepdim=True)
        self.encoder_wave = wave.clone()
        ctf = self.enh_model.frozen_ctf(wave, torch.tensor([wave.shape[1]]))
        if self.fail:
            raise RuntimeError("intentional failure")
        return [(wave + ctf.mean()).numpy(), (wave - ctf.mean()).numpy()]


class InterventionTest(unittest.TestCase):
    def setUp(self):
        self.separator = Separator()
        self.wave = np.linspace(-.4, .4, 512, dtype=np.float32)[None]

    def test_identity_and_only_predictor_changes(self):
        separator = self.separator
        baseline = separator(self.wave, 8000)
        encoder = separator.encoder_wave.clone()
        weights = separator.enh_model.frozen_ctf.weight.clone()
        identity, ctf1 = conditioning_override(separator, self.wave, self.wave)
        for a, b in zip(baseline, identity):
            np.testing.assert_array_equal(a, b)
        alternate = self.wave * -.5 + .2
        changed, ctf2 = conditioning_override(separator, self.wave, alternate)
        self.assertNotEqual(ctf1, ctf2)
        self.assertFalse(np.array_equal(changed[0], baseline[0]))
        torch.testing.assert_close(separator.encoder_wave, encoder, rtol=0, atol=0)
        torch.testing.assert_close(separator.enh_model.frozen_ctf.weight, weights, rtol=0, atol=0)
        expected = torch.from_numpy(alternate)
        expected = expected / torch.std(expected, dim=1, keepdim=True)
        torch.testing.assert_close(separator.enh_model.frozen_ctf.last_wave, expected, rtol=0, atol=0)
        for a, b in zip(separator(self.wave, 8000), baseline):
            np.testing.assert_array_equal(a, b)

    def test_hook_cleanup_on_failure(self):
        self.separator.fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            conditioning_override(self.separator, self.wave, self.wave)
        module = self.separator.enh_model.frozen_ctf
        self.assertFalse(module._forward_hooks)
        self.assertFalse(module._forward_pre_hooks)

    def test_invalid_interventions_rejected(self):
        for alternate in (self.wave[:, :-1], np.full_like(self.wave, np.nan),
                          np.zeros_like(self.wave)):
            with self.assertRaises(ValueError):
                conditioning_override(self.separator, self.wave, alternate)
        self.separator.segmenting = True
        with self.assertRaises(ValueError):
            conditioning_override(self.separator, self.wave, self.wave)


def prepared_fixture(root, wrong_rir=False):
    settings = wrong_fixtures.fixture(root) if wrong_rir else fixtures.fixture(root)
    dump = root / "generated"
    generate.prepare(**settings, output=dump,
                     conditions=["same_utt_other_rir"] if wrong_rir else None)
    moved = root / "transferred"
    shutil.copytree(dump, moved)
    keys = sorted(generate.read_scp(settings["baseline_test"] / "wav.scp"))
    model = root / "model"
    model.mkdir()
    (model / "config.yaml").write_text("model: test\n")
    (model / "129epoch.pth").write_bytes(b"immutable-fixture")
    scores = {}
    for label in ("baseline", "matched"):
        directory = root / label / "score"
        directory.mkdir(parents=True)
        for metric in evaluate.METRICS:
            for k in (1, 2):
                evaluate.save_index(directory / f"{metric}_spk{k}",
                                    {uid: 17. + (label == "matched") for uid in keys})
        scores[label] = directory
    for k in (1, 2):
        shutil.copyfile(settings["baseline_test"] / f"spk{k}.scp",
                        scores["matched"].parent / f"spk{k}.scp")
    args = argparse.Namespace(output=root / "evaluation", dump_root=moved,
                              original_test=settings["baseline_test"],
                              original_recipe=settings["baseline_recipe"],
                              model_exp=model, baseline_score=scores["baseline"],
                              matched_score=scores["matched"], expected_count=4)
    return args, settings


class EvaluationPreparationTest(unittest.TestCase):
    def test_single_wrong_rir_evaluation_keeps_original_refs(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_eval_test_", dir=ROOT) as directory:
            root = Path(directory)
            args, settings = prepared_fixture(root, wrong_rir=True)
            evaluate.prepare(args)
            spec = evaluate.load_spec(args.output)
            self.assertEqual(spec["conditions"], {"same_utt_other_rir": "tt_same_utt_other_rir_8k"})
            for name in evaluate.NAMES:
                self.assertEqual(evaluate.read_index(args.output / "inputs/original" / name),
                                 evaluate.read_index(settings["baseline_test"] / name))
            values = evaluate.read_index(args.output / "inputs/same_utt_other_rir.scp")
            self.assertTrue(all(Path(p).is_relative_to(args.dump_root) for p in values.values()))
            for name in evaluate.CONDITIONS:
                self.assertFalse((args.output / "inputs" / f"{name}.scp").exists())
            target = args.output / "same_utt_other_rir/score"
            target.mkdir(parents=True)
            for metric in evaluate.METRICS:
                for k in (1, 2):
                    evaluate.save_index(target / f"{metric}_spk{k}", {uid: 16. for uid in spec["keys"]})
            evaluate.summarize(args.output, spec)
            with (args.output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["same_utt_other_rir_delta_matched"]), -2.)
            self.assertNotIn("same_rir_other_utt", row)

    def test_wrong_rir_audit_rejects_fixedpoint_and_broken_pair(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_eval_test_", dir=ROOT) as directory:
            args, _ = prepared_fixture(Path(directory), wrong_rir=True)
            with (args.dump_root / "audit.tsv").open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            audits = {(r["condition"], r["scene_uid"]): r for r in rows}
            keys = {r["scene_uid"] for r in rows}
            evaluate.validate_other_rir_audit(audits, keys)
            changed = copy.deepcopy(audits)
            uid = rows[0]["scene_uid"]
            changed["same_utt_other_rir", uid]["donor_rir_uid"] = uid
            with self.assertRaisesRegex(ValueError, "another original test"):
                evaluate.validate_other_rir_audit(changed, keys)
            changed = copy.deepcopy(audits)
            changed["same_utt_other_rir", uid]["rir1_reverb_sha256"] = "not-the-donor"
            with self.assertRaisesRegex(ValueError, "coherent pair"):
                evaluate.validate_other_rir_audit(changed, keys)
            changed = copy.deepcopy(audits)
            duplicate = next(row for row in rows if row["scene_uid"] != uid and row["donor_rir_uid"] != uid)
            target = changed["same_utt_other_rir", uid]
            target["donor_rir_uid"] = duplicate["donor_rir_uid"]
            target["source_t60"] = duplicate["source_t60"]
            for k in (1, 2):
                for kind in ("anechoic", "reverb"):
                    name = f"rir{k}_{kind}"
                    target[name] = duplicate[name]
                    target[f"{name}_sha256"] = duplicate[f"{name}_sha256"]
            with self.assertRaisesRegex(ValueError, "bijection"):
                evaluate.validate_other_rir_audit(changed, keys)

    def test_condition_selection_is_backwards_compatible_and_strict(self):
        self.assertEqual(evaluate.selected_conditions({}), evaluate.CONDITIONS)
        self.assertEqual(evaluate.selected_conditions({"condition_definitions": {"same_utt_other_rir": "definition"}},
                                                     generation=True),
                         {"same_utt_other_rir": "tt_same_utt_other_rir_8k"})
        for source in ({"conditions": {}}, {"conditions": {"unknown": "bad"}},
                       {"conditions": {"same_utt_other_rir": "wrong_dataset"}}):
            with self.assertRaises(ValueError):
                evaluate.selected_conditions(source)

    def test_transfer_rebase_and_fixed_original_inputs(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_eval_test_", dir=ROOT) as directory:
            root = Path(directory)
            args, settings = prepared_fixture(root)
            evaluate.prepare(args)
            spec = evaluate.load_spec(args.output)
            self.assertEqual(spec["count"], 4)
            original = evaluate.read_index(settings["baseline_test"] / "wav.scp")
            self.assertEqual(evaluate.read_index(args.output / "inputs/original/wav.scp"), original)
            for condition in evaluate.CONDITIONS:
                values = evaluate.read_index(args.output / "inputs" / f"{condition}.scp")
                self.assertEqual(set(values), set(original))
                self.assertTrue(all(Path(p).is_relative_to(args.dump_root) for p in values.values()))
                self.assertFalse(set(values.values()) & set(original.values()))
            for k in (1, 2):
                self.assertEqual(evaluate.read_index(args.output / f"inputs/original/spk{k}.scp"),
                                 evaluate.read_index(settings["baseline_test"] / f"spk{k}.scp"))
            with self.assertRaises(FileExistsError):
                evaluate.prepare(args)
            with (args.output / "inputs/original/wav.scp").open("a") as stream:
                stream.write("corrupt entry\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate.load_spec(args.output)

    def test_changed_transfer_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_eval_test_", dir=ROOT) as directory:
            root = Path(directory)
            args, _ = prepared_fixture(root)
            path = next(args.dump_root.rglob("*.wav"))
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate.prepare(args)
            self.assertFalse(args.output.exists())

    def test_changed_original_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_eval_test_", dir=ROOT) as directory:
            root = Path(directory)
            args, settings = prepared_fixture(root)
            path = next(iter(evaluate.read_index(settings["baseline_test"] / "spk1.scp").values()))
            with Path(path).open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate.prepare(args)
            self.assertFalse(args.output.exists())

    def test_summary_uses_paired_original_keys(self):
        with tempfile.TemporaryDirectory(prefix=".ctf_eval_test_", dir=ROOT) as directory:
            root = Path(directory)
            args, _ = prepared_fixture(root)
            evaluate.prepare(args)
            spec = evaluate.load_spec(args.output)
            for condition, value in zip(evaluate.CONDITIONS, (18.5, 18.25)):
                target = args.output / condition / "score"
                target.mkdir(parents=True)
                for metric in evaluate.METRICS:
                    for k in (1, 2):
                        evaluate.save_index(target / f"{metric}_spk{k}",
                                            {uid: value for uid in spec["keys"]})
            evaluate.summarize(args.output, spec)
            with (args.output / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            row = next(row for row in rows if row["metric"] == "SI_SNR")
            self.assertEqual(float(row["baseline"]), 17.)
            self.assertEqual(float(row["matched"]), 18.)
            self.assertEqual(float(row["same_rir_other_utt_delta_matched"]), .5)
            self.assertEqual(float(row["same_utt_swapped_rir_delta_baseline"]), 1.25)


if __name__ == "__main__":
    unittest.main()
