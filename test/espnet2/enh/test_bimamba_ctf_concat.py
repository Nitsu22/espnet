"""DAI-43 concat with frozen estimated CTF, including the actual Task build path."""

import copy
import unittest
from unittest import mock

import torch
import yaml

from espnet2.enh.separator.tflocoformer_separator_nocashe_ctf_concat import (
    TFLocoformerSeparator,
)
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask
from test.espnet2.enh import test_frozen_bimamba_conditioning as film


CONFIG = film.RECIPE / "conf/tuning/train_enh_tflocoformer_s_nf8k_bimamba_ctf_concat.yaml"
ORACLE = (
    film.ROOT / "egs2/whamr/enh9_condition/conf/tuning/"
    "train_enh_tflocoformer_s_nf8k_oracle_ctf_concat.yaml"
)
FILM_OPTIONS = film.options


def options(small=True, sample_rate=8000):
    args = FILM_OPTIONS(small=small, sample_rate=sample_rate)
    config = yaml.safe_load(CONFIG.read_text())
    args.separator = config["separator"]
    args.separator_conf.update(ctf_taps=60, random_ctf_swap=True)
    args.model_conf["bimamba_preserve_initialization_rng"] = True
    return args


class EstimatedCTFConcatTests(film.Conditioning8kTests):
    """Also inherit tap/axis, true-length, source, and inference regressions."""

    def setUp(self):
        patch = mock.patch.object(film, "options", side_effect=options)
        patch.start()
        self.addCleanup(patch.stop)
        super().setUp()

    def test_source_survives_xavier_and_stays_frozen_in_train(self):
        self.model.train()
        self.assertIsInstance(self.model.separator, TFLocoformerSeparator)
        self.assertFalse(self.model.frozen_ctf.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.frozen_ctf.parameters()))
        for name, weight in self.model.frozen_ctf.predictor.state_dict().items():
            torch.testing.assert_close(
                weight, self.expected["ctf_predictor." + name], rtol=0, atol=0
            )

    def test_training_step_updates_film_but_not_predictor(self):
        # Retain the inherited test's name to replace its FiLM-specific check.
        self.model.train()
        frozen = copy.deepcopy(self.model.frozen_ctf.state_dict())
        projection = self.model.separator.ctf_projection.weight.detach().clone()
        speech = torch.randn(2, 1024)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        loss, _, _ = self.model(
            speech_mix=speech, speech_mix_lengths=torch.tensor([1024, 1024]),
            speech_ref1=torch.randn_like(speech), speech_ref2=torch.randn_like(speech),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.frozen_ctf.parameters()))
        for name, parameter in self.model.separator.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(projection, self.model.separator.ctf_projection.weight))
        for name, weight in self.model.frozen_ctf.state_dict().items():
            torch.testing.assert_close(weight, frozen[name], rtol=0, atol=0)

    def test_identity_film_matches_baseline_with_same_trunk(self):
        # This mode is deliberately NOT identity FiLM: test matched Oracle
        # architecture/initialization and a measurable CTF effect instead.
        args = copy.deepcopy(self.args)
        args.separator = "tflocoformer_nocashe_ctf_concat"
        args.model_conf = dict(normalize_variance=True, rir_condition_type="rir_ctf")
        torch.manual_seed(73)
        oracle = EnhancementRIRCTFTask.build_model(args)
        torch.manual_seed(73)
        estimated = EnhancementTask.build_model(self.args).eval()
        for name, weight in oracle.separator.state_dict().items():
            torch.testing.assert_close(
                weight, estimated.separator.state_dict()[name], rtol=0, atol=0
            )
        speech, lengths = torch.randn(1, 1024), torch.tensor([1024])
        with torch.no_grad():
            real = estimated.forward_enhance(speech, lengths)[0]
            condition = estimated.prepare_conditioning(speech, lengths)["rir_ctf"]
            with mock.patch.object(estimated, "prepare_conditioning", return_value={
                "rir_ctf": torch.zeros_like(condition)
            }):
                zero = estimated.forward_enhance(speech, lengths)[0]
        self.assertGreater(sum((a - b).abs().sum().item() for a, b in zip(real, zero)), 0)


class ConcatConfigTests(unittest.TestCase):
    def test_training_and_architecture_match_oracle_and_baseline(self):
        new, oracle = [yaml.safe_load(path.read_text()) for path in (CONFIG, ORACLE)]
        self.assertEqual(new.keys(), oracle.keys())
        for key in oracle.keys() - {"separator", "model_conf"}:
            self.assertEqual(new[key], oracle[key], key)
        baseline_path = film.ROOT / (
            "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_8k.yaml"
        )
        baseline = yaml.safe_load(baseline_path.read_text())
        for key in baseline.keys() - {"separator", "separator_conf", "model_conf"}:
            self.assertEqual(new[key], baseline[key], key)
        self.assertEqual(
            {k: v for k, v in new["separator_conf"].items()
             if k not in ("ctf_taps", "random_ctf_swap")}, baseline["separator_conf"]
        )

    def test_full_config_loads_real_frozen_checkpoint(self):
        args = EnhancementTask.get_parser().parse_args(
            ["--config", str(CONFIG), "--output_dir", str(CONFIG.parent)]
        )
        source = film.RECIPE / args.model_conf["bimamba_checkpoint"]
        if not source.is_file():
            self.skipTest("Pretrained 8-kHz checkpoint not available")
        args.model_conf["bimamba_checkpoint"] = str(source)
        model = EnhancementTask.build_model(args)
        self.assertTrue(model.frozen_ctf.source_loaded.item())
        self.assertEqual(sum(p.numel() for p in model.frozen_ctf.parameters()), 439532)
        self.assertEqual(sum(p.numel() for p in model.separator.ctf_projection.parameters()), 288)
        torch.manual_seed(73)
        estimated = EnhancementTask.build_model(args)
        args.separator = "tflocoformer_nocashe_ctf_concat"
        args.model_conf = dict(normalize_variance=True, rir_condition_type="rir_ctf")
        torch.manual_seed(73)
        oracle = EnhancementRIRCTFTask.build_model(args)
        for name, weight in oracle.separator.state_dict().items():
            torch.testing.assert_close(
                weight, estimated.separator.state_dict()[name], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
