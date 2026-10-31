"""Frozen Rec-RIR uses the DAI-117 separator, including checkpoint inference."""
import copy
import torch
import yaml

from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask
from test.espnet2.enh import test_frozen_recrir_conditioning as base


CONFIG = base.ROOT / "egs2/whamr/enh_rir/conf/tuning/train_enh_tflocoformer_s_nf8k_recrir_ctf_concat_glue.yaml"


class RecRIRConcatGlueTests(base.RecRIRConditioningTests):
    def setUp(self):
        super().setUp()
        _, args, source = self.conditioned
        args = copy.deepcopy(args)
        args.separator = "tflocoformer_recrir_ctf_concat"
        args.separator_conf.update(glue_frames=4, glue_value=0.0)
        self.conditioned = EnhancementTask.build_model(args), args, source

    def test_source_survives_xavier_and_optimizer_step(self):
        model, args, source = self.conditioned
        expected = torch.load(source, map_location="cpu")
        model.train()
        self.assertFalse(model.frozen_ctf.training)
        self.assertFalse(model.frozen_ctf.model.training)
        before = model.separator.ctf_projection.weight.clone()
        speech = torch.randn(2, 1024)
        optimizer = torch.optim.AdamW(model.parameters(), **args.optim_conf)
        loss, _, _ = model(speech_mix=speech, speech_mix_lengths=torch.tensor([1024, 1024]),
                           speech_ref1=torch.randn_like(speech), speech_ref2=torch.randn_like(speech))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.separator.ctf_projection.weight.grad.abs().sum().item(), 0)
        for name, p in model.named_parameters():
            if name.startswith("frozen_ctf."):
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
            elif p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.separator.ctf_projection.weight))
        for name, value in model.frozen_ctf.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_baseline_and_original_estimator_configs_match(self):
        new = yaml.safe_load(CONFIG.read_text())
        baseline = yaml.safe_load((base.ROOT / "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_8k.yaml").read_text())
        for key in baseline.keys() - {"separator", "separator_conf", "model_conf"}:
            self.assertEqual(new[key], baseline[key], key)
        conf = dict(new["separator_conf"])
        self.assertEqual(conf.pop("glue_frames"), 4)
        self.assertEqual(conf.pop("glue_value"), 0.0)
        self.assertEqual(conf, baseline["separator_conf"])
        original = base.ROOT / "egs2/whamr/rir/conf/tuning/train_rec_rir_2spk_pit_clean_8192.yaml"
        self.assertEqual(new["model_conf"]["rec_rir_model_conf"], yaml.safe_load(original.read_text())["model_conf"])

    def test_separator_initialization_exactly_matches_oracle_glue(self):
        _, args, _ = self.conditioned
        oracle_args = copy.deepcopy(args)
        oracle_args.separator = "tflocoformer_nocashe_ctf_concat"
        oracle_args.model_conf = dict(normalize_variance=True, rir_condition_type="rir_ctf")
        torch.manual_seed(23)
        estimated = EnhancementTask.build_model(args)
        torch.manual_seed(23)
        oracle = EnhancementRIRCTFTask.build_model(oracle_args)
        actual = {k: v for k, v in estimated.state_dict().items() if not k.startswith("frozen_ctf.")}
        self.assertEqual(set(actual), set(oracle.state_dict()))
        for name, value in oracle.state_dict().items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)

    def test_restored_model_keeps_zero_glue(self):
        model, args, source = self.conditioned
        state = copy.deepcopy(model.state_dict())
        source.unlink()
        clone = EnhancementTask.build_model(args)
        clone.load_state_dict(state, strict=True)
        self.assertEqual(clone.separator.glue_frames, 4)
        self.assertEqual(clone.separator.glue_value, 0.0)
