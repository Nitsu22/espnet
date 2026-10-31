"""Check Rec-RIR API parity, freeze/restore, and separation training on CPU."""
import copy
import argparse
import hashlib
from pathlib import Path
import tempfile
from unittest import mock

import unittest
import torch
from torch import nn
import yaml

from espnet2.enh.espnet_model_frozen_recrir import FrozenRecRIRCTF
from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.tasks.enh import EnhancementTask

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "egs2/whamr/enh_rir/conf/tuning/train_enh_tflocoformer_s_nf8k_recrir_ctf_film_all.yaml"


class CPUMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class RecRIRConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.temp = tempfile.TemporaryDirectory(prefix="recrir_conditioning_")
        self.addCleanup(self.temp.cleanup)
        self.patch = mock.patch("espnet2.rir.rec_rir.model._build_mamba",
                                side_effect=lambda dim, attention: CPUMamba(dim))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        args = EnhancementTask.get_parser().parse_args(
            ["--config", str(CONFIG), "--output_dir", self.temp.name])
        args.separator_conf.update(n_layers=1, emb_dim=8, num_groups=2,
                                   n_heads=2, attention_dim=8,
                                   ffn_hidden_dim=[8, 8], conv1d_kernel=3)
        conf = args.model_conf["rec_rir_model_conf"]
        conf.update(dim_hidden=8, dim_squeeze=2, num_layers_spch=1,
                    num_layers_noise=1, num_layers_CTF=1, conv_groups=[2, 2])
        source = Path(self.temp.name) / "source.pth"
        torch.save(FrozenRecRIRCTF(conf).model.state_dict(), source)
        args.model_conf.update(rec_rir_checkpoint=str(source),
                              rec_rir_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        self.conditioned = EnhancementTask.build_model(args), args, source

    def test_public_ctf_api_and_true_lengths(self):
        model, _, _ = self.conditioned
        speech = torch.randn(2, 1024)
        lengths = torch.tensor([1024, 768])
        actual = model.prepare_conditioning(speech, lengths)["rir_ctf"]
        assert actual.shape == (2, 60, 2, 129, 2)
        assert not actual.requires_grad
        for i, length in enumerate(lengths):
            native = model.frozen_ctf.model.estimate_ctf(speech[i:i+1, :length])
            expected = torch.view_as_real(native).permute(0, 3, 1, 2, 4)
            torch.testing.assert_close(actual[i:i+1], expected, rtol=0, atol=0)
        speech[1, 768:] = 1e6
        torch.testing.assert_close(actual, model.prepare_conditioning(speech, lengths)["rir_ctf"], rtol=0, atol=0)


    def test_source_survives_xavier_and_optimizer_step(self):
        model, args, source = self.conditioned
        expected = torch.load(source, map_location="cpu")
        for name, value in model.frozen_ctf.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        model.train()
        assert not model.frozen_ctf.training
        assert not model.frozen_ctf.model.training
        speech = torch.randn(2, 1024)
        gamma = model.separator.rir_films[0].gamma_proj.weight.clone()
        assert torch.all(gamma == 0)
        optimizer = torch.optim.AdamW(model.parameters(), **args.optim_conf)
        loss, _, _ = model(speech_mix=speech, speech_mix_lengths=torch.tensor([1024, 1024]),
                           speech_ref1=torch.randn_like(speech), speech_ref2=torch.randn_like(speech))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(not p.requires_grad and p.grad is None for p in model.frozen_ctf.parameters())
        for name, p in model.separator.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
        optimizer.step()
        assert not torch.equal(gamma, model.separator.rir_films[0].gamma_proj.weight)
        for name, value in model.frozen_ctf.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


    def test_complete_checkpoint_and_missing_source(self):
        model, args, source = self.conditioned
        state = copy.deepcopy(model.state_dict())
        source.unlink()
        clone = EnhancementTask.build_model(args)
        speech, lengths = torch.randn(1, 1024), torch.tensor([1024])
        with self.assertRaisesRegex(RuntimeError, "not been loaded"):
            clone.prepare_conditioning(speech, lengths)
        clone.load_state_dict(state, strict=True)
        torch.testing.assert_close(clone.prepare_conditioning(speech, lengths)["rir_ctf"],
                                   model.prepare_conditioning(speech, lengths)["rir_ctf"], rtol=0, atol=0)


    def test_bad_hash_and_rate_fail(self):
        model, args, _ = self.conditioned
        args.model_conf["rec_rir_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            EnhancementTask.build_model(args)
        with self.assertRaisesRegex(ValueError, "8-kHz"):
            model.prepare_conditioning(torch.randn(1, 1024), torch.tensor([1024]), 16000)

    def test_full_and_segmented_inference_use_rec_rir(self):
        model, _, _ = self.conditioned
        with mock.patch.object(EnhancementTask, "build_model_from_file",
                               return_value=(model, argparse.Namespace())):
            for segment_size, expected_calls in ((None, 1), (0.064, 2)):
                inference = SeparateSpeech(segment_size=segment_size, hop_size=0.032)
                with mock.patch.object(model, "prepare_conditioning",
                                       wraps=model.prepare_conditioning) as prepare:
                    waves = inference(torch.randn(1, 768), fs=8000)
                    self.assertEqual(prepare.call_count, expected_calls)
                    self.assertEqual(len(waves), 2)
                    self.assertEqual(waves[0].shape, (1, 768))


    def test_baseline_and_original_estimator_configs_match(self):
        new = yaml.safe_load(CONFIG.read_text())
        base = yaml.safe_load((ROOT / "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_8k.yaml").read_text())
        for key in base.keys() - {"separator", "model_conf"}:
            assert new[key] == base[key], key
        source = ROOT / "egs2/whamr/rir/conf/tuning/train_rec_rir_2spk_pit_clean_8192.yaml"
        assert new["model_conf"]["rec_rir_model_conf"] == yaml.safe_load(source.read_text())["model_conf"]

if __name__ == "__main__":
    unittest.main()
