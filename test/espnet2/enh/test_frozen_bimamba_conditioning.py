"""CPU checks for frozen weights, same-observation conditioning, and inference.

The CPU substitute exercises the complete separation/conditioning plumbing.
The actual pretrained Mamba weights are separately checked without executing
CUDA-only kernels. Real-Mamba forward/backward is checked by the recipe's
``local/check_bimamba_conditioning_training.py`` in a GPU allocation.
"""

import argparse
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
from torch import nn

from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.enh.espnet_model_frozen_bimamba import FrozenBiMambaCTF
from espnet2.tasks.enh import EnhancementTask


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "egs2/whamr/enh_rir"
CONFIG = RECIPE / "conf/tuning/train_enh_tflocoformer_s_nf16k_bimamba_ctf_film_all.yaml"
BASELINE = ROOT / "egs2/whamr/enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_16k.yaml"


def options(small=True):
    args = EnhancementTask.get_parser().parse_args(
        ["--config", str(CONFIG), "--output_dir", "/tmp/bimamba_conditioning_test"]
    )
    args.model_conf = copy.deepcopy(args.model_conf)
    if small:
        args.separator_conf.update(
            n_layers=1, emb_dim=8, num_groups=2, n_heads=2,
            attention_dim=8, ffn_hidden_dim=[8, 8], conv1d_kernel=3,
        )
        args.model_conf["bimamba_predictor_conf"].update(
            emb_dim=8, pre_layers=1, post_layers=1, d_state=4, d_conv=2,
            freq_bottleneck=4, freq_rank=4, pooling_hidden=4,
            n_heads=2, ffn_dim=8, head_dim=8,
        )
    return args


class CPUMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, sequence):
        return self.linear(sequence)


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.temp = tempfile.TemporaryDirectory(prefix="bimamba_ctf_test_")
        self.patch = mock.patch(
            "espnet2.rir.rec_rir.pooled_bimamba._build_mamba",
            side_effect=lambda dim, specification: CPUMamba(dim),
        )
        self.patch.start()
        self.args = options()
        component = FrozenBiMambaCTF(self.args.model_conf["bimamba_predictor_conf"])
        state = {"ctf_predictor." + key: value
                 for key, value in component.predictor.state_dict().items()}
        self.source = Path(self.temp.name) / "source.pth"
        torch.save(state, self.source)
        self.args.model_conf.update(
            bimamba_checkpoint=str(self.source),
            bimamba_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
        )
        self.expected = state
        self.model = EnhancementTask.build_model(self.args)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_source_survives_xavier_and_stays_frozen_in_train(self):
        self.model.train()
        self.assertFalse(self.model.frozen_ctf.training)
        for name, value in self.model.frozen_ctf.predictor.state_dict().items():
            torch.testing.assert_close(value, self.expected["ctf_predictor." + name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in self.model.frozen_ctf.parameters()))
        for film in self.model.separator.rir_films:
            self.assertTrue(torch.all(film.gamma_proj.weight == 0))
            self.assertTrue(torch.all(film.gamma_proj.bias == 1))
            self.assertTrue(torch.all(film.beta_proj.weight == 0))
            self.assertTrue(torch.all(film.beta_proj.bias == 0))

    def test_true_length_and_ctf_axis_order(self):
        speech = torch.randn(2, 1024)
        lengths = torch.tensor([1024, 768])
        component = self.model.frozen_ctf
        result = component(speech, lengths)
        self.assertEqual(result.shape, (2, 60, 2, 257, 2))
        self.assertFalse(result.requires_grad)
        changed = speech.clone()
        changed[1, 768:] = 1e5
        torch.testing.assert_close(result, component(changed, lengths), rtol=0, atol=0)
        for i, length in enumerate(lengths):
            wave = speech[i:i+1, :length]
            wave = wave / wave.abs().amax(1, keepdim=True).clamp_min(1e-8)
            spec = component.transforms.stft(wave[:, None], "complex")[:, 0].transpose(1, 2)
            with torch.no_grad():
                raw = component.predictor(spec)[0].flip(-1)
            expected = torch.view_as_real(raw).permute(0, 3, 1, 2, 4)
            torch.testing.assert_close(result[i:i+1], expected, rtol=0, atol=0)

    def test_training_step_updates_film_but_not_predictor(self):
        self.model.train()
        frozen = {n: p.clone() for n, p in self.model.frozen_ctf.named_parameters()}
        gamma = self.model.separator.rir_films[0].gamma_proj.weight.clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        speech = torch.randn(2, 1024)
        loss, _, _ = self.model(
            speech_mix=speech, speech_mix_lengths=torch.tensor([1024, 1024]),
            speech_ref1=torch.randn_like(speech), speech_ref2=torch.randn_like(speech),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.frozen_ctf.parameters()))
        for name, p in self.model.separator.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(gamma, self.model.separator.rir_films[0].gamma_proj.weight))
        for name, p in self.model.frozen_ctf.named_parameters():
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)

    def test_identity_film_matches_baseline_with_same_trunk(self):
        base_args = copy.deepcopy(self.args)
        base_args.separator = "tflocoformer_nocashe"
        base_args.model_conf = {"normalize_variance": True}
        baseline = EnhancementTask.build_model(base_args).eval()
        trunk = baseline.separator.state_dict()
        conditioned = self.model.separator.state_dict()
        for name in trunk:
            trunk[name] = conditioned[name]
        baseline.separator.load_state_dict(trunk)
        speech = torch.randn(1, 1024)
        lengths = torch.tensor([1024])
        with torch.no_grad():
            actual = self.model.eval().forward_enhance(speech, lengths)[0]
            expected = baseline.forward_enhance(speech, lengths)[0]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_complete_checkpoint_without_source(self):
        saved = copy.deepcopy(self.model.state_dict())
        args = copy.deepcopy(self.args)
        args.model_conf["bimamba_checkpoint"] = str(Path(self.temp.name) / "missing.pth")
        clone = EnhancementTask.build_model(args)
        with self.assertRaisesRegex(RuntimeError, "not been loaded"):
            clone.prepare_conditioning(torch.randn(1, 1024), torch.tensor([1024]))
        clone.load_state_dict(saved, strict=True)
        speech, lengths = torch.randn(1, 1024), torch.tensor([1024])
        torch.testing.assert_close(
            clone.prepare_conditioning(speech, lengths)["rir_ctf"],
            self.model.prepare_conditioning(speech, lengths)["rir_ctf"], rtol=0, atol=0,
        )

    def test_bad_digest_and_wrong_sample_rate_fail(self):
        args = copy.deepcopy(self.args)
        args.model_conf["bimamba_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            EnhancementTask.build_model(args)
        with self.assertRaisesRegex(ValueError, "16-kHz"):
            self.model.prepare_conditioning(torch.randn(1, 1024), torch.tensor([1024]), 8000)

    def test_inference_uses_conditioning_for_full_and_segment_inputs(self):
        with mock.patch.object(EnhancementTask, "build_model_from_file",
                               return_value=(self.model, argparse.Namespace())):
            for segment, hop, expected_calls in ((None, None, 1), (0.048, 0.032, 2)):
                inference = SeparateSpeech(segment_size=segment, hop_size=hop)
                with mock.patch.object(self.model, "prepare_conditioning",
                                       wraps=self.model.prepare_conditioning) as prepare:
                    waves = inference(torch.randn(1, 1024), fs=16000)
                    self.assertEqual(prepare.call_count, expected_calls)
                    self.assertEqual(len(waves), 2)
                    self.assertEqual(waves[0].shape, (1, 1024))
                    self.assertTrue(all(torch.isfinite(torch.from_numpy(w)).all() for w in waves))


class RealCheckpointAndConfigTests(unittest.TestCase):
    def test_baseline_conditions_match(self):
        base, new = [yaml.safe_load(path.read_text()) for path in (BASELINE, CONFIG)]
        for key in base.keys() - {"separator", "model_conf"}:
            self.assertEqual(base[key], new[key], key)
        self.assertEqual(base["model_conf"], {"normalize_variance": new["model_conf"]["normalize_variance"]})

    def test_real_checkpoint_strict_load_and_exact_weights(self):
        args = options(small=False)
        path = RECIPE / args.model_conf["bimamba_checkpoint"]
        if not path.is_file():
            self.skipTest("Pretrained BiMamba artifact is not present in this checkout")
        args.model_conf["bimamba_checkpoint"] = str(path)
        model = EnhancementTask.build_model(args)
        source = torch.load(path, map_location="cpu")
        self.assertEqual(sum(p.numel() for p in model.frozen_ctf.parameters()), 456428)
        for name, value in model.frozen_ctf.predictor.state_dict().items():
            torch.testing.assert_close(value, source["ctf_predictor." + name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
