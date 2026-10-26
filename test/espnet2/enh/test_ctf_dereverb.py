"""End-to-end CTF model, shared PIT, normalization and real-context checks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model_ctf_dereverb import ESPnetCTFDereverbModel
from espnet2.enh.layers.ctf_deconvolution import ctf_forward
from espnet2.enh.separator.tflocoformer_ctf_dereverb import (
    TFLocoformerCTFDereverbBaselineSeparator,
    TFLocoformerCTFDereverbSeparator,
)
from espnet2.train.preprocessor_ctf_dereverb import CTFDereverbPreprocessor


def model(stage=1, baseline=False, gradients=0):
    conf = dict(n_layers=1, emb_dim=8, n_heads=2, attention_dim=8,
                num_groups=2, ffn_hidden_dim=[8, 8], conv1d_kernel=4)
    if baseline:
        separator = TFLocoformerCTFDereverbBaselineSeparator(33, **conf)
    else:
        separator = TFLocoformerCTFDereverbSeparator(
            33, ctf_taps=4, ctf_hidden_dim=8, mamba_d_state=2,
            mamba_d_conv=2, mamba_backend="torch", cg_iterations=8,
            use_deconvolution=stage == 2, **conf,
        )
    return ESPnetCTFDereverbModel(
        encoder=STFTEncoder(n_fft=64, hop_length=16),
        separator=separator, decoder=STFTDecoder(n_fft=64, hop_length=16),
        mask_module=None, loss_wrappers=[], normalize_variance=True,
        training_stage=stage, baseline_mode=baseline,
        component_grad_interval=gradients,
    )


class TestCTFContext(unittest.TestCase):
    def test_actual_context_and_target_alignment(self):
        proc = CTFDereverbPreprocessor(
            train=True, ctf_taps=4, ctf_n_fft=64, ctf_hop_length=16,
            speech_segment=128, num_spk=2, force_single_channel=True,
        )
        waveform = np.arange(1024, dtype=np.float32) / 2048
        original = {name: waveform.copy() * weight for name, weight in (
            ("speech_mix", 1), ("speech_ref1", .3), ("speech_ref2", .4),
            ("speech_reverb1", .5), ("speech_reverb2", .6),
        )}
        with patch("numpy.random.randint", return_value=20):
            actual = proc("example", {name: value.copy() for name, value in original.items()})
        left = 320 - (3 * 16 + 64)
        right = 320 + 128 + 64
        self.assertEqual(actual["ctf_loss_start"].item(), 112)
        self.assertEqual(actual["ctf_loss_end"].item(), 240)
        self.assertEqual(actual["ctf_crop_start"].item(), left)
        self.assertEqual(actual["ctf_original_length"].item(), 1024)
        for name in original:
            np.testing.assert_array_equal(actual[name], original[name][left:right])
        validation = CTFDereverbPreprocessor(
            train=False, ctf_taps=4, ctf_n_fft=64, ctf_hop_length=16,
            num_spk=2, force_single_channel=True,
        )
        result = validation("mix_only", {"speech_mix": waveform.copy()})
        self.assertEqual(result["ctf_loss_start"].item(), 0)
        self.assertEqual(result["ctf_loss_end"].item(), 1024)

    def test_baseline_has_original_parameter_count_and_no_mamba(self):
        from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator
        args = dict(n_layers=1, emb_dim=8, attention_dim=8, n_heads=2,
                    num_groups=2, ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
                    ffn_hidden_dim=[8, 8], conv1d_kernel=4, norm_type="rmsgroupnorm")
        expected = TFLocoformerSeparator(33, **args)
        actual = model(baseline=True).separator
        self.assertEqual(sum(p.numel() for p in expected.parameters()),
                         sum(p.numel() for p in actual.parameters()))
        self.assertFalse(hasattr(actual, "ctf_estimator"))


@unittest.skipUnless(importlib.util.find_spec("mamba_ssm"), "mamba_ssm unavailable")
class TestCTFDereverb(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_stage1_stage2_backprop_and_silence(self):
        for stage in (1, 2):
            network = model(stage, gradients=1)
            lengths = torch.tensor([256, 192])
            refs = [torch.randn(2, 256) for _ in range(2)]
            for ref in refs:
                ref[1, 192:] = 0
            mixture = refs[0] + refs[1]
            kwargs = dict(speech_ref1=refs[0], speech_ref2=refs[1],
                          speech_reverb1=1.1*refs[0], speech_reverb2=1.1*refs[1])
            loss, stats, weight = network(mixture, lengths, **kwargs)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(int(weight), 2)
            self.assertIn("grad_norm_ctf", stats)
            loss.backward()
            for name, parameter in network.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            torch.optim.AdamW(network.parameters(), lr=1e-4).step()
            network.eval()
            zero = torch.zeros(1, 256)
            loss, _, _ = network(zero, speech_ref1=zero, speech_ref2=zero,
                                 speech_reverb1=zero, speech_reverb2=zero)
            self.assertTrue(torch.isfinite(loss))

    def test_padding_scale_and_mixture_only_prediction(self):
        network = model(stage=2).eval()
        x = torch.randn(2, 256)
        lengths = torch.tensor([256, 192])
        with torch.no_grad():
            original = network.predict_all(x, lengths)
            x[1, 192:] = 1e4
            changed = network.predict_all(x, lengths)
            single = network.predict_all(x[1:2, :192])
            scaled = network.predict_all(7 * x, lengths)
        for key in ("reverb", "clean_prior", "clean_deconv", "clean_final"):
            torch.testing.assert_close(original[key], changed[key])
            torch.testing.assert_close(original[key][1:2, ..., :13], single[key])
            torch.testing.assert_close(scaled[key], 7*original[key], rtol=2e-4, atol=2e-4)
            self.assertEqual(original["waveforms"][key][1, :, 192:].abs().sum(), 0)
        torch.testing.assert_close(original["ctf"], scaled["ctf"], rtol=1e-4, atol=1e-5)
        self.assertEqual(original["ctf"].shape, (2, 2, 33, 4))
        self.assertEqual(original["clean_final"].dtype, torch.complex64)
        self.assertEqual(original["waveforms"]["clean_final"].shape, (2, 2, 256))

    def test_shared_pit_and_teacher_ctf_reconstruction(self):
        network = model(stage=2)
        refs = torch.randn(1, 2, 256)
        lengths = torch.tensor([256])
        spectrum = network._encode_sources(refs, lengths)
        ctf = torch.zeros(1, 2, 33, 4, dtype=torch.complex64)
        ctf[..., 0] = 1
        target = torch.ones(1, 1, 1, 17, dtype=torch.bool)
        results = dict(clean_prior=spectrum, clean_final=spectrum,
                       reverb=spectrum, ctf=ctf)
        waves = dict(clean_prior=refs, clean_final=refs)
        expected = network._loss_one(results, waves, refs, refs, lengths, 0, 256, target)
        flipped = network._loss_one(
            {k: v.flip(1) for k, v in results.items()},
            {k: v.flip(1) for k, v in waves.items()}, refs, refs, lengths, 0, 256, target,
        )
        for key in expected:
            torch.testing.assert_close(expected[key], flipped[key])
        self.assertLess(float(expected["ctf"]), 1e-5)
        mismatched = dict(results, reverb=spectrum.flip(1))
        bad = network._loss_one(mismatched, waves, refs, refs, lengths, 0, 256, target)
        self.assertGreater(float(bad["reverb"]), .1)
        # A predicted Reverb error is not silently used as the CTF teacher.
        torch.testing.assert_close(bad["ctf"], expected["ctf"])

    def test_crop_guard_and_actual_context_reconstruction(self):
        network = model(stage=2)
        metadata = {key: torch.tensor([[value]]) for key, value in (
            ("ctf_loss_start", 112), ("ctf_loss_end", 240),
            ("ctf_crop_start", 208), ("ctf_crop_end", 512),
            ("ctf_original_length", 1024),
        )}
        start, end, target, observed = network._interval(metadata, 0, 304, 20, "cpu")
        self.assertEqual((start, end), (112, 240))
        self.assertFalse(observed[..., :5].any())
        self.assertTrue((target & observed).equal(target))
        spectrum = torch.randn(1, 2, 33, 65, dtype=torch.complex64)
        ctf = torch.randn(1, 2, 33, 4, dtype=torch.complex64)
        full = ctf_forward(spectrum, ctf)
        cropped = ctf_forward(spectrum[..., 13:33], ctf)
        torch.testing.assert_close(cropped[..., 7:15], full[..., 20:28], atol=3e-6, rtol=3e-6)
        metadata["ctf_loss_start"] = torch.tensor([[0]])
        with self.assertRaises(ValueError):
            network._interval(metadata, 0, 304, 20, "cpu")

    def test_task_registration_and_checkpoint(self):
        from espnet2.tasks.enh import EnhancementTask
        with tempfile.TemporaryDirectory() as directory:
            options = EnhancementTask.get_parser().parse_args(["--output_dir", directory])
            options.encoder = "stft"
            options.encoder_conf = dict(n_fft=64, hop_length=16)
            options.decoder = "stft"
            options.decoder_conf = dict(n_fft=64, hop_length=16)
            options.separator = "tflocoformer_ctf_dereverb"
            options.separator_conf = dict(
                n_layers=1, emb_dim=8, attention_dim=8, n_heads=2, num_groups=2,
                ffn_hidden_dim=[8, 8], conv1d_kernel=4, ctf_taps=4,
                ctf_hidden_dim=8, mamba_d_state=2, mamba_backend="torch",
                use_deconvolution=True,
            )
            options.model_conf = dict(training_stage=2, normalize_variance=True)
            network = EnhancementTask.build_model(options).eval()
            config = Path(directory) / "config.yaml"
            config.write_text(yaml.safe_dump(vars(options)))
            checkpoint = Path(directory) / "model.pth"
            torch.save(network.state_dict(), checkpoint)
            loaded, _ = EnhancementTask.build_model_from_file(str(config), str(checkpoint), "cpu")
            loaded.eval()
            x = torch.randn(1, 256)
            with torch.no_grad():
                a, b = network.predict_all(x), loaded.predict_all(x)
            for key in ("ctf", "reverb", "clean_prior", "clean_final"):
                torch.testing.assert_close(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
