"""Real Mamba reference-backend checks; CUDA equivalence requires a GPU."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from espnet2.enh.layers.bidirectional_mamba import BidirectionalMamba
from espnet2.enh.separator.tflocoformer_separator_nocashe import (
    MultiHeadSelfAttention,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_bimamba import (
    TFLocoformerBiMambaSeparator,
)
from espnet2.torch_utils.initialize import initialize


@unittest.skipUnless(importlib.util.find_spec("mamba_ssm"), "mamba_ssm unavailable")
class TestBiMamba(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_reference_matches_official_forward_and_gradients(self):
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

        layer = BidirectionalMamba(8, d_state=4, backend="torch")
        official = copy.deepcopy(layer.forward_mamba)
        official.use_fast_path = False
        x = torch.randn(2, 7, 8, requires_grad=True)
        probe = torch.randn_like(x)
        actual = layer._torch_forward(layer.forward_mamba, x)
        # Exercise the official Mamba forward with its genuine reference scan.
        with patch("mamba_ssm.modules.mamba_simple.causal_conv1d_fn", None), patch(
            "mamba_ssm.modules.mamba_simple.selective_scan_fn", selective_scan_ref
        ):
            expected = official(x)
        torch.testing.assert_close(actual, expected)
        args = [x] + list(layer.forward_mamba.parameters())
        refs = [x] + list(official.parameters())
        actual_grads = torch.autograd.grad((actual * probe).sum(), args)
        expected_grads = torch.autograd.grad((expected * probe).sum(), refs)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)

    def test_initialization_bidirectionality_and_no_state_leak(self):
        layer = BidirectionalMamba(8, d_state=4, backend="torch")
        initial = {
            name: value.clone() for name, value in layer.state_dict().items()
            if any(key in name for key in ("dt_proj", "A_log", ".D"))
        }
        initialize(layer, "xavier_uniform")
        for name, value in initial.items():
            torch.testing.assert_close(layer.state_dict()[name], value)
        x = torch.randn(1, 7, 8, requires_grad=True)
        y = layer(x)
        gradient = torch.autograd.grad(y[:, 0].square().sum(), x)[0]
        self.assertGreater(gradient[:, -1].abs().sum().item(), 0)
        swapped = copy.deepcopy(layer)
        swapped.forward_mamba, swapped.backward_mamba = (
            swapped.backward_mamba, swapped.forward_mamba
        )
        with torch.no_grad():
            swapped.fusion.weight.copy_(
                layer.fusion.weight.reshape(8, 2, 8).flip(1).reshape(8, 16)
            )
        torch.testing.assert_close(layer(x), swapped(x.flip(1)).flip(1))
        layer(torch.randn(2, 9, 8))
        torch.testing.assert_close(layer(x), y)
        layer(x).square().mean().backward()
        for name, parameter in layer.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_topology_padding_and_input_validation(self):
        for order in ("ft", "tf"):
            model = TFLocoformerBiMambaSeparator(
                9, n_layers=1, emb_dim=8, attention_dim=8, n_heads=2,
                num_groups=2, ffn_hidden_dim=[8, 8], tf_order=order,
                mamba_d_state=4, mamba_backend="torch",
            ).eval()
            self.assertIsInstance(model.blocks[0].freq_path.attn, MultiHeadSelfAttention)
            self.assertIsInstance(model.blocks[0].frame_path.attn, BidirectionalMamba)
            x = torch.randn(2, 7, 9, dtype=torch.complex64)
            lengths = torch.tensor([7, 4])
            padded, returned, _ = model(x, lengths)
            single, _, _ = model(x[1:2, :4], torch.tensor([4]))
            torch.testing.assert_close(returned, lengths)
            channel, _, _ = model(x.unsqueeze(2), lengths)
            for a, b, c in zip(padded, single, channel):
                torch.testing.assert_close(a[1:2, :4], b)
                torch.testing.assert_close(a, c)
                self.assertEqual(a[1, 4:].abs().sum().item(), 0)
            changed = x.clone()
            changed[1, 4:] = 1000
            altered, _, _ = model(changed, lengths)
            for a, b in zip(altered, padded):
                torch.testing.assert_close(a, b)
            for bad in (torch.tensor([0, 4]), torch.tensor([8, 4]), lengths.float()):
                with self.assertRaises(ValueError):
                    model(x, bad)
            with self.assertRaises(ValueError):
                model(x.unsqueeze(2).expand(-1, -1, 2, -1), lengths)

    def test_recipe_waveform_training_and_checkpoint(self):
        from espnet2.tasks.enh import EnhancementTask

        config = Path(__file__).resolve().parents[4] / (
            "egs2/whamr/enh7_baseline/conf/tuning/"
            "train_enh_tflocoformer_bimamba_s_whamr_8k.yaml"
        )
        baseline = yaml.safe_load(config.with_name(
            "train_enh_tflocoformer_s_whamr_8k.yaml"
        ).read_text())
        modified = yaml.safe_load(config.read_text())
        modified["separator"] = baseline["separator"]
        for key in list(modified["separator_conf"]):
            if key.startswith("mamba_"):
                del modified["separator_conf"][key]
        self.assertEqual(modified, baseline)
        with tempfile.TemporaryDirectory() as directory:
            args = EnhancementTask.get_parser().parse_args([
                "--config", str(config), "--output_dir", directory,
            ])
            args.separator_conf["mamba_backend"] = "torch"
            model = EnhancementTask.build_model(args)
            self.assertEqual(len(model.separator.blocks), 4)
            refs = [torch.randn(2, 512) for _ in range(2)]
            lengths = torch.tensor([512, 384])
            for reference in refs:
                reference[1, 384:] = 0
            mixture = refs[0] + refs[1] + 0.01 * torch.randn(2, 512)
            loss, _, _ = model(
                speech_mix=mixture, speech_mix_lengths=lengths,
                speech_ref1=refs[0], speech_ref2=refs[1],
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.optim.AdamW(model.parameters(), lr=1e-3).step()
            config_file = Path(directory) / "config.yaml"
            config_file.write_text(yaml.safe_dump(vars(args)))
            checkpoint = Path(directory) / "model.pth"
            torch.save(model.state_dict(), checkpoint)
            restored, _ = EnhancementTask.build_model_from_file(
                str(config_file), str(checkpoint), device="cpu",
            )
            model.eval()
            restored.eval()
            with torch.no_grad():
                encoded, frames = model.encoder(mixture, lengths)
                expected, _, _ = model.separator(encoded, frames)
                actual, _, _ = restored.separator(encoded, frames)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)
                    audio, audio_lengths = restored.decoder(a, lengths)
                    self.assertEqual(audio.shape, mixture.shape)
                    self.assertTrue(torch.isfinite(audio).all())
                    torch.testing.assert_close(audio_lengths, lengths)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_cuda_matches_reference_outputs_and_gradients(self):
        reference = BidirectionalMamba(8, d_state=4, backend="torch").cuda()
        cuda = copy.deepcopy(reference)
        cuda.backend = "cuda"
        x = torch.randn(2, 17, 8, device="cuda", requires_grad=True)
        a, b = reference(x), cuda(x)
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
        probe = torch.randn_like(a)
        ga = torch.autograd.grad((a * probe).sum(), [x] + list(reference.parameters()))
        gb = torch.autograd.grad((b * probe).sum(), [x] + list(cuda.parameters()))
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
