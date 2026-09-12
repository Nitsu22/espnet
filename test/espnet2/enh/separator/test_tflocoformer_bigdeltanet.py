"""CPU checks; also runnable with unittest when pytest is unavailable."""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

import torch
from torch.nn import functional as F

from espnet2.enh.layers.bidirectional_gated_deltanet import (
    BidirectionalGatedDeltaNet,
    gated_delta_rule,
    chunk_gated_delta_rule,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_bigdeltanet import (
    TFLocoformerBiGatedDeltaNetSeparator,
)
from espnet2.torch_utils.initialize import initialize


class TestBiGatedDeltaNet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_rule_matrix_form_and_gradients(self):
        # Independent matrix-transition form of the paper's update.
        q, k = [F.normalize(torch.randn(2, 5, 2, 3, dtype=torch.double), dim=-1)
                for _ in range(2)]
        v = torch.randn(2, 5, 2, 4, dtype=torch.double)
        g = -torch.rand(2, 5, 2, dtype=torch.double)
        beta = torch.rand(2, 5, 2, dtype=torch.double)
        state = torch.zeros(2, 2, 3, 4, dtype=torch.double)
        eye = torch.eye(3, dtype=torch.double)
        expected = []
        for t in range(5):
            kt = k[:, t, :, :, None]
            transition = (eye - beta[:, t, :, None, None] * (kt @ kt.transpose(-1, -2)))
            state = g[:, t, :, None, None].exp() * (transition @ state)
            state = state + beta[:, t, :, None, None] * (kt @ v[:, t, :, None, :])
            expected.append((q[:, t, :, None, :] @ state).squeeze(-2) / 3**0.5)
        torch.testing.assert_close(
            gated_delta_rule(q, k, v, g, beta), torch.stack(expected, 1)
        )
        args = tuple(
            t[:1, :2, :1].detach().requires_grad_() for t in (q, k, v, g, beta)
        )
        self.assertTrue(torch.autograd.gradcheck(gated_delta_rule, args))

    def test_chunk_matches_recurrent_across_boundaries(self):
        for time in (1, 7, 32, 33, 65):
            q, k = [F.normalize(torch.randn(2, time, 2, 4), dim=-1)
                    for _ in range(2)]
            tensors = [q, k, torch.randn(2, time, 2, 6),
                       -torch.rand(2, time, 2), torch.rand(2, time, 2)]
            tensors = [t.requires_grad_() for t in tensors]
            recurrent = gated_delta_rule(*tensors)
            chunked = chunk_gated_delta_rule(*tensors)
            torch.testing.assert_close(chunked, recurrent, atol=2e-6, rtol=2e-5)
            probe = torch.randn_like(recurrent)
            grads_ref = torch.autograd.grad((recurrent * probe).sum(), tensors)
            grads_chunk = torch.autograd.grad((chunked * probe).sum(), tensors)
            for a, b in zip(grads_chunk, grads_ref):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        # Strong forgetting must not overflow masked upper-triangular entries.
        tensors[3] = torch.full_like(tensors[3], -100).requires_grad_()
        chunked = chunk_gated_delta_rule(*tensors)
        self.assertTrue(torch.isfinite(chunked).all())
        self.assertTrue(all(torch.isfinite(g).all() for g in
                            torch.autograd.grad(chunked.sum(), tensors)))

    def test_bidirectionality_and_no_state_leak(self):
        layer = BidirectionalGatedDeltaNet(8, heads=2, head_dim=4, value_dim=4)
        x = torch.randn(1, 7, 8, requires_grad=True)
        y = layer(x)
        # The first output depends on a future frame through the reverse branch.
        grad = torch.autograd.grad(y[:, 0].square().sum(), x)[0]
        self.assertGreater(grad[:, -1].abs().sum().item(), 0)
        swapped = copy.deepcopy(layer)
        swapped.forward_net, swapped.backward_net = (
            swapped.backward_net, swapped.forward_net
        )
        with torch.no_grad():
            swapped.fusion.weight.copy_(
                layer.fusion.weight.reshape(8, 2, 8).flip(1).reshape(8, 16)
            )
        torch.testing.assert_close(layer(x), swapped(x.flip(1)).flip(1))
        layer(torch.randn(2, 9, 8))
        torch.testing.assert_close(layer(x), y)
        for name, p in layer.named_parameters():
            self.assertTrue(torch.isfinite(p).all(), name)

    def test_initialization_survives_espnet(self):
        layer = BidirectionalGatedDeltaNet(8, heads=2, head_dim=4, value_dim=4)
        initialize(layer, "xavier_uniform")
        for net in (layer.forward_net, layer.backward_net):
            dt = F.softplus(net.dt_bias)
            self.assertTrue(((dt >= 0.001) & (dt <= 0.1)).all())
            self.assertTrue(torch.isfinite(net.A_log).all())
            torch.testing.assert_close(
                net.norm_weight, torch.ones_like(net.norm_weight)
            )
        layer(torch.randn(2, 5, 8)).square().mean().backward()
        for name, p in layer.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_small_espnet_waveform_training_and_reload(self):
        from espnet2.tasks.enh import EnhancementTask

        config = (Path(__file__).resolve().parents[4] /
                  "egs2/whamr/enh1/conf/tuning/"
                  "train_enh_tflocoformer_small_nocashe_bigdeltanet.yaml")
        with tempfile.TemporaryDirectory() as directory:
            args = EnhancementTask.get_parser().parse_args([
                "--config", str(config), "--output_dir", directory,
            ])
            model = EnhancementTask.build_model(args)
            self.assertEqual(len(model.separator.blocks), 4)
            refs = [torch.randn(1, 1024) for _ in range(2)]
            mixture = refs[0] + refs[1] + 0.01 * torch.randn(1, 1024)
            lengths = torch.tensor([1024])
            loss, _, _ = model(
                speech_mix=mixture, speech_mix_lengths=lengths,
                speech_ref1=refs[0], speech_ref2=refs[1],
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
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

    def test_lengths_channels_and_checkpoint(self):
        model = TFLocoformerBiGatedDeltaNetSeparator(
            9, n_layers=1, emb_dim=8, attention_dim=8, n_heads=2,
            num_groups=2, ffn_hidden_dim=[8, 8],
            gdn_heads=2, gdn_head_dim=4, gdn_value_dim=4,
        ).eval()
        x = torch.randn(2, 7, 9, dtype=torch.complex64)
        lengths = torch.tensor([7, 4])
        padded, actual_lengths, _ = model(x, lengths)
        single, _, _ = model(x[1:2, :4], torch.tensor([4]))
        torch.testing.assert_close(actual_lengths, lengths)
        for a, b in zip(padded, single):
            torch.testing.assert_close(a[1:2, :4], b)
            self.assertEqual(a[1, 4:].abs().sum().item(), 0)
        clone = copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(), strict=True)
        mono, _, _ = clone(x.unsqueeze(2), lengths)
        for a, b in zip(padded, mono):
            torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            model(x.unsqueeze(2).expand(-1, -1, 2, -1), lengths)
        with self.assertRaises(ValueError):
            model(x, torch.tensor([0, 4]))
        sum(a.abs().square().mean() for a in padded).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
