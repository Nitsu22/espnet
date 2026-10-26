"""Tests using genuine Mamba-1 modules and the CPU reference selective scan."""

import io
import unittest

import torch

from espnet2.enh.layers.speaker_ctf_estimator import SpeakerCTFEstimator
from espnet2.torch_utils.initialize import initialize


def build():
    return SpeakerCTFEstimator(
        feature_dim=4,
        hidden_dim=4,
        ctf_taps=3,
        d_state=2,
        d_conv=2,
        expand=1,
        backend="torch",
    )


class TestSpeakerCTFEstimator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = build()
        self.features = torch.randn(2, 4, 6, 3)
        self.reverb = torch.randn(2, 2, 3, 6, dtype=torch.cfloat)
        self.lengths = torch.tensor([6, 4])

    def test_shapes_and_speaker_permutation_equivariance(self):
        expected = self.model(self.features, self.reverb, self.lengths)
        swapped = self.model(self.features, self.reverb.flip(1), self.lengths)
        self.assertEqual(expected.shape, (2, 2, 3, 3))
        self.assertEqual(expected.dtype, torch.complex64)
        torch.testing.assert_close(swapped, expected.flip(1))

    def test_conditioning_on_each_speaker(self):
        expected = self.model(self.features, self.reverb, self.lengths)
        changed_reverb = self.reverb.clone()
        changed_reverb[:, 0] += 3.0 - 2.0j
        actual = self.model(self.features, changed_reverb, self.lengths)
        self.assertGreater((actual[:, 0] - expected[:, 0]).abs().max().item(), 1e-5)
        torch.testing.assert_close(actual[:, 1], expected[:, 1])

    def test_adversarial_padding_and_batch_independence(self):
        expected = self.model(self.features, self.reverb, self.lengths)
        features = self.features.clone()
        reverb = self.reverb.clone()
        features[1, :, 4:] = float("nan")
        reverb[1, :, :, 4:] = complex(float("nan"), float("nan"))
        actual = self.model(features, reverb, self.lengths)
        torch.testing.assert_close(actual, expected)
        individual = self.model(
            self.features[1:2, :, :4],
            self.reverb[1:2, :, :, :4],
            torch.tensor([4]),
        )
        torch.testing.assert_close(individual[0], expected[1])

    def test_real_reference_backward_and_optimizer_step(self):
        features = self.features.clone().requires_grad_()
        reverb = self.reverb.clone().requires_grad_()
        before = self.model.output_projection.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        ctf = self.model(features, reverb, self.lengths)
        target = torch.randn_like(ctf)
        loss = (ctf - target).abs().square().mean()
        loss.backward()
        for value in (features.grad, reverb.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.abs().sum().item(), 0.0)
        self.assertEqual(features.grad[1, :, 4:].abs().sum().item(), 0.0)
        self.assertEqual(reverb.grad[1, :, :, 4:].abs().sum().item(), 0.0)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertGreater(
            (self.model.output_projection.weight.detach() - before).abs().sum().item(),
            0.0,
        )

    def test_independent_directions_and_initialization_checkpoint(self):
        stages = (
            self.model.time_mamba,
            self.model.frequency_mamba,
            self.model.post_pool_frequency_mamba,
        )
        special = {}
        pointers = set()
        for i, stage in enumerate(stages):
            for direction in ("forward_mamba", "backward_mamba"):
                module = getattr(stage.mixer, direction)
                pointers.add(module.in_proj.weight.data_ptr())
                special[(i, direction)] = (
                    module.dt_proj.weight.detach().clone(),
                    module.dt_proj.bias.detach().clone(),
                    module.A_log.detach().clone(),
                    module.D.detach().clone(),
                )
        self.assertEqual(len(pointers), 6)
        initialize(self.model, "xavier_uniform")
        for (i, direction), expected in special.items():
            module = getattr(stages[i].mixer, direction)
            for actual, initial in zip(
                (module.dt_proj.weight, module.dt_proj.bias, module.A_log, module.D),
                expected,
            ):
                torch.testing.assert_close(actual, initial, rtol=0, atol=0)
        expected = self.model(self.features, self.reverb, self.lengths)
        checkpoint = io.BytesIO()
        torch.save(self.model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = build()
        initialize(restored, "xavier_uniform")
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        actual = restored(self.features, self.reverb, self.lengths)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_lengths_and_shapes(self):
        for lengths in (
            torch.tensor([6, 0]),
            torch.tensor([6, 7]),
            torch.tensor([6.0, 4.0]),
            torch.tensor([[6, 4]]),
        ):
            with self.assertRaises(ValueError):
                self.model(self.features, self.reverb, lengths)
        with self.assertRaises(ValueError):
            self.model(self.features, self.reverb[:, :, :, :-1], self.lengths)
        with self.assertRaises(ValueError):
            self.model(self.features, self.reverb.real, self.lengths)


if __name__ == "__main__":
    unittest.main()
