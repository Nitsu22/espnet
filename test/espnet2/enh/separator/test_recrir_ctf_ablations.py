"""Numerical and experimental-control checks; runnable without pytest."""

import unittest

import torch
from torch import nn

from espnet2.enh.separator.tflocoformer_separator_nocashe_rir_cmha_film_all import (
    RIRCMHAFiLMBlock,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_rir_cmha_film_all_ctf import (
    TFLocoformerSeparator as Reference,
)
from espnet2.enh.separator.tflocoformer_separator_recrir_ctf_ablations import (
    TFLocoformerSeparator as Ablation,
)
from espnet2.torch_utils.initialize import initialize


def build(variant=None):
    torch.manual_seed(0)
    options = dict(input_dim=9, emb_dim=8, n_layers=1, n_heads=2,
                   attention_dim=8, norm_type="rmsgroupnorm", num_groups=2,
                   ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
                   ffn_hidden_dim=[16, 16], conv1d_kernel=4)
    model = (Reference(**options) if variant is None else
             Ablation(conditioning_variant=variant, **options))
    initialize(model, "xavier_uniform")
    return model


class ConditioningTests(unittest.TestCase):
    def test_common_initialization_and_identity_outputs(self):
        reference = build()
        x = torch.randn(1, 11, 9, dtype=torch.complex64)
        lengths = torch.tensor([11])
        additional = {"rir_ctf": torch.randn(1, 60, 2, 9, 2)}
        with torch.no_grad():
            expected = reference(x, lengths, additional)[0]
        for variant in ("reference", "lag_position", "joint_mlp", "multiscale"):
            with self.subTest(variant=variant):
                model = build(variant)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
                with torch.no_grad():
                    actual = model(x, lengths, additional)[0]
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                if variant != "joint_mlp":
                    self.assertEqual(sum(p.numel() for p in reference.parameters()),
                                     sum(p.numel() for p in model.parameters()))

    def test_reference_film_algebra_after_nonidentity_update(self):
        block = RIRCMHAFiLMBlock(8, n_heads=2)
        nn.init.normal_(block.gamma_proj.weight)
        nn.init.normal_(block.beta_proj.weight)
        x, r = torch.randn(2, 8, 11, 3), torch.randn(2, 8, 7, 3)
        q = x.permute(0, 3, 2, 1).reshape(6, 11, 8)
        kv = r.permute(0, 3, 2, 1).reshape(6, 7, 8)
        c = block.mha(q, kv, kv, need_weights=False)[0]
        c = c.reshape(2, 3, 11, 8).permute(0, 3, 2, 1).contiguous()
        expected = block.gamma_proj(c) * x + block.beta_proj(c)
        torch.testing.assert_close(block(x, r), expected, rtol=0, atol=0)

    def test_lag_positions_break_content_permutation_invariance(self):
        torch.manual_seed(4)
        mha = nn.MultiheadAttention(8, 2, batch_first=True)
        q, r = torch.randn(1, 5, 8), torch.randn(1, 9, 8)
        p = Ablation.lag_positions(torch.zeros(1, 8, 9, 1))[0, :, :, 0].T[None]
        order = torch.arange(8, -1, -1)
        a = mha(q, r, r, need_weights=False)[0]
        b = mha(q, r[:, order], r[:, order], need_weights=False)[0]
        torch.testing.assert_close(a, b)
        a = mha(q, r + p, r + p, need_weights=False)[0]
        b = mha(q, r[:, order] + p, r[:, order] + p, need_weights=False)[0]
        self.assertFalse(torch.allclose(a, b))

    def test_multiscale_reaches_distant_taps_without_extra_parameters(self):
        model = build("multiscale")
        model.rir_conv[1] = nn.Identity()
        with torch.no_grad():
            model.rir_conv[0].weight.fill_(1)
            model.rir_conv[0].bias.zero_()
        r = torch.zeros(1, 4, 60, 9)
        r[:, :, 30, 4] = 1
        encoded = model._encode_rir(r)
        self.assertEqual(tuple(encoded.shape), (1, 8, 60, 9))
        self.assertEqual(encoded[:, :4, 38, 4].abs().sum().item(), 0)
        self.assertGreater(encoded[:, 4:, 38, 4].abs().sum().item(), 0)

    def test_joint_fusion_learns_after_identity_first_step(self):
        model = build("joint_mlp")
        x, r = torch.randn(2, 8, 11, 3), torch.randn(2, 8, 7, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        target = torch.randn_like(x)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model._condition_speech(0, x, r)
            if step == 0:
                torch.testing.assert_close(result, x, rtol=0, atol=0)
            (result - target).square().mean().backward()
            grad = model.joint_fusions[0][0].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            if step == 1:
                self.assertGreater(grad.abs().sum().item(), 0)
                self.assertGreater(model.rir_films[0].mha.in_proj_weight.grad.abs().sum().item(), 0)
            optimizer.step()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
