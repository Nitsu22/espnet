"""Unfold geometry, residuals, common initialization and full-frame attention."""
import io
from unittest.mock import patch

import torch
from torch import nn

from espnet2.rir.rec_rir.spmamba_blocks import UnfoldBiMamba, UnfoldAxisBiMamba, FrameAttention
from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor


def substitute(dim, specification):
    return nn.Linear(dim, dim)


def test_axis_preserves_short_odd_sequences_and_residual():
    for axis in ("time", "frequency"):
        with patch("espnet2.rir.rec_rir.spmamba_blocks._build_mamba", substitute):
            module = UnfoldAxisBiMamba(4, 3, 4, 2, 1e-5, axis)
        for frames, freqs in [(1, 1), (5, 7), (8, 6)]:
            x = torch.randn(2, frames, freqs, 4, requires_grad=True)
            y = module(x)
            assert y.shape == x.shape
            y.square().mean().backward()
            assert torch.isfinite(x.grad).all()
        nn.init.zeros_(module.sequence.restore.weight)
        nn.init.zeros_(module.sequence.restore.bias)
        x = torch.randn(2, 5, 7, 4)
        torch.testing.assert_close(module(x), x, rtol=0, atol=0)


def test_unfold_never_mixes_independent_sequence_batches():
    with patch("espnet2.rir.rec_rir.spmamba_blocks._build_mamba", substitute):
        module = UnfoldBiMamba(4, 3, 4, 2, 1e-5)
    x = torch.randn(3, 7, 4)
    expected = torch.cat([module(row[None]) for row in x], dim=0)
    torch.testing.assert_close(module(x), expected)


def test_frame_attention_matches_explicit_full_frame_reference():
    module = FrameAttention(4, 2, 1, 1e-5)
    x = torch.randn(2, 3, 5, 4, requires_grad=True)
    q, k, v = module.qkv(module.norm(x)).split((2, 2, 4), dim=-1)
    heads = []
    for head in range(2):
        queries = q[..., head].reshape(2, 3, 5)
        keys = k[..., head].reshape(2, 3, 5)
        values = v[..., head * 2:(head + 1) * 2]
        weights = (torch.einsum("btf,buf->btu", queries, keys) / 5**0.5).softmax(-1)
        heads.append(torch.einsum("btu,bufd->btfd", weights, values))
    expected = x + module.projection(torch.cat(heads, -1))
    torch.testing.assert_close(module(x), expected)
    module(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def make_predictor(attention):
    return PooledBiMambaCTFPredictor(input_dim=7, ctf_taps=5, emb_dim=8, pre_layers=2,
        post_layers=1, d_state=4, d_conv=2, n_heads=2, ffn_dim=8, freq_rank=2,
        freq_bottleneck=2, head_dim=8, pooling_hidden=4, local_kernel=3,
        pre_backbone="spmamba", pre_frame_attention=attention, unfold_kernel=3)


def test_common_weights_identical_and_ctf_checkpoint_roundtrip():
    with patch("espnet2.rir.rec_rir.spmamba_blocks._build_mamba", substitute), \
         patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        torch.manual_seed(0)
        baseline = make_predictor(False)
        torch.manual_seed(0)
        attention = make_predictor(True)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, attention.state_dict()[name], rtol=0, atol=0)
        x = torch.randn(2, 5, 7, dtype=torch.complex64)
        result = attention(x)[0]
        assert result.shape == (2, 2, 7, 5)
        result.abs().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in attention.parameters() if p.requires_grad)
        stream = io.BytesIO()
        torch.save(attention.state_dict(), stream)
        stream.seek(0)
        restored = make_predictor(True)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(x)[0], result)


def test_invalid_unfold_geometry():
    for kernel in (0, 2, True):
        try:
            UnfoldBiMamba(4, kernel, 4, 2, 1e-5)
        except ValueError as error:
            assert "positive odd" in str(error)
        else:
            raise AssertionError("Invalid kernel accepted")
