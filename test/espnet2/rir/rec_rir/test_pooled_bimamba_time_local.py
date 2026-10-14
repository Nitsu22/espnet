"""Local temporal attention: receptive field, numerical oracle and shared loss.

The dense reference intentionally uses explicit softmax, independent of SDPA
and of the implementation's chunking. CPU sequence substitutes check model
wiring only; selective-scan kernels require the laboratory CUDA validation.
"""

import copy
import io
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.pooled_bimamba import (
    PooledBiMambaCTFPredictor,
    TimeLocalAttention,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)


SMALL = dict(
    emb_dim=8, pre_layers=2, post_layers=2, d_state=4, d_conv=2,
    freq_bottleneck=4, freq_rank=4, local_kernel=3, pooling_hidden=4,
    n_heads=2, ffn_dim=8, head_dim=8,
)


class CpuSequenceSubstitute(nn.Module):
    """Trainable sequence stand-in, not a test of Mamba kernels."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x).cumsum(1) / x.shape[1]


def substitute(dim, specification):
    return CpuSequenceSubstitute(dim)


def predictor(**options):
    return PooledBiMambaCTFPredictor(
        input_dim=33, ctf_taps=5, **(SMALL | options)
    )


def sweep_model(**options):
    return ESPnetPooledBiMambaSweepV2PITModel(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=0.1,
        predictor_conf=SMALL | {"time_module": "bilstm"} | options,
    )


def dense_reference(block, x):
    """Full-T explicit attention oracle, with an explicit temporal band."""
    batch, frames, freqs, dim = x.shape
    sequence = x.permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
    normalized = F.layer_norm(
        sequence, (dim,), block.norm_attention.weight,
        block.norm_attention.bias, block.norm_attention.eps,
    )
    qkv = F.linear(normalized, block.qkv.weight, block.qkv.bias)
    head_dim = dim // block.heads
    qkv = qkv.reshape(batch * freqs, frames, 3, block.heads, head_dim)
    query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    positions = torch.arange(frames, device=x.device, dtype=torch.float32)
    frequencies = 10000.0 ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
        / head_dim
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    def rotate(values):
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), -1).flatten(-2)

    query, key = rotate(query), rotate(key)
    logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    offsets = positions[None, :] - positions[:, None]
    allowed = ((offsets >= -block.past_frames) & (offsets <= block.future_frames))
    logits = logits.masked_fill(~allowed, -torch.inf)
    update = torch.matmul(logits.softmax(-1), value)
    update = update.transpose(1, 2).reshape(batch * freqs, frames, dim)
    update = F.linear(update, block.projection.weight, block.projection.bias)
    update = update.reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)
    return x + update


def test_local_matches_dense_output_and_gradients_across_chunk_boundaries():
    # Window 1, windows larger than the sequence, and a long sequence cover
    # edges, nonzero RoPE offsets, unequal final chunks and sequence batches.
    for frames, window, query_chunk, sequence_batch, past, future in (
        (11, 1, 3, 2, None, None), (13, 3, 4, 3, None, None),
        (9, 125, 2, 1, None, None), (133, 125, 17, 4, None, None),
        (133, 81, 17, 4, 16, 64), (133, 81, 19, 4, 64, 16),
        (11, 81, 3, 2, 16, 64), (11, 5, 3, 2, 0, 4),
    ):
        torch.manual_seed(11)
        block = TimeLocalAttention(
            8, 2, window, 1e-5, query_chunk_size=query_chunk,
            sequence_batch_size=sequence_batch, past_frames=past, future_frames=future,
        )
        oracle = copy.deepcopy(block)
        x = torch.randn(2, frames, 3, 8, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual, expected = block(x), dense_reference(oracle, reference_x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=5e-5, atol=5e-6)
        parameters = dict(oracle.named_parameters())
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad, parameters[name].grad, rtol=5e-5, atol=3e-5,
                msg=lambda message: f"{name}: {message}",
            )


def test_window_excludes_outside_frames_and_keeps_batch_frequency_independent():
    torch.manual_seed(12)
    block = TimeLocalAttention(8, 2, 5, 1e-5, query_chunk_size=3,
                               sequence_batch_size=2).eval()
    x = torch.randn(2, 11, 3, 8)
    baseline = block(x)
    # Frame 5 can see precisely frames 3..7 in its own batch/frequency.
    for batch, frame, frequency in ((0, 2, 1), (0, 8, 1),
                                     (1, 5, 1), (0, 5, 2)):
        changed = x.clone()
        changed[batch, frame, frequency, 0] += 8
        torch.testing.assert_close(
            block(changed)[0, 5, 1], baseline[0, 5, 1], rtol=0, atol=0,
        )
    for frame in (3, 7):
        changed = x.clone()
        changed[0, frame, 1, 0] += 8
        assert (block(changed)[0, 5, 1] - baseline[0, 5, 1]).abs().max() > 1e-6
    # The input Jacobian confirms no hidden path carries distant frames.
    differentiable = x.detach().clone().requires_grad_()
    block(differentiable)[0, 5, 1].square().sum().backward()
    gradient = differentiable.grad.abs().sum(-1)
    assert gradient[0, 3:8, 1].sum() > 0
    gradient[0, 3:8, 1] = 0
    assert torch.count_nonzero(gradient) == 0


def test_sdpa_uses_bounded_query_key_and_sequence_batches():
    block = TimeLocalAttention(8, 2, 5, 1e-5, query_chunk_size=8,
                               sequence_batch_size=2)
    actual_sdpa, observed = F.scaled_dot_product_attention, []

    def record(query, key, value, *args, **kwargs):
        observed.append((query.shape, key.shape))
        return actual_sdpa(query, key, value, *args, **kwargs)

    with patch("espnet2.rir.rec_rir.pooled_bimamba.F.scaled_dot_product_attention",
               side_effect=record):
        output = block(torch.randn(2, 65, 3, 8))
    assert output.shape == (2, 65, 3, 8)
    assert observed
    for query, key in observed:
        assert query[0] <= 2 and key[0] <= 2
        assert query[-2] <= 8
        assert key[-2] <= 8 + 5 - 1
        assert query[-2] < 65 and key[-2] < 65


def test_disabled_default_common_weights_and_rng_are_preserved():
    models, rng_states = [], []
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        for options in ({}, {"time_local_attention_window": 0},
                        {"time_local_attention_window": 125},
                        {"time_local_attention_window": 81,
                         "time_local_attention_past_frames": 16,
                         "time_local_attention_future_frames": 64}):
            torch.manual_seed(17)
            models.append(predictor(**options).eval())
            rng_states.append(torch.get_rng_state())
            assert torch.initial_seed() == 17
    for state in rng_states:
        torch.testing.assert_close(state, rng_states[0], rtol=0, atol=0)
    baseline = models[0].state_dict()
    assert not any(name.startswith("time_local_attention_blocks.") for name in baseline)
    assert baseline.keys() == models[1].state_dict().keys()
    for model in models[1:]:
        for name, tensor in baseline.items():
            torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    torch.testing.assert_close(models[0](observation)[0], models[1](observation)[0],
                               rtol=0, atol=0)


def test_local_time_frequency_stage_order_and_disabled_time_stages():
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        model = predictor(pre_layers=4, time_local_attention_window=5).eval()
    events, handles = [], []
    for index in range(4):
        for label, block in (("local", model.time_local_attention_blocks[index]),
                             ("time", model.time_blocks[index]),
                             ("frequency", model.frequency_blocks[index])):
            handles.append(block.register_forward_pre_hook(
                lambda module, inputs, i=index, name=label: events.append((i, name))
            ))
    output = model(torch.randn(2, 9, 33, dtype=torch.complex64))[0]
    for handle in handles:
        handle.remove()
    assert events == [(i, name) for i in range(4)
                      for name in ("local", "time", "frequency")]
    assert output.shape == (2, 2, 33, 5) and torch.is_complex(output)
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        partial = predictor(pre_layers=4, time_block_indices=[0, 2],
                            time_local_attention_window=5)
    for index in (1, 3):
        assert isinstance(partial.time_local_attention_blocks[index], nn.Identity)
        assert not any(name.startswith(f"time_local_attention_blocks.{index}.")
                       for name, _ in partial.named_parameters())


def test_invalid_local_attention_configuration_fails():
    for window in (0, -1, 2, True, 3.0, "3"):
        with unittest.TestCase().assertRaises(ValueError):
            TimeLocalAttention(8, 2, window, 1e-5)
    for dim, heads in ((8, 0), (8, 3), (6, 2), (8, True)):
        with unittest.TestCase().assertRaises(ValueError):
            TimeLocalAttention(dim, heads, 3, 1e-5)
    for field in ("query_chunk_size", "sequence_batch_size"):
        for value in (0, -1, True, 1.0):
            with unittest.TestCase().assertRaises(ValueError):
                TimeLocalAttention(8, 2, 3, 1e-5, **{field: value})
    for window in (-1, 2, True, 3.0):
        with unittest.TestCase().assertRaises(ValueError):
            predictor(time_local_attention_window=window)


def test_config_changes_only_local_time_attention_window():
    configs = Path(__file__).resolve().parents[4] / "egs2/whamr/rir_2spk/conf/tuning"
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2"
    baseline = yaml.safe_load((configs / f"{prefix}.yaml").read_text())
    actual = yaml.safe_load((configs / f"{prefix}_time_local_attention125.yaml").read_text())
    expected = copy.deepcopy(baseline)
    expected["model_conf"]["predictor_conf"]["time_local_attention_window"] = 125
    assert actual == expected


def check_sweep_pit(**context):
    torch.manual_seed(23)
    batch = dict(speech_mix=torch.randn(2, 256),
                 speech_mix_lengths=torch.tensor([256, 192]))
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone()
        reverb[:, 180 + speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    options = dict(time_local_attention_window=5,
                   time_local_attention_query_chunk_size=3,
                   time_local_attention_sequence_batch_size=7)
    options.update(context)
    model, reference = sweep_model(**options).train(), sweep_model().eval()
    loss, stats, count = model(**batch)
    assert torch.isfinite(loss) and count == 2
    torch.testing.assert_close(loss.detach(), stats["loss_sweep"])
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    model.eval()
    swapped = dict(batch)
    for prefix in ("rir_direct", "rir_ref"):
        swapped[prefix + "1"], swapped[prefix + "2"] = batch[prefix + "2"], batch[prefix + "1"]
    with torch.no_grad():
        torch.testing.assert_close(model(**batch)[0], model(**swapped)[0])
        spectrum = model.transforms.stft(batch["speech_mix"][:, None], "complex")
        ctf, _ = model._estimate_ctf_and_aux_from_complex(spectrum)
        direct = torch.stack([batch[f"rir_direct{s}"] for s in (1, 2)], 1)
        reverb = torch.stack([batch[f"rir_ref{s}"] for s in (1, 2)], 1)
        actual = model.paired_pit_loss(ctf, direct, reverb)
        expected = reference.paired_pit_loss(ctf, direct, reverb)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        before = model.estimate_ctf(batch["speech_mix"][0])
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        clone = sweep_model(**options).eval()
        clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(before, clone.estimate_ctf(batch["speech_mix"][0]))
        rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
        assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()



def test_local_attention_shared_sweep_pit_gradients_checkpoint_and_two_rirs():
    check_sweep_pit()
    check_sweep_pit(time_local_attention_past_frames=1,
                    time_local_attention_future_frames=3)


def test_asymmetric_window_direction_edges_and_independence():
    torch.manual_seed(29)
    block = TimeLocalAttention(8, 2, 81, 1e-5, query_chunk_size=17,
                               sequence_batch_size=2, past_frames=16, future_frames=64)
    x = torch.randn(2, 133, 3, 8)
    output = block(x)
    # At t=40, the allowed times are 24..104 inclusive; not a centered window.
    for batch, frame, frequency in ((0, 23, 1), (0, 105, 1),
                                     (1, 40, 1), (0, 40, 2)):
        changed = x.clone()
        changed[batch, frame, frequency, 0] += 8
        torch.testing.assert_close(block(changed)[0, 40, 1], output[0, 40, 1],
                                   rtol=0, atol=0)
    for frame in (24, 104):
        changed = x.clone()
        changed[0, frame, 1, 0] += 8
        assert (block(changed)[0, 40, 1] - output[0, 40, 1]).abs().max() > 1e-6
    differentiable = x.clone().requires_grad_()
    block(differentiable)[0, 40, 1].square().sum().backward()
    gradient = differentiable.grad.abs().sum(-1)
    assert gradient[0, 24, 1] > 0 and gradient[0, 104, 1] > 0
    gradient[0, 24:105, 1] = 0
    assert torch.count_nonzero(gradient) == 0


def test_asymmetric_configuration_and_symmetric_checkpoint_compatibility():
    configs = Path(__file__).resolve().parents[4] / "egs2/whamr/rir_2spk/conf/tuning"
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2_time_local_attention"
    expected = yaml.safe_load((configs / f"{prefix}125.yaml").read_text())
    actual = yaml.safe_load((configs / f"{prefix}_p16_f64.yaml").read_text())
    expected["model_conf"]["predictor_conf"].update(
        time_local_attention_window=81, time_local_attention_past_frames=16,
        time_local_attention_future_frames=64)
    assert actual == expected
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        torch.manual_seed(31)
        symmetric = predictor(time_local_attention_window=125)
        torch.manual_seed(31)
        asymmetric = predictor(time_local_attention_window=81,
                               time_local_attention_past_frames=16,
                               time_local_attention_future_frames=64)
    assert symmetric.state_dict().keys() == asymmetric.state_dict().keys()
    for key, value in symmetric.state_dict().items():
        torch.testing.assert_close(value, asymmetric.state_dict()[key], rtol=0, atol=0)
    implicit = TimeLocalAttention(8, 2, 125, 1e-5)
    explicit = TimeLocalAttention(8, 2, 125, 1e-5, past_frames=62, future_frames=62)
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    x = torch.randn(2, 133, 3, 8)
    torch.testing.assert_close(implicit(x), explicit(x), rtol=0, atol=0)
    for past, future in ((None, 64), (16, None), (-1, 81), (True, 79),
                         (16.0, 64), (16, 63)):
        with unittest.TestCase().assertRaises(ValueError):
            TimeLocalAttention(8, 2, 81, 1e-5, past_frames=past, future_frames=future)
        with unittest.TestCase().assertRaises(ValueError):
            predictor(time_local_attention_window=81,
                      time_local_attention_past_frames=past,
                      time_local_attention_future_frames=future)
    with unittest.TestCase().assertRaises(ValueError):
        predictor(time_local_attention_window=0, time_local_attention_past_frames=0,
                  time_local_attention_future_frames=0)


if __name__ == "__main__":
    tests = [unittest.FunctionTestCase(value) for name, value in list(globals().items())
             if name.startswith("test_")]
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(tests)).wasSuccessful())
