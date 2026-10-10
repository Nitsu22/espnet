"""Standalone post-pooling attention regressions.

Dense attention references check actual communication semantics, independently
of the optimized layout. Integration tests use the real CPU BiLSTM option;
small trainable Mamba substitutes only check initialization and dispatch. They
do not validate CUDA Mamba kernels, which the laboratory smoke check exercises.
"""

import copy
import io
import math
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.pooled_bimamba import (
    PooledBiMambaCTFPredictor,
    PostPoolingAttention,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)


TINY = dict(
    emb_dim=8,
    pre_layers=1,
    post_layers=2,
    d_state=4,
    d_conv=2,
    freq_bottleneck=4,
    freq_rank=4,
    local_kernel=3,
    pooling_hidden=4,
    n_heads=2,
    ffn_dim=8,
    head_dim=8,
)
VARIANTS = {
    "post_speaker": {
        "post_interaction": "speaker",
        "post_interaction_order": "after_frequency",
    },
    "post_joint": {
        "post_interaction": "joint",
        "post_interaction_order": "after_frequency",
    },
    "post_frequency": {
        "post_interaction": "frequency",
        "post_interaction_order": "after_frequency",
    },
    "pre_speaker": {
        "post_interaction": "speaker",
        "post_interaction_order": "before_frequency",
    },
}


def _cpu_mamba_substitute(dim, specification):
    """Topology and RNG substitute only; real recurrent integration uses LSTM."""
    return nn.Sequential(nn.Linear(dim, dim), nn.SiLU())


def _predictor(**options):
    return PooledBiMambaCTFPredictor(
        input_dim=33, ctf_taps=5, **(TINY | options)
    )


def _sweep_model(**options):
    return ESPnetPooledBiMambaSweepV2PITModel(
        sr=16000,
        n_fft=64,
        win_len=64,
        hop_len=32,
        num_freqs=33,
        ctf_taps=5,
        pim_sweep_duration=0.1,
        predictor_conf=TINY | {"time_module": "bilstm"} | options,
    )


def _dense_reference(block, x, concatenate_speaker_positions=False):
    """Use all S*F tokens with an explicit connectivity mask for every axis.

    This reference never reshapes speakers/frequencies into separate batches.
    Complex multiplication implements RoPE independently of the module helper.
    """
    batch, speakers, freqs, dim = x.shape
    length, heads = speakers * freqs, block.heads
    sequence = x.reshape(batch, length, dim)
    normalized = (sequence - sequence.mean(-1, keepdim=True)) / torch.sqrt(
        sequence.var(-1, keepdim=True, unbiased=False)
        + block.norm_attention.eps
    )
    normalized = normalized * block.norm_attention.weight + block.norm_attention.bias
    qkv = F.linear(normalized, block.qkv.weight, block.qkv.bias).reshape(
        batch, length, 3, heads, dim // heads
    )
    query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    frequencies = torch.arange(freqs, device=x.device).repeat(speakers)
    slots = torch.arange(speakers, device=x.device).repeat_interleave(freqs)
    if block.axis != "speaker":
        positions = (
            torch.arange(length, device=x.device)
            if concatenate_speaker_positions else frequencies
        )
        head_dim = dim // heads
        inverse = 10000 ** (
            -torch.arange(0, head_dim, 2, device=x.device).float() / head_dim
        )
        phase = positions.float()[:, None] * inverse[None]
        rotation = torch.polar(torch.ones_like(phase), phase)

        def rotate(tensor):
            pairs = torch.view_as_complex(
                tensor.contiguous().reshape(*tensor.shape[:-1], -1, 2)
            )
            return torch.view_as_real(pairs * rotation).flatten(-2)

        query, key = rotate(query), rotate(key)
    logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim // heads)
    if block.axis == "speaker":
        allowed = frequencies[:, None] == frequencies[None]
    elif block.axis == "frequency":
        allowed = slots[:, None] == slots[None]
    else:
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device)
    weights = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
    update = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
    update = F.linear(update, block.projection.weight, block.projection.bias)
    return x + update.reshape(batch, speakers, freqs, dim)


def test_attention_axes_match_independent_dense_reference():
    torch.manual_seed(19)
    x = torch.randn(2, 2, 5, 8)
    for axis in ("speaker", "joint", "frequency"):
        block = PostPoolingAttention(8, 2, 1e-5, axis).eval()
        torch.testing.assert_close(
            block(x), _dense_reference(block, x), atol=2e-6, rtol=2e-6
        )


def test_attention_communication_footprints():
    """A changed cell can affect only the cells permitted by the selected axis."""
    x = torch.zeros(1, 2, 5, 8)
    modified = x.clone()
    modified[0, 1, 2, 0] = 3
    for axis in ("speaker", "joint", "frequency"):
        block = PostPoolingAttention(8, 2, 1e-5, axis).eval()
        with torch.no_grad():
            # Uniform attention makes every permitted communication observable.
            block.qkv.weight.zero_()
            block.qkv.bias.zero_()
            block.qkv.weight[16:].copy_(torch.eye(8))
            block.projection.weight.copy_(torch.eye(8))
            block.projection.bias.zero_()
            changed = (block(modified) - block(x)).abs().sum(-1)[0] > 1e-6
        expected = torch.zeros(2, 5, dtype=torch.bool)
        if axis == "speaker":
            expected[:, 2] = True
        elif axis == "frequency":
            expected[1] = True
        else:
            expected[:] = True
        torch.testing.assert_close(changed, expected)


def test_joint_rope_repeats_frequency_positions_and_slot_permutation():
    torch.manual_seed(23)
    x = torch.randn(2, 2, 5, 8)
    for axis in ("speaker", "joint", "frequency"):
        block = PostPoolingAttention(8, 2, 1e-5, axis).eval()
        actual = block(x)
        # Slot order carries no artificial positional label inside the module.
        torch.testing.assert_close(
            block(x.flip(1)), actual.flip(1), atol=2e-6, rtol=2e-6
        )
        if axis == "joint":
            torch.testing.assert_close(
                actual, _dense_reference(block, x), atol=2e-6, rtol=2e-6
            )
            wrong = _dense_reference(block, x, concatenate_speaker_positions=True)
            assert not torch.allclose(actual, wrong, atol=1e-5, rtol=1e-5)


def test_variants_preserve_common_initialization_and_equal_parameter_counts():
    # Reuse the frozen pre-ablation regression for unchanged default behavior.
    old_path = Path(__file__).with_name("test_pooled_bimamba_ablations.py")
    old_test = runpy.run_path(str(old_path))
    old_test["test_default_legacy_checkpoint_and_unmasked_output"]()
    models = []
    with patch(
        "espnet2.rir.rec_rir.pooled_bimamba._build_mamba", _cpu_mamba_substitute
    ):
        torch.manual_seed(31)
        baseline = _predictor()
        for options in VARIANTS.values():
            torch.manual_seed(31)
            models.append(_predictor(**options))
    assert len(baseline.post_interaction_blocks) == 0
    base_state = baseline.state_dict()
    counts = []
    for model in models:
        state = model.state_dict()
        for name, value in base_state.items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        added = state.keys() - base_state.keys()
        assert added and all(
            name.startswith("post_interaction_blocks.") for name in added
        )
        counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts)) == 1
    baseline_count = sum(p.numel() for p in baseline.parameters())
    assert counts[0] - baseline_count == 2 * (4 * 8**2 + 6 * 8)
    # Axis and processing order must be the only changes to the seeded weights.
    for model in models[1:]:
        assert model.state_dict().keys() == models[0].state_dict().keys()
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)


def test_frequency_interaction_order_dispatch_and_output():
    torch.manual_seed(37)
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    outputs = []
    with patch(
        "espnet2.rir.rec_rir.pooled_bimamba._build_mamba", _cpu_mamba_substitute
    ):
        after = _predictor(**VARIANTS["post_speaker"]).eval()
        before = _predictor(**VARIANTS["pre_speaker"]).eval()
    before.load_state_dict(after.state_dict(), strict=True)
    for model, order in (
        (after, ("frequency", "interaction")),
        (before, ("interaction", "frequency")),
    ):
        events, handles = [], []
        for kind, blocks in (
            ("frequency", model.post_blocks),
            ("interaction", model.post_interaction_blocks),
        ):
            for index, block in enumerate(blocks):
                handles.append(block.register_forward_pre_hook(
                    lambda module, inputs, label=(kind, index): events.append(
                        (label, tuple(inputs[0].shape))
                    )
                ))
        ctf, lengths, _ = model(observation, torch.tensor([7, 5]))
        for handle in handles:
            handle.remove()
        assert [label for label, shape in events] == [
            (kind, i) for i in range(2) for kind in order
        ]
        for (kind, index), shape in events:
            assert shape == ((4, 33, 8) if kind == "frequency" else (2, 2, 33, 8))
        assert ctf.shape == (2, 2, 33, 5)
        assert torch.is_complex(ctf) and torch.isfinite(ctf).all()
        torch.testing.assert_close(lengths, torch.tensor([7, 5]))
        outputs.append(ctf)
    assert not torch.allclose(outputs[0], outputs[1])


def test_invalid_interactions_fail_before_model_construction():
    for options in (
        {"post_interaction": "both"},
        {"post_interaction": False},
        {"post_interaction_order": "after_pool"},
        {"post_interaction": "speaker", "frequency_refinement_position": "before_pool"},
        {"post_interaction": "joint", "post_layers": 0},
        {"post_interaction": "frequency", "emb_dim": 6, "n_heads": 2},
    ):
        with unittest.TestCase().assertRaises(ValueError):
            _predictor(**options)
    for args in ((8, 2, 1e-5, "both"), (8, 3, 1e-5, "speaker"), (6, 2, 1e-5, "joint")):
        with unittest.TestCase().assertRaises(ValueError):
            PostPoolingAttention(*args)


def test_attention_configs_change_only_requested_architecture():
    configs = Path(__file__).resolve().parents[4] / "egs2/whamr/rir_2spk/conf/tuning"
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2"
    baseline = yaml.safe_load((configs / f"{prefix}.yaml").read_text())
    for suffix, options in VARIANTS.items():
        actual = yaml.safe_load((configs / f"{prefix}_{suffix}.yaml").read_text())
        expected = copy.deepcopy(baseline)
        expected["model_conf"]["predictor_conf"].update(options)
        assert actual == expected, suffix


def test_four_variants_actual_cpu_bilstm_pit_checkpoint_and_two_rirs():
    """Real recurrent integration verifies gradients, shared PIT and PIM APIs."""
    torch.manual_seed(41)
    batch = dict(
        speech_mix=torch.randn(2, 256),
        speech_mix_lengths=torch.tensor([256, 192]),
    )
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone()
        reverb[:, 180 + speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    for options in VARIANTS.values():
        with patch(
            "espnet2.rir.rec_rir.pooled_bimamba._build_mamba",
            side_effect=AssertionError("CPU LSTM requested Mamba"),
        ):
            model = _sweep_model(**options).train()
        loss, stats, count = model(**batch)
        assert torch.isfinite(loss) and count == 2
        torch.testing.assert_close(loss.detach(), stats["loss_sweep"])
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        for interaction in model.ctf_predictor.post_interaction_blocks:
            assert sum(p.grad.abs().sum() for p in interaction.parameters()) > 0
        model.eval()
        swapped = dict(batch)
        for prefix in ("rir_direct", "rir_ref"):
            swapped[prefix + "1"], swapped[prefix + "2"] = (
                batch[prefix + "2"], batch[prefix + "1"]
            )
        with torch.no_grad():
            torch.testing.assert_close(model(**batch)[0], model(**swapped)[0])
            expected = model.estimate_ctf(batch["speech_mix"][0])
            assert expected.shape == (2, 33, 5)
            clone = _sweep_model(**options).eval()
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            clone.load_state_dict(torch.load(buffer), strict=True)
            torch.testing.assert_close(
                expected, clone.estimate_ctf(batch["speech_mix"][0])
            )
            rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
            assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


def test_identical_paired_sweep_objective_and_full_tail():
    baseline = ESPnetTFLocoformerSweepV2PITModel(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=0.1, n_layers=1, emb_dim=8,
        num_groups=2, n_heads=2, attention_dim=8, pos_enc="nope",
        ffn_hidden_dim=[8, 8], conv1d_kernel=3,
    )
    direct = torch.zeros(2, 2, 256)
    direct[:, 0, 20], direct[:, 1, 30] = 1, -2
    target = torch.zeros_like(direct)
    target[:, 0, 20], target[:, 1, 158] = 3, -1
    ctf = torch.zeros(2, 2, 33, 5, dtype=torch.complex64)
    ctf[:, 0, :, 0], ctf[:, 1, :, 4] = 3, 0.5
    reference_loss, reference_perm = baseline.paired_pit_loss(ctf, direct, target)
    for options in VARIANTS.values():
        model = _sweep_model(**options)
        actual, perm = model.paired_pit_loss(ctf, direct, target)
        torch.testing.assert_close(actual, reference_loss, atol=0, rtol=0)
        torch.testing.assert_close(perm, reference_perm)
        torch.testing.assert_close(
            actual, model.paired_pit_loss(ctf.flip(1), direct, target)[0]
        )
        wrong = torch.zeros_like(ctf, requires_grad=True)
        bad, _ = model.paired_pit_loss(wrong, direct, target)
        bad.backward()
        assert bad > 0 and wrong.grad[..., 4].abs().sum() > 0
        # A final-sample impulse must retain the whole delayed sweep, including
        # the part after the original excitation duration plus STFT guards.
        impulse = torch.zeros(1, 256)
        impulse[:, -1] = 1
        expected_wave = F.pad(model.sweep, (255 + 64, 64))[None, None]
        expected_spec = model.transforms.stft(expected_wave, "complex")
        torch.testing.assert_close(
            model.response_spectrum(impulse), expected_spec, atol=2e-5, rtol=2e-5
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for key, value in list(globals().items()) if key.startswith("test_")
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
