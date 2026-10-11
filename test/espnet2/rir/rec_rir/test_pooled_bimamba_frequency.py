"""Frequency replacements: layout, initialization, shared loss and checkpoint.

CPU substitutes verify frequency-Mamba topology, not selective-scan kernels.
The laboratory check uses the real CUDA Mamba with the full experiment YAML.
"""

import copy
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn

from espnet2.rir.rec_rir.pooled_bimamba import (
    FramewiseFrequencyBlock,
    FrequencyAttentionBlock,
    FrequencyBiMambaBlock,
    PooledBiMambaCTFPredictor,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)


OPTIONS = {
    "freq_pre_attention": {
        "pre_frequency_module": "attention",
        "post_frequency_module": "attention",
    },
    "freq_pre_bimamba": {
        "pre_frequency_module": "bimamba",
        "post_frequency_module": "attention",
    },
    "freq_post_bimamba": {
        "pre_frequency_module": "light",
        "post_frequency_module": "bimamba",
    },
    "freq_both_bimamba": {
        "pre_frequency_module": "bimamba",
        "post_frequency_module": "bimamba",
    },
}
SMALL = dict(
    emb_dim=8, pre_layers=2, post_layers=2, d_state=4, d_conv=2,
    freq_bottleneck=4, freq_rank=4, local_kernel=3, pooling_hidden=4,
    n_heads=2, ffn_dim=8, head_dim=8,
)


class CpuSequenceSubstitute(nn.Module):
    """A trainable frequency-dependent stand-in, never a Mamba validation."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
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


def test_configs_change_only_frequency_families():
    configs = Path(__file__).resolve().parents[4] / "egs2/whamr/rir_2spk/conf/tuning"
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2"
    baseline = yaml.safe_load((configs / f"{prefix}.yaml").read_text())
    for variant, options in OPTIONS.items():
        actual = yaml.safe_load((configs / f"{prefix}_{variant}.yaml").read_text())
        expected = copy.deepcopy(baseline)
        expected["model_conf"]["predictor_conf"].update(options)
        assert actual == expected, variant


def test_frequency_mamba_reverses_bins_and_keeps_frames_independent():
    torch.manual_seed(9)
    source = FrequencyAttentionBlock(8, 2, 8, 3, 1e-5)
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        block = FrequencyBiMambaBlock(8, 4, 2, 1e-5, source)
    wrapper = FramewiseFrequencyBlock(block).eval()
    x = torch.randn(2, 3, 5, 8)
    output = wrapper(x)
    normalized = block.norm_mamba(x.reshape(6, 5, 8))
    torch.testing.assert_close(block.forward_mamba.seen, normalized)
    torch.testing.assert_close(block.backward_mamba.seen, normalized.flip(1))
    changed = x.clone()
    changed[0, 1, 2, 0] += 4
    difference = (wrapper(changed) - output).abs().sum((-1, -2))
    assert difference[0, 1] > 0
    difference[0, 1] = 0
    assert torch.count_nonzero(difference) == 0
    output.square().mean().backward()
    for branch in (block.forward_mamba, block.backward_mamba):
        assert branch.linear.weight.grad.abs().sum() > 0


def test_common_weights_and_rng_are_preserved():
    models = {}
    rng_states = []
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        for name, options in {"baseline": {}, **OPTIONS}.items():
            torch.manual_seed(17)
            models[name] = predictor(**options)
            rng_states.append(torch.get_rng_state())
            assert torch.initial_seed() == 17
    for state in rng_states:
        torch.testing.assert_close(state, rng_states[0], rtol=0, atol=0)
    base = models["baseline"].state_dict()
    for model in models.values():
        state = model.state_dict()
        for name, value in base.items():
            common = name.startswith(("encoder.", "time_blocks.", "pooling.",
                                      "head.", "slot_embeddings"))
            common |= name.startswith("post_blocks.") and name.split(".")[2] in {
                "norm_ffn", "ffn_in", "local", "ffn_out"
            }
            if common:
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    # Both replacement stages use identical initial weights in the combined run.
    for prefix, one in (("frequency_blocks.", "freq_pre_bimamba"),
                        ("post_blocks.", "freq_post_bimamba")):
        standalone = models[one].state_dict()
        combined = models["freq_both_bimamba"].state_dict()
        for name, value in standalone.items():
            if name.startswith(prefix):
                torch.testing.assert_close(value, combined[name], rtol=0, atol=0)
    # Attention versus Mamba in the pre-stage retains the same Conv-FFN.
    for index in range(2):
        attention = models["freq_pre_attention"].frequency_blocks[index].sequence_block
        mamba = models["freq_pre_bimamba"].frequency_blocks[index].sequence_block
        for component in ("norm_ffn", "ffn_in", "local", "ffn_out"):
            a, b = getattr(attention, component), getattr(mamba, component)
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)


def test_axes_and_stage_counts_are_unchanged():
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        for variant, options in OPTIONS.items():
            model = predictor(**options).eval()
            assert len(model.time_blocks) == len(model.frequency_blocks) == 2
            assert len(model.post_blocks) == 2
            events, handles = [], []
            for index, block in enumerate(model.frequency_blocks):
                if isinstance(block, FramewiseFrequencyBlock):
                    handles.append(block.sequence_block.register_forward_pre_hook(
                        lambda module, inputs: events.append(("pre", tuple(inputs[0].shape)))
                    ))
            for block in model.post_blocks:
                handles.append(block.register_forward_pre_hook(
                    lambda module, inputs: events.append(("post", tuple(inputs[0].shape)))
                ))
            ctf, _, _ = model(observation, torch.tensor([7, 5]))
            for handle in handles:
                handle.remove()
            pre_count = 0 if variant == "freq_post_bimamba" else 2
            assert events == [("pre", (14, 33, 8))] * pre_count + [
                ("post", (4, 33, 8))
            ] * 2
            assert ctf.shape == (2, 2, 33, 5) and torch.is_complex(ctf)


def test_replacements_respect_disabled_pre_stages():
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        model = predictor(pre_frequency_module="bimamba", frequency_block_indices=[0])
    assert isinstance(model.frequency_blocks[0], FramewiseFrequencyBlock)
    assert isinstance(model.frequency_blocks[1], nn.Identity)
    assert not any(name.startswith("frequency_blocks.1.") for name, _ in model.named_parameters())


def test_invalid_replacement_settings_fail():
    for options in (
        {"pre_frequency_module": "mamba"},
        {"pre_frequency_module": False},
        {"post_frequency_module": "light"},
        {"pre_frequency_module": "attention", "pre_frequency_global": False},
        {"post_frequency_module": "bimamba", "post_attention": False},
        {"post_frequency_module": "bimamba", "post_layers": 0},
    ):
        with unittest.TestCase().assertRaises(ValueError):
            predictor(**options)


def test_shared_sweep_pit_gradients_checkpoint_and_two_rirs():
    torch.manual_seed(23)
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 192]))
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone()
        reverb[:, 180 + speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", substitute):
        reference = sweep_model().eval()
        for options in OPTIONS.values():
            model = sweep_model(**options).train()
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
                clone.load_state_dict(torch.load(stream), strict=True)
                torch.testing.assert_close(before, clone.estimate_ctf(batch["speech_mix"][0]))
                rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
                assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


if __name__ == "__main__":
    tests = [unittest.FunctionTestCase(value) for name, value in list(globals().items())
             if name.startswith("test_")]
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(tests)).wasSuccessful())
