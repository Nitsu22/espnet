"""Standalone ablation regressions; CPU stand-ins do not validate Mamba kernels.

The LSTM training/inference test uses the actual recurrent implementation.
Only the topology/checkpoint regression tests replace Mamba with a small
trainable module, so they can run without CUDA or selective-scan kernels.
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
    FrequencyAttentionBlock,
    LightFrequencyBlock,
    PooledBiMambaCTFPredictor,
    TimeBiMamba,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)


SMALL_PREDICTOR = dict(
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


def cpu_mamba_standin(dim, specification):
    """A topology-only substitute, never presented as real Mamba validation."""
    return nn.Sequential(nn.Linear(dim, dim), nn.SiLU())


def predictor(**changes):
    return PooledBiMambaCTFPredictor(
        input_dim=33, ctf_taps=5, **(SMALL_PREDICTOR | changes)
    )


def sweep_model(**changes):
    return ESPnetPooledBiMambaSweepV2PITModel(
        sr=16000,
        n_fft=64,
        win_len=64,
        hop_len=32,
        num_freqs=33,
        ctf_taps=5,
        pim_sweep_duration=0.1,
        predictor_conf=SMALL_PREDICTOR | changes,
    )


class LegacyPredictor(nn.Module):
    """Pre-ablation state layout and forward path, frozen for compatibility."""

    def __init__(self):
        super().__init__()
        dim, freqs, speakers, taps = 8, 33, 2, 5
        self.encoder = nn.Conv2d(2, dim, kernel_size=3, padding=1)
        self.time_blocks = nn.ModuleList([TimeBiMamba(dim, 4, 2, 1e-5)])
        self.frequency_blocks = nn.ModuleList(
            [LightFrequencyBlock(dim, freqs, 4, 4, 3, 1e-5)]
        )
        self.pooling = nn.Sequential(
            nn.Linear(dim, 4), nn.GELU(), nn.Linear(4, speakers * dim)
        )
        self.slot_embeddings = nn.Parameter(torch.empty(speakers, dim))
        nn.init.normal_(self.slot_embeddings, std=0.02)
        self.post_blocks = nn.ModuleList(
            [FrequencyAttentionBlock(dim, 2, 8, 3, 1e-5) for _ in range(2)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-5),
            nn.Linear(dim, 8),
            nn.SiLU(),
            nn.Linear(8, 2 * taps),
        )

    def forward(self, observation):
        x = self.encoder(
            torch.stack((observation.real, observation.imag), 1).float()
        ).permute(0, 2, 3, 1)
        for temporal, frequency in zip(self.time_blocks, self.frequency_blocks):
            x = frequency(temporal(x))
        batch, frames, freqs, dim = x.shape
        weights = self.pooling(x).reshape(batch, frames, freqs, 2, dim).softmax(1)
        x = (weights * x.unsqueeze(3)).sum(1).permute(0, 2, 1, 3)
        x = (x + self.slot_embeddings[None, :, None]).reshape(batch * 2, freqs, dim)
        for frequency in self.post_blocks:
            x = frequency(x)
        output = self.head(x).reshape(batch, 2, freqs, 5, 2).float()
        return torch.complex(output[..., 0], output[..., 1]).contiguous()


def test_default_legacy_checkpoint_and_unmasked_output():
    """Adding switches must preserve the old default checkpoint and padding."""
    with patch(
        "espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin
    ):
        torch.manual_seed(41)
        legacy = LegacyPredictor().eval()
        torch.manual_seed(41)
        current = predictor().eval()
    old_state, new_state = legacy.state_dict(), current.state_dict()
    assert old_state.keys() == new_state.keys()
    for name in old_state:
        torch.testing.assert_close(old_state[name], new_state[name], rtol=0, atol=0)
    current.load_state_dict(old_state, strict=True)
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    observation[1, 5:] = 0
    with torch.no_grad():
        expected = legacy(observation)
        actual, returned_lengths, _ = current(observation, torch.tensor([7, 5]))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(returned_lengths, torch.tensor([7, 5]))
        torch.testing.assert_close(
            actual, current(observation, torch.tensor([7, 7]))[0], rtol=0, atol=0
        )


def test_frequency_placement_uses_same_weights_with_different_sequence_counts():
    """Placement alone changes B*T versus B*S work, with shared speaker weights."""
    with patch(
        "espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin
    ):
        after = predictor().eval()
        before = predictor(frequency_refinement_position="before_pool").eval()
    before.load_state_dict(after.state_dict(), strict=True)
    assert sum(p.numel() for p in after.parameters()) == sum(
        p.numel() for p in before.parameters()
    )
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    outputs = []
    for model, expected_sequences in ((after, 4), (before, 14)):
        seen = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs: seen.append(tuple(inputs[0].shape))
            )
            for block in model.post_blocks
        ]
        ctf = model(observation)[0]
        for handle in handles:
            handle.remove()
        assert seen == [(expected_sequences, 33, 8)] * 2, seen
        assert ctf.shape == (2, 2, 33, 5) and torch.isfinite(ctf).all()
        ctf.abs().square().mean().backward()
        for block in model.post_blocks:
            assert all(p.grad is not None for p in block.parameters() if p.requires_grad)
            assert sum(p.grad.abs().sum() for p in block.parameters() if p.requires_grad) > 0
        outputs.append(ctf.detach())
    # An accidentally unused switch would produce identical output here.
    assert not torch.allclose(outputs[0], outputs[1])


def test_attention_removal_keeps_trainable_frequency_ffn():
    """The new ablation removes attention, unlike the older whole-block removal."""
    with patch(
        "espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin
    ):
        full = predictor()
        ffn_only = predictor(post_attention=False)
    full_parameters = dict(full.named_parameters())
    ffn_parameters = dict(ffn_only.named_parameters())
    removed = full_parameters.keys() - ffn_parameters.keys()
    assert removed and all(
        name.startswith("post_blocks.")
        and name.split(".")[2] in {"norm_attention", "qkv", "rope", "projection"}
        for name in removed
    ), removed
    for name, parameter in ffn_parameters.items():
        assert name in full_parameters and parameter.shape == full_parameters[name].shape
    output = ffn_only(torch.randn(2, 7, 33, dtype=torch.complex64))[0]
    output.abs().square().mean().backward()
    for index in range(2):
        for component in ("norm_ffn", "ffn_in", "local", "ffn_out"):
            prefix = f"post_blocks.{index}.{component}."
            gradients = [
                p.grad for name, p in ffn_parameters.items() if name.startswith(prefix)
            ]
            assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(g.abs().sum() for g in gradients) > 0, prefix


def test_real_bilstm_training_pit_checkpoint_and_rir_inference():
    """Exercise actual CPU BiLSTM, full paired-sweep loss, and inference API."""
    torch.manual_seed(7)
    # This construction must not depend on the optional Mamba implementation.
    with patch(
        "espnet2.rir.rec_rir.pooled_bimamba._build_mamba",
        side_effect=AssertionError("BiLSTM requested Mamba"),
    ):
        model = sweep_model(time_module="bilstm").train()
    batch = dict(
        speech_mix=torch.randn(2, 256),
        speech_mix_lengths=torch.tensor([256, 192]),
    )
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone()
        reverb[:, 155 + speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    loss, stats, count = model(**batch)
    assert torch.isfinite(loss) and count == 2
    torch.testing.assert_close(stats["loss"], stats["loss_sweep"])
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for direction in ("weight_ih_l0", "weight_ih_l0_reverse"):
        assert getattr(model.ctf_predictor.time_blocks[0].lstm, direction).grad.abs().sum() > 0
    for component in ("encoder", "time_blocks", "frequency_blocks", "pooling",
                      "slot_embeddings", "post_blocks", "head"):
        gradients = [
            p.grad.abs().sum() for name, p in model.named_parameters()
            if p.requires_grad and name.startswith("ctf_predictor." + component)
        ]
        assert gradients and sum(gradients) > 0, component
    model.eval()
    with torch.no_grad():
        spectrum = model.transforms.stft(batch["speech_mix"][:, None], "complex")
        ctf, _ = model._estimate_ctf_and_aux_from_complex(spectrum)
        direct = torch.stack([batch[f"rir_direct{s}"] for s in (1, 2)], 1)
        reverb = torch.stack([batch[f"rir_ref{s}"] for s in (1, 2)], 1)
        baseline = ESPnetTFLocoformerSweepV2PITModel(
            sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
            ctf_taps=5, pim_sweep_duration=0.1, n_layers=1, emb_dim=8,
            num_groups=2, n_heads=2, attention_dim=8, pos_enc="nope",
            ffn_hidden_dim=[8, 8], conv1d_kernel=3,
        )
        actual_loss, actual_permutation = model.paired_pit_loss(ctf, direct, reverb)
        reference_loss, reference_permutation = baseline.paired_pit_loss(ctf, direct, reverb)
        torch.testing.assert_close(actual_loss, reference_loss, rtol=0, atol=0)
        torch.testing.assert_close(actual_permutation, reference_permutation)
        torch.testing.assert_close(
            actual_loss, model.paired_pit_loss(ctf.flip(1), direct, reverb)[0]
        )
        estimated = model.estimate_ctf(batch["speech_mix"][0])
        assert estimated.shape == (2, 33, 5) and torch.isfinite(estimated).all()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = sweep_model(time_module="bilstm").eval()
        clone.load_state_dict(torch.load(buffer), strict=True)
        torch.testing.assert_close(estimated, clone.estimate_ctf(batch["speech_mix"][0]))
        rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
        assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


def test_ablation_configs_change_only_requested_architecture():
    configs = Path(__file__).resolve().parents[4] / "egs2/whamr/rir_2spk/conf/tuning"
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2"
    baseline = yaml.safe_load((configs / f"{prefix}.yaml").read_text())
    changes = {
        "ffn_only": {"post_attention": False},
        "freq_before_pool": {"frequency_refinement_position": "before_pool"},
        "2blocks": {"pre_layers": 2},
        "bilstm": {"time_module": "bilstm"},
        "time2_freq4": {"time_block_indices": [0, 2]},
        "time4_freq2": {"frequency_block_indices": [0, 2]},
        "scalar_pool": {"pooling_channelwise": False},
        "local_freq_only": {"pre_frequency_global": False},
    }
    for suffix, options in changes.items():
        actual = yaml.safe_load((configs / f"{prefix}_{suffix}.yaml").read_text())
        expected = copy.deepcopy(baseline)
        expected["model_conf"]["predictor_conf"].update(options)
        assert actual == expected, suffix


def test_invalid_architecture_selectors_fail_early():
    for options in (
        {"post_attention": "false"},
        {"post_attention": 0},
        {"frequency_refinement_position": "after_time"},
        {"time_module": "lstm"},
        {"time_block_indices": [1]},
        {"time_block_indices": [0, 0]},
        {"frequency_block_indices": [-1]},
        {"frequency_block_indices": [True]},
        {"time_block_indices": "0"},
        {"pooling_channelwise": "false"},
        {"pre_frequency_global": 0},
    ):
        with unittest.TestCase().assertRaises(ValueError):
            predictor(**options)


def test_independent_depth_removal_preserves_other_stages_and_order():
    """Two-stage masks remove only the requested branch, never zip-truncate."""
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin):
        full = predictor(pre_layers=4)
        time2 = predictor(pre_layers=4, time_block_indices=[0, 2])
        freq2 = predictor(pre_layers=4, frequency_block_indices=[0, 2])
    complete = dict(full.named_parameters())
    for model, removed_branch in ((time2, "time_blocks"), (freq2, "frequency_blocks")):
        active = dict(model.named_parameters())
        removed = complete.keys() - active.keys()
        assert removed and all(name.startswith((f"{removed_branch}.1.", f"{removed_branch}.3."))
                               for name in removed), removed
        assert all(name in complete and value.shape == complete[name].shape
                   for name, value in active.items())
        assert len(model.time_blocks) == len(model.frequency_blocks) == 4
        events, handles = [], []
        for branch in ("time_blocks", "frequency_blocks"):
            for index, block in enumerate(getattr(model, branch)):
                handles.append(block.register_forward_pre_hook(
                    lambda module, inputs, label=(branch, index): events.append(label)))
        result = model(torch.randn(2, 7, 33, dtype=torch.complex64))[0]
        for handle in handles:
            handle.remove()
        assert events == [(branch, i) for i in range(4)
                          for branch in ("time_blocks", "frequency_blocks")]
        result.abs().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        for i in (1, 3):
            assert isinstance(getattr(model, removed_branch)[i], nn.Identity)


def test_scalar_pooling_keeps_speaker_specific_weights_and_broadcasts_channels():
    """Only the channel dimension shares a time distribution, not speakers."""
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin):
        model = predictor(pooling_channelwise=False, post_layers=0).eval()
    captured = {}
    pool_handle = model.pooling.register_forward_hook(
        lambda module, inputs, output: captured.update(x=inputs[0].detach(), scores=output.detach()))
    head_handle = model.head.register_forward_pre_hook(
        lambda module, inputs: captured.update(pooled=inputs[0].detach()))
    output = model(torch.randn(2, 7, 33, dtype=torch.complex64))[0]
    pool_handle.remove(); head_handle.remove()
    x = captured["x"]
    scores = captured["scores"].reshape(2, 7, 33, 2, 1)
    weights = scores.softmax(1)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 33, 2, 1))
    assert not torch.allclose(weights[..., 0, :], weights[..., 1, :])
    expected = (weights * x.unsqueeze(3)).sum(1).permute(0, 2, 1, 3)
    expected = (expected + model.slot_embeddings[None, :, None]).reshape(4, 33, 8)
    torch.testing.assert_close(captured["pooled"], expected)
    output.abs().square().mean().backward()
    assert model.pooling[-1].out_features == 2
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_global_frequency_removal_keeps_local_parameters_and_gradient():
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin):
        full = predictor(pre_layers=4)
        local = predictor(pre_layers=4, pre_frequency_global=False)
    full_parameters, local_parameters = dict(full.named_parameters()), dict(local.named_parameters())
    removed = full_parameters.keys() - local_parameters.keys()
    assert removed and all(name.startswith("frequency_blocks.") and
                           name.split(".")[2] in {"global_norm", "compress", "frequency_mlp", "expand"}
                           for name in removed)
    assert all(name in full_parameters and p.shape == full_parameters[name].shape
               for name, p in local_parameters.items())
    result = local(torch.randn(2, 7, 33, dtype=torch.complex64))[0]
    result.abs().square().mean().backward()
    for block in local.frequency_blocks:
        assert not hasattr(block, "frequency_mlp")
        assert block.local.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in local.parameters() if p.requires_grad)


def test_four_architecture_variants_full_sweep_pit_and_checkpoint():
    """All active branches train through the shared objective and decode two RIRs."""
    variants = ({"time_block_indices": [0, 2]}, {"frequency_block_indices": [0, 2]},
                {"pooling_channelwise": False}, {"pre_frequency_global": False})
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 192]))
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone(); reverb[:, 130 + speaker] = .3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    for options in variants:
        with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin):
            model = sweep_model(pre_layers=4, **options).train()
        loss, stats, _ = model(**batch)
        assert torch.isfinite(loss)
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
            expected = model.estimate_ctf(batch["speech_mix"][0])
            with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin):
                clone = sweep_model(pre_layers=4, **options).eval()
            buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
            clone.load_state_dict(torch.load(buffer), strict=True)
            torch.testing.assert_close(expected, clone.estimate_ctf(batch["speech_mix"][0]))
            rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
            assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for key, value in list(globals().items()) if key.startswith("test_")
    )
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
