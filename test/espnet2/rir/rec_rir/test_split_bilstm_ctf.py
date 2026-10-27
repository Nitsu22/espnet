"""CPU checks of intermediate speaker splitting and shared CTF decoding."""

import copy
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from espnet2.rir.rec_rir.pooled_bimamba import (
    FrequencyBiLSTMBlock,
    LightFrequencyBlock,
    TimeBiLSTM,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from espnet2.rir.rec_rir.split_bilstm import (
    FixedSpeakerSplit,
    SpeakerInteraction,
    SplitBiLSTMCTFPredictor,
)
from test.espnet2.rir.rec_rir.test_sweep_v2_drr_pit import fixture


REPO = Path(__file__).resolve().parents[4]
CONFIG = REPO / (
    "egs2/whamr/rir_2spk/conf/tuning/"
    "train_split_bilstm_2spk_nf_8k_sweep_v2_enc4_dec4_post4_rir_mse.yaml"
)


def configuration():
    return yaml.safe_load(CONFIG.read_text())


def small_model(**changes):
    conf = copy.deepcopy(configuration()["model_conf"])
    conf.update(n_fft=64, win_len=64, hop_len=32, num_freqs=33,
                ctf_taps=5, pim_sweep_duration=0.1)
    conf.update(changes)
    conf["predictor_conf"].update(
        emb_dim=8, freq_bottleneck=4, local_kernel=3, pooling_hidden=4,
        n_heads=2, ffn_dim=8, head_dim=8,
    )
    # Fixed integer overrides in the production YAML should also stay small.
    for name in ("split_hidden", "speaker_ffn_dim"):
        if conf["predictor_conf"].get(name) is not None:
            conf["predictor_conf"][name] = 8
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba",
               side_effect=AssertionError("The split model must not build Mamba")):
        return ESPnetPooledBiMambaSweepV2PITModel(**conf)


def test_configuration_preserves_training_and_selects_split_mse():
    config = configuration()
    conf = config["model_conf"]
    predictor = conf["predictor_conf"]
    assert conf["predictor_type"] == "split_bilstm"
    assert conf["sr"] == 8000 and conf["num_freqs"] == 129
    assert conf["ctf_taps"] == 60 and conf["num_spk"] == 2
    assert conf["time_sweep_loss_weight"] == 0
    assert conf["rir_mse_loss_weight"] == 10000
    assert predictor["split_after"] == 4
    assert predictor["pre_layers"] == 8 and predictor["post_layers"] == 4
    assert predictor["emb_dim"] == 64 and predictor["freq_rank"] == 16
    assert config["init_param"] == [] and config["resume"] is False
    assert config["max_epoch"] == 200 and config["optim_conf"]["lr"] == 0.001
    assert config["patience"] == 14
    assert config["scheduler_conf"]["patience"] == 4
    assert config["early_stopping_criterion"] == ["valid", "loss_sweep_min", "min"]


def test_split_produces_two_full_width_streams_and_distinct_gradients():
    torch.manual_seed(1)
    split = FixedSpeakerSplit(dim=8, num_spk=2, hidden_dim=12, eps=1e-5)
    x = torch.randn(2, 7, 11, 8, requires_grad=True)
    streams = split(x)
    assert streams.shape == (2, 2, 7, 11, 8)
    assert not torch.allclose(streams[:, 0], streams[:, 1])
    left_grad, = torch.autograd.grad(streams[:, 0].square().mean(), x,
                                    retain_graph=True)
    right_grad, = torch.autograd.grad(streams[:, 1].square().mean(), x,
                                     retain_graph=True)
    assert left_grad.abs().sum() > 0 and right_grad.abs().sum() > 0
    assert not torch.allclose(left_grad, right_grad)
    streams.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in split.parameters() if p.requires_grad)


def test_speaker_attention_is_local_cross_speaker_and_equivariant():
    torch.manual_seed(2)
    interaction = SpeakerInteraction(dim=8, heads=2, ffn_dim=12, eps=1e-5).eval()
    x = torch.randn(2, 2, 5, 7, 8, requires_grad=True)
    output = interaction(x)
    assert output.shape == x.shape
    torch.testing.assert_close(interaction(x.flip(1)), output.flip(1))
    # One output branch must use the other speaker at that same TF location.
    gradient, = torch.autograd.grad(output[0, 0, 2, 3].square().sum(), x)
    assert gradient[0, 1, 2, 3].abs().sum() > 0
    # This block must not accidentally attend along time, frequency, or batch.
    changed = x.detach().clone()
    changed[0, :, 1, 3] += 5 * torch.randn(2, 8)
    changed[0, :, 2, 4] += 5 * torch.randn(2, 8)
    changed[1] += 5 * torch.randn_like(changed[1])
    torch.testing.assert_close(interaction(changed)[0, :, 2, 3],
                               output.detach()[0, :, 2, 3], rtol=0, atol=0)


def test_speaker_attention_chunking_preserves_outputs_and_gradients():
    torch.manual_seed(5)
    reference = SpeakerInteraction(dim=8, heads=2, ffn_dim=12, eps=1e-5).double()
    assert reference._max_attention_batch_heads == 65535
    chunked = copy.deepcopy(reference)
    # Force several chunks and a partial final chunk on a small CPU fixture.
    chunked._max_attention_batch_heads = 12
    x = torch.randn(2, 2, 5, 7, 8, dtype=torch.float64, requires_grad=True)
    split_x = x.detach().clone().requires_grad_(True)
    calls = {"reference": [], "chunked": []}

    def record(name):
        def hook(module, args):
            value = args[0]
            assert value.shape[2] == 1
            calls[name].append(value.shape[0] * module.heads)
        return hook

    handles = [reference.attention.register_forward_pre_hook(record("reference")),
               chunked.attention.register_forward_pre_hook(record("chunked"))]
    try:
        output, split_output = reference(x), chunked(split_x)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls["reference"]) == 1
    assert len(calls["chunked"]) > 1
    assert max(calls["chunked"]) <= chunked._max_attention_batch_heads
    assert calls["chunked"][-1] < chunked._max_attention_batch_heads
    assert sum(calls["reference"]) == sum(calls["chunked"]) == 2 * 5 * 7 * 2
    torch.testing.assert_close(split_output, output, rtol=1e-10, atol=1e-10)

    # Check the earlier frame/frequency grouping too: attention stays TF-local.
    with torch.no_grad():
        previous = reference.attention(x.permute(0, 2, 1, 3, 4).reshape(10, 2, 7, 8))
        previous = previous.reshape(2, 5, 2, 7, 8).permute(0, 2, 1, 3, 4)
        previous = previous + reference.ffn(reference.norm_ffn(previous))
    torch.testing.assert_close(output, previous, rtol=1e-10, atol=1e-10)

    weights = torch.randn_like(output)
    (output * weights).sum().backward()
    (split_output * weights).sum().backward()
    torch.testing.assert_close(split_x.grad, x.grad, rtol=1e-9, atol=1e-10)
    for (name, parameter), (other_name, other) in zip(
            reference.named_parameters(), chunked.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all(), name
        torch.testing.assert_close(other.grad, parameter.grad, rtol=1e-9, atol=1e-10)


def test_recurrent_axes_change_only_after_the_speaker_split():
    model = small_model().ctf_predictor.eval()
    assert isinstance(model, SplitBiLSTMCTFPredictor)
    assert len(model.time_blocks) == 8 and len(model.frequency_blocks) == 8
    assert len(model.post_blocks) == 4 and len(model.speaker_interactions) == 4
    assert all(isinstance(block, TimeBiLSTM) for block in model.time_blocks)
    assert all(isinstance(block, LightFrequencyBlock) for block in model.frequency_blocks)
    assert all(isinstance(block, FrequencyBiLSTMBlock) for block in model.post_blocks)
    assert not any("slot_embeddings" in name for name, _ in model.named_parameters())
    time_seen, frequency_seen, attention_seen = [], [], []
    handles = []
    for block in model.time_blocks:
        handles.append(block.lstm.register_forward_pre_hook(
            lambda module, args: time_seen.append(tuple(args[0].shape))))
    for block in model.post_blocks:
        handles.append(block.lstm.register_forward_pre_hook(
            lambda module, args: frequency_seen.append(tuple(args[0].shape))))
    for block in model.speaker_interactions:
        handles.append(block.register_forward_pre_hook(
            lambda module, args: attention_seen.append(tuple(args[0].shape))))
    lengths = torch.tensor([9, 7])
    try:
        ctf, actual_lengths, other = model(
            torch.randn(2, 9, 33, dtype=torch.complex64), lengths)
    finally:
        for handle in handles:
            handle.remove()
    assert time_seen == [(66, 9, 8)] * 4 + [(132, 9, 8)] * 4
    assert frequency_seen == [(4, 33, 8)] * 4
    assert attention_seen == [(2, 2, 9, 33, 8)] * 4
    assert ctf.shape == (2, 2, 33, 5) and torch.is_complex(ctf)
    assert actual_lengths is lengths and not other


def test_whole_shared_decoder_commutes_with_speaker_permutation():
    torch.manual_seed(3)
    model = small_model().ctf_predictor.eval()
    streams = torch.randn(2, 2, 7, 33, 8)
    with torch.no_grad():
        output = model.decode_streams(streams)
        swapped = model.decode_streams(streams.flip(1))
    assert output.shape == (2, 2, 33, 5)
    torch.testing.assert_close(swapped, output.flip(1))


def test_split_model_uses_matched_mse_and_one_paired_pit_assignment():
    model = small_model()
    ctf, direct, target = fixture()
    loss, best, stats = model.paired_pit_loss_with_stats(ctf, direct, target)
    assert loss < 1e-4 and stats["loss_rir_mse"] < 1e-8
    assert "loss_time_sweep_l1" not in stats
    torch.testing.assert_close(loss, stats["loss_sweep"] + 10000 * stats["loss_rir_mse"])
    d2, t2 = direct.clone(), target.clone()
    d2[0], t2[0] = direct[0].flip(0), target[0].flip(0)
    swapped_loss, swapped_best = model.paired_pit_loss(ctf, d2, t2)
    torch.testing.assert_close(swapped_loss, loss)
    assert best[0] != swapped_best[0] and best[1] == swapped_best[1]
    # Competing terms must choose one weighted assignment, never their own minima.
    spectral = [torch.tensor(x) for x in ([0., 0.], [3., 3.], [3., 3.], [0., 0.])]
    mse = [torch.tensor(x) for x in ([.0004, .0002], [0., 0.], [0., 0.], [.0004, .0002])]
    with patch.object(model, "_complex_loss", side_effect=spectral), patch.object(
            model, "rir_peak_mse", side_effect=mse):
        weighted, chosen, weighted_stats = model.paired_pit_loss_with_stats(ctf, direct, target)
    torch.testing.assert_close(chosen, torch.tensor([1, 0]))
    torch.testing.assert_close(weighted, torch.tensor(2.5))
    torch.testing.assert_close(weighted, weighted_stats["loss_sweep"]
                               + 10000 * weighted_stats["loss_rir_mse"])


def test_split_model_backpropagates_reloads_and_infers_two_rirs():
    torch.manual_seed(4)
    model = small_model()
    _, direct, target = fixture()
    batch = dict(speech_mix=torch.randn(2, 256),
                 speech_mix_lengths=torch.tensor([256, 224]))
    for speaker in range(2):
        batch[f"rir_direct{speaker + 1}"] = direct[:, speaker]
        batch[f"rir_ref{speaker + 1}"] = target[:, speaker]
    loss, stats, count = model(**batch)
    assert count == 2 and torch.isfinite(loss)
    assert "loss_time_sweep_l1" not in stats
    torch.testing.assert_close(loss.detach(), stats["loss_sweep"]
                               + 10000 * stats["loss_rir_mse"])
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for module in (model.ctf_predictor.splitter, *model.ctf_predictor.speaker_interactions):
        assert sum(p.grad.abs().sum() for p in module.parameters()
                   if p.requires_grad) > 0
    model.eval()
    with torch.no_grad():
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        clone = small_model().eval()
        clone.load_state_dict(torch.load(checkpoint), strict=True)
        torch.testing.assert_close(model.estimate_ctf(batch["speech_mix"][0]),
                                   clone.estimate_ctf(batch["speech_mix"][0]))
        rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
        assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(value, description=name)
                               for name, value in list(globals().items())
                               if name.startswith("test_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
