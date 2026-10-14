"""Real BiLSTM frequency-axis gradients, shared initialization and sweep PIT."""
import io
import unittest
from unittest.mock import patch

import torch

from espnet2.rir.rec_rir.pooled_bimamba import FrequencyBiLSTMBlock
from test.espnet2.rir.rec_rir.test_pooled_bimamba_ablations import (
    cpu_mamba_standin, predictor, sweep_model,
)


VARIANTS = (
    dict(time_module="bilstm", pre_frequency_module="light", post_frequency_module="bimamba"),
    dict(time_module="bilstm", pre_frequency_module="light", post_frequency_module="bilstm"),
    dict(time_module="bilstm", pre_frequency_module="bilstm", post_frequency_module="bilstm"),
)


def test_bilstm_frequency_axis_and_shared_post_speaker_weights():
    model = predictor(**VARIANTS[2])
    seen_pre, seen_post = [], []
    handles = []
    for block in model.frequency_blocks:
        assert isinstance(block.sequence_block, FrequencyBiLSTMBlock)
        handles.append(block.sequence_block.lstm.register_forward_pre_hook(
            lambda module, inputs: seen_pre.append(tuple(inputs[0].shape))))
    for block in model.post_blocks:
        handles.append(block.lstm.register_forward_pre_hook(
            lambda module, inputs: seen_post.append(tuple(inputs[0].shape))))
    output = model(torch.randn(2, 7, 33, dtype=torch.complex64))[0]
    for handle in handles:
        handle.remove()
    assert seen_pre == [(14, 33, 8)]
    assert seen_post == [(4, 33, 8)] * 2
    assert output.shape == (2, 2, 33, 5)
    output.abs().square().mean().backward()
    for block in [model.frequency_blocks[0].sequence_block] + list(model.post_blocks):
        for direction in ("weight_ih_l0", "weight_ih_l0_reverse"):
            assert getattr(block.lstm, direction).grad.abs().sum() > 0


def test_common_initial_weights_and_frequency_ffn_are_preserved():
    models = []
    for options in VARIANTS:
        with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", cpu_mamba_standin):
            torch.manual_seed(42)
            models.append(predictor(**options))
    states = [model.state_dict() for model in models]
    for name in states[0]:
        if name.startswith(("encoder.", "time_blocks.", "pooling.", "slot_embeddings", "head.")):
            for other in states[1:]:
                torch.testing.assert_close(states[0][name], other[name], rtol=0, atol=0)
    for index in range(2):
        for component in ("norm_ffn", "ffn_in", "local", "ffn_out"):
            prefix = f"post_blocks.{index}.{component}."
            for name in states[0]:
                if name.startswith(prefix):
                    for other in states[1:]:
                        torch.testing.assert_close(states[0][name], other[name], rtol=0, atol=0)
    for name in states[1]:
        if name.startswith("post_blocks."):
            torch.testing.assert_close(states[1][name], states[2][name], rtol=0, atol=0)


def test_all_three_train_through_sweep_pit_and_reload_two_rirs():
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 192]))
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = float(speaker)
        reverb = direct.clone()
        reverb[:, 145 + speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    for options in VARIANTS:
        replacement = cpu_mamba_standin if options["post_frequency_module"] == "bimamba" else None
        def factory(dim, specification):
            if replacement is None:
                raise AssertionError("All-BiLSTM construction requested Mamba")
            return replacement(dim, specification)
        with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", factory):
            model = sweep_model(**options).train()
        loss, stats, _ = model(**batch)
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss.detach(), stats["loss_sweep"])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        model.eval()
        swapped = dict(batch)
        for prefix in ("rir_direct", "rir_ref"):
            swapped[prefix + "1"], swapped[prefix + "2"] = batch[prefix + "2"], batch[prefix + "1"]
        with torch.no_grad():
            torch.testing.assert_close(model(**batch)[0], model(**swapped)[0])
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", factory):
                clone = sweep_model(**options).eval()
            clone.load_state_dict(torch.load(buffer), strict=True)
            torch.testing.assert_close(model.estimate_ctf(batch['speech_mix'][0]),
                                       clone.estimate_ctf(batch['speech_mix'][0]))
            rirs = clone.estimate_rir(batch['speech_mix'][0], rir_length=256)
            assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for key, value in list(globals().items()) if key.startswith("test_"))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
