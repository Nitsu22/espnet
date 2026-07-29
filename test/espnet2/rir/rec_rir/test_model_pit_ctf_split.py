import argparse

import pytest
import torch

import espnet2.rir.rec_rir.model as rec_rir_model
from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel
from espnet2.rir.rec_rir.espnet_model_pit_ctf_split import (
    ESPnetRecRIRPITCTFSplitModel,
)
from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT
from espnet2.rir.rec_rir.model_pit_ctf_split import BiSpatialNetPITCTFSplit
from espnet2.tasks.rir import RIRTask


@pytest.fixture
def identity_mamba(monkeypatch):
    monkeypatch.setattr(
        rec_rir_model,
        "_build_mamba",
        lambda dim_hidden, attention: torch.nn.Identity(),
    )


def _network_conf(dim_output_ctf=6):
    return {
        "dim_input": 2,
        "dim_output_spch": 4,
        "dim_output_CTF": dim_output_ctf,
        "dim_hidden": 4,
        "dim_squeeze": 2,
        "num_freqs": 3,
        "num_layers_spch": 1,
        "num_layers_noise": 1,
        "num_layers_CTF": 1,
        "encoder_kernel_size": 1,
        "dropout": (0.0, 0.0, 0.0),
        "kernel_size": (3, 3),
        "conv_groups": (1, 1),
        "norms": ("LN", "LN", "LN", "LN", "LN", "LN"),
        "attention": "mamba(16,4)",
        "num_spk": 2,
    }


def test_ctf_split_forward_paths_shapes_and_gradients(identity_mamba):
    batch, speakers, freqs, frames, hidden = 2, 2, 3, 5, 4
    ctf_taps = 3
    model = BiSpatialNetPITCTFSplit(**_network_conf())

    speaker_split_calls = []
    decoder_spch_inputs = []
    decoder_rev_inputs = []
    ctf_inputs = []
    time_weights = []
    hooks = [
        model.speaker_split.register_forward_hook(
            lambda module, args, output: speaker_split_calls.append(
                (module, args[0].shape, output.shape)
            )
        ),
        model.decoder_spch.register_forward_pre_hook(
            lambda module, args: decoder_spch_inputs.append(args[0].shape)
        ),
        model.decoder_rev.register_forward_pre_hook(
            lambda module, args: decoder_rev_inputs.append(args[0].shape)
        ),
        model.ctf_layers[0].register_forward_pre_hook(
            lambda module, args: ctf_inputs.append(args[0].shape)
        ),
        model.weight_layer.register_forward_hook(
            lambda module, args, output: time_weights.append(output.detach())
        ),
    ]

    speech, ctf, reverb = model(torch.randn(batch, 2, freqs, frames))
    for hook in hooks:
        hook.remove()

    assert speech.shape == (batch, speakers, 2, freqs, frames)
    assert reverb.shape == (batch, speakers, 2, freqs, frames)
    assert ctf.shape == (batch, speakers, 2, freqs, ctf_taps)

    assert len(speaker_split_calls) == 2
    assert all(call[0] is model.speaker_split for call in speaker_split_calls)
    assert [call[1] for call in speaker_split_calls] == [
        torch.Size((batch, freqs, frames, hidden)),
        torch.Size((batch, freqs, frames, hidden)),
    ]
    assert [call[2] for call in speaker_split_calls] == [
        torch.Size((batch, freqs, frames, speakers * hidden)),
        torch.Size((batch, freqs, frames, speakers * hidden)),
    ]

    # Clean/reverb heads remain on the unsplit baseline path.
    assert decoder_spch_inputs == [torch.Size((batch, freqs, frames, hidden))]
    assert decoder_rev_inputs == [torch.Size((batch, freqs, frames, hidden))]
    # Only the CTF path folds the speaker axis into the batch axis.
    assert ctf_inputs == [torch.Size((batch * speakers, freqs, frames, hidden))]
    assert len(time_weights) == 1
    assert time_weights[0].shape == (batch * speakers, freqs, frames, 1)
    torch.testing.assert_close(
        time_weights[0].sum(dim=2),
        torch.ones(batch * speakers, freqs, 1),
    )

    (speech.mean() + ctf.mean() + reverb.mean()).backward()
    parameters = (
        model.speaker_split.weight,
        model.compress_CTF.alpha,
        model.compress_CTF.beta,
        model.ctf_layers[0].norm_mamba_t_f.weight,
        model.weight_layer[0].weight,
        model.decoder_CTF[2].weight,
        model.decoder_spch[2].weight,
        model.decoder_rev[2].weight,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)

    with torch.no_grad():
        embedding = model(
            torch.randn(batch, 2, freqs, frames),
            return_embedding=True,
        )
    assert embedding.shape == (batch, speakers, freqs, 1, hidden)


def test_clean_reverb_outputs_bypass_ctf_speaker_split(identity_mamba):
    model = BiSpatialNetPITCTFSplit(**_network_conf())
    speech, _, reverb = model(torch.randn(2, 2, 3, 5))

    (speech.mean() + reverb.mean()).backward()

    assert model.speaker_split.weight.grad is None
    assert model.speaker_split.bias.grad is None
    assert model.decoder_spch[2].weight.grad is not None
    assert model.decoder_rev[2].weight.grad is not None


def test_rir_task_selects_ctf_split_without_changing_pit_baseline(
    identity_mamba,
):
    split_args = argparse.Namespace(
        rir_model_type="rec_rir_pit_ctf_split",
        model_conf=_network_conf(),
        init=None,
    )
    split_model = RIRTask.build_model(split_args)
    assert isinstance(split_model, ESPnetRecRIRPITCTFSplitModel)
    assert type(split_model.rec_rir) is BiSpatialNetPITCTFSplit

    baseline_conf = _network_conf(dim_output_ctf=12)
    baseline_args = argparse.Namespace(
        rir_model_type="rec_rir_pit",
        model_conf=baseline_conf,
        init=None,
    )
    baseline_model = RIRTask.build_model(baseline_args)
    assert isinstance(baseline_model, ESPnetRecRIRPITModel)
    assert type(baseline_model.rec_rir) is BiSpatialNetPIT
