import argparse

import pytest
import torch

import espnet2.rir.rec_rir.model as rec_rir_model
import espnet2.rir.rec_rir.model_lag_narrow as lag_narrow_module
from espnet2.rir.rec_rir.espnet_model import ESPnetRecRIRModel
from espnet2.rir.rec_rir.espnet_model_lag_narrow import (
    ESPnetRecRIRLagNarrowModel,
)
from espnet2.rir.rec_rir.model import BiSpatialNet
from espnet2.rir.rec_rir.model_lag_narrow import LagAwareBiSpatialNet
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
        "dim_output_spch": 2,
        "dim_output_CTF": dim_output_ctf,
        "dim_hidden": 4,
        "dim_squeeze": 2,
        "num_freqs": 3,
        "num_layers_spch": 1,
        "num_layers_noise": 1,
        "num_layers_CTF": 2,
        "encoder_kernel_size": 1,
        "dropout": (0.0, 0.0, 0.0),
        "kernel_size": (3, 3),
        "conv_groups": (1, 1),
        "norms": ("LN", "LN", "LN", "LN", "LN", "LN"),
        "attention": "mamba(16,4)",
    }


def test_lag_narrow_forward_backward_and_ctf_axis(identity_mamba, monkeypatch):
    checkpoint_calls = []
    original_checkpoint = lag_narrow_module.checkpoint

    def checkpoint_spy(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(lag_narrow_module, "checkpoint", checkpoint_spy)
    model = LagAwareBiSpatialNet(**_network_conf())
    ctf_inputs = []
    hook = model.ctf_layers[0].register_forward_pre_hook(
        lambda module, args: ctf_inputs.append(args[0].shape)
    )

    speech, ctf, reverb = model(torch.randn(2, 2, 3, 5))
    hook.remove()

    assert speech.shape == (2, 2, 3, 5)
    assert reverb.shape == (2, 2, 3, 5)
    assert ctf.shape == (2, 2, 3, 3)
    assert ctf_inputs == [torch.Size((2, 3, 3, 4))]
    assert len(model.ctf_layers) == 2
    assert len(checkpoint_calls) == 3
    assert all(call["use_reentrant"] is False for call in checkpoint_calls)

    (speech.mean() + ctf.mean() + reverb.mean()).backward()
    parameters = (
        model.compress_CTF.alpha,
        model.compress_CTF.beta,
        model.weight_layer[0].weight,
        model.decoder_CTF[2].weight,
        model.ctf_layers[0].norm_mamba_t_f.weight,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)

    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 2, 3, 5))
    assert len(checkpoint_calls) == 3


def test_lag_alignment_and_uniform_time_pooling(identity_mamba):
    conf = _network_conf()
    conf.update(
        dim_hidden=2,
        dim_squeeze=1,
        num_freqs=1,
        num_layers_spch=0,
        num_layers_noise=0,
        num_layers_CTF=0,
    )
    model = LagAwareBiSpatialNet(**conf).eval()
    with torch.no_grad():
        for parameter in model.weight_layer.parameters():
            parameter.zero_()

    clean = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    reverb = (100 + torch.arange(10, dtype=torch.float32)).reshape(1, 1, 5, 2)

    with torch.no_grad():
        model.compress_CTF.alpha.zero_()
        model.compress_CTF.beta.fill_(1.0)
        pooled_reverb = model._pool_lag_sequence(clean, reverb)
    expected_reverb = torch.stack(
        [reverb[:, :, lag:, :].mean(dim=2) for lag in range(3)],
        dim=2,
    )
    torch.testing.assert_close(pooled_reverb, expected_reverb)

    with torch.no_grad():
        model.compress_CTF.alpha.fill_(1.0)
        model.compress_CTF.beta.zero_()
        pooled_clean = model._pool_lag_sequence(clean, reverb)
    expected_clean = torch.stack(
        [clean[:, :, : 5 - lag, :].mean(dim=2) for lag in range(3)],
        dim=2,
    )
    torch.testing.assert_close(pooled_clean, expected_clean)


def test_lag_narrow_rejects_invalid_ctf_size_and_short_input(identity_mamba):
    with pytest.raises(ValueError, match="positive even"):
        LagAwareBiSpatialNet(**_network_conf(dim_output_ctf=5))

    model = LagAwareBiSpatialNet(**_network_conf())
    short = torch.randn(1, 3, 2, 4)
    with pytest.raises(ValueError, match=r"frames=2, lags=3"):
        model._pool_lag_sequence(short, short)


def test_rir_task_selects_new_model_without_changing_baseline(identity_mamba):
    model_conf = _network_conf()
    lag_args = argparse.Namespace(
        rir_model_type="rec_rir_lag_narrow",
        model_conf=model_conf,
        init=None,
    )
    lag_model = RIRTask.build_model(lag_args)
    assert isinstance(lag_model, ESPnetRecRIRLagNarrowModel)
    assert type(lag_model.rec_rir) is LagAwareBiSpatialNet

    baseline_args = argparse.Namespace(
        rir_model_type="rec_rir",
        model_conf=model_conf,
        init=None,
    )
    baseline_model = RIRTask.build_model(baseline_args)
    assert isinstance(baseline_model, ESPnetRecRIRModel)
    assert type(baseline_model.rec_rir) is BiSpatialNet
