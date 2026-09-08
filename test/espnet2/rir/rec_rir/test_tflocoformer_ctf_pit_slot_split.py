import pytest
import torch

from espnet2.rir.rec_rir.tflocoformer_ctf_pit import (
    ESPnetRecRIRTFLocoformerPITModel,
    TFLocoformerCTFPredictor,
    TFLocoformerSlotSplitCTFPredictor,
)


def _predictor_conf():
    return {
        "input_dim": 5,
        "num_spk": 2,
        "ctf_taps": 3,
        "n_layers": 4,
        "emb_dim": 8,
        "norm_type": "rmsgroupnorm",
        "num_groups": 2,
        "tf_order": "ft",
        "n_heads": 2,
        "flash_attention": False,
        "attention_dim": 8,
        "pos_enc": "nope",
        "ffn_type": ["swiglu_conv1d", "swiglu_conv1d"],
        "ffn_hidden_dim": [8, 8],
        "conv1d_kernel": 3,
        "conv1d_shift": 1,
        "dropout": 0.0,
        "eps": 1.0e-5,
    }


def _complex_input(batch=2, frames=6, freqs=5):
    return torch.complex(
        torch.randn(batch, frames, freqs),
        torch.randn(batch, frames, freqs),
    )


def test_slot_split_shapes_weight_sharing_and_independent_heads():
    model = TFLocoformerSlotSplitCTFPredictor(
        **_predictor_conf(),
        num_shared_layers=2,
        slot_embedding_std=0.02,
    )
    decoder_inputs = []
    time_weights = [[], []]
    hooks = [
        model.blocks[2].register_forward_pre_hook(
            lambda module, args: decoder_inputs.append(args[0].shape)
        )
    ]
    for slot_idx, scorer in enumerate(model.weight_layers):
        hooks.append(
            scorer.register_forward_hook(
                lambda module, args, output, idx=slot_idx: time_weights[idx].append(
                    output.detach()
                )
            )
        )

    input_complex = _complex_input()
    ctf, _, _ = model(input_complex)
    for hook in hooks:
        hook.remove()

    assert ctf.shape == (2, 2, 5, 3)
    assert torch.is_complex(ctf)
    # There are still four block modules.  The first private-side block sees
    # both slots folded into one batch, proving the weights are reused.
    assert len(model.blocks) == 4
    assert decoder_inputs == [torch.Size((2 * 2, 8, 6, 5))]

    for slot_idx in range(2):
        assert len(time_weights[slot_idx]) == 1
        assert time_weights[slot_idx][0].shape == (2, 5, 6, 1)
        torch.testing.assert_close(
            time_weights[slot_idx][0].sum(dim=2),
            torch.ones(2, 5, 1),
        )

    assert model.weight_layers[0] is not model.weight_layers[1]
    assert model.ctf_heads[0] is not model.ctf_heads[1]
    assert (
        model.weight_layers[0][0].weight.data_ptr()
        != model.weight_layers[1][0].weight.data_ptr()
    )
    assert (
        model.ctf_heads[0][0].weight.data_ptr()
        != model.ctf_heads[1][0].weight.data_ptr()
    )

    with torch.no_grad():
        batch0 = input_complex.unsqueeze(1)
        encoded = model.conv(torch.cat((batch0.real, batch0.imag), dim=1).float())
        slot_features = model._apply_slot_blocks(encoded)
    assert slot_features.shape == (2, 2, 8, 6, 5)
    assert not torch.allclose(slot_features[:, 0], slot_features[:, 1])


def test_slot_split_backward_reaches_shared_decoder_slots_and_both_heads():
    model = TFLocoformerSlotSplitCTFPredictor(
        **_predictor_conf(),
        num_shared_layers=2,
        slot_embedding_std=0.02,
    )
    ctf, _, _ = model(_complex_input())
    (ctf.real.square().mean() + ctf.imag.square().mean()).backward()

    parameters = [
        model.slot_embeddings,
        model.blocks[0].freq_path.attn.qkv.weight,
        model.blocks[2].freq_path.attn.qkv.weight,
        model.weight_layers[0][0].weight,
        model.weight_layers[1][0].weight,
        model.ctf_heads[0][0].weight,
        model.ctf_heads[1][0].weight,
    ]
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_baseline_predictor_path_is_unchanged():
    baseline = TFLocoformerCTFPredictor(**_predictor_conf())
    ctf, _, _ = baseline(_complex_input())

    assert ctf.shape == (2, 2, 5, 3)
    assert hasattr(baseline, "weight_layer")
    assert hasattr(baseline, "ctf_head")
    assert not hasattr(baseline, "slot_embeddings")


def test_wrapper_selects_slot_split_only_when_requested():
    common = dict(
        sr=8000,
        n_fft=8,
        win_len=8,
        hop_len=4,
        num_freqs=5,
        num_spk=2,
        ctf_taps=3,
        n_layers=4,
        emb_dim=8,
        norm_type="rmsgroupnorm",
        num_groups=2,
        tf_order="ft",
        n_heads=2,
        flash_attention=False,
        attention_dim=8,
        pos_enc="nope",
        ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
        ffn_hidden_dim=[8, 8],
        conv1d_kernel=3,
        conv1d_shift=1,
        pim_sweep_duration=0.1,
    )
    baseline = ESPnetRecRIRTFLocoformerPITModel(**common, slot_split=False)
    slot_split = ESPnetRecRIRTFLocoformerPITModel(
        **common,
        slot_split=True,
        num_shared_layers=2,
        slot_embedding_std=0.02,
    )

    assert type(baseline.ctf_predictor) is TFLocoformerCTFPredictor
    assert type(slot_split.ctf_predictor) is TFLocoformerSlotSplitCTFPredictor


@pytest.mark.parametrize("num_shared_layers", [0, 4, 5])
def test_slot_split_rejects_invalid_layer_boundaries(num_shared_layers):
    with pytest.raises(ValueError, match="num_shared_layers"):
        TFLocoformerSlotSplitCTFPredictor(
            **_predictor_conf(),
            num_shared_layers=num_shared_layers,
        )
