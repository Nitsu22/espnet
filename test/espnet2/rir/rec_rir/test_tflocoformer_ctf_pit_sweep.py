import numpy as np
import torch

from espnet2.rir.rec_rir.tflocoformer_ctf_pit import (
    ESPnetRecRIRTFLocoformerPITModel,
)
from espnet2.train.preprocessor_rec_rir_pit import RecRIRPITPreprocessor


def _sweep_model(reconstruction_signal="sweep"):
    return ESPnetRecRIRTFLocoformerPITModel(
        sr=8000,
        n_fft=8,
        win_len=8,
        hop_len=4,
        num_freqs=5,
        num_spk=2,
        ctf_taps=3,
        n_layers=1,
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
        reconstruction_signal=reconstruction_signal,
    )


def test_sweep_preprocessor_uses_mix_and_two_rirs_only():
    preprocessor = RecRIRPITPreprocessor(
        train=False,
        use_sweep_target=True,
        rir_length=12,
        force_single_channel=True,
    )
    mix = np.arange(20, dtype=np.float64).reshape(10, 2)
    rir1 = np.arange(10, dtype=np.float64).reshape(5, 2)
    rir2 = np.arange(16, dtype=np.float64).reshape(8, 2)

    output = preprocessor(
        "utt",
        {
            "speech_mix": mix,
            "rir_ref1": rir1,
            "rir_ref2": rir2,
        },
    )

    np.testing.assert_array_equal(output["speech_mix"], mix[:, 0])
    np.testing.assert_array_equal(output["rir_ref1"][:5], rir1[:, 0])
    np.testing.assert_array_equal(output["rir_ref2"][:8], rir2[:, 0])
    assert output["rir_ref1"].shape == (12,)
    assert output["rir_ref2"].shape == (12,)
    assert np.count_nonzero(output["rir_ref1"][5:]) == 0
    assert np.count_nonzero(output["rir_ref2"][8:]) == 0


def test_sweep_preprocessor_peak_aligns_and_normalizes_rirs():
    preprocessor = RecRIRPITPreprocessor(
        train=False,
        use_sweep_target=True,
        rir_length=10,
        canonicalize_rir_by_peak=True,
        rir_peak_index=2,
    )
    rir1 = np.asarray([0.0, 0.5, -4.0, 2.0, 1.0], dtype=np.float64)
    rir2 = np.asarray([0.0, 1.0, 0.5, 0.0, 0.0, 3.0], dtype=np.float64)

    output = preprocessor(
        "utt",
        {
            "speech_mix": np.ones(12, dtype=np.float64),
            "rir_ref1": rir1,
            "rir_ref2": rir2,
        },
    )

    assert int(np.argmax(np.abs(output["rir_ref1"]))) == 2
    assert int(np.argmax(np.abs(output["rir_ref2"]))) == 2
    assert np.max(np.abs(output["rir_ref1"])) == 1.0
    assert np.max(np.abs(output["rir_ref2"])) == 1.0
    assert output["rir_ref1"][2] == -1.0
    assert output["rir_ref2"][2] == 1.0


def test_sweep_delta_rirs_match_ctf_tap_zero_with_pit():
    model = _sweep_model().to(dtype=torch.float32)
    assert not any(torch.is_complex(buffer) for buffer in model.buffers())
    rir = torch.zeros(1, 2, 16)
    rir[:, 0, 0] = 1.0
    rir[:, 1, 0] = 2.0
    reference = model._sweep_reference_stft(rir)

    # Reverse the two reference gains so PIT must select permutation 1.
    est_ctf = torch.zeros(1, 2, 5, 3, dtype=torch.complex64)
    est_ctf[:, 0, :, 0] = 2.0
    est_ctf[:, 1, :, 0] = 1.0
    loss, best_perm = model._pit_sweep_rec_loss(est_ctf, reference)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2.0e-6, rtol=0.0)
    assert best_perm.tolist() == [1]


def test_sweep_loss_backpropagates_to_estimated_ctf():
    model = _sweep_model()
    rir = torch.zeros(1, 2, 16)
    rir[:, 0, 0] = 1.0
    rir[:, 1, 0] = 2.0
    reference = model._sweep_reference_stft(rir)

    est_ctf = torch.zeros(
        1, 2, 5, 3, dtype=torch.complex64, requires_grad=True
    )
    loss, _ = model._pit_sweep_rec_loss(est_ctf, reference)
    loss.backward()

    assert loss.item() > 0.0
    assert est_ctf.grad is not None
    assert torch.isfinite(est_ctf.grad).all()
    assert torch.count_nonzero(est_ctf.grad) > 0


def test_sweep_forward_uses_mix_and_rir_references():
    model = _sweep_model()
    speech_mix = torch.randn(1, 32)
    speech_mix_lengths = torch.tensor([32], dtype=torch.long)
    rir_ref1 = torch.zeros(1, 16)
    rir_ref2 = torch.zeros(1, 16)
    rir_ref1[:, 0] = 1.0
    rir_ref2[:, 0] = 0.5

    loss, stats, weight = model(
        speech_mix=speech_mix,
        speech_mix_lengths=speech_mix_lengths,
        rir_ref1=rir_ref1,
        rir_ref2=rir_ref2,
    )
    loss.backward()

    assert loss.numel() == 1
    assert torch.isfinite(loss)
    assert torch.isfinite(stats["loss_sweep"])
    assert weight.item() == 1
    assert model.ctf_predictor.ctf_head[-1].weight.grad is not None


def test_existing_speech_forward_remains_available():
    model = _sweep_model(reconstruction_signal="speech")
    speech_mix = torch.randn(1, 32)
    lengths = torch.tensor([32], dtype=torch.long)
    direct1 = torch.randn(1, 32)
    direct2 = torch.randn(1, 32)
    reverb1 = torch.randn(1, 32)
    reverb2 = torch.randn(1, 32)

    loss, stats, _ = model(
        speech_mix=speech_mix,
        speech_mix_lengths=lengths,
        speech_direct1=direct1,
        speech_direct1_lengths=lengths,
        speech_direct2=direct2,
        speech_direct2_lengths=lengths,
        speech_reverb1=reverb1,
        speech_reverb1_lengths=lengths,
        speech_reverb2=reverb2,
        speech_reverb2_lengths=lengths,
    )

    assert loss.numel() == 1
    assert torch.isfinite(loss)
    assert "loss_sweep" not in stats
