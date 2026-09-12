import torch
from torch import nn

from espnet2.rir.rec_rir.espnet_model import ESPnetRecRIRModel
from espnet2.rir.rec_rir.espnet_model_sweep import ESPnetRecRIRSweepModel


def make_model(sweep=False):
    cls = ESPnetRecRIRSweepModel if sweep else ESPnetRecRIRModel
    return cls(
        sr=8000, n_fft=8, win_len=8, hop_len=4, num_freqs=5,
        dim_output_CTF=6, network_type="tflocoformer",
        network_conf=dict(n_layers=1, emb_dim=8, num_groups=2, n_heads=2,
                          attention_dim=8, pos_enc="nope",
                          ffn_hidden_dim=[8, 8], conv1d_kernel=3),
        loss_w_cln=0.0, loss_w_rvb=0.0, pim_sweep_duration=0.1,
    )


def check_single_source_backward_and_inference_layout(sweep):
    torch.manual_seed(0)
    model = make_model(sweep)
    speech = torch.randn(1, 64)
    batch = dict(speech_mix=speech, speech_mix_lengths=torch.tensor([64]))
    if sweep:
        rir = torch.zeros(1, 16)
        rir[:, 0] = 1.0
        batch["rir_ref"] = rir
    else:
        batch.update(speech_direct=speech, speech_reverb=speech)
    loss, stats, _ = model(**batch)
    torch.testing.assert_close(loss, stats["loss_sweep" if sweep else "loss_rec"])
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    # estimate_ctf normalizes its input before predicting.
    actual = model.estimate_ctf(speech)
    normalized = speech / speech.abs().amax(dim=1, keepdim=True)
    spec = model.transforms.stft(normalized[:, None], "complex")
    _, features, _ = model.rec_rir(model.transforms.preprocess(spec))
    expected = torch.complex(features[:, 0], features[:, 1]).flip(-1)
    assert actual.shape == (1, 5, 3)
    torch.testing.assert_close(actual, expected)


class IdentityCTF(nn.Module):
    def forward(self, features):
        ctf = features.new_zeros(features.shape[0], 2, features.shape[2], 3)
        ctf[:, 0, :, 0] = 1.0
        return None, ctf, None


def check_identity_ctf_matches_identity_rir_and_speech(sweep):
    model = make_model(sweep)
    model.rec_rir = IdentityCTF()
    speech = torch.randn(1, 64)
    batch = dict(speech_mix=speech, speech_mix_lengths=torch.tensor([64]))
    if sweep:
        rir = torch.zeros(1, 16)
        rir[:, 0] = 1.0
        batch["rir_ref"] = rir
    else:
        batch.update(speech_direct=speech, speech_reverb=speech)
    loss, _, _ = model(**batch)
    assert loss < 1e-5


def test_ctf_backward_and_inference():
    check_single_source_backward_and_inference_layout(False)


def test_sweep_backward_and_inference():
    check_single_source_backward_and_inference_layout(True)


def test_ctf_identity():
    check_identity_ctf_matches_identity_rir_and_speech(False)


def test_sweep_identity():
    check_identity_ctf_matches_identity_rir_and_speech(True)
