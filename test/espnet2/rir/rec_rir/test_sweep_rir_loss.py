import pytest
import torch
from scipy.signal import fftconvolve

from espnet2.rir.rec_rir.espnet_model_sweep import ESPnetRecRIRSweepModel


def model(sw=1.0, rw=0.0):
    return ESPnetRecRIRSweepModel(
        sr=8000, n_fft=8, win_len=8, hop_len=4, num_freqs=5,
        dim_output_CTF=6, network_type='tflocoformer',
        network_conf=dict(n_layers=1, emb_dim=8, num_groups=2, n_heads=2,
                          attention_dim=8, pos_enc='nope',
                          ffn_hidden_dim=[8, 8], conv1d_kernel=3),
        loss_w_cln=0., loss_w_rvb=0., pim_sweep_duration=0.1,
        sweep_loss_weight=sw, rir_loss_weight=rw)


@pytest.mark.parametrize('sw,rw', [(1., 0.), (0., 1.), (1., 1.), (.2, 3.)])
def test_weights_and_network_gradients(sw, rw):
    torch.manual_seed(2)
    m = model(sw, rw)
    rir = torch.zeros(2, 64)
    rir[:, 20] = 1
    loss, stats, _ = m(speech_mix=torch.randn(2, 80),
                       speech_mix_lengths=torch.tensor([80, 80]), rir_ref=rir,
                       rir_ref_lengths=torch.tensor([64, 64]))
    torch.testing.assert_close(loss, sw * stats['loss_sweep'] + rw * stats['loss_rir_l1'])
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_fixed_delay_inverse_filter_against_scipy():
    m = model(0., 1.)
    h = torch.zeros(64)
    h[20], h[35] = -1., .3
    response = torch.tensor(fftconvolve(m.sweep.numpy(), h.numpy()))
    spec = m.transforms.stft(response, 'complex')[None, None].requires_grad_()
    actual = m.response_to_rir(spec, 64)
    waveform = m.transforms.istft(spec.detach(), 'complex')[0, 0].numpy()
    expected = torch.tensor(fftconvolve(waveform, m.inverse_sweep.numpy()))
    expected = expected[m.inverse_delay:m.inverse_delay + 64]
    expected /= expected.abs().max()
    torch.testing.assert_close(actual[0, 0], expected, atol=2e-5, rtol=2e-4)
    assert actual[0, 0].abs().argmax() == 20
    assert actual[0, 0, 20] < 0  # Preserve polarity.
    (actual - h).abs().mean().backward()
    assert torch.isfinite(spec.grad).all() and spec.grad.abs().sum() > 0


def test_legacy_sweep_loss_unchanged():
    m = model()
    speech = torch.randn(1, 80)
    h = torch.zeros(1, 64)
    h[:, 20] = 1
    normalized = speech / speech.abs().amax(dim=1, keepdim=True)
    _, features, _ = m.rec_rir(m.transforms.preprocess(m.transforms.stft(normalized[:, None], 'complex')))
    ctf = m.transforms.postprocess(features)
    excitation = torch.complex(m.sweep_real, m.sweep_imag)[None, None]
    reference = m.reference_spectrum(h)
    old_loss = m._complex_loss(m._complex_convolve(excitation, ctf)[..., :reference.shape[-1]], reference)
    loss, _, _ = m(speech_mix=speech, speech_mix_lengths=torch.tensor([80]), rir_ref=h)
    torch.testing.assert_close(loss.squeeze(), old_loss)


@pytest.mark.parametrize('weights', [(0, 0), (-1, 1), (1, float('nan'))])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        model(*weights)
