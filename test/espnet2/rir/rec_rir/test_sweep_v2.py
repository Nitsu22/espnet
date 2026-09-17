import numpy as np
import unittest
import torch

from espnet2.rir.rec_rir.espnet_model_sweep_v2 import ESPnetRecRIRSweepV2Model
from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.train.preprocessor_rec_rir_sweep_v2 import RecRIRSweepV2Preprocessor


def loss_model():
    model = ESPnetRecRIRSweepV2Model.__new__(ESPnetRecRIRSweepV2Model)
    torch.nn.Module.__init__(model)
    model.transforms = RecRIRTransforms(16000, 64, 32, 'sqrthann', 64)
    model.loss_type = 'RIMag'
    # Excitation near the END makes truncation of delayed responses observable.
    sweep = torch.zeros(512)
    sweep[-96:-64] = torch.linspace(-1, 1, 32)
    model.register_buffer('sweep', sweep)
    return model


def test_full_response_and_delayed_ctf_gradient():
    model = loss_model()
    direct = torch.zeros(1, 192)
    direct[:, 0] = 1
    target = torch.zeros_like(direct)
    target[:, 128] = 1
    response = model.response_spectrum(target)
    # Compare against independent time-domain convolution, including late energy.
    expected = torch.zeros(1, 512 + 192 - 1)
    expected[:, 128:128 + 512] = model.sweep
    assert expected[:, 512:].abs().sum() > 0
    expected = torch.nn.functional.pad(expected, (64, 64))
    torch.testing.assert_close(response, model.transforms.stft(expected[:, None]))
    ctf = torch.zeros(1, 1, 33, 5, dtype=torch.complex64)
    ctf[..., 4] = 1
    assert model.sweep_loss(ctf, direct, target) < 1e-5
    wrong = torch.zeros_like(ctf, requires_grad=True)
    loss = model.sweep_loss(wrong, direct, target)
    loss.backward()
    assert loss > 0
    assert torch.isfinite(wrong.grad).all()
    assert wrong.grad[..., 4].abs().sum() > 0


def test_paired_response_gain_and_extra_prediction_tail():
    model = loss_model()
    direct = torch.zeros(1, 192)
    direct[:, 60] = 2
    direct[:, 61] = -0.4
    target = direct * 3
    ctf = torch.full((1, 1, 33, 1), 3, dtype=torch.complex64)
    assert model.sweep_loss(ctf, direct, target) < 1e-5
    wrong = ctf / 3
    assert model.sweep_loss(wrong, direct, target) > 0.01
    # A late predicted reflection must be penalized even after target support.
    extended = torch.zeros(1, 1, 33, 20, dtype=torch.complex64)
    extended[..., 0] = 3
    extended[..., -1] = 1
    assert model.sweep_loss(extended, direct, target) > 0.01


def test_preprocessor_preserves_pair_and_rejects_truncation():
    proc = RecRIRSweepV2Preprocessor(False, rir_length=32)
    direct, reverb = np.zeros((32, 2)), np.zeros((32, 2))
    direct[5, 0] = -2
    reverb[5, 0], reverb[20, 0] = -2, 4
    data = {'speech_mix': np.ones((100, 2)), 'rir_direct': direct, 'rir_ref': reverb}
    out = proc('example', data)
    assert out['rir_direct'][5] == -1
    assert out['rir_ref'][5] == -1 and out['rir_ref'][20] == 2
    assert np.argmax(abs(out['rir_direct'])) == 5
    assert np.argmax(abs(out['rir_ref'])) == 20
    data['rir_ref'] = np.ones((33, 2))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'exceeds rir_length'):
        proc('example', data)


def combined_model(sw=1.0, rw=1.0):
    return ESPnetRecRIRSweepV2Model(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        dim_output_CTF=10, network_type='tflocoformer',
        network_conf=dict(n_layers=1, emb_dim=8, num_groups=2, n_heads=2,
                          attention_dim=8, pos_enc='nope',
                          ffn_hidden_dim=[8, 8], conv1d_kernel=3),
        loss_w_cln=0., loss_w_rvb=0., pim_sweep_duration=0.2,
        sweep_loss_weight=sw, rir_loss_weight=rw)


def test_inverse_delay_gain_and_gradient_against_scipy():
    from scipy.signal import fftconvolve
    m = combined_model()
    h = torch.zeros(1, 512)
    h[0, 101] = -0.4
    spec = m.response_spectrum(h).detach().requires_grad_()
    actual = m.response_to_rir(spec, 512)[0, 0]
    # Independent waveform-domain reference, including both guards.
    waveform = np.pad(fftconvolve(m.sweep.numpy(), h[0].numpy()), (64, 64))
    expected = fftconvolve(waveform, m.inverse_sweep.numpy())
    start = len(m.sweep) - 1 + 64
    torch.testing.assert_close(actual, torch.tensor(expected[start:start+512]),
                               atol=2e-6, rtol=2e-4)
    assert actual.abs().argmax() == 101
    assert abs(actual[101].item() + .4) < 2e-5
    twice = m.response_to_rir(spec * 2, 512)[0, 0]
    torch.testing.assert_close(twice, actual * 2)
    actual.abs().mean().backward()
    assert torch.isfinite(spec.grad).all() and spec.grad.abs().sum() > 0


def test_combined_weights_network_gradients_and_zero_weight_compatibility():
    torch.manual_seed(2)
    data = dict(speech_mix=torch.randn(2, 256),
                speech_mix_lengths=torch.tensor([256, 256]))
    direct = torch.zeros(2, 256)
    direct[:, 50] = 1
    reverb = direct.clone()
    reverb[:, 114] = .3
    data.update(rir_direct=direct, rir_ref=reverb)
    for sw, rw in ((1., 0.), (1., 1.), (.2, 3.), (0., 1.)):
        m = combined_model(sw, rw)
        loss, stats, _ = m(**data)
        torch.testing.assert_close(loss, sw * stats['loss_sweep'] + rw * stats['loss_rir_l1'])
        if rw == 0:
            speech = data['speech_mix'] / data['speech_mix'].abs().amax(1, keepdim=True)
            _, features, _ = m.rec_rir(m.transforms.preprocess(m.transforms.stft(speech[:, None])))
            ctf = m.transforms.postprocess(features)
            torch.testing.assert_close(loss.squeeze(), m.sweep_loss(ctf, direct, reverb))
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    with unittest.TestCase().assertRaises(ValueError):
        combined_model(0, 0)


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items()) if name.startswith("test_")
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
