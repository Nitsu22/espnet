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


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items()) if name.startswith("test_")
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
