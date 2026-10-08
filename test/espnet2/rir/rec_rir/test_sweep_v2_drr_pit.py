"""Matched sweep measurement, DRR gradients and one shared PIT assignment."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import ESPnetTFLocoformerSweepV2PITModel


def model(weight=0.1):
    return ESPnetTFLocoformerSweepV2PITModel(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, n_layers=1, emb_dim=8, num_groups=2, n_heads=2,
        attention_dim=8, pos_enc='nope', ffn_hidden_dim=[8, 8],
        conv1d_kernel=3, pim_sweep_duration=.1, drr_loss_weight=weight)


def fixture():
    # Non-impulse direct paths: predicting the relative filter must recover
    # the corresponding physical RIR, rather than comparing that filter alone.
    direct = torch.zeros(2, 2, 256)
    direct[:, 0, 20] = 1
    direct[:, 0, 27] = .2
    direct[:, 1, 35] = -2
    direct[:, 1, 42] = -.3
    reverb = direct.clone()
    reverb[:, 0, 128:] += .4 * direct[:, 0, :-128]
    reverb[:, 1, 96:] += .6 * direct[:, 1, :-96]
    ctf = torch.zeros(2, 2, 33, 5, dtype=torch.complex64)
    ctf[..., 0] = 1
    ctf[:, 0, :, 4] = .4
    ctf[:, 1, :, 3] = .6
    return ctf, direct, reverb


def test_drr_energy_windows_gain_polarity_and_stability():
    m = model()
    rir = torch.zeros(1, 256)
    rir[0, 90] = 1
    rir[0, 180] = .5
    rir[0, 20] = .25  # pre-direct energy must be in the denominator
    center = torch.tensor([90])
    expected = 10 * np.log10(1 / (.5**2 + .25**2))
    value = m.drr_db(rir, center)
    torch.testing.assert_close(value, torch.tensor([expected], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value, m.drr_db(-3 * rir, center))
    torch.testing.assert_close(value, m.drr_db(1e-9 * rir, center))
    for scale in (0, 1e-9):
        quiet = (scale * rir).requires_grad_()
        quiet_drr = m.drr_db(quiet, center)
        assert torch.isfinite(quiet_drr).all()
        quiet_drr.sum().backward()
        assert torch.isfinite(quiet.grad).all()


def test_paired_measurement_equivalence_permutation_and_old_weights():
    m = model()
    ctf, direct, reverb = fixture()
    loss, perm, stats = m.paired_pit_loss_with_stats(ctf, direct, reverb)
    assert loss < 1e-5 and stats['loss_drr'] < 1e-8, (loss, stats)
    torch.testing.assert_close(loss, m.paired_pit_loss(ctf.flip(1), direct, reverb)[0])
    d2, r2 = direct.clone(), reverb.clone()
    d2[0], r2[0] = direct[0].flip(0), reverb[0].flip(0)
    swapped, p2 = m.paired_pit_loss(ctf, d2, r2)
    torch.testing.assert_close(loss, swapped)
    assert p2[0] != p2[1]
    old = model(0)
    assert old.state_dict().keys() == m.state_dict().keys()
    old.load_state_dict(m.state_dict(), strict=True)
    a = old.paired_pit_loss(ctf, direct, reverb)[0]
    torch.testing.assert_close(a, stats['loss_sweep'], rtol=0, atol=0)


def test_drr_alone_has_ctf_gradients():
    m = model()
    ctf, direct, reverb = fixture()
    wrong = ctf[:, 0].clone()
    wrong[..., 4] = .1
    wrong.requires_grad_()
    excitation = m.response_spectrum(direct[:, 0])[:, 0]
    predicted = m._complex_convolve(excitation, wrong)
    recovered = m.response_to_rir(predicted[:, None], 256)[:, 0]
    with torch.no_grad():
        reference = m.response_to_rir(m.response_spectrum(reverb[:, 0]), 256)[:, 0]
        centers = reverb[:, 0].abs().argmax(-1)
        target_drr = m.drr_db(reference, centers)
    loss = F.smooth_l1_loss(m.drr_db(recovered, centers), target_drr)
    loss.backward()
    assert loss > 0 and torch.isfinite(wrong.grad).all()
    assert wrong.grad[..., 4].abs().sum() > 0
    assert wrong.grad[..., 0].abs().sum() > 0


def test_one_assignment_for_sweep_and_drr():
    direct = torch.zeros(1, 2, 256)
    direct[..., 20] = 1
    reverb = direct.clone()
    reverb[:, 0, 20], reverb[:, 0, 148] = 8, 4
    reverb[:, 1, 148] = .1
    ctf = torch.zeros(1, 2, 33, 5, dtype=torch.complex64)
    ctf[:, 0, :, 0], ctf[:, 0, :, 4] = 8, .8
    ctf[:, 1, :, 0], ctf[:, 1, :, 4] = 1, .5
    _, sweep_perm = model(0).paired_pit_loss(ctf, direct, reverb)
    m = model(100)
    total, joint_perm, stats = m.paired_pit_loss_with_stats(ctf, direct, reverb)
    assert sweep_perm.item() == 0 and joint_perm.item() == 1
    torch.testing.assert_close(total.detach(), stats['loss_sweep'] + 100 * stats['loss_drr'])


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for key, value in list(globals().items()) if key.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
