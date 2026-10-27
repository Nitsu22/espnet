"""DAMSEP native causal CTF -> common oracle-CD CA measurement checks."""
import numpy as np
import torch

from test.espnet2.rir.rec_rir.test_ca_evaluation import physical_measurement
from test.espnet2.rir.rec_rir.test_ca_evaluation_pit import EVALUATE
from espnet2.rir.rec_rir.ca_evaluation import CAReconstructor
from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.pim import RecRIRPIM


def test_damsep_native_order_and_real_imag_axes():
    native = torch.zeros(2, 2, 257, 60)
    native[0, 0, :, 0] = 2
    native[0, 1, :, 0] = -3
    native[1, 0, :, 7] = -4
    native[1, 1, :, 7] = 5
    converted = EVALUATE.damsep_ctf_to_pim_order(native)
    assert converted.shape == (2, 257, 60)
    torch.testing.assert_close(converted[0, :, -1], torch.full((257,), 2-3j))
    torch.testing.assert_close(converted[1, :, 52], torch.full((257,), -4+5j))
    assert torch.count_nonzero(converted) == 2*257


def test_damsep_loss_stft_synthesis_matches_independent_waveform_delay():
    pim = RecRIRPIM(8000, 0.05)
    transform = RecRIRTransforms(8000, 512, 128, 'sqrthann', 256)
    recon = CAReconstructor(pim, transform, 2048)
    direct = torch.zeros(2, 512)
    direct[0, 31], direct[0, 49] = 0.3, -0.06
    direct[1, 79], direct[1, 98] = -0.8, 0.04
    native = torch.zeros(2, 2, 257, 60)
    native[0, 0, :, 0] = 1
    native[1, 0, :, 4] = -0.75
    predictions = recon.candidate_rirs(EVALUATE.damsep_ctf_to_pim_order(native), direct)
    for r in range(2):
        expected = physical_measurement(pim, direct[r].numpy(), 2048)
        # Delay the physical filter BEFORE measurement: shifting an already
        # cropped inverse response would discard pre-ringing that now lies
        # inside the output window.
        delayed = physical_measurement(pim, -0.75*np.pad(direct[r].numpy(), (512, 0)), 2048)
        np.testing.assert_allclose(predictions[0, r].numpy(), expected, atol=2e-6)
        np.testing.assert_allclose(predictions[1, r].numpy(), delayed, atol=2e-6)
    assert predictions[0].abs().argmax(-1).tolist() == [31, 79]
    assert predictions[1].abs().argmax(-1).tolist() == [543, 591]


def test_damsep_rejects_wrong_layout_and_nonfinite_ctfs():
    for bad in (torch.zeros(2, 257, 60), torch.full((2, 2, 257, 60), float('nan'))):
        try:
            EVALUATE.damsep_ctf_to_pim_order(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid DAMSEP CTF was accepted')
