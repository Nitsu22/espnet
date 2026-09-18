import unittest
import torch
from scipy.signal import fftconvolve
from espnet2.rir.rec_rir.pim import RecRIRPIM
from espnet2.rir.rec_rir.feature import RecRIRTransforms


class TestDirectSweep(unittest.TestCase):
    def test_delta_matches_original_and_keeps_relative_reflection(self):
        pim = RecRIRPIM(sr=8000, sweep_duration=.1)
        transform = RecRIRTransforms(8000, 64, 32, 'sqrthann', 64)
        ctf = torch.ones(33, 1, dtype=torch.complex64)
        ordinary = pim.ctf_to_rir(ctf, transform, torch.device('cpu'), 512)
        delta = pim.ctf_to_rir(ctf, transform, torch.device('cpu'), 512, torch.ones(1))
        torch.testing.assert_close(delta, ordinary, atol=2e-6, rtol=2e-4)
        direct = torch.zeros(128)
        direct[40], direct[104] = -.4, .1
        actual = pim.ctf_to_rir(ctf, transform, torch.device('cpu'), 512, direct)
        response = fftconvolve(pim.sinesweep.numpy(), direct.numpy())
        expected = torch.tensor(fftconvolve(response, pim.invfilter.numpy()))
        start = max(0, int(expected.abs().argmax()) - 20)
        expected = expected[start:start+512]
        if expected.abs().max() > 1:
            expected /= expected.abs().max()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
        self.assertEqual(int(actual.abs().argmax()), 20)
        self.assertLess(actual[20], 0)
        self.assertGreater(actual[84], 0)

    def test_invalid_direct_rir(self):
        pim = RecRIRPIM(sr=8000, sweep_duration=.1)
        transform = RecRIRTransforms(8000, 64, 32, 'sqrthann', 64)
        for direct in (torch.zeros(3), torch.ones(1, 3), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                pim.ctf_to_rir(torch.ones(33, 1, dtype=torch.complex64),
                               transform, torch.device('cpu'), 512, direct)


if __name__ == '__main__':
    unittest.main()
