"""Oracle-CD clean-to-all RIR reconstruction for evaluation only.

Input CTF taps use estimate_ctf()'s PIM (reversed) ordering. The network sees
only the mixture; known CD RIRs are used afterwards for every slot/ref pair.
"""
import torch
import torch.nn.functional as F
import torchaudio


class CAReconstructor:
    def __init__(self, pim, transform, rir_length):
        self.transform = transform
        self.sweep = pim.sinesweep
        gain = torch.dot(self.sweep.double(), pim.invfilter.flip(0).double())
        if not torch.isfinite(gain) or gain.abs() < 1e-8:
            raise ValueError("Invalid known sweep/inverse calibration gain")
        self.inverse = pim.invfilter / gain.float()
        self.delay = self.sweep.numel() - 1 + transform.n_fft
        self.rir_length = int(rir_length)
        if self.rir_length <= 0:
            raise ValueError("rir_length must be positive")

    @staticmethod
    def convolve(signal, filt):
        """Full real linear convolution, with broadcasting over leading axes."""
        total = signal.shape[-1] + filt.shape[-1] - 1
        nfft = 1 << (total - 1).bit_length()
        return torch.fft.irfft(torch.fft.rfft(signal, n=nfft)
                               * torch.fft.rfft(filt, n=nfft), n=nfft)[..., :total]

    def response_spectrum(self, rirs):
        if rirs.ndim != 2 or not torch.isfinite(rirs).all():
            raise ValueError("Expected finite [reference, samples] RIRs")
        response = self.convolve(rirs.float(), self.sweep.to(rirs.device))
        guard = self.transform.n_fft
        response = F.pad(response, (guard, guard))
        return self.transform.stft(response[:, None], 'complex')[:, 0]

    def recover_spectrum(self, spectra):
        """Fixed known-system delay/gain compensation; no predicted peak fit."""
        shape = spectra.shape[:-2]
        flat = spectra.reshape(-1, 1, *spectra.shape[-2:])
        response = self.transform.istft(flat, 'complex')[:, 0].float()
        recovered = self.convolve(response, self.inverse.to(response.device))
        end = self.delay + self.rir_length
        if end > recovered.shape[-1]:
            raise ValueError("RIR window exceeds full inverse-filter response")
        return recovered[:, self.delay:end].reshape(*shape, self.rir_length)

    @torch.no_grad()
    def candidate_rirs(self, pim_ctf, direct_rirs):
        """Return [predicted slot, reference speaker, samples] CA candidates.

        Each candidate uses the same reference speaker's CD excitation. PIT
        must choose ONE slot permutation against that speaker's raw CA RIR.
        """
        if (pim_ctf.ndim != 3 or not torch.is_complex(pim_ctf)
                or not torch.isfinite(pim_ctf).all()):
            raise ValueError("Expected finite complex [slot, frequency, taps] CTF")
        if torch.any(direct_rirs.abs().amax(-1) == 0):
            raise ValueError("CD RIRs must be nonzero")
        excitation = self.response_spectrum(direct_rirs)
        if excitation.shape[-2] != pim_ctf.shape[-2]:
            raise ValueError("CTF frequency count does not match STFT")
        # Public estimate_ctf reverses the learned causal taps for PIM's
        # unfold/matmul implementation. Undo that reversal exactly once.
        causal_ctf = pim_ctf.flip(-1)
        spectra = torch.stack([
            torchaudio.functional.convolve(excitation, filt[None], mode='full')
            for filt in causal_ctf
        ])
        return self.recover_spectrum(spectra)
