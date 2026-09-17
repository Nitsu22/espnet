"""CTF supervision using full direct and reverberant sweep responses."""

import math

import torch
import torch.nn.functional as F

from espnet2.rir.rec_rir.ctf_only import CTFOnlyBiSpatialNet
from espnet2.rir.rec_rir.espnet_model import ESPnetRecRIRModel
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetRecRIRSweepV2Model(ESPnetRecRIRModel):
    rec_rir_network_class = CTFOnlyBiSpatialNet

    def __init__(self, sweep_loss_weight=1.0, rir_loss_weight=0.0, **kwargs):
        weights = (float(sweep_loss_weight), float(rir_loss_weight))
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("Loss weights must be finite, nonnegative and not both zero")
        super().__init__(**kwargs)
        self.sweep_loss_weight, self.rir_loss_weight = weights
        if self.loss_w_cln != 0 or self.loss_w_rvb != 0 or self.loss_w_rec != 1:
            raise ValueError("Sweep v2 requires auxiliary weights 0 and reconstruction weight 1")
        self.register_buffer("sweep", self.pim.sinesweep.clone(), persistent=False)
        inverse = self.pim.invfilter.clone()
        # Calibrate only the KNOWN measurement system, never the predicted peak.
        gain = torch.dot(self.sweep.double(), inverse.flip(0).double())
        if not torch.isfinite(gain) or gain.abs() < 1e-8:
            raise ValueError("Invalid sweep/inverse calibration gain")
        self.register_buffer("inverse_sweep", inverse / gain.float(), persistent=False)
        self.inverse_delay = self.sweep.numel() - 1 + self.transforms.n_fft

    @torch.no_grad()
    def response_spectrum(self, rir):
        """Linear convolution with its complete tail and zero STFT boundary guard."""
        total = self.sweep.numel() + rir.shape[-1] - 1
        nfft = 1 << (total - 1).bit_length()
        response = torch.fft.irfft(
            torch.fft.rfft(self.sweep.float(), n=nfft)
            * torch.fft.rfft(rir.float(), n=nfft), n=nfft
        )[..., :total]
        # A common guard avoids reflecting a nonzero endpoint in centered STFT.
        response = F.pad(response, (self.transforms.n_fft, self.transforms.n_fft))
        return self.transforms.stft(response[:, None], "complex")

    def response_spectra(self, ctf, rir_direct, rir_ref):
        excitation = self.response_spectrum(rir_direct)
        reference = self.response_spectrum(rir_ref)
        prediction = self._complex_convolve(excitation, ctf)
        frames = max(prediction.shape[-1], reference.shape[-1])
        # Retain ALL target frames and ALL predicted tail frames.
        prediction = F.pad(prediction, (0, frames - prediction.shape[-1]))
        reference = F.pad(reference, (0, frames - reference.shape[-1]))
        return prediction, reference

    def sweep_loss(self, ctf, rir_direct, rir_ref):
        prediction, reference = self.response_spectra(ctf, rir_direct, rir_ref)
        return self._complex_loss(prediction, reference)

    def response_to_rir(self, prediction, rir_length):
        """Recover clean-to-reverb RIR from the predicted DIRECT-sweep response.

        Use the original sweep inverse, with fixed gain and delay compensation.
        Do not peak-align or normalize the predicted RIR. The extra delay is the
        left waveform guard added by response_spectrum, not the centered STFT.
        """
        response = self.transforms.istft(prediction, "complex").float()
        total = response.shape[-1] + self.inverse_sweep.numel() - 1
        nfft = 1 << (total - 1).bit_length()
        recovered = torch.fft.irfft(
            torch.fft.rfft(response, n=nfft)
            * torch.fft.rfft(self.inverse_sweep.float(), n=nfft), n=nfft
        )
        end = self.inverse_delay + rir_length
        if end > total:
            raise ValueError("RIR window exceeds the inverse-filter response")
        return recovered[..., self.inverse_delay:end]

    def forward(self, speech_mix, speech_mix_lengths, rir_ref, rir_direct,
                rir_ref_lengths=None, rir_direct_lengths=None, **kwargs):
        speech = self._to_mono(speech_mix)
        speech = speech[:, :int(speech_mix_lengths.max())]
        if self.normalize_by_mix:
            speech = speech / speech.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        spec = self.transforms.stft(speech[:, None], "complex")
        _, features, _ = self.rec_rir(self.transforms.preprocess(spec))
        ctf = self.transforms.postprocess(features).to(torch.complex64)
        direct, reverb = self._to_mono(rir_direct), self._to_mono(rir_ref)
        for rir, lengths in ((direct, rir_direct_lengths), (reverb, rir_ref_lengths)):
            if lengths is not None and not torch.all(lengths == rir.shape[-1]):
                raise ValueError("Sweep v2 requires fixed-length prepared RIRs")
        prediction, reference = self.response_spectra(ctf, direct, reverb)
        loss_sweep = self._complex_loss(prediction, reference)
        loss_rir = speech.new_zeros(())
        if self.rir_loss_weight:
            predicted_rir = self.response_to_rir(prediction, reverb.shape[-1])[:, 0]
            loss_rir = (predicted_rir - reverb).abs().mean()
        loss = self.sweep_loss_weight * loss_sweep + self.rir_loss_weight * loss_rir
        stats = {"loss": loss.detach(), "loss_sweep": loss_sweep.detach(),
                 "loss_rir_l1": loss_rir.detach()}
        return force_gatherable((loss, stats, speech.shape[0]), loss.device)

    def collect_feats(self, speech_mix, speech_mix_lengths, **kwargs):
        return {"speech_mix": speech_mix, "speech_mix_lengths": speech_mix_lengths}
