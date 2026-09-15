"""Rec-RIR feature network with a CTF-only output and sweep supervision."""
import math

import torch
from espnet2.rir.rec_rir.ctf_only import CTFOnlyBiSpatialNet
from espnet2.rir.rec_rir.espnet_model import ESPnetRecRIRModel
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetRecRIRSweepModel(ESPnetRecRIRModel):
    rec_rir_network_class = CTFOnlyBiSpatialNet

    def __init__(self, sweep_loss_weight=1.0, rir_loss_weight=0.0, **kwargs):
        weights = (float(sweep_loss_weight), float(rir_loss_weight))
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("Loss weights must be finite, nonnegative and not both zero")
        super().__init__(**kwargs)
        self.sweep_loss_weight, self.rir_loss_weight = weights
        sweep = self.pim.sinesweep.clone()
        self.register_buffer('sweep', sweep, persistent=False)
        self.register_buffer('inverse_sweep', self.pim.invfilter.clone(), persistent=False)
        # Both sweep and inverse include 512-sample padding on each side.
        # Their linear-convolution impulse is at len(sweep)-1.
        self.inverse_delay = len(sweep) - 1
        spec = self.transforms.stft(sweep, 'complex')
        self.register_buffer('sweep_real', spec.real, persistent=False)
        self.register_buffer('sweep_imag', spec.imag, persistent=False)

    @torch.no_grad()
    def reference_spectrum(self, rir):
        sweep = self.sweep.float()
        total = len(sweep) + rir.shape[-1] - 1
        nfft = 1 << (total - 1).bit_length()
        response = torch.fft.irfft(
            torch.fft.rfft(sweep, n=nfft) * torch.fft.rfft(rir.float(), n=nfft), n=nfft)
        # Match the existing sweep experiment: score the excitation-length window.
        return self.transforms.stft(response[..., :len(sweep)], 'complex')[:, None]

    def response_to_rir(self, response_spectrum, rir_length):
        """Differentiable PIM with fixed delay removal, not predicted-peak alignment.

        Use the full CTF-convolved sweep spectrum so that the response tail is
        retained. Peak amplitude normalization preserves polarity and gradients.
        """
        response = self.transforms.istft(response_spectrum, 'complex').float()
        total = response.shape[-1] + self.inverse_sweep.numel() - 1
        nfft = 1 << (total - 1).bit_length()
        recovered = torch.fft.irfft(
            torch.fft.rfft(response, n=nfft)
            * torch.fft.rfft(self.inverse_sweep.float(), n=nfft), n=nfft)
        end = self.inverse_delay + rir_length
        if end > total:
            raise ValueError("RIR window exceeds the linear inverse-filter response")
        rir = recovered[..., self.inverse_delay:end]
        return rir / rir.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, speech_mix, speech_mix_lengths, rir_ref, rir_ref_lengths=None, **kwargs):
        speech = self._to_mono(speech_mix)
        speech = speech[:, :int(speech_mix_lengths.max())]
        if self.normalize_by_mix:
            speech = speech / speech.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        spec = self.transforms.stft(speech[:, None], 'complex')
        _, features, _ = self.rec_rir(self.transforms.preprocess(spec))
        ctf = self.transforms.postprocess(features).to(torch.complex64)
        target_rir = self._to_mono(rir_ref)
        if rir_ref_lengths is not None and not torch.all(rir_ref_lengths == target_rir.shape[-1]):
            raise ValueError("RIR supervision requires fixed-length prepared references")
        excitation = torch.complex(self.sweep_real, self.sweep_imag)[None, None]
        prediction = self._complex_convolve(excitation.expand(ctf.shape[0], -1, -1, -1), ctf)
        loss_sweep = speech.new_zeros(())
        loss_rir = speech.new_zeros(())
        if self.sweep_loss_weight:
            reference = self.reference_spectrum(target_rir)
            loss_sweep = self._complex_loss(prediction[..., :reference.shape[-1]], reference)
        if self.rir_loss_weight:
            predicted_rir = self.response_to_rir(prediction, target_rir.shape[-1])[:, 0]
            loss_rir = (predicted_rir - target_rir).abs().mean()
        loss = self.sweep_loss_weight * loss_sweep + self.rir_loss_weight * loss_rir
        stats = {'loss': loss.detach(), 'loss_sweep': loss_sweep.detach(),
                 'loss_rir_l1': loss_rir.detach()}
        return force_gatherable((loss, stats, speech.shape[0]), loss.device)

    def collect_feats(self, speech_mix, speech_mix_lengths, **kwargs):
        return {'speech_mix': speech_mix, 'speech_mix_lengths': speech_mix_lengths}
