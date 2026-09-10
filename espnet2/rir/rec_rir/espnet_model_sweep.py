"""Rec-RIR feature network with a CTF-only output and sweep supervision."""
import torch
from espnet2.rir.rec_rir.model import BiSpatialNet
from espnet2.rir.rec_rir.espnet_model import ESPnetRecRIRModel
from espnet2.torch_utils.device_funcs import force_gatherable


class CTFOnlyBiSpatialNet(BiSpatialNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.decoder_spch
        del self.decoder_rev

    def forward(self, input, return_embedding=False):
        x = torch.nn.functional.pad(input, (
            self.padding_size[1], self.padding_size[1],
            self.padding_size[0], self.padding_size[0]))
        x = self.encoder(x).permute(0, 2, 3, 1)
        for layer in self.noise_layers:
            x = layer(x)
        reverb_features = x
        for layer in self.spch_layers:
            x = layer(x)
        x = self.compress_CTF(x, reverb_features)
        for layer in self.ctf_layers:
            x = layer(x)
        x = (x * self.weight_layer(x)).sum(-2).unsqueeze(2)
        if return_embedding:
            return x
        batch, freq, _, _ = x.shape
        ctf = self.decoder_CTF(x).reshape(batch, freq, 2, -1).permute(0, 2, 1, 3)
        # Preserve the CTF position expected by the inherited inference method.
        return None, ctf, None


class ESPnetRecRIRSweepModel(ESPnetRecRIRModel):
    rec_rir_network_class = CTFOnlyBiSpatialNet

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sweep = self.pim.sinesweep.clone()
        self.register_buffer('sweep', sweep, persistent=False)
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

    def forward(self, speech_mix, speech_mix_lengths, rir_ref, rir_ref_lengths=None, **kwargs):
        speech = self._to_mono(speech_mix)
        speech = speech[:, :int(speech_mix_lengths.max())]
        if self.normalize_by_mix:
            speech = speech / speech.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        spec = self.transforms.stft(speech[:, None], 'complex')
        _, features, _ = self.rec_rir(self.transforms.preprocess(spec))
        ctf = self.transforms.postprocess(features).to(torch.complex64)
        reference = self.reference_spectrum(self._to_mono(rir_ref))
        excitation = torch.complex(self.sweep_real, self.sweep_imag)[None, None]
        prediction = self._complex_convolve(excitation.expand(ctf.shape[0], -1, -1, -1), ctf)
        prediction = prediction[..., :reference.shape[-1]]
        loss = self._complex_loss(prediction, reference)
        return force_gatherable((loss, {'loss': loss.detach(), 'loss_sweep': loss.detach()}, speech.shape[0]), loss.device)

    def collect_feats(self, speech_mix, speech_mix_lengths, **kwargs):
        return {'speech_mix': speech_mix, 'speech_mix_lengths': speech_mix_lengths}
