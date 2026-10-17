"""Rec-RIR CTF-only trunk with the shared full-response Sweep v2 PIT loss."""

from collections import OrderedDict

import torch
from torch import nn

from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)


class CTFOnlyBiSpatialNetPIT(BiSpatialNetPIT):
    """Keep the existing two-source Rec-RIR CTF path; remove speech decoders."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.decoder_spch
        del self.decoder_rev

    def forward(self, observation):
        x = torch.nn.functional.pad(observation, (
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
        batch, freqs, _, _ = x.shape
        ctf = self.decoder_CTF(x).reshape(
            batch, freqs, self.num_spk, 2, self.ctf_taps)
        return ctf.permute(0, 2, 3, 1, 4).contiguous()


class RecRIRCTFPredictor(nn.Module):
    """Adapt complex [B,T,F] input to Rec-RIR and return causal [B,S,F,L]."""

    def __init__(self, num_freqs, num_spk, ctf_taps, network_conf=None):
        super().__init__()
        conf = dict(network_conf or {})
        reserved = {
            'dim_input', 'dim_output_spch', 'dim_output_CTF', 'num_freqs', 'num_spk'
        } & conf.keys()
        if reserved:
            raise ValueError(f'Specify {sorted(reserved)} using model dimensions')
        defaults = dict(
            dim_hidden=96, dim_squeeze=8, num_layers_spch=6,
            num_layers_noise=2, num_layers_CTF=6, encoder_kernel_size=5,
            dropout=(0.0, 0.0, 0.0), kernel_size=(5, 3), conv_groups=(8, 8),
            norms=('LN',) * 6, full_share=0, attention='mamba(16,4)')
        defaults.update(conf)
        self.num_freqs = num_freqs
        self.network = CTFOnlyBiSpatialNetPIT(
            dim_input=2, dim_output_spch=2 * num_spk,
            dim_output_CTF=2 * num_spk * ctf_taps,
            num_freqs=num_freqs, num_spk=num_spk, **defaults)

    def forward(self, observation):
        if (observation.ndim != 3 or not observation.is_complex()
                or observation.shape[-1] != self.num_freqs):
            raise ValueError('Expected complex observation [B,T,F]')
        # Exactly the RI layout used by RecRIRTransforms.preprocess().
        ri = torch.stack((observation.real, observation.imag), 1)
        ri = ri.transpose(-1, -2).contiguous().float()
        coefficients = self.network(ri)
        ctf = torch.complex(coefficients[:, :, 0], coefficients[:, :, 1])
        # Shared wrapper handles causal convolution, PIT, and the PIM tap flip.
        return ctf.contiguous(), None, OrderedDict()


class ESPnetRecRIRSweepV2PITModel(ESPnetTFLocoformerSweepV2PITModel):
    """Change only the predictor; reuse Baseline's teacher/loss/inference code."""

    def __init__(self, num_freqs=129, num_spk=2, ctf_taps=60,
                 network_conf=None, **kwargs):
        predictor = RecRIRCTFPredictor(
            num_freqs, num_spk, ctf_taps, network_conf=network_conf)
        super().__init__(num_freqs=num_freqs, num_spk=num_spk,
                         ctf_taps=ctf_taps, ctf_predictor=predictor, **kwargs)
