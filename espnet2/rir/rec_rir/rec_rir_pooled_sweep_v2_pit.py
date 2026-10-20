"""Rec-RIR features with Baseline's speaker pooling and frequency CTF head.

The six CTF-only temporal layers and all legacy output heads are removed.
Teacher construction, full-response Sweep v2 PIT, padding policy, and PIM
inference are inherited without changing their definition.
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT
from espnet2.rir.rec_rir.pooled_bimamba import FrequencyAttentionBlock
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)


class RecRIRFeatureTrunk(BiSpatialNetPIT):
    """Keep Rec-RIR's encoder, noise/speech feature layers, and feature fusion."""

    def __init__(self, **kwargs):
        if kwargs.get('num_layers_CTF') != 0:
            raise ValueError('The pooled model replaces all CTF temporal layers')
        super().__init__(**kwargs)
        for name in ('ctf_layers', 'weight_layer', 'decoder_CTF',
                     'decoder_spch', 'decoder_rev'):
            delattr(self, name)

    def forward(self, observation):
        x = F.pad(observation, (
            self.padding_size[1], self.padding_size[1],
            self.padding_size[0], self.padding_size[0]))
        x = self.encoder(x).permute(0, 2, 3, 1)
        for layer in self.noise_layers:
            x = layer(x)
        reverb_features = x
        for layer in self.spch_layers:
            x = layer(x)
        return self.compress_CTF(x, reverb_features)


class BaselinePooledCTFDecoder(nn.Module):
    """Baseline's shared pooling -> frequency Attention -> CTF head.

    Only the input projection is new: Rec-RIR supplies 96-dimensional
    [B,F,T,D] features, while the Baseline tail operates on 64 channels.
    No padding mask is added, matching the existing comparison models.
    """

    def __init__(self, input_dim=96, num_spk=2, ctf_taps=60, emb_dim=64,
                 pooling_hidden=32, post_layers=2, n_heads=4, ffn_dim=128,
                 local_kernel=5, head_dim=128, slot_embedding_std=0.02,
                 eps=1e-5):
        super().__init__()
        dimensions = (input_dim, num_spk, ctf_taps, emb_dim, pooling_hidden,
                      n_heads, ffn_dim, local_kernel, head_dim)
        if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                for v in dimensions)
                or isinstance(post_layers, bool) or not isinstance(post_layers, int)
                or post_layers < 0 or local_kernel % 2 != 1 or num_spk != 2):
            raise ValueError('Use positive dimensions, two speakers, and an odd kernel')
        if not torch.isfinite(torch.tensor([slot_embedding_std, eps])).all() or (
                slot_embedding_std < 0 or eps <= 0):
            raise ValueError('Use a finite nonnegative slot scale and positive epsilon')
        self.input_dim = input_dim
        self.num_spk = num_spk
        self.ctf_taps = ctf_taps
        self.emb_dim = emb_dim
        self.projection = nn.Linear(input_dim, emb_dim)
        self.pooling = nn.Sequential(
            nn.Linear(emb_dim, pooling_hidden), nn.GELU(),
            nn.Linear(pooling_hidden, num_spk * emb_dim))
        self.slot_embeddings = nn.Parameter(torch.empty(num_spk, emb_dim))
        nn.init.normal_(self.slot_embeddings, std=slot_embedding_std)
        self.post_blocks = nn.ModuleList(
            FrequencyAttentionBlock(emb_dim, n_heads, ffn_dim, local_kernel, eps)
            for _ in range(post_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim, eps=eps), nn.Linear(emb_dim, head_dim),
            nn.SiLU(), nn.Linear(head_dim, 2 * ctf_taps))

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != self.input_dim or x.shape[2] == 0:
            raise ValueError('Expected nonempty Rec-RIR features [B,F,T,input_dim]')
        x = self.projection(x).permute(0, 2, 1, 3)
        batch, frames, freqs, dim = x.shape
        scores = self.pooling(x).reshape(batch, frames, freqs, self.num_spk, dim)
        weights = scores.softmax(dim=1)
        pooled = (weights * x.unsqueeze(3)).sum(1).permute(0, 2, 1, 3)
        pooled = pooled + self.slot_embeddings[None, :, None, :]
        pooled = pooled.reshape(batch * self.num_spk, freqs, dim)
        for block in self.post_blocks:
            pooled = block(pooled)
        output = self.head(pooled).reshape(
            batch, self.num_spk, freqs, self.ctf_taps, 2).float()
        return torch.complex(output[..., 0], output[..., 1]).contiguous()


class RecRIRPooledCTFPredictor(nn.Module):
    """Return causal complex [B,S,F,L] CTFs from complex [B,T,F] audio STFT."""

    def __init__(self, num_freqs, num_spk, ctf_taps,
                 network_conf=None, decoder_conf=None):
        super().__init__()
        conf = dict(network_conf or {})
        reserved = {
            'dim_input', 'dim_output_spch', 'dim_output_CTF', 'num_freqs', 'num_spk'
        } & conf.keys()
        if reserved:
            raise ValueError(f'Specify {sorted(reserved)} using model dimensions')
        defaults = dict(
            dim_hidden=96, dim_squeeze=8, num_layers_spch=6,
            num_layers_noise=2, num_layers_CTF=0, encoder_kernel_size=5,
            dropout=(0.0, 0.0, 0.0), kernel_size=(5, 3), conv_groups=(8, 8),
            norms=('LN',) * 6, full_share=0, attention='mamba(16,4)')
        defaults.update(conf)
        self.num_freqs = num_freqs
        self.trunk = RecRIRFeatureTrunk(
            dim_input=2, dim_output_spch=2 * num_spk,
            dim_output_CTF=2 * num_spk * ctf_taps,
            num_freqs=num_freqs, num_spk=num_spk, **defaults)
        self.decoder = BaselinePooledCTFDecoder(
            input_dim=defaults['dim_hidden'], num_spk=num_spk,
            ctf_taps=ctf_taps, **dict(decoder_conf or {}))

    def forward(self, observation):
        if (observation.ndim != 3 or not observation.is_complex()
                or observation.shape[-1] != self.num_freqs):
            raise ValueError('Expected complex observation [B,T,F]')
        ri = torch.stack((observation.real, observation.imag), 1)
        ri = ri.transpose(-1, -2).contiguous().float()
        ctf = self.decoder(self.trunk(ri))
        return ctf, None, OrderedDict()


class ESPnetRecRIRPooledSweepV2PITModel(ESPnetTFLocoformerSweepV2PITModel):
    """Use Sweep v2 alone; change only the CTF predictor architecture."""

    def __init__(self, num_freqs=129, num_spk=2, ctf_taps=60,
                 network_conf=None, decoder_conf=None, **kwargs):
        if kwargs.get('drr_loss_weight', 0) != 0:
            raise ValueError('This experiment uses Sweep v2 alone')
        predictor = RecRIRPooledCTFPredictor(
            num_freqs, num_spk, ctf_taps, network_conf, decoder_conf)
        super().__init__(num_freqs=num_freqs, num_spk=num_spk,
                         ctf_taps=ctf_taps, ctf_predictor=predictor, **kwargs)
