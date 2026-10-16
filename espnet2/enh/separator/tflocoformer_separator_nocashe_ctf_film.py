"""Oracle CTF cross-attention and identity-initialized all-block FiLM."""
import torch
from torch import nn

from espnet2.enh.separator.tflocoformer_separator_nocashe_ctf_concat import (
    TFLocoformerSeparator as _CTFSeparator,
)
from espnet2.enh.separator.tflocoformer_separator_nocashe_rir_cmha_film_all import (
    RIRCMHAFiLMBlock,
)


class TFLocoformerSeparator(_CTFSeparator):
    """Use the existing frequency-wise Cross-MHA/FiLM with Oracle CTFs.

    Inherit the validated CTF axes, speaker-order augmentation and per-example
    handling of unequal speech lengths. Speech parameters keep the baseline's
    names, registration order and seeded initialization. Only conditioning
    modules differ from the temporal-prefix model.
    """

    def __init__(self, input_dim, emb_dim=128, n_heads=4, dropout=0.0, eps=1e-5, **kwargs):
        super().__init__(input_dim, emb_dim=emb_dim, n_heads=n_heads,
                         dropout=dropout, eps=eps, **kwargs)
        # Replace the prefix projection with the existing FiLM CTF encoder.
        # Keeping all speech modules registered first preserves Xavier draws.
        with torch.random.fork_rng(devices=[]):
            self.ctf_projection = nn.Sequential(
                nn.Conv2d(self.num_spk * 2, emb_dim, (3, 3), padding=(1, 1)),
                nn.GroupNorm(1, emb_dim, eps=eps),
            )
            self.rir_films = nn.ModuleList([
                RIRCMHAFiLMBlock(emb_dim, n_heads=n_heads, dropout=dropout)
                for _ in self.blocks
            ])

    def _separate(self, speech, ctf):
        batch_size, _, frames, frequencies = speech.shape
        # Match the previous Rec-RIR FiLM encoder's channel order:
        # [speaker1.real, speaker2.real, speaker1.imag, speaker2.imag].
        ctf_channels = torch.cat(
            (ctf[..., 0].transpose(1, 2), ctf[..., 1].transpose(1, 2)), dim=1
        )
        with torch.cuda.amp.autocast(enabled=False):
            features = self.conv(speech)
            context = self.ctf_projection(ctf_channels)
        for film, block in zip(self.rir_films, self.blocks):
            with torch.cuda.amp.autocast(enabled=False):
                features = film(features, context)
            features = block(features)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.deconv(features)
        return output.reshape(batch_size, self.num_spk, 2, frames, frequencies)
