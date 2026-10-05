"""TF-Locoformer-S with speech and utterance-level complex CTF outputs."""

from collections import OrderedDict

import torch
from torch import nn

from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator


class TFLocoformerCTFSeparator(TFLocoformerSeparator):
    """Add a CTF head after the last block, preserving the speech branch.

    CTF taps are in causal order: tap zero multiplies the current STFT frame.
    The output in ``others['ctf']`` is complex [B, speaker, frequency, tap].
    """

    def __init__(self, input_dim, ctf_taps: int = 120, **kwargs):
        super().__init__(input_dim, **kwargs)
        if ctf_taps < 1:
            raise ValueError("ctf_taps must be positive")
        self.ctf_taps = int(ctf_taps)
        emb_dim = self.deconv.in_channels
        self.ctf_weight = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
        )
        self.ctf_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, self.num_spk * 2 * self.ctf_taps),
        )

    def estimate_ctf(self, features, ilens):
        # [B,C,T,F] -> [B,F,T,C]; exclude padding from temporal softmax.
        x = features.permute(0, 3, 2, 1).float()
        logits = self.ctf_weight(x)
        if ilens is not None:
            if (ilens <= 0).any() or (ilens > x.shape[2]).any():
                raise ValueError("Invalid STFT lengths for CTF pooling")
            valid = torch.arange(x.shape[2], device=x.device)[None] < ilens[:, None]
            logits = logits.masked_fill(~valid[:, None, :, None], -torch.inf)
        pooled = (x * logits.softmax(dim=2)).sum(dim=2)
        b, f, _ = pooled.shape
        ri = self.ctf_head(pooled).reshape(b, f, self.num_spk, 2, self.ctf_taps)
        ri = ri.permute(0, 2, 3, 1, 4).contiguous()
        return torch.complex(ri[:, :, 0], ri[:, :, 1])

    def forward(self, input, ilens, additional=None):
        if input.ndim == 3:
            batch0 = input.unsqueeze(1)
        elif input.ndim == 4 and input.shape[2] == 1:
            batch0 = input.transpose(1, 2)
        else:
            raise ValueError("Expected mono complex STFT [B,T,F] or [B,T,1,F]")
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        b, _, t, f = batch.shape
        with torch.cuda.amp.autocast(enabled=False):
            features = self.conv(batch.float())
        for block in self.blocks:
            features = block(features)
        with torch.cuda.amp.autocast(enabled=False):
            ctf = self.estimate_ctf(features, ilens)
            speech = self.deconv(features.float()).reshape(b, self.num_spk, 2, t, f)
        speech = new_complex_like(batch0, (speech[:, :, 0], speech[:, :, 1]))
        return list(speech.unbind(1)), ilens, OrderedDict(ctf=ctf)
