"""TF-Locoformer with alternating shared and split feature processing."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator


class FeatureSplit(nn.Module):
    """Make two gated feature streams and fold their axis into the batch.

    Pointwise SwiGLU projections are inspired by the gated speaker split in
    TISDiSS and SepReformer. These streams are latent branches; no intermediate
    speaker supervision is imposed. Input/output layouts are [B, D, T, F] and
    [2B, D, T, F], with the two branches of each example adjacent.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.input_projection = nn.Conv2d(emb_dim, 4 * emb_dim, 1)
        self.output_projection = nn.Conv2d(2 * emb_dim, 2 * emb_dim, 1)

    def forward(self, features):
        batch, channels, frames, freqs = features.shape
        value, gate = self.input_projection(features).chunk(2, dim=1)
        branches = self.output_projection(value * F.silu(gate))
        return branches.reshape(batch * 2, channels, frames, freqs)


class FeatureFusion(nn.Module):
    """Concatenate each example's two branches, project, and add its shortcut."""

    def __init__(self, emb_dim):
        super().__init__()
        self.projection = nn.Conv2d(2 * emb_dim, emb_dim, 1)

    def forward(self, branches, shortcut):
        batch, channels, frames, freqs = shortcut.shape
        expected = (batch * 2, channels, frames, freqs)
        if tuple(branches.shape) != expected:
            raise ValueError(
                f"Expected split features {expected}, got {branches.shape}"
            )
        paired = branches.reshape(batch, channels * 2, frames, freqs)
        return shortcut + self.projection(paired)


class TFLocoformerSplitSeparator(TFLocoformerSeparator):
    """Four original TF-Locoformer blocks with two Split/Fusion stages.

    Conv -> block 1 -> Split -> block 2 -> Fusion -> block 3 -> Split
    -> block 4 -> Fusion -> the original joint two-speaker output head.
    Blocks 2 and 4 share weights across branches, while all four blocks and
    both Split/Fusion pairs have separate parameters. The original no-cache
    RoPE, normalization, attention, and FFNs are reused without modification.
    """

    def __init__(self, input_dim, n_layers: int = 4, **kwargs):
        if n_layers != 4:
            raise ValueError("TF-Locoformer-Split-S requires exactly four blocks")
        super().__init__(input_dim, n_layers=n_layers, **kwargs)
        emb_dim = self.deconv.in_channels
        self.splits = nn.ModuleList([FeatureSplit(emb_dim) for _ in range(2)])
        self.fusions = nn.ModuleList([FeatureFusion(emb_dim) for _ in range(2)])

    def forward(self, input, ilens, additional=None):
        if input.ndim == 3:
            batch0 = input.unsqueeze(1)
        elif input.ndim == 4 and input.shape[2] == 1:
            batch0 = input.transpose(1, 2)
        else:
            raise ValueError("Expected mono complex STFT [B,T,F] or [B,T,1,F]")

        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        n_batch, _, n_frames, n_freqs = batch.shape
        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch)

        for stage in range(2):
            shortcut = self.blocks[2 * stage](batch)
            branches = self.splits[stage](shortcut)
            branches = self.blocks[2 * stage + 1](branches)
            batch = self.fusions[stage](branches, shortcut)

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.deconv(batch)
        batch = batch.reshape(n_batch, self.num_spk, 2, n_frames, n_freqs)
        speech = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        return (
            [speech[:, speaker] for speaker in range(self.num_spk)],
            ilens,
            OrderedDict(),
        )
