"""Early speaker splitting and shared BiLSTM refinement for CTF estimation.

Inspired by SR-CorrNet's fixed-speaker split and cross-speaker refinement.
This model keeps the existing CTF input, output, and unmasked padding behavior.
"""

from collections import OrderedDict

import torch
from torch import nn

from espnet2.rir.rec_rir.pooled_bimamba import (
    PooledBiMambaCTFPredictor,
    PostPoolingAttention,
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class FixedSpeakerSplit(nn.Module):
    """Expand each TF feature into separate, full-width speaker streams."""

    def __init__(self, dim, num_spk, hidden_dim, eps):
        super().__init__()
        for name, value in (("dim", dim), ("num_spk", num_spk),
                            ("hidden_dim", hidden_dim)):
            _positive_integer(value, name)
        self.dim = dim
        self.num_spk = num_spk
        self.projection = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_spk * dim),
        )
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError("Expected encoded features [B, T, F, dim]")
        batch, frames, freqs, _ = x.shape
        streams = self.projection(x).reshape(
            batch, frames, freqs, self.num_spk, self.dim)
        return self.norm(streams.permute(0, 3, 1, 2, 4))


class SpeakerInteraction(nn.Module):
    """Residual speaker self-attention and FFN at each individual TF bin.

    No speaker positional encoding or speaker-specific projections are used.
    Exchanging the input streams therefore exchanges the output streams.
    """

    def __init__(self, dim, heads, ffn_dim, eps):
        super().__init__()
        _positive_integer(ffn_dim, "speaker_ffn_dim")
        self.dim = dim
        # Conservatively bound both the batch and batch*head grid sizes used
        # by CUDA SDPA kernels. TF positions are independent, so chunking this
        # batch axis preserves the attention operation and its gradients.
        self._max_attention_batch_heads = 65535
        self.attention = PostPoolingAttention(dim, heads, eps, axis="speaker")
        self.norm_ffn = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        if x.ndim != 5 or x.shape[-1] != self.dim:
            raise ValueError("Expected speaker features [B, S, T, F, dim]")
        batch, speakers, frames, freqs, dim = x.shape
        # Attend only to S. The singleton F axis lets each chunk contain an
        # arbitrary number of independent TF positions without a large CUDA
        # batch (e.g. B=4,T=251,F=129 would otherwise make 129516 sequences).
        tf_first = x.permute(0, 2, 3, 1, 4).reshape(
            batch * frames * freqs, speakers, 1, dim)
        chunk_size = max(1, self._max_attention_batch_heads // self.attention.heads)
        attended = torch.cat([
            self.attention(chunk) for chunk in tf_first.split(chunk_size, dim=0)
        ], dim=0)
        output = attended.reshape(
            batch, frames, freqs, speakers, dim).permute(0, 3, 1, 2, 4)
        return output + self.ffn(self.norm_ffn(output))


class SplitBiLSTMCTFPredictor(PooledBiMambaCTFPredictor):
    """Shared TF encoders -> speaker split -> shared TF decoders -> CTFs.

    Common layers are initialized through the baseline constructor before new
    modules are added. The obsolete late-split pooling and slot embeddings
    are replaced/removed, so no unused trainable parameters remain.
    """

    def __init__(self, split_after=4, pre_layers=8, post_layers=4,
                 split_hidden=None, speaker_ffn_dim=None, **kwargs):
        _positive_integer(pre_layers, "pre_layers")
        _positive_integer(post_layers, "post_layers")
        _positive_integer(split_after, "split_after")
        if split_after >= pre_layers:
            raise ValueError("split_after must leave at least one decoder block")
        required = dict(
            time_module="bilstm", pre_frequency_module="light",
            post_frequency_module="bilstm", post_interaction="none",
            frequency_refinement_position="after_pool", pooling_mode="legacy",
            pooling_channelwise=True, pre_frequency_global=True,
            post_attention=True,
        )
        for name, value in required.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"split_bilstm requires {name}={value!r}")
            kwargs[name] = value
        for name in ("time_block_indices", "frequency_block_indices",
                     "pre_frequency_modules"):
            if kwargs.get(name) is not None:
                raise ValueError(f"split_bilstm does not support {name}")
        super().__init__(pre_layers=pre_layers, post_layers=post_layers, **kwargs)
        self.split_after = split_after
        self.dim = kwargs.get("emb_dim", 64)
        heads = kwargs.get("n_heads", 4)
        eps = kwargs.get("eps", 1e-5)
        split_hidden = (4 * self.num_spk * self.dim if split_hidden is None
                        else split_hidden)
        speaker_ffn_dim = (4 * self.dim if speaker_ffn_dim is None
                           else speaker_ffn_dim)
        _positive_integer(split_hidden, "split_hidden")
        _positive_integer(speaker_ffn_dim, "speaker_ffn_dim")
        # Isolate added parameters from all common baseline initialization.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(
                (torch.initial_seed() + 700001) % (2**63 - 1))
            self.splitter = FixedSpeakerSplit(
                self.dim, self.num_spk, split_hidden, eps)
            self.speaker_interactions = nn.ModuleList(
                SpeakerInteraction(self.dim, heads, speaker_ffn_dim, eps)
                for _ in range(pre_layers - split_after))
            # One score per feature channel, shared between speaker branches.
            self.pooling[-1] = nn.Linear(self.pooling[-1].in_features, self.dim)
        del self.slot_embeddings

    def decode_streams(self, x):
        """Refine already split [B,S,T,F,D] streams and return [B,S,F,K]."""
        if (x.ndim != 5 or x.shape[1] != self.num_spk
                or x.shape[3] != self.input_dim or x.shape[-1] != self.dim):
            raise ValueError("Expected split features [B, num_spk, T, input_dim, dim]")
        batch, speakers, frames, freqs, dim = x.shape
        for index, interaction in enumerate(self.speaker_interactions,
                                             start=self.split_after):
            sequence = x.reshape(batch * speakers, frames, freqs, dim)
            sequence = self.frequency_blocks[index](self.time_blocks[index](sequence))
            x = interaction(sequence.reshape(batch, speakers, frames, freqs, dim))
        # Preserve the baseline's unmasked time axis, but never split S again.
        weights = self.pooling(x).softmax(dim=2)
        pooled = (weights * x).sum(dim=2).reshape(batch * speakers, freqs, dim)
        for block in self.post_blocks:
            pooled = block(pooled)
        output = self.head(pooled).reshape(
            batch, speakers, freqs, self.ctf_taps, 2).float()
        return torch.complex(output[..., 0], output[..., 1]).contiguous()

    def forward(self, input, ilens=None, additional=None):
        if (input.ndim != 3 or not torch.is_complex(input)
                or input.shape[-1] != self.input_dim):
            raise ValueError("Expected complex STFT [B, T, input_dim]")
        x = self.encoder(torch.stack((input.real, input.imag), 1).float())
        x = x.permute(0, 2, 3, 1)
        for index in range(self.split_after):
            x = self.frequency_blocks[index](self.time_blocks[index](x))
        ctf = self.decode_streams(self.splitter(x))
        return ctf, ilens, OrderedDict()
