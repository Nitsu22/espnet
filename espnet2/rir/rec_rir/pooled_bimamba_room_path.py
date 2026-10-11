"""Shared room features and two path features for a shared CTF decoder.

The three frequency-resolved codes are learned from the same input. Their
names specify the information-sharing structure; no supervision forces the
room code to contain only physical room properties. Padding behavior and CTF
tap ordering match the pooled BiMamba Sweep v2 baseline.
"""
from collections import OrderedDict

import torch
from torch import nn

from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor


class RoomPathPooledBiMambaCTFPredictor(PooledBiMambaCTFPredictor):
    """Estimate CTFs with ``Decoder(room, path_s)`` for each output slot.

    The shared room code has shape ``[B, F, D]``. Each path code has the same
    shape. The decoder processes each room/path pair independently with shared
    weights, so one path code is never an input to another path's decoder.
    """

    def __init__(self, *args, **kwargs):
        # Initialize every common baseline parameter first, preserving their
        # seeded values and checkpoint keys before initializing new modules.
        super().__init__(*args, **kwargs)
        if self.frequency_refinement_position != "after_pool":
            raise ValueError("Room/path decoding requires frequency_refinement_position='after_pool'")
        if self.post_interaction != "none":
            raise ValueError("Independent room/path decoding requires post_interaction='none'")
        self.emb_dim = self.encoder.out_channels
        pooling_hidden = self.pooling[0].out_features
        eps = self.head[0].eps
        # All three score heads use the existing shared pooling hidden layer.
        self.room_pooling = nn.Linear(pooling_hidden, self.emb_dim)
        self.room_path_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.emb_dim, eps=eps),
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.SiLU(),
        )

    def encode_room_path(self, input, ilens=None):
        """Return one shared room code and two separate path-slot codes."""
        if (not isinstance(input, torch.Tensor) or input.ndim != 3
                or not torch.is_complex(input) or input.shape[-1] != self.input_dim):
            raise ValueError("Expected complex STFT [B, T, input_dim]")
        if input.shape[0] == 0 or input.shape[1] == 0:
            raise ValueError("Complex STFT must have nonempty batch and time axes")
        x = self.encoder(torch.stack((input.real, input.imag), 1).float()).permute(0, 2, 3, 1)
        for time_block, frequency_block in zip(self.time_blocks, self.frequency_blocks):
            x = frequency_block(time_block(x))
        batch, frames, freqs, dim = x.shape
        hidden = self.pooling[1](self.pooling[0](x))
        path_scores = self.pooling[2](hidden).reshape(
            batch, frames, freqs, self.num_spk, self.pooling_channels)
        # Keep the baseline's full-time-axis softmax, including batch padding.
        paths = (path_scores.softmax(dim=1) * x.unsqueeze(3)).sum(1)
        paths = paths.permute(0, 2, 1, 3) + self.slot_embeddings[None, :, None, :]
        room_scores = self.room_pooling(hidden)
        room = (room_scores.softmax(dim=1) * x).sum(1)
        return OrderedDict(room=room, path1=paths[:, 0], path2=paths[:, 1])

    def decode_room_path(self, room, path):
        """Decode one path ``[B,F,D]`` or multiple paths ``[B,S,F,D]``.

        The output is complex ``[B,F,L]`` or ``[B,S,F,L]`` respectively, with
        the same frequency/tap ordering as the baseline predictor.
        """
        if (not isinstance(room, torch.Tensor) or room.ndim != 3
                or room.shape[1:] != (self.input_dim, self.emb_dim)
                or room.shape[0] == 0 or torch.is_complex(room)):
            raise ValueError("Expected real room code [B, input_dim, emb_dim]")
        if (not isinstance(path, torch.Tensor) or path.ndim not in (3, 4)
                or path.shape[0] != room.shape[0]
                or path.shape[-2:] != (self.input_dim, self.emb_dim)
                or torch.is_complex(path)):
            raise ValueError("Expected real path code [B, F, D] or [B, S, F, D] matching room")
        if path.device != room.device or path.dtype != room.dtype:
            raise ValueError("Room and path codes must use the same device and dtype")
        single_path = path.ndim == 3
        paths = path.unsqueeze(1) if single_path else path
        if paths.shape[1] == 0:
            raise ValueError("Path code must have a nonempty slot axis")
        batch, slots, freqs, dim = paths.shape
        rooms = room.unsqueeze(1).expand(batch, slots, freqs, dim)
        features = self.room_path_fusion(torch.cat((rooms, paths), dim=-1))
        features = features.reshape(batch * slots, freqs, dim)
        for block in self.post_blocks:
            features = block(features)
        output = self.head(features).reshape(batch, slots, freqs, self.ctf_taps, 2).float()
        ctf = torch.complex(output[..., 0], output[..., 1]).contiguous()
        return ctf[:, 0] if single_path else ctf

    def forward(self, input, ilens=None, additional=None):
        codes = self.encode_room_path(input, ilens)
        paths = torch.stack((codes["path1"], codes["path2"]), dim=1)
        ctf = self.decode_room_path(codes["room"], paths)
        return ctf, ilens, OrderedDict(room_code=codes["room"], path_codes=paths)
