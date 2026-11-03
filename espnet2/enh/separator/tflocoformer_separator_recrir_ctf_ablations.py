"""Independent CTF conditioning ablations with identical common initialization."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.enh.separator.tflocoformer_separator_nocashe_rir_cmha_film_all_ctf import (
    TFLocoformerSeparator as _CTFFiLMSeparator,
)


class TFLocoformerSeparator(_CTFFiLMSeparator):
    """Change exactly one component of the existing frequency-wise MHCA-FiLM.

    lag_position: fixed sinusoidal lag positions, added after CTF normalization.
    joint_mlp: gamma/beta heads receive SiLU(W[X; context] + b).
    multiscale: the existing CTF convolution's channel halves use lag dilations
        one and eight. Weights, channel count and parameter count are unchanged.
    """

    def __init__(self, input_dim, conditioning_variant="reference", **kwargs):
        variants = ("reference", "lag_position", "joint_mlp", "multiscale")
        if conditioning_variant not in variants:
            raise ValueError(f"conditioning_variant must be one of {variants}")
        super().__init__(input_dim=input_dim, **kwargs)
        self.conditioning_variant = conditioning_variant
        channels = self.rir_conv[0].out_channels
        if channels % 2:
            raise ValueError("CTF conditioning ablations require even emb_dim")
        # Append extra parameters after ALL existing separator parameters. Keep
        # construction RNG unchanged, so common Xavier draws also remain equal.
        if conditioning_variant == "joint_mlp":
            with torch.random.fork_rng(devices=[]):
                self.joint_fusions = nn.ModuleList(
                    nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.SiLU())
                    for _ in range(self.n_layers)
                )

    @staticmethod
    def lag_positions(features):
        """[1,D,L,1]; tap l means l*16 ms for the frozen native-8k Rec-RIR.

        p[l,2i]=sin(l/10000**(2i/D)), p[l,2i+1]=cos(...).
        These are lag coordinates, never audio-frame t-l relative positions.
        """
        channels, taps = features.shape[1:3]
        position = torch.arange(taps, device=features.device, dtype=torch.float32)
        scales = torch.exp(
            torch.arange(0, channels, 2, device=features.device, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )
        angles = position[:, None] * scales[None, :]
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1)
        return encoding.reshape(taps, channels).T[None, :, :, None].to(features.dtype)

    def _encode_rir(self, rir_feat):
        if self.conditioning_variant == "multiscale":
            conv, norm = self.rir_conv
            split = conv.out_channels // 2
            short = F.conv2d(
                rir_feat, conv.weight[:split], conv.bias[:split], padding=(1, 1)
            )
            long = F.conv2d(
                rir_feat, conv.weight[split:], conv.bias[split:],
                padding=(8, 1), dilation=(8, 1),
            )
            # One shared GroupNorm, exactly as in the reference encoder.
            return norm(torch.cat((short, long), dim=1))
        features = super()._encode_rir(rir_feat)
        if self.conditioning_variant == "lag_position":
            features = features + self.lag_positions(features)
        return features

    def _condition_speech(self, index, speech, rir):
        transform = (
            self.joint_fusions[index]
            if self.conditioning_variant == "joint_mlp" else None
        )
        return self.rir_films[index](speech, rir, context_transform=transform)
