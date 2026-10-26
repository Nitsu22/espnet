"""Speaker-conditioned, utterance-level complex CTF estimation."""

import torch
from torch import nn

from espnet2.enh.layers.bidirectional_mamba import BidirectionalMamba


class _ResidualBiMamba(nn.Module):
    """Apply a pre-normalized update to a [batch, sequence, feature] tensor."""

    def __init__(self, hidden_dim, d_state, d_conv, expand, backend, eps):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.mixer = BidirectionalMamba(
            hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=backend,
        )

    def forward(self, sequence):
        return sequence + self.mixer(self.norm(sequence))


class SpeakerCTFEstimator(nn.Module):
    """Estimate one filter per speaker from its predicted reverberant speech.

    Args:
        features: Shared backbone features, real [B, D, T, F].
        reverb: Predicted complex reverberant spectra [B, N, F, T].
        frame_lengths: Number of valid frames for each utterance [B].

    Returns:
        Complex causal filters [B, N, F, L], where tap zero is the
        current-frame multiplier. The two real output groups are real/imaginary,
        each containing all L taps.

    All speaker slots share every parameter. Each utterance is sliced before
    processing so that padded frames cannot affect either direction of Mamba,
    frequency processing, or the temporal softmax. This also makes a short
    utterance's result independent of the longest utterance in its batch.
    """

    def __init__(
        self,
        feature_dim=96,
        hidden_dim=64,
        ctf_taps=128,
        d_state=16,
        d_conv=4,
        expand=2,
        backend="auto",
        eps=1.0e-5,
    ):
        super().__init__()
        if min(feature_dim, hidden_dim, ctf_taps, d_state, d_conv, expand) <= 0:
            raise ValueError("CTF estimator dimensions must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.ctf_taps = int(ctf_taps)
        self.input_projection = nn.Conv2d(feature_dim + 2, hidden_dim, 1)
        arguments = (hidden_dim, d_state, d_conv, expand, backend, eps)
        self.time_mamba = _ResidualBiMamba(*arguments)
        self.frequency_mamba = _ResidualBiMamba(*arguments)
        self.pooling = nn.Linear(hidden_dim, 1)
        self.post_pool_frequency_mamba = _ResidualBiMamba(*arguments)
        self.output_projection = nn.Linear(hidden_dim, 2 * ctf_taps)

    def _forward_utterance(self, features, reverb):
        # Treat speaker slots as a batch, without a slot-specific embedding.
        speakers, freqs, frames = reverb.shape
        shared = features.expand(speakers, -1, -1, -1)
        ri = torch.stack((reverb.real, reverb.imag), dim=1).transpose(-1, -2)
        x = self.input_projection(torch.cat((shared, ri), dim=1).float())
        # [N,H,T,F] -> time sequences [N*F,T,H].
        x = x.permute(0, 3, 2, 1).reshape(
            speakers * freqs, frames, self.hidden_dim
        )
        x = self.time_mamba(x)
        # Each valid time frame then traverses the frequency axis.
        x = x.reshape(speakers, freqs, frames, self.hidden_dim).transpose(1, 2)
        x = self.frequency_mamba(
            x.reshape(speakers * frames, freqs, self.hidden_dim)
        ).reshape(speakers, frames, freqs, self.hidden_dim)
        # This tensor contains valid frames only; the softmax never sees padding.
        weights = self.pooling(x).softmax(dim=1)
        pooled = (weights * x).sum(dim=1)
        pooled = self.post_pool_frequency_mamba(pooled)
        ri = self.output_projection(pooled).float().reshape(
            speakers, freqs, 2, self.ctf_taps
        )
        return torch.complex(ri[:, :, 0], ri[:, :, 1])

    def forward(self, features, reverb, frame_lengths):
        if features.ndim != 4 or torch.is_complex(features):
            raise ValueError("features must be real [B,D,T,F]")
        if reverb.ndim != 4 or not torch.is_complex(reverb):
            raise ValueError("reverb must be complex [B,N,F,T]")
        batch, channels, frames, freqs = features.shape
        if (
            channels != self.feature_dim
            or min(batch, frames, freqs, reverb.shape[1]) < 1
            or reverb.shape[0] != batch
            or reverb.shape[2:] != (freqs, frames)
        ):
            raise ValueError("Backbone and reverberant-speech shapes do not match")
        if features.device != reverb.device:
            raise ValueError("features and reverb must use the same device")
        if (
            frame_lengths.ndim != 1
            or frame_lengths.shape[0] != batch
            or frame_lengths.dtype not in (torch.int32, torch.int64)
            or (frame_lengths < 1).any()
            or (frame_lengths > frames).any()
        ):
            raise ValueError("frame_lengths must contain valid positive frame counts")
        # CTFs are linear complex spectra. Keep their head in float32 even when
        # the surrounding backbone runs under mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            outputs = [
                self._forward_utterance(
                    features[b : b + 1, :, :length], reverb[b, :, :, :length]
                )
                for b, length in enumerate(frame_lengths.tolist())
            ]
        return torch.stack(outputs, dim=0)
