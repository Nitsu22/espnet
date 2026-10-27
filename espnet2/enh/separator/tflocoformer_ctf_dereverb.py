"""Three TF-Locoformer blocks, CTF inversion, and one shared restoration block."""

import copy
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.enh.layers.ctf_deconvolution import regularized_deconvolution
from espnet2.enh.layers.speaker_ctf_estimator import SpeakerCTFEstimator
from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator


def decode_ri(decoder, features, speakers):
    """[speaker, real/imag] channels -> complex [B,N,F,T]."""
    batch, _, frames, freqs = features.shape
    ri = decoder(features.float()).reshape(batch, speakers, 2, frames, freqs)
    return torch.complex(ri[:, :, 0], ri[:, :, 1]).transpose(-1, -2)


class TFLocoformerCTFDereverbBaselineSeparator(TFLocoformerSeparator):
    """Original parameterization, with the same valid-length handling as A0/A1."""

    def __init__(self, input_dim, **kwargs):
        defaults = dict(
            num_spk=2, n_layers=4, emb_dim=96, norm_type="rmsgroupnorm",
            num_groups=4, tf_order="ft", n_heads=4,
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=[128, 128], conv1d_kernel=8,
        )
        defaults.update(kwargs)
        super().__init__(input_dim, **defaults)
        if self.num_spk != 2:
            raise ValueError("This initial CTF experiment requires two speakers")

    def _single(self, spectrum, observation_mask=None):
        features = self.conv(torch.stack((spectrum.real, spectrum.imag), 1).float())
        for block in self.blocks:
            features = block(features)
        clean = decode_ri(self.deconv, features, self.num_spk)
        return dict(clean_final=clean)

    def separate_all(self, spectrum, frame_lengths, observation_mask=None):
        if spectrum.ndim == 4 and spectrum.shape[2] == 1:
            spectrum = spectrum[:, :, 0]
        if spectrum.ndim != 3 or not torch.is_complex(spectrum):
            raise ValueError("Expected native complex mono STFT [B,T,F]")
        if frame_lengths.shape != (spectrum.shape[0],):
            raise ValueError("Expected one frame length per utterance")
        if frame_lengths.is_floating_point() or frame_lengths.dtype == torch.bool:
            raise ValueError("Frame lengths must be integers")
        if (frame_lengths <= 0).any() or (frame_lengths > spectrum.shape[1]).any():
            raise ValueError("Invalid frame lengths")
        results = {}
        # Avoid padding in gLN, both dual paths and bidirectional recurrence.
        with torch.cuda.amp.autocast(enabled=False):
            for index, length in enumerate(frame_lengths.tolist()):
                mask = (
                    None if observation_mask is None
                    else observation_mask[index:index+1, ..., :length]
                )
                result = self._single(spectrum[index:index+1, :length].cfloat(), mask)
                for name, value in result.items():
                    if name != "ctf":
                        value = F.pad(value, (0, spectrum.shape[1] - length))
                    results.setdefault(name, []).append(value)
        return {name: torch.cat(values) for name, values in results.items()}

    def forward(self, input, ilens, additional=None):
        result = self.separate_all(input, ilens)
        estimates = result["clean_final"].transpose(-1, -2).unbind(1)
        return list(estimates), ilens, OrderedDict(result)


class TFLocoformerCTFDereverbSeparator(TFLocoformerCTFDereverbBaselineSeparator):
    """Stage 1 predicts Reverb/CTF; Stage 2 refines their CG reconstruction.

    Stages are architectural parts, jointly trained from scratch. There is no
    Clean decoder before CG and no S0. The single restoration block/decoder is
    shared by speakers and predicts a residual on the CG output.
    """

    def __init__(
        self, input_dim, ctf_taps=128, ctf_hidden_dim=64,
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_backend="auto", use_deconvolution=True,
        cg_iterations=8, cg_gamma=0.1, cg_eps=1e-6, **kwargs,
    ):
        kwargs.setdefault("n_layers", 3)
        super().__init__(input_dim, **kwargs)
        if not self.blocks:
            raise ValueError("The front end requires at least one TF-Locoformer block")
        if (not isinstance(cg_iterations, int) or isinstance(cg_iterations, bool)
                or cg_iterations < 1 or not math.isfinite(cg_gamma)
                or not math.isfinite(cg_eps) or cg_gamma < 0 or cg_eps <= 0):
            raise ValueError("Use positive iterations/epsilon and nonnegative gamma")
        self.ctf_taps = ctf_taps
        self.use_deconvolution = use_deconvolution
        self.cg_iterations = cg_iterations
        self.cg_gamma = cg_gamma
        self.cg_eps = cg_eps
        # Reuse the ordinary output head exclusively for Reverb. A Clean head
        # is present only in the restoration stage below.
        feature_dim = self.deconv.in_channels
        self.reverb_decoder = self.deconv
        del self.deconv
        self.ctf_estimator = SpeakerCTFEstimator(
            feature_dim=feature_dim, hidden_dim=ctf_hidden_dim,
            ctf_taps=ctf_taps, d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand=mamba_expand, backend=mamba_backend,
        )
        self.restoration_projection = nn.Conv2d(feature_dim + 2, feature_dim, 1)
        # Copy the configuration, with independent parameters. ESPnet's global
        # initialization subsequently initializes both blocks independently.
        self.restoration_block = copy.deepcopy(self.blocks[-1])
        self.restoration_decoder = nn.ConvTranspose2d(
            feature_dim, 2, 3, stride=1, padding=1
        )
        self.espnet_initialization_fn()

    def espnet_initialization_fn(self):
        # Initially Stage 2 preserves the CG output. Restore this after ESPnet's
        # global Xavier initialization as well as direct module construction.
        nn.init.zeros_(self.restoration_decoder.weight)
        nn.init.zeros_(self.restoration_decoder.bias)

    def restore(self, features, deconvolved):
        """One utterance: shared [1,D,T,F] features + complex [1,N,F,T]."""
        speakers = deconvolved.shape[1]
        shared = features.expand(speakers, -1, -1, -1)
        ri = torch.stack(
            (deconvolved[0].real, deconvolved[0].imag), dim=1
        ).transpose(-1, -2)
        x = self.restoration_projection(torch.cat((shared, ri), dim=1))
        x = self.restoration_block(x)
        residual = decode_ri(self.restoration_decoder, x, 1)[:, 0].unsqueeze(0)
        return deconvolved + residual

    def _single(self, spectrum, observation_mask=None):
        features = self.conv(torch.stack((spectrum.real, spectrum.imag), 1).float())
        for block in self.blocks:
            features = block(features)
        reverb = decode_ri(self.reverb_decoder, features, self.num_spk)
        lengths = torch.tensor([spectrum.shape[1]], device=spectrum.device)
        ctf = self.ctf_estimator(features, reverb, lengths)
        if self.use_deconvolution:
            clean = regularized_deconvolution(
                reverb, ctf, frame_lengths=lengths,
                observation_mask=observation_mask, gamma=self.cg_gamma,
                eps=self.cg_eps, iterations=self.cg_iterations,
            )
        else:
            clean = reverb
        final = self.restore(features, clean)
        return dict(
            reverb=reverb, ctf=ctf, clean_deconv=clean, clean_final=final,
        )
