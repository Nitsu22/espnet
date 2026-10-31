"""TF-Locoformer conditioned by a temporal prefix of Direct-to-Reverb CTFs."""

from collections import OrderedDict
import math
from numbers import Real
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.separator.tflocoformer_separator_nocashe import (
    TFLocoformerSeparator as _BaselineSeparator,
)


class TFLocoformerSeparator(_BaselineSeparator):
    """Prepend two speaker CTFs to the unchanged TF-Locoformer speech trunk.

    ``additional["rir_ctf"]`` is a real tensor ``[B, L, 2, F, 2]``. The last
    axis stores real and imaginary parts; taps run from zero to positive lag.
    One shared pointwise projection encodes both speakers without assigning a
    speaker identity to either prefix. Training randomly exchanges the prefixes
    per utterance; the enhancement model retains its ordinary speech PIT loss.

    The baseline trunk has global normalization and unmasked time attention.
    Unequal input lengths are therefore processed separately, with padding only
    after separation, so padded speech cannot affect valid outputs or prompts.
    Equal-length training batches retain the normal batched execution path.
    """

    def __init__(
        self,
        input_dim,
        emb_dim: int = 128,
        ctf_taps: int = 60,
        random_ctf_swap: bool = True,
        glue_frames: int = 0,
        glue_value: float = 0.0,
        **kwargs,
    ):
        # Construct the complete baseline first, preserving its parameter names
        # and initialization order. Only the shared prompt projection is new.
        super().__init__(input_dim, emb_dim=emb_dim, **kwargs)
        if self.num_spk != 2:
            raise ValueError("CTF concatenation currently requires num_spk=2.")
        if isinstance(ctf_taps, bool) or not isinstance(ctf_taps, int) or ctf_taps < 1:
            raise ValueError("ctf_taps must be a positive integer.")
        if not isinstance(random_ctf_swap, bool):
            raise ValueError("random_ctf_swap must be a bool.")
        if isinstance(glue_frames, bool) or not isinstance(glue_frames, int) or glue_frames < 0:
            raise ValueError("glue_frames must be a nonnegative integer.")
        if (isinstance(glue_value, bool) or not isinstance(glue_value, Real)
                or not math.isfinite(glue_value)
                or abs(glue_value) > torch.finfo(torch.float32).max):
            raise ValueError("glue_value must be a finite float32-compatible number.")
        self.ctf_taps = ctf_taps
        self.random_ctf_swap = random_ctf_swap
        self.glue_frames = glue_frames
        self.glue_value = float(glue_value)
        # ESPnet globally initializes the model after construction. Preserve the
        # CPU RNG here so adding a projection does not shift the shared trunk's
        # subsequent Xavier draws relative to the baseline with the same seed.
        with torch.random.fork_rng(devices=[]):
            self.ctf_projection = nn.Conv2d(2, emb_dim, kernel_size=1)

    @staticmethod
    def _validate_lengths(lengths, batch_size, maximum, name):
        if not isinstance(lengths, torch.Tensor) or lengths.shape != (batch_size,):
            raise ValueError(f"{name} must be a tensor of shape [B].")
        if lengths.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{name} must contain integer lengths.")
        if torch.any(lengths <= 0) or torch.any(lengths > maximum):
            raise ValueError(f"{name} must lie in [1, {maximum}].")

    def _get_ctf(self, additional, batch_size, frequencies, reference):
        if additional is None or "rir_ctf" not in additional:
            raise ValueError("rir_ctf is required for CTF concatenation.")
        ctf = additional["rir_ctf"]
        expected = (batch_size, self.ctf_taps, self.num_spk, frequencies, 2)
        if not isinstance(ctf, torch.Tensor) or tuple(ctf.shape) != expected:
            raise ValueError(f"rir_ctf must have shape {expected} [B,L,S,F,RI].")
        if not torch.is_floating_point(ctf) or not torch.isfinite(ctf).all():
            raise ValueError("rir_ctf must contain finite real floating-point values.")
        if "rir_ctf_lengths" in additional:
            lengths = additional["rir_ctf_lengths"]
            self._validate_lengths(
                lengths, batch_size, self.ctf_taps, "rir_ctf_lengths"
            )
            if torch.any(lengths != self.ctf_taps):
                raise ValueError("rir_ctf_lengths must equal the fixed ctf_taps.")
        ctf = ctf.to(device=reference.device, dtype=reference.dtype)
        if self.training and self.random_ctf_swap:
            swap = torch.rand(batch_size, device=ctf.device) < 0.5
            ctf = torch.where(swap[:, None, None, None, None], ctf.flip(2), ctf)
        return ctf

    def _separate(self, speech, ctf):
        batch_size, _, frames, frequencies = speech.shape
        with torch.cuda.amp.autocast(enabled=False):
            features = self.conv(speech)
            prompts = ctf.permute(0, 2, 4, 1, 3).reshape(
                batch_size * self.num_spk, 2, self.ctf_taps, frequencies
            )
            prompts = self.ctf_projection(prompts)
            prompts = prompts.reshape(
                batch_size, self.num_spk, -1, self.ctf_taps, frequencies
            ).permute(0, 2, 1, 3, 4)
            prompts = prompts.reshape(
                batch_size, features.shape[1], self.num_spk * self.ctf_taps, frequencies
            )
            if self.glue_frames:
                # Insert constants AFTER projection: zero RI input would instead
                # become the learned projection bias. These are valid delimiter
                # tokens, not padding, and remain visible to every block.
                glue = prompts.new_full(
                    (batch_size, features.shape[1], self.glue_frames, frequencies),
                    self.glue_value,
                )
                prompts = torch.cat(
                    [piece for speaker in prompts.split(self.ctf_taps, dim=2)
                     for piece in (speaker, glue)], dim=2,
                )
            features = torch.cat((prompts, features), dim=2)
        for block in self.blocks:
            features = block(features)
        # The output convolution sees speech frames only. In particular its
        # temporal kernel never treats a CTF tap as reconstructed speech.
        prefix_frames = self.num_spk * (self.ctf_taps + self.glue_frames)
        features = features[:, :, prefix_frames:]
        with torch.cuda.amp.autocast(enabled=False):
            output = self.deconv(features)
        return output.reshape(batch_size, self.num_spk, 2, frames, frequencies)

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        if not is_complex(input):
            raise ValueError("Input must be a complex STFT tensor.")
        if input.ndim == 3:
            spectra = input.unsqueeze(1)  # [B,1,T,F]
        elif input.ndim == 4 and input.shape[2] == 1:
            spectra = input.transpose(1, 2)  # [B,T,1,F] -> [B,1,T,F]
        else:
            raise ValueError("Only monaural [B,T,F] or [B,T,1,F] input is supported.")
        speech = torch.cat((spectra.real, spectra.imag), dim=1)
        batch_size, _, frames, frequencies = speech.shape
        self._validate_lengths(ilens, batch_size, frames, "ilens")
        ctf = self._get_ctf(additional, batch_size, frequencies, speech)
        lengths = ilens.tolist()
        if all(length == lengths[0] for length in lengths):
            output = self._separate(speech[:, :, : lengths[0]], ctf)
            output = F.pad(output, (0, 0, 0, frames - lengths[0]))
        else:
            output = torch.cat(
                [
                    F.pad(
                        self._separate(speech[i : i + 1, :, :length], ctf[i : i + 1]),
                        (0, 0, 0, frames - length),
                    )
                    for i, length in enumerate(lengths)
                ],
                dim=0,
            )
        separated = new_complex_like(input, (output[:, :, 0], output[:, :, 1]))
        return (
            [separated[:, speaker] for speaker in range(self.num_spk)],
            ilens,
            OrderedDict(),
        )
