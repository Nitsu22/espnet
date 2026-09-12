"""Adapt the existing TF-Locoformer CTF predictor to single-source Rec-RIR."""

import torch
from torch import nn

from espnet2.rir.rec_rir.tflocoformer_ctf_pit import TFLocoformerCTFPredictor


class SingleSourceTFLocoformer(nn.Module):
    """Return CTF real/imag features in the Rec-RIR training/inference format."""

    def __init__(self, num_freqs, ctf_taps, **kwargs):
        super().__init__()
        self.predictor = TFLocoformerCTFPredictor(
            input_dim=num_freqs, num_spk=1, ctf_taps=ctf_taps, **kwargs
        )

    def forward(self, features):
        # Rec-RIR input [B, 2, F, T] -> TF-Locoformer complex [B, T, F].
        spectrum = torch.complex(features[:, 0], features[:, 1]).transpose(1, 2)
        ctf, _, _ = self.predictor(spectrum)
        ctf = ctf[:, 0]  # [B, F, L], with the same tap order as Rec-RIR.
        return None, torch.stack((ctf.real, ctf.imag), dim=1), None
