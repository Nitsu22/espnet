"""Shared Rec-RIR trunk with only a CTF output head."""
import torch
from espnet2.rir.rec_rir.model import BiSpatialNet


class CTFOnlyBiSpatialNet(BiSpatialNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.decoder_spch
        del self.decoder_rev

    def forward(self, input, return_embedding=False):
        x = torch.nn.functional.pad(input, (
            self.padding_size[1], self.padding_size[1],
            self.padding_size[0], self.padding_size[0]))
        x = self.encoder(x).permute(0, 2, 3, 1)
        for layer in self.noise_layers:
            x = layer(x)
        reverb_features = x
        for layer in self.spch_layers:
            x = layer(x)
        x = self.compress_CTF(x, reverb_features)
        for layer in self.ctf_layers:
            x = layer(x)
        x = (x * self.weight_layer(x)).sum(-2).unsqueeze(2)
        if return_embedding:
            return x
        batch, freq, _, _ = x.shape
        ctf = self.decoder_CTF(x).reshape(batch, freq, 2, -1).permute(0, 2, 1, 3)
        # Preserve the CTF position expected by the inherited inference method.
        return None, ctf, None
