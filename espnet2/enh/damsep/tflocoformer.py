"""TF-Locoformer separation followed by DAMSEP's shared dereverb/CTF network."""

from typing import Dict, Optional

import torch
from torch import nn

from espnet2.enh.damsep.espnet_model import (
    ESPnetDAMSEPModel,
    complex_convolve,
    ri_mag_loss,
)
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.separator.tflocoformer_separator_nocashe import TFLocoformerSeparator
from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel


class TFLocoformerDAMSEP(nn.Module):
    """Preserve the baseline TF network and the released DAMSEP second stage."""

    def __init__(self, separator_conf=None, ctf_enabled=True):
        super().__init__()
        from espnet2.enh.damsep.vendor.models.RecRIR import BiSpatialNet
        from espnet2.enh.damsep.vendor.models.SPMamba import STFTDecoder, STFTEncoder

        conf = dict(
            num_spk=2,
            n_layers=4,
            emb_dim=96,
            norm_type="rmsgroupnorm",
            num_groups=4,
            tf_order="ft",
            n_heads=4,
            flash_attention=False,
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=[128, 128],
            conv1d_kernel=8,
            conv1d_shift=1,
            dropout=0.0,
            eps=1e-5,
        )
        conf.update(separator_conf or {})
        if conf["num_spk"] != 2:
            raise ValueError("TF-DAMSEP requires two speakers")
        self.enc = STFTEncoder(256, 256, 64, window="hann", use_builtin_complex=True)
        self.dec = STFTDecoder(256, 256, 64, window="hann")
        self.separator = TFLocoformerSeparator(input_dim=129, **conf)
        # Initialize the TF branch before creating the new modules. This keeps
        # the baseline's TF initialization and both ablations' shared weights.
        initialize(self.separator, "xavier_uniform")
        self.ctf_n_fft = 512
        self.ctf_enc = STFTEncoder(
            512, 512, 128, window="hann", use_builtin_complex=True
        )
        self.ctf_dec = STFTDecoder(512, 512, 128, window="hann")
        self.recrir = BiSpatialNet(
            dim_input=2,
            dim_output_spch=2,
            dim_output_CTF=120,
            dim_hidden=64,
            dim_squeeze=8,
            num_freqs=257,
            num_layers_spch=6,
            num_layers_noise=2,
            encoder_kernel_size=5,
            dropout=(0, 0, 0),
            kernel_size=(5, 3),
            conv_groups=(8, 8),
            norms=["LN"] * 6,
            full_share=0,
            attention="mamba(16,4)",
        )
        self.ctf_enabled = ctf_enabled
        if not ctf_enabled:
            for module in (
                self.recrir.ctf_layers,
                self.recrir.decoder_CTF,
                self.recrir.compress_CTF,
                self.recrir.weight_layer,
            ):
                module.requires_grad_(False)

    def forward(self, mixture, estimate_ctf=True):
        b, n = mixture.shape
        lengths = torch.full((b,), n, dtype=torch.long, device=mixture.device)
        # Same per-mixture variance normalization as the TF baseline. Restore
        # scale before the shared second stage and absolute spectral losses.
        scale = mixture.std(dim=1, keepdim=True).clamp_min(1e-12)
        spectrum, frame_lengths = self.enc(mixture / scale, lengths)
        separated, _, _ = self.separator(spectrum, frame_lengths)
        spectrum = torch.stack(separated, dim=1)
        t, f = spectrum.shape[-2:]
        source_lengths = lengths.repeat_interleave(2)
        reverb = self.dec(spectrum.reshape(b * 2, t, f), source_lengths)[0]
        reverb = reverb.reshape(b, 2, n) * scale[:, None]
        spectrum, _ = self.ctf_enc(reverb.reshape(b * 2, n), source_lengths)
        features = torch.stack([spectrum.real, spectrum.imag], dim=1)
        features = features.permute(0, 1, 3, 2).contiguous()
        use_ctf = self.ctf_enabled and estimate_ctf
        ctf, clean = self.recrir(features, estimate_ctf=use_ctf)
        spectrum = torch.complex(clean[:, 0], clean[:, 1]).transpose(1, 2)
        clean = self.ctf_dec(spectrum, source_lengths)[0].reshape(b, 2, n)
        output = {"x_sep": reverb, "x_derev": clean}
        if use_ctf:
            output["rir"] = ctf
        return output


class ESPnetTFLocoformerDAMSEPModel(ESPnetDAMSEPModel):
    """Ablate CTF reconstruction while keeping clean+reverb supervision fixed."""

    def __init__(
        self,
        separator_conf: Optional[Dict] = None,
        loss_w_reverb: float = 0.1,
        loss_w_reconstruction: float = 0.5,
        si_snr_eps: float = 1e-7,
        extract_feats_in_collect_stats: bool = False,
    ):
        AbsESPnetModel.__init__(self)
        if loss_w_reverb <= 0 or loss_w_reconstruction < 0:
            raise ValueError(
                "Require positive reverb and nonnegative reconstruction weights"
            )
        self.loss_w_reverb = loss_w_reverb
        self.loss_w_reconstruction = loss_w_reconstruction
        self.ctf_enabled = loss_w_reconstruction > 0
        self.clean_only = False  # Both experiments export clean and reverb audio.
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.num_spk = 2
        self.network = TFLocoformerDAMSEP(separator_conf, self.ctf_enabled)
        self.clean_criterion = SISNRLoss(eps=si_snr_eps)
        # Preserve DAMSEP's auxiliary-loss STFT separately from its analysis STFT.
        self.transforms = RecRIRTransforms(
            sr=8000, n_fft=512, win_len=256, hop_len=128, win_type="sqrthann"
        )

    def _loss(self, output, clean, reverb):
        estimate = output["x_derev"]
        identity = self.clean_criterion(clean, estimate).mean(-1)
        swapped = self.clean_criterion(clean, estimate.flip(1)).mean(-1)
        swap = swapped < identity
        order = torch.tensor([0, 1], device=clean.device).expand(clean.shape[0], -1)
        order = torch.where(swap[:, None], order.flip(1), order)

        def align(x):
            index = order.reshape(order.shape + (1,) * (x.ndim - 2)).expand_as(x)
            return x.gather(1, index)

        loss_clean = torch.where(swap, swapped, identity)
        target = self.transforms.stft(reverb)
        loss_reverb = ri_mag_loss(self.transforms.stft(align(output["x_sep"])), target)
        total = loss_clean + self.loss_w_reverb * loss_reverb
        stats = {
            "loss_clean": loss_clean.mean().detach(),
            "si_snr": -loss_clean.mean().detach(),
            "loss_reverb": loss_reverb.mean().detach(),
            "pit_swap_ratio": swap.float().mean().detach(),
        }
        if self.ctf_enabled:
            ri = output["rir"].reshape(clean.shape[0], 2, 2, 257, 60)
            ctf = align(torch.complex(ri[:, :, 0], ri[:, :, 1]))
            # GT clean sources, not predictions or noisy mixture, are filtered.
            reconstruction = complex_convolve(self.transforms.stft(clean), ctf)
            loss_reconstruction = ri_mag_loss(reconstruction, target)
            total = total + self.loss_w_reconstruction * loss_reconstruction
            stats["loss_reconstruction"] = loss_reconstruction.mean().detach()
        stats["loss"] = total.mean().detach()
        return total.mean(), stats
