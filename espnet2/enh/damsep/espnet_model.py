"""ESPnet adapter for the released DAMSEP network and training objectives."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


def negative_sdr(estimate, target, scale_invariant=False):
    """Source-aligned, zero-mean negative SNR/SI-SDR, as in DAMSEP."""
    estimate = estimate - estimate.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    reference = target
    if scale_invariant:
        reference = (
            (estimate * target).sum(-1, keepdim=True)
            / (target.square().sum(-1, keepdim=True) + 1e-8)
            * target
        )
    ratio = reference.square().sum(-1) / (
        (estimate - reference).square().sum(-1) + 1e-8
    )
    return -10 * torch.log10(ratio + 1e-8)


def complex_convolve(signal, ctf):
    """Causal convolution over frames; tap zero multiplies the current frame.

    Shapes are [batch, speaker, frequency, frame/tap]. Grouped real convolutions
    avoid the release's batch-one reshape and broadcast errors.
    """
    if signal.shape[:-1] != ctf.shape[:-1]:
        raise ValueError("Signal and CTF batch/speaker/frequency axes must match")
    if not torch.is_complex(signal) or not torch.is_complex(ctf):
        raise ValueError("Signal and CTF must be complex tensors")
    taps = ctf.shape[-1]
    groups = signal.numel() // signal.shape[-1]
    real = F.pad(signal.real.reshape(1, groups, -1), (taps - 1, 0))
    imag = F.pad(signal.imag.reshape(1, groups, -1), (taps - 1, 0))
    kr = ctf.real.reshape(groups, 1, taps).flip(-1)
    ki = ctf.imag.reshape(groups, 1, taps).flip(-1)
    rr = F.conv1d(real, kr, groups=groups)
    ii = F.conv1d(imag, ki, groups=groups)
    ri = F.conv1d(real, ki, groups=groups)
    ir = F.conv1d(imag, kr, groups=groups)
    return torch.complex(rr - ii, ri + ir).reshape_as(signal)


def ri_mag_loss(estimate, target):
    return (
        (estimate.real - target.real).abs()
        + (estimate.imag - target.imag).abs()
        + (estimate.abs() - target.abs()).abs()
    ).mean(dim=(-1, -2, -3))


class ESPnetDAMSEPModel(AbsESPnetModel):
    def __init__(
        self,
        network_conf: Optional[Dict] = None,
        loss_w_reverb: float = 0.1,
        loss_w_reconstruction: float = 0.5,
        validation_scale_invariant: bool = True,
        loss_n_fft: int = 512,
        loss_win_length: int = 256,
        loss_hop_length: int = 128,
        loss_window: str = "sqrthann",
    ):
        super().__init__()
        # Lazy import: task help/data utilities do not require Mamba CUDA packages.
        from espnet2.enh.damsep.vendor.models.SPMamba import SPMamba

        conf = dict(
            input_dim=64,
            n_srcs=2,
            n_fft=256,
            stride=64,
            window="hann",
            n_imics=1,
            n_layers=6,
            lstm_hidden_units=256,
            attn_n_head=4,
            attn_approx_qk_dim=512,
            emb_dim=16,
            emb_ks=8,
            emb_hs=1,
            activation="prelu",
            eps=1e-5,
            use_builtin_complex=False,
            sample_rate=8000,
        )
        conf.update(network_conf or {})
        if conf["n_srcs"] != 2 or conf["n_imics"] != 1 or conf["sample_rate"] != 8000:
            raise ValueError("This DAMSEP recipe requires two sources, mono, 8 kHz")
        if loss_w_reverb < 0 or loss_w_reconstruction < 0:
            raise ValueError("Loss weights must be nonnegative")
        self.network = SPMamba(**conf)
        self.num_spk = 2
        self.loss_w_reverb = loss_w_reverb
        self.loss_w_reconstruction = loss_w_reconstruction
        self.validation_scale_invariant = validation_scale_invariant
        # Preserve the release's auxiliary-loss STFT, which differs from the
        # network's 512-point, 512-sample Hann analysis. Do not silently unify it.
        self.transforms = RecRIRTransforms(
            sr=8000,
            n_fft=loss_n_fft,
            win_len=loss_win_length,
            hop_len=loss_hop_length,
            win_type=loss_window,
        )
        if loss_n_fft != self.network.ctf_n_fft:
            raise ValueError("Loss frequency bins must match the CTF network")

    @staticmethod
    def _mono(signal):
        if signal.ndim == 3:
            return signal[..., 0]
        if signal.ndim != 2:
            raise ValueError(
                f"Expected [batch,time] or [batch,time,channel]: {signal.shape}"
            )
        return signal

    def _loss(self, output, clean, reverb):
        estimate = output["x_derev"]
        # Choose one permutation from the clean waveform objective and reuse it
        # for reverberant-source supervision and CTF reconstruction.
        scale_invariant = not self.training and self.validation_scale_invariant
        identity = negative_sdr(estimate, clean, scale_invariant).mean(-1)
        swapped = negative_sdr(estimate.flip(1), clean, scale_invariant).mean(-1)
        swap = swapped < identity
        order = torch.tensor([0, 1], device=clean.device).expand(clean.shape[0], -1)
        order = torch.where(swap[:, None], order.flip(1), order)

        def align(x):
            index = order.reshape(order.shape + (1,) * (x.ndim - 2)).expand_as(x)
            return x.gather(1, index)

        loss_clean = torch.where(swap, swapped, identity)
        clean_spec = self.transforms.stft(clean)
        target_spec = self.transforms.stft(reverb)
        loss_reverb = ri_mag_loss(
            self.transforms.stft(align(output["x_sep"])), target_spec
        )
        ctf_ri = output["rir"].reshape(clean.shape[0], 2, 2, 257, 60)
        ctf = align(torch.complex(ctf_ri[:, :, 0], ctf_ri[:, :, 1]))
        reconstruction = complex_convolve(clean_spec, ctf)
        loss_reconstruction = ri_mag_loss(reconstruction, target_spec)
        total = (
            loss_clean
            + self.loss_w_reverb * loss_reverb
            + self.loss_w_reconstruction * loss_reconstruction
        )
        return total.mean(), {
            "loss": total.mean().detach(),
            "loss_clean": loss_clean.mean().detach(),
            "loss_reverb": loss_reverb.mean().detach(),
            "loss_reconstruction": loss_reconstruction.mean().detach(),
            "pit_swap_ratio": swap.float().mean().detach(),
        }

    def forward(
        self,
        speech_mix,
        speech_mix_lengths,
        speech_ref1,
        speech_ref2,
        speech_reverb1,
        speech_reverb2,
        **kwargs,
    ):
        dtype = next(self.network.parameters()).dtype
        signals = [
            self._mono(x).to(dtype=dtype)
            for x in (
                speech_mix,
                speech_ref1,
                speech_ref2,
                speech_reverb1,
                speech_reverb2,
            )
        ]
        names = [
            "speech_mix",
            "speech_ref1",
            "speech_ref2",
            "speech_reverb1",
            "speech_reverb2",
        ]
        for name, signal in zip(names, signals):
            lengths = kwargs.get(name + "_lengths", speech_mix_lengths)
            if not torch.equal(lengths, speech_mix_lengths):
                raise ValueError(f"{name}: lengths differ from the mixture")
            if signal.shape[:2] != signals[0].shape[:2]:
                raise ValueError(f"{name}: padded shape differs from the mixture")
        # Process differing lengths separately so padding cannot affect Mamba,
        # normalization, attention, temporal pooling or the losses.
        groups = [
            torch.where(speech_mix_lengths == n)[0] for n in speech_mix_lengths.unique()
        ]
        losses, statistics, weights = [], [], []
        for indices in groups:
            n = int(speech_mix_lengths[indices[0]])
            if n <= self.network.ctf_n_fft // 2:
                raise ValueError("DAMSEP needs more than 256 waveform samples")
            mix, c1, c2, r1, r2 = [s[indices, :n] for s in signals]
            output = self.network(mix)
            loss, stats = self._loss(
                output, torch.stack([c1, c2], 1), torch.stack([r1, r2], 1)
            )
            losses.append(loss * indices.numel())
            statistics.append(stats)
            weights.append(indices.numel())
        batch_size = speech_mix.shape[0]
        loss = sum(losses) / batch_size
        stats = {
            k: sum(s[k] * w for s, w in zip(statistics, weights)) / batch_size
            for k in statistics[0]
        }
        return force_gatherable((loss, stats, batch_size), loss.device)

    def collect_feats(self, speech_mix, speech_mix_lengths, **kwargs):
        return {"feats": self._mono(speech_mix), "feats_lengths": speech_mix_lengths}

    @torch.no_grad()
    def separate(self, speech_mix):
        """Return clean/reverberant waveforms and native complex CTFs."""
        dtype = next(self.network.parameters()).dtype
        return self.network(self._mono(speech_mix).to(dtype=dtype))
