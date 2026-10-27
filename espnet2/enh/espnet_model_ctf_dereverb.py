"""Joint 3+1-block CTF restoration with baseline SI-SNR and common PIT."""

import math
from itertools import permutations

import torch
from torch.nn import functional as F

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.layers.ctf_deconvolution import ctf_forward
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.torch_utils.device_funcs import force_gatherable


def spectral_distance(estimate, reference, mask):
    """RI-L1 + magnitude-L1, averaged over valid frequency/frame bins."""
    if estimate.shape != reference.shape:
        raise ValueError("Spectral estimate and reference shapes differ")
    error = (estimate.real - reference.real).abs()
    error = error + (estimate.imag - reference.imag).abs()
    error = error + (estimate.abs() - reference.abs()).abs()
    weight = torch.broadcast_to(mask, error.shape)
    if not weight.any():
        raise ValueError("The target interval contains no valid STFT frames")
    return (error * weight).sum() / weight.sum()


class ESPnetCTFDereverbModel(ESPnetEnhancementModel):
    """Final-Clean SI-SNR PIT, shared by Reverb and teacher-CTF reconstruction.

    Each utterance is encoded at its own boundary. The mixture-derived scale
    is shared by every teacher and estimate; all spectral losses use linear
    complex STFTs. No teacher signal enters the prediction path.
    """

    def __init__(
        self,
        *args,
        loss_w_clean=1.0,
        loss_w_reverb=1.0,
        loss_w_ctf=1.0,
        baseline_mode=False,
        component_grad_interval=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        loss_weights = (loss_w_clean, loss_w_reverb, loss_w_ctf)
        if not all(math.isfinite(x) and x >= 0 for x in loss_weights):
            raise ValueError("Loss weights must be nonnegative")
        if self.num_spk != 2 or self.mask_module is not None:
            raise ValueError("The initial model requires two fixed mono sources")
        if self.normalize_variance_per_ch or self.flexible_numspk or self.categories:
            raise ValueError("Use fixed mono data and one common normalization scale")
        if not hasattr(self.separator, "separate_all"):
            raise ValueError("Use a CTF-dereverb separator")
        if not baseline_mode and not hasattr(self.separator, "ctf_taps"):
            raise ValueError("Use the CTF restoration separator for auxiliary losses")
        # Honor the actual YAML criterion, including baseline clamp/zero-mean
        # settings. Other losses must not be silently constructed and ignored.
        if self.loss_wrappers:
            if (
                len(self.loss_wrappers) != 1
                or not isinstance(self.loss_wrappers[0], PITSolver)
                or not isinstance(self.loss_wrappers[0].criterion, SISNRLoss)
                or self.loss_wrappers[0].weight != 1
                or self.loss_wrappers[0].flexible_numspk
            ):
                raise ValueError("Configure one weight-1 SI-SNR PIT criterion")
            self.speech_criterion = self.loss_wrappers[0].criterion
        else:
            self.speech_criterion = SISNRLoss(eps=1e-7)
        for module in (self.encoder, self.decoder):
            if getattr(module, "spec_transform_type", None) not in (None, "none"):
                raise ValueError("CTF/CG require uncompressed linear complex STFT")
            if not getattr(module, "center", False):
                raise ValueError("This recipe uses the baseline centered STFT")
        for attribute in ("n_fft", "win_length", "hop_length", "window", "center"):
            if getattr(self.encoder, attribute) != getattr(self.decoder, attribute):
                raise ValueError("Encoder and decoder STFT settings must match")
        for attribute in ("normalized", "onesided"):
            if getattr(self.encoder.stft, attribute) != getattr(
                self.decoder.stft, attribute
            ):
                raise ValueError("Encoder and decoder STFT conventions must match")
        if not getattr(self.encoder, "use_builtin_complex", False):
            raise ValueError("Use native complex STFT")
        if component_grad_interval < 0:
            raise ValueError("component_grad_interval must be nonnegative")
        self.loss_w_clean = loss_w_clean
        self.loss_w_reverb = loss_w_reverb
        self.loss_w_ctf = loss_w_ctf
        self.baseline_mode = baseline_mode
        self.component_grad_interval = component_grad_interval
        self._loss_calls = 0

    def _encode_sources(self, sources, lengths):
        return torch.stack(
            [self.encoder(source, lengths)[0].transpose(-1, -2)
             for source in sources.unbind(1)],
            dim=1,
        )

    def _decode_sources(self, spectrum, lengths):
        return torch.stack(
            [self.decoder(x.transpose(-1, -2), lengths)[0]
             for x in spectrum.unbind(1)],
            dim=1,
        )

    def _predict_one(self, mixture, observation_mask=None):
        lengths = torch.tensor([mixture.shape[-1]], device=mixture.device)
        scale = mixture.std(-1, keepdim=True).clamp_min(1e-8)
        if not self.normalize_variance:
            scale = torch.ones_like(scale)
        normalized = mixture.float() / scale
        spectrum, frames = self.encoder(normalized, lengths)
        result = self.separator.separate_all(spectrum, frames, observation_mask)
        waves = {
            name: self._decode_sources(value, lengths)
            for name, value in result.items() if name != "ctf"
        }
        return result, waves, scale, frames

    def predict_all(self, speech_mix, speech_mix_lengths=None):
        """Mixture-only inference; spectra/waves are restored to original scale.

        Returns all named complex branches, ``waveforms`` (same branch names),
        frame/sample lengths, and the common ``normalization_scale``. CTFs are
        dimensionless. Outputs retain the model's own shared speaker ordering.
        """
        if speech_mix.ndim != 2:
            raise ValueError("Expected mono waveforms [B,samples]")
        lengths = self._lengths(speech_mix, speech_mix_lengths)
        spectra, waves, scales, flens = {}, {}, [], []
        for index, length in enumerate(lengths.tolist()):
            result, audio, scale, frames = self._predict_one(
                speech_mix[index : index + 1, :length]
            )
            scales.append(scale)
            flens.append(frames)
            for name, value in result.items():
                restored = value if name == "ctf" else value * scale[..., None, None]
                spectra.setdefault(name, []).append(restored)
            for name, value in audio.items():
                waves.setdefault(name, []).append(value * scale[..., None])
        frame_lengths = torch.cat(flens)
        max_frames, max_samples = int(frame_lengths.max()), int(lengths.max())
        output = {
            name: torch.cat([
                x if name == "ctf" else F.pad(x, (0, max_frames - x.shape[-1]))
                for x in values
            ])
            for name, values in spectra.items()
        }
        output["waveforms"] = {
            name: torch.cat([F.pad(x, (0, max_samples - x.shape[-1])) for x in values])
            for name, values in waves.items()
        }
        output.update(
            frame_lengths=frame_lengths,
            sample_lengths=lengths,
            normalization_scale=torch.cat(scales),
        )
        return output

    @staticmethod
    def _lengths(mixture, lengths):
        if lengths is None:
            lengths = torch.full(
                (mixture.shape[0],), mixture.shape[-1],
                dtype=torch.long, device=mixture.device,
            )
        if (lengths.shape != (mixture.shape[0],) or lengths.is_floating_point()
                or lengths.dtype == torch.bool):
            raise ValueError("Expected one integer sample length per utterance")
        if (lengths <= 1).any() or (lengths > mixture.shape[-1]).any():
            raise ValueError("Invalid waveform lengths")
        return lengths

    def _interval(self, kwargs, index, length, frames, device):
        def value(name, default):
            x = kwargs.get(name)
            return default if x is None else int(x[index].reshape(-1)[0])

        start = value("ctf_loss_start", 0)
        end = value("ctf_loss_end", length)
        if not 0 <= start < end <= length or end - start < 2:
            raise ValueError("Invalid target interval")
        positions = torch.arange(frames, device=device) * self.encoder.hop_length
        target = (positions >= start) & (positions < end)
        observed = torch.ones(frames, dtype=torch.bool, device=device)
        crop_start = value("ctf_crop_start", 0)
        crop_end = value("ctf_crop_end", length)
        original = value("ctf_original_length", length)
        if crop_start < 0 or crop_end - crop_start != length or crop_end > original:
            raise ValueError("Inconsistent physical crop boundaries")
        guard = self.encoder.n_fft // 2
        if crop_start > 0:
            taps = getattr(self.separator, "ctf_taps", 128)
            observed &= positions >= (taps - 1) * self.encoder.hop_length + guard
        if crop_end < original:
            observed &= positions + guard < length
        if not target.any() or not (target & observed).equal(target):
            raise ValueError("Target lacks sufficient real CTF/STFT context")
        return start, end, target[None, None, None], observed[None, None, None]

    def _loss_one(
        self, result, waves, clean_wave, reverb_wave, lengths, start, end, target
    ):
        clean = self._encode_sources(clean_wave, lengths)
        reverb = (
            None if reverb_wave is None else self._encode_sources(reverb_wave, lengths)
        )
        matrices = {}
        names = ("clean_final",) if self.baseline_mode else (
            "clean_final", "reverb", "clean_deconv"
        )
        for name in names:
            teacher = reverb_wave if name == "reverb" else clean_wave
            rows = []
            for predicted in range(self.num_spk):
                row = []
                for reference in range(self.num_spk):
                    td = self.speech_criterion(
                        teacher[:, reference, start:end],
                        waves[name][:, predicted, start:end],
                    ).mean()
                    row.append(td)
                rows.append(torch.stack(row))
            matrices[name] = torch.stack(rows)
        perms = torch.tensor(
            list(permutations(range(self.num_spk))), device=clean.device
        )
        pred = torch.arange(self.num_spk, device=clean.device)
        costs = torch.stack([
            matrices["clean_final"][pred, perm].mean() for perm in perms
        ])
        selected = perms[costs.argmin()]
        components = {
            "clean": matrices["clean_final"][pred, selected].mean(),
            "si_snr": -matrices["clean_final"][pred, selected].mean(),
        }
        if not self.baseline_mode:
            # The SAME clean-derived permutation is used for Y and H.
            aligned_clean = clean[:, selected]
            aligned_reverb = reverb[:, selected]
            reconstructed = ctf_forward(aligned_clean, result["ctf"])
            components["reverb"] = matrices["reverb"][pred, selected].mean()
            components["si_snr_reverb"] = -components["reverb"]
            components["si_snr_deconv"] = -matrices["clean_deconv"][
                pred, selected
            ].mean()
            components["ctf"] = spectral_distance(reconstructed, aligned_reverb, target)
        return components

    def forward(self, speech_mix, speech_mix_lengths=None, **kwargs):
        if speech_mix.ndim != 2:
            raise ValueError("Expected mono waveforms [B,samples]")
        lengths = self._lengths(speech_mix, speech_mix_lengths)
        losses = {}
        for index, length in enumerate(lengths.tolist()):
            single_lengths = lengths[index:index+1]
            frame_count = length // self.encoder.hop_length + 1
            start, end, target, observed = self._interval(
                kwargs, index, length, frame_count, speech_mix.device
            )
            result, waves, scale, _ = self._predict_one(
                speech_mix[index:index+1, :length], observed
            )
            refs = []
            for prefix in ("speech_ref", "speech_reverb"):
                if prefix == "speech_reverb" and self.baseline_mode:
                    refs.append(None)
                    continue
                signals = []
                for speaker in range(1, self.num_spk + 1):
                    key = prefix + str(speaker)
                    if key not in kwargs or kwargs[key].ndim != 2:
                        raise ValueError(f"{key} must be a paired mono waveform")
                    if not torch.equal(kwargs.get(key + "_lengths", lengths), lengths):
                        raise ValueError(f"{key} lengths differ from mixture")
                    if (
                        kwargs[key].shape[0] != speech_mix.shape[0]
                        or kwargs[key].shape[1] < length
                    ):
                        raise ValueError(f"{key} shape differs from mixture")
                    signals.append(kwargs[key][index:index+1, :length].float() / scale)
                refs.append(torch.stack(signals, 1))
            for name, value in self._loss_one(
                result, waves, refs[0], refs[1], single_lengths, start, end, target
            ).items():
                losses.setdefault(name, []).append(value)
        components = {
            name: torch.stack(values).mean() for name, values in losses.items()
        }
        weights = {"clean": self.loss_w_clean}
        if not self.baseline_mode:
            weights.update(reverb=self.loss_w_reverb, ctf=self.loss_w_ctf)
        loss = sum(weights[key] * components[key] for key in weights)
        stats = {"loss": loss.detach()}
        stats.update({
            key if key.startswith("si_snr") else "loss_" + key: value.detach()
            for key, value in components.items()
        })
        if (self.training and torch.is_grad_enabled()
                and self.component_grad_interval
                and self._loss_calls % self.component_grad_interval == 0):
            parameters = tuple(p for p in self.parameters() if p.requires_grad)
            for name in weights:
                gradients = torch.autograd.grad(
                    weights[name] * components[name], parameters,
                    retain_graph=True, allow_unused=True,
                )
                norm = sum(
                    g.detach().float().square().sum()
                    for g in gradients if g is not None
                )
                stats["grad_norm_" + name] = norm.sqrt()
        self._loss_calls += int(self.training)
        return force_gatherable((loss, stats, speech_mix.shape[0]), loss.device)
