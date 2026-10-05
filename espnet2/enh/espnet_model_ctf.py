"""Joint clean-speech and CTF-reconstructed reverberant-speech SI-SNR."""

from itertools import permutations

import torch
from torch.nn import functional as F

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetEnhancementCTFModel(ESPnetEnhancementModel):
    """Apply one speaker assignment to the sum of two waveform losses.

    No oracle speech/RIR enters the separator. Reverberant waveforms are made
    from the *predicted* clean waveforms and predicted CTF, with no detach.
    Inference retains the ordinary enhancement interface and clean outputs.
    """

    def __init__(self, *args, loss_w_clean=1.0, loss_w_reverb=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_w_clean <= 0 or loss_w_reverb <= 0:
            raise ValueError("Both joint loss weights must be positive")
        if self.num_spk != 2 or self.mask_module is not None:
            raise ValueError("Joint CTF enhancement supports two fixed speech outputs")
        if (
            len(self.loss_wrappers) != 1
            or not isinstance(self.loss_wrappers[0], PITSolver)
            or not isinstance(self.loss_wrappers[0].criterion, SISNRLoss)
            or self.loss_wrappers[0].criterion.only_for_test
        ):
            raise ValueError("Configure one training SI-SNR PIT criterion")
        if self.flexible_numspk or self.categories or self.normalize_variance_per_ch:
            raise ValueError("Joint CTF enhancement requires fixed mono training")
        if not hasattr(self.encoder, "hop_length"):
            raise ValueError("Joint CTF enhancement requires an STFT encoder")
        self.loss_w_clean = float(loss_w_clean)
        self.loss_w_reverb = float(loss_w_reverb)

    @staticmethod
    def convolve_ctf(spectrum, ctf):
        """Full causal frame convolution: [B,S,F,T] * [B,S,F,L]."""
        if spectrum.shape[:-1] != ctf.shape[:-1]:
            raise ValueError("Speech spectrum and CTF dimensions do not match")
        length = spectrum.shape[-1] + ctf.shape[-1] - 1
        size = 1 << (length - 1).bit_length()
        return torch.fft.ifft(
            torch.fft.fft(spectrum, n=size) * torch.fft.fft(ctf, n=size),
            n=size,
        )[..., :length]

    def reconstruct_reverb(self, speech_pre, speech_lengths, ctf):
        # Re-encode decoded speech to enforce STFT consistency. Undoing the
        # mix normalization before this function keeps both branches on the
        # same waveform scale; CTF itself is a dimensionless transfer.
        output = [[] for _ in range(self.num_spk)]
        maximum = int(speech_lengths.max())
        for b, length in enumerate(speech_lengths.tolist()):
            # STFT each utterance at its actual boundary. Transforming a padded
            # batch then masking frames changes the shorter utterance's tail.
            lens = speech_lengths[b : b + 1]
            spectra = [self.encoder(x[b : b + 1, :length], lens)[0] for x in speech_pre]
            spectrum = torch.stack(spectra, dim=1).transpose(-1, -2)
            reverb = self.convolve_ctf(spectrum, ctf[b : b + 1])
            # WHAMR references stop at the utterance boundary. Retain full
            # convolution internally, then compare the observed interval.
            reverb = reverb[..., : spectrum.shape[-1]].transpose(-1, -2)
            for s, x in enumerate(reverb.unbind(1)):
                wave = self.decoder(x, lens)[0]
                output[s].append(F.pad(wave, (0, maximum - length)))
        return [torch.cat(x, dim=0) for x in output]

    def joint_loss(self, clean_pre, reverb_pre, clean_ref, reverb_ref, lengths, starts):
        """Minimize the weighted sum BEFORE choosing the PIT permutation."""
        criterion = self.loss_wrappers[0].criterion
        clean_pairs, reverb_pairs = [], []
        for b, (length, start) in enumerate(zip(lengths.tolist(), starts.tolist())):
            if not 0 <= start < length - 1:
                raise ValueError("No valid reverberant-speech loss interval")
            cp, rp = [], []
            for pred in range(self.num_spk):
                cp.append(
                    torch.stack(
                        [
                            criterion(
                                clean_ref[ref][b : b + 1, :length],
                                clean_pre[pred][b : b + 1, :length],
                            ).squeeze(0)
                            for ref in range(self.num_spk)
                        ]
                    )
                )
                rp.append(
                    torch.stack(
                        [
                            criterion(
                                reverb_ref[ref][b : b + 1, start:length],
                                reverb_pre[pred][b : b + 1, start:length],
                            ).squeeze(0)
                            for ref in range(self.num_spk)
                        ]
                    )
                )
            clean_pairs.append(torch.stack(cp))
            reverb_pairs.append(torch.stack(rp))
        clean_pairs = torch.stack(clean_pairs)
        reverb_pairs = torch.stack(reverb_pairs)
        perms = list(permutations(range(self.num_spk)))
        pred = torch.arange(self.num_spk, device=lengths.device)
        clean = torch.stack([clean_pairs[:, pred, list(p)].mean(1) for p in perms], 1)
        reverb = torch.stack([reverb_pairs[:, pred, list(p)].mean(1) for p in perms], 1)
        total = self.loss_w_clean * clean + self.loss_w_reverb * reverb
        best = total.argmin(1, keepdim=True)
        lc = clean.gather(1, best).mean()
        lr = reverb.gather(1, best).mean()
        # Compare clean quality with the baseline's clean-only PIT assignment.
        # This metric does not change the joint training loss.
        clean_pit = clean.min(dim=1).values.mean()
        return self.loss_w_clean * lc + self.loss_w_reverb * lr, lc, lr, best, clean_pit

    def forward(self, speech_mix, speech_mix_lengths=None, **kwargs):
        if speech_mix.ndim != 2:
            raise ValueError("Joint CTF model expects mono waveforms [B,N]")
        if speech_mix_lengths is None:
            speech_mix_lengths = torch.full(
                (speech_mix.shape[0],),
                speech_mix.shape[1],
                dtype=torch.long,
                device=speech_mix.device,
            )
        lengths = speech_mix_lengths
        if (lengths <= 1).any() or (lengths > speech_mix.shape[1]).any():
            raise ValueError("Invalid waveform lengths")
        clean_ref, reverb_ref = [], []
        for s in range(1, self.num_spk + 1):
            for prefix, signals in [
                ("speech_ref", clean_ref),
                ("speech_reverb", reverb_ref),
            ]:
                name = f"{prefix}{s}"
                if name not in kwargs:
                    raise ValueError(f"{name} is required for joint CTF training")
                signal = kwargs[name]
                if signal.ndim != 2 or signal.shape[0] != speech_mix.shape[0]:
                    raise ValueError(f"{name} must be mono [B,N]")
                ref_lengths = kwargs.get(f"{name}_lengths", lengths)
                if not torch.equal(ref_lengths, lengths):
                    raise ValueError(f"{name} and mixture lengths differ")
                if signal.shape[1] < lengths.max():
                    raise ValueError(f"{name} is shorter than the mixture")
                signals.append(signal[:, : lengths.max()])
        speech_mix = speech_mix[:, : lengths.max()]
        scale = speech_mix.std(dim=1, keepdim=True).clamp_min(1.0e-8)
        model_input = speech_mix / scale if self.normalize_variance else speech_mix
        clean_pre, _, _, others = self.forward_enhance(model_input, lengths, {})
        if self.normalize_variance:
            clean_pre = [x * scale for x in clean_pre]
        reverb_pre = self.reconstruct_reverb(clean_pre, lengths, others["ctf"])

        # Keep the baseline's exact four-second input/crop. A crop starting
        # mid-utterance lacks earlier clean speech: exclude the CTF warmup
        # and one STFT window from the reverb loss, without oracle context.
        crop_start = kwargs.get("reverb_crop_start", torch.zeros_like(lengths))
        crop_start = crop_start.reshape(-1)
        warmup = (
            self.separator.ctf_taps - 1
        ) * self.encoder.hop_length + self.encoder.win_length
        starts = torch.where(crop_start > 0, warmup, 0)
        loss, lc, lr, _, clean_pit = self.joint_loss(
            clean_pre,
            reverb_pre,
            clean_ref,
            reverb_ref,
            lengths,
            starts,
        )
        stats = dict(
            loss=loss.detach(),
            loss_clean=lc.detach(),
            loss_reverb=lr.detach(),
            loss_clean_pit=clean_pit.detach(),
            si_snr=-clean_pit.detach(),
            si_snr_reverb=-lr.detach(),
        )
        return force_gatherable((loss, stats, speech_mix.shape[0]), loss.device)
