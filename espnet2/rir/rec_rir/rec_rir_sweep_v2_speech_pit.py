"""CTF-only Rec-RIR with jointly assigned Sweep v2 and speech reconstruction."""
import math

import torch
import torch.nn.functional as F

from espnet2.rir.rec_rir.rec_rir_sweep_v2_pit import ESPnetRecRIRSweepV2PITModel
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetRecRIRSweepV2SpeechPITModel(ESPnetRecRIRSweepV2PITModel):
    def __init__(self, sweep_loss_weight=1.0, speech_loss_weight=1.0, **kwargs):
        weights = (float(sweep_loss_weight), float(speech_loss_weight))
        if (any(not math.isfinite(w) or w < 0 for w in weights)
                or not sum(weights) > 0):
            raise ValueError('Use finite nonnegative loss weights, at least one positive')
        super().__init__(**kwargs)
        if self.drr_loss_weight:
            raise ValueError('This experiment uses only Sweep v2 and speech reconstruction')
        self.sweep_loss_weight, self.speech_loss_weight = weights

    def select_joint_costs(self, sweep_costs, speech_costs):
        if (sweep_costs.shape != speech_costs.shape or sweep_costs.ndim != 3
                or sweep_costs.shape[1:] != (self.num_spk, self.num_spk)):
            raise ValueError('Expected paired costs [B,predicted speaker,reference speaker]')

        def permutations(costs):
            return torch.stack([
                torch.stack([costs[:, pred, ref] for pred, ref in enumerate(perm)], -1).mean(-1)
                for perm in self.permutations], 0)

        sweep, speech = permutations(sweep_costs), permutations(speech_costs)
        totals = self.sweep_loss_weight * sweep + self.speech_loss_weight * speech
        best = totals.argmin(0)

        def selected(values):
            return values.gather(0, best[None]).mean()

        return selected(totals), best, dict(
            loss_sweep=selected(sweep).detach(),
            loss_speech_rec=selected(speech).detach())

    def paired_joint_loss(self, ctf, direct_rir, reverb_rir, direct_speech, reverb_speech):
        batch, speakers, samples = direct_rir.shape
        excitation = self.response_spectrum(direct_rir.reshape(batch * speakers, samples))
        target = self.response_spectrum(reverb_rir.reshape(batch * speakers, reverb_rir.shape[-1]))
        excitation = excitation[:, 0].reshape(batch, speakers, *excitation.shape[-2:])
        target = target[:, 0].reshape(batch, speakers, *target.shape[-2:])
        sweep_rows, speech_rows = [], []
        for pred in range(speakers):
            sweep_row, speech_row = [], []
            for ref in range(speakers):
                response = self._complex_convolve(excitation[:, ref], ctf[:, pred])
                frames = max(response.shape[-1], target.shape[-1])
                sweep_row.append(self._complex_loss(
                    F.pad(response, (0, frames - response.shape[-1])),
                    F.pad(target[:, ref], (0, frames - target.shape[-1])), reduction='none'))
                # Same speech reconstruction definition as Rec-RIR: retain
                # target utterance frames, with the shared batching policy.
                reconstructed = self._complex_convolve(direct_speech[:, ref], ctf[:, pred])
                speech_row.append(self._complex_loss(
                    reconstructed[..., :reverb_speech.shape[-1]],
                    reverb_speech[:, ref], reduction='none'))
            sweep_rows.append(torch.stack(sweep_row, -1))
            speech_rows.append(torch.stack(speech_row, -1))
        return self.select_joint_costs(
            torch.stack(sweep_rows, 1), torch.stack(speech_rows, 1))

    def forward(self, speech_mix, speech_mix_lengths, rir_ref1, rir_ref2,
                rir_direct1, rir_direct2, speech_direct1, speech_direct2,
                speech_reverb1, speech_reverb2, **kwargs):
        speech = self._to_mono(speech_mix)[:, :int(speech_mix_lengths.max())]
        direct = torch.stack([self._to_mono(x) for x in (rir_direct1, rir_direct2)], 1)
        reverb = torch.stack([self._to_mono(x) for x in (rir_ref1, rir_ref2)], 1)
        for name in ('rir_direct1', 'rir_direct2', 'rir_ref1', 'rir_ref2'):
            lengths = kwargs.get(name + '_lengths')
            if lengths is not None and not torch.all(lengths == direct.shape[-1]):
                raise ValueError('Use fixed-length paired RIRs')
        direct_audio = torch.stack([self._to_mono(x)[:, :speech.shape[-1]]
                                    for x in (speech_direct1, speech_direct2)], 1)
        reverb_audio = torch.stack([self._to_mono(x)[:, :speech.shape[-1]]
                                    for x in (speech_reverb1, speech_reverb2)], 1)
        if direct_audio.shape[-1] != speech.shape[-1] or reverb_audio.shape[-1] != speech.shape[-1]:
            raise ValueError('Speech pairs must share the mixture crop')
        for name in ('speech_direct1', 'speech_direct2', 'speech_reverb1', 'speech_reverb2'):
            lengths = kwargs.get(name + '_lengths')
            if lengths is not None and not torch.equal(lengths, speech_mix_lengths):
                raise ValueError('Speech pair lengths must match mixture lengths')
        if self.normalize_by_mix:
            scale = speech.abs().amax(1, keepdim=True).clamp_min(1e-8)
            speech = speech / scale
            direct_audio = direct_audio / scale[:, None]
            reverb_audio = reverb_audio / scale[:, None]
        spec = self.transforms.stft(speech[:, None], 'complex')
        ctf = self._estimate_ctf_from_complex(spec)
        loss, best, stats = self.paired_joint_loss(ctf, direct, reverb,
            self.transforms.stft(direct_audio, 'complex'),
            self.transforms.stft(reverb_audio, 'complex'))
        stats.update(loss=loss.detach(), pit_perm0_ratio=(best == 0).float().mean())
        return force_gatherable((loss, stats, speech.shape[0]), loss.device)
