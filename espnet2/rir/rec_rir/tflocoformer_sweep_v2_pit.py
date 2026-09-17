"""Two-source full direct-sweep CTF supervision with utterance-level PIT."""
import torch
import torch.nn.functional as F

from espnet2.rir.rec_rir.tflocoformer_ctf_pit import ESPnetRecRIRTFLocoformerPITModel
from espnet2.rir.rec_rir.espnet_model_sweep_v2 import ESPnetRecRIRSweepV2Model
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetTFLocoformerSweepV2PITModel(ESPnetRecRIRTFLocoformerPITModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.rt60_auxiliary or self.reconstruction_signal != 'speech':
            raise ValueError('Use the basic CTF predictor; sweep v2 supplies its own loss')
        self.register_buffer('sweep', self.pim.sinesweep.clone(), persistent=False)

    response_spectrum = ESPnetRecRIRSweepV2Model.response_spectrum

    def paired_pit_loss(self, ctf, direct, reverb):
        batch, speakers, samples = direct.shape
        excitation = self.response_spectrum(direct.reshape(batch * speakers, samples))
        target = self.response_spectrum(reverb.reshape(batch * speakers, reverb.shape[-1]))
        excitation = excitation[:, 0].reshape(batch, speakers, *excitation.shape[-2:])
        target = target[:, 0].reshape(batch, speakers, *target.shape[-2:])
        costs = []
        for pred in range(speakers):
            row = []
            for ref in range(speakers):
                # BOTH the excitation and response belong to the reference
                # speaker; permuting only target responses is incorrect.
                output = self._complex_convolve(excitation[:, ref], ctf[:, pred])
                frames = max(output.shape[-1], target.shape[-1])
                row.append(self._complex_loss(
                    F.pad(output, (0, frames-output.shape[-1])),
                    F.pad(target[:, ref], (0, frames-target.shape[-1])),
                    reduction='none'))
            costs.append(row)
        totals = torch.stack([sum(costs[p][r] for p, r in enumerate(perm)) / speakers
                              for perm in self.permutations])
        best = totals.argmin(0)
        return totals.gather(0, best[None]).mean(), best

    def forward(self, speech_mix, speech_mix_lengths, rir_ref1, rir_ref2,
                rir_direct1, rir_direct2, **kwargs):
        speech = self._to_mono(speech_mix)[:, :int(speech_mix_lengths.max())]
        if self.normalize_by_mix:
            speech = speech / speech.abs().amax(1, keepdim=True).clamp_min(1e-8)
        direct = torch.stack([self._to_mono(x) for x in (rir_direct1, rir_direct2)], 1)
        reverb = torch.stack([self._to_mono(x) for x in (rir_ref1, rir_ref2)], 1)
        for name in ('rir_ref1', 'rir_ref2', 'rir_direct1', 'rir_direct2'):
            lengths = kwargs.get(name + '_lengths')
            if lengths is not None and not torch.all(lengths == direct.shape[-1]):
                raise ValueError('PIT sweep v2 requires fixed-length paired RIRs')
        spec = self.transforms.stft(speech[:, None], 'complex')
        ctf, _ = self._estimate_ctf_and_aux_from_complex(spec)
        loss, best = self.paired_pit_loss(ctf, direct, reverb)
        return force_gatherable((loss, dict(loss=loss.detach(), loss_sweep=loss.detach(),
                                          pit_perm0_ratio=(best == 0).float().mean()),
                                speech.shape[0]), loss.device)
