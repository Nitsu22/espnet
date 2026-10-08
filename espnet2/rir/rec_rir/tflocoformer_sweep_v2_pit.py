"""Two-source full direct-sweep CTF supervision with utterance-level PIT."""
import math

import torch
import torch.nn.functional as F

from espnet2.rir.rec_rir.tflocoformer_ctf_pit import ESPnetRecRIRTFLocoformerPITModel
from espnet2.rir.rec_rir.espnet_model_sweep_v2 import ESPnetRecRIRSweepV2Model
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetTFLocoformerSweepV2PITModel(ESPnetRecRIRTFLocoformerPITModel):
    def __init__(self, drr_loss_weight=0.0, drr_direct_window_ms=2.5,
                 drr_huber_beta=1.0, drr_energy_floor=1e-8, **kwargs):
        settings = (drr_loss_weight, drr_direct_window_ms,
                    drr_huber_beta, drr_energy_floor)
        if (not all(math.isfinite(value) for value in settings)
                or drr_loss_weight < 0 or drr_direct_window_ms < 0
                or drr_huber_beta <= 0 or not 0 < drr_energy_floor < 1):
            raise ValueError('Invalid DRR loss weight, window, Huber beta or energy floor')
        super().__init__(**kwargs)
        if self.rt60_auxiliary or self.reconstruction_signal != 'speech':
            raise ValueError('Use the basic CTF predictor; sweep v2 supplies its own loss')
        self.register_buffer('sweep', self.pim.sinesweep.clone(), persistent=False)
        self.drr_loss_weight = float(drr_loss_weight)
        self.drr_direct_window_ms = float(drr_direct_window_ms)
        self.drr_huber_beta = float(drr_huber_beta)
        self.drr_energy_floor = float(drr_energy_floor)
        if self.drr_loss_weight:
            inverse = self.pim.invfilter.clone()
            gain = torch.dot(self.sweep.double(), inverse.flip(0).double())
            if not torch.isfinite(gain) or gain.abs() < 1e-8:
                raise ValueError('Invalid sweep/inverse calibration gain')
            self.register_buffer('inverse_sweep', inverse / gain.float(), persistent=False)
            self.inverse_delay = self.sweep.numel() - 1 + self.transforms.n_fft

    response_spectrum = ESPnetRecRIRSweepV2Model.response_spectrum
    response_to_rir = ESPnetRecRIRSweepV2Model.response_to_rir

    def drr_db(self, rir, centers):
        """Differentiable stabilized DRR with teacher-defined, fixed windows.

        The numerator is the energy within +/-drr_direct_window_ms of the
        teacher peak. The denominator includes ALL other samples, including
        the pre-peak region. No student argmax, peak shift, gain fit or detach.
        The floor scales with total energy to preserve gain/polarity invariance.
        """
        if rir.ndim != 2 or centers.shape != rir.shape[:1]:
            raise ValueError('Expected RIR [B, samples] and centers [B]')
        if torch.any(centers < 0) or torch.any(centers >= rir.shape[-1]):
            raise ValueError('DRR center is outside the RIR window')
        half = int(round(self.sr * self.drr_direct_window_ms / 1000))
        positions = torch.arange(rir.shape[-1], device=rir.device)
        mask = (positions[None] - centers[:, None]).abs() <= half
        energy = rir.float().square()
        direct = energy.masked_fill(~mask, 0).sum(-1)
        reverb = energy.masked_fill(mask, 0).sum(-1)
        # Subtract logs instead of dividing: the quotient's backward can
        # underflow its squared denominator for a zero/very quiet response.
        # The minimal floor also bounds log derivatives within float32.
        floor = ((direct + reverb) * self.drr_energy_floor).clamp_min(
            16 * torch.finfo(energy.dtype).tiny)
        return 10 * (torch.log10(direct + floor) - torch.log10(reverb + floor))

    def paired_pit_loss(self, ctf, direct, reverb):
        loss, best, _ = self.paired_pit_loss_with_stats(ctf, direct, reverb)
        return loss, best

    def paired_pit_loss_with_stats(self, ctf, direct, reverb):
        batch, speakers, samples = direct.shape
        excitation = self.response_spectrum(direct.reshape(batch * speakers, samples))
        target = self.response_spectrum(reverb.reshape(batch * speakers, reverb.shape[-1]))
        excitation = excitation[:, 0].reshape(batch, speakers, *excitation.shape[-2:])
        target = target[:, 0].reshape(batch, speakers, *target.shape[-2:])
        if self.drr_loss_weight:
            # Both sides include the SAME original-sweep measurement/inverse
            # response. A raw physical RIR is not a band-limited PIM reference.
            with torch.no_grad():
                recovered_target = self.response_to_rir(
                    target.reshape(batch * speakers, 1, *target.shape[-2:]),
                    reverb.shape[-1])[:, 0].reshape(batch, speakers, -1)
                centers = reverb.abs().argmax(-1)
                target_drr = self.drr_db(
                    recovered_target.reshape(batch * speakers, -1),
                    centers.reshape(-1)).reshape(batch, speakers)
        costs, drr_costs, drr_errors = [], [], []
        for pred in range(speakers):
            row, drr_row, error_row = [], [], []
            for ref in range(speakers):
                # BOTH the excitation and response belong to the reference
                # speaker; permuting only target responses is incorrect.
                output = self._complex_convolve(excitation[:, ref], ctf[:, pred])
                frames = max(output.shape[-1], target.shape[-1])
                row.append(self._complex_loss(
                    F.pad(output, (0, frames-output.shape[-1])),
                    F.pad(target[:, ref], (0, frames-target.shape[-1])),
                    reduction='none'))
                if self.drr_loss_weight:
                    # Restore the clean-to-reverb basis from the predicted
                    # DIRECT-sweep response, using the ORIGINAL sweep inverse.
                    recovered = self.response_to_rir(output[:, None], reverb.shape[-1])[:, 0]
                    predicted_drr = self.drr_db(recovered, centers[:, ref])
                    drr_row.append(F.smooth_l1_loss(
                        predicted_drr, target_drr[:, ref], reduction='none',
                        beta=self.drr_huber_beta))
                    error_row.append((predicted_drr - target_drr[:, ref]).abs())
            costs.append(row)
            drr_costs.append(drr_row)
            drr_errors.append(error_row)
        def permute_costs(values):
            return torch.stack([sum(values[p][r] for p, r in enumerate(perm)) / speakers
                                for perm in self.permutations])
        sweep_totals = permute_costs(costs)
        totals = sweep_totals
        if self.drr_loss_weight:
            drr_totals = permute_costs(drr_costs)
            totals = sweep_totals + self.drr_loss_weight * drr_totals
        best = totals.argmin(0)
        def select(values):
            return values.gather(0, best[None]).mean()
        stats = dict(loss_sweep=select(sweep_totals).detach())
        if self.drr_loss_weight:
            stats.update(loss_drr=select(drr_totals).detach(),
                         drr_mae_db=select(permute_costs(drr_errors)).detach())
        return select(totals), best, stats

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
        loss, best, stats = self.paired_pit_loss_with_stats(ctf, direct, reverb)
        stats.update(loss=loss.detach(), pit_perm0_ratio=(best == 0).float().mean())
        return force_gatherable((loss, stats, speech.shape[0]), loss.device)
