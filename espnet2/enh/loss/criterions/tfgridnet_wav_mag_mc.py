"""TF-GridNet TASLP 2023, equation (12): Wav+Mag+MC."""

from itertools import permutations

import torch

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class TFGridNetWavMag(TimeDomainLoss):
    """Waveform and re-synthesized magnitude L1, with no RI loss."""

    def __init__(self, n_fft=256, stride=64, window="sqrt_hann"):
        super().__init__("wav_mag_mc")
        self.encoder = STFTEncoder(n_fft, n_fft, stride, window=window)

    def forward(self, ref, inf, ref_mag, inf_mag):
        return (ref - inf).abs().mean(dim=1) + (ref_mag - inf_mag).abs().mean(dim=(1, 2))


class TFGridNetWavMagMCPIT(AbsLossWrapper):
    """Joint utterance PIT plus waveform AND magnitude mixture constraints.

    MC compares sums of direct-path targets/estimates, not the noisy,
    reverberant observed mixture. The speaker costs are summed, as in (12).
    """

    def __init__(self, criterion, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others):
        scale = others["tfgridnet_mix_std"]
        lengths = others["tfgridnet_lengths"]
        if len(ref) != len(inf):
            raise ValueError("Reference and estimate speaker counts differ")
        # DDP uses one utterance per GPU in this recipe. Trim padding as well
        # to keep the loss correct for batched variable-length validation.
        orders = list(permutations(range(len(ref))))
        losses, assignments, pit_values, mc_values = [], [], [], []
        for b, length in enumerate(lengths.tolist()):
            refs = [r[b:b + 1, :length] / scale[b:b + 1] for r in ref]
            estimates = [s[b:b + 1, :length] / scale[b:b + 1] for s in inf]
            lens = lengths[b:b + 1]
            ref_mag = [self.criterion.encoder(r, lens)[0].abs() for r in refs]
            inf_mag = [self.criterion.encoder(s, lens)[0].abs() for s in estimates]
            pairs = [
                [self.criterion(r, s, rm, sm) for s, sm in zip(estimates, inf_mag)]
                for r, rm in zip(refs, ref_mag)
            ]
            costs = torch.stack([
                sum(pairs[i][j] for i, j in enumerate(order)) for order in orders
            ], dim=1)
            best, index = costs.min(dim=1)
            ref_sum, inf_sum = sum(refs), sum(estimates)
            mc = self.criterion(
                ref_sum, inf_sum,
                self.criterion.encoder(ref_sum, lens)[0].abs(),
                self.criterion.encoder(inf_sum, lens)[0].abs(),
            )
            losses.append(best + mc)
            assignments.append(index)
            pit_values.append(best.detach())
            mc_values.append(mc.detach())
        loss = torch.cat(losses).mean()
        perm = torch.tensor(orders, device=loss.device)[torch.cat(assignments)]
        stats = {self.criterion.name: loss.detach(),
                 "wav_mag_pit": torch.cat(pit_values).mean(),
                 "mixture_constraint": torch.cat(mc_values).mean()}
        return loss, stats, {"perm": perm}
