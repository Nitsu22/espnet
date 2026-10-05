"""Roland Koudaisai separation objective with the DOA term removed."""

from itertools import permutations

import torch

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class RolandTFL1(TimeDomainLoss):
    """Waveform L1 + RI L1 + magnitude L1 in mixture-normalized units."""

    def __init__(self, n_fft=256, stride=64, window="hann"):
        super().__init__("roland_tf_wav_mc")
        self.encoder = STFTEncoder(n_fft, n_fft, stride, window=window)

    def forward(self, ref, inf, ref_tf, inf_tf):
        wav = (ref - inf).abs().mean(dim=1)
        spec = (
            (ref_tf.real - inf_tf.real).abs()
            + (ref_tf.imag - inf_tf.imag).abs()
            + (ref_tf.abs() - inf_tf.abs()).abs()
        ).mean(dim=(1, 2))
        return wav + spec


class RolandTFMCPIT(AbsLossWrapper):
    """Speaker-mean PIT + unaveraged sum-signal MC, as in Roland's solver.

    The external weight preserves the original separation coefficient (0.95).
    No DOA labels or predictions enter either the objective or the permutation.
    This recipe uses fixed-size chunks, including during validation, like the
    original chunk iterator. Raw predicted spectra must not be re-analyzed.
    """

    def __init__(self, criterion, weight=0.95):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others):
        scale = others["tfgridnet_mix_std"]
        lengths = others["tfgridnet_lengths"]
        refs = [r / scale for r in ref]
        estimates = [s / scale for s in inf]
        ref_tf = [self.criterion.encoder(r, lengths)[0] for r in refs]
        inf_tf = others["tfgridnet_spectra"]
        if len(refs) != len(estimates):
            raise ValueError("Speaker counts differ")
        orders = list(permutations(range(len(refs))))
        costs = torch.stack([
            sum(self.criterion(refs[i], estimates[j], ref_tf[i], inf_tf[j])
                for i, j in enumerate(order)) / len(refs)
            for order in orders
        ], dim=1)
        pit, index = costs.min(dim=1)
        mc = self.criterion(sum(refs), sum(estimates), sum(ref_tf), sum(inf_tf))
        loss = (pit + mc).mean()
        perm = torch.tensor(orders, device=loss.device)[index]
        return loss, {
            self.criterion.name: loss.detach(),
            "roland_pit": pit.detach().mean(),
            "roland_mc": mc.detach().mean(),
        }, {"perm": perm}
