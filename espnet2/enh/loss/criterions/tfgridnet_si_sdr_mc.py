"""TF-GridNet TASLP 2023 equations (9)-(10), scaling the estimate."""

from itertools import permutations

import torch

from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class TFGridNetSISDRSE(TimeDomainLoss):
    """Equation (9), without mean subtraction or reference-side projection.

    Epsilon floors protect silent references/estimates and perfect reconstruction.
    Alpha remains in the autograd graph, including in the mixture constraint.
    """

    def __init__(self, eps=1e-8):
        super().__init__("si_sdr_se_mc")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def project(self, ref, inf):
        alpha = (ref * inf).sum(-1, keepdim=True) / inf.square().sum(
            -1, keepdim=True
        ).clamp_min(self.eps)
        return alpha * inf

    def forward(self, ref, inf):
        residual = self.project(ref, inf) - ref
        return 10 * (
            residual.square().sum(-1).clamp_min(self.eps).log10()
            - ref.square().sum(-1).clamp_min(self.eps).log10()
        )


class TFGridNetSISDRMCPIT(AbsLossWrapper):
    """Joint PIT over speaker-summed SI-SDR-SE plus scaled-estimate MC.

    Inputs arrive in the original waveform units. Both references and predictions
    are divided by the same input-mixture sample standard deviation. Padding is
    excluded. MC targets the sum of direct-path references, never the observation.
    """

    def __init__(self, criterion, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others):
        if not ref or len(ref) != len(inf):
            raise ValueError("Reference and estimate speaker counts must match")
        scale = others["tfgridnet_mix_std"]
        lengths = others["tfgridnet_lengths"]
        orders = list(permutations(range(len(ref))))
        losses, permutations_out, se_values, mc_values = [], [], [], []
        for b, length in enumerate(lengths.tolist()):
            if length <= 0:
                raise ValueError("Waveform lengths must be positive")
            refs = torch.stack([r[b, :length] for r in ref]) / scale[b]
            estimates = torch.stack([s[b, :length] for s in inf]) / scale[b]
            se_costs, mc_costs = [], []
            for order in orders:
                ordered = estimates[list(order)]
                se_costs.append(self.criterion(refs, ordered).sum())
                projected = self.criterion.project(refs, ordered)
                mc_costs.append((projected.sum(0) - refs.sum(0)).abs().mean())
            se_costs, mc_costs = torch.stack(se_costs), torch.stack(mc_costs)
            best, index = (se_costs + mc_costs).min(0)
            losses.append(best)
            permutations_out.append(index)
            se_values.append(se_costs[index].detach())
            mc_values.append(mc_costs[index].detach())
        loss = torch.stack(losses).mean()
        perm = torch.tensor(orders, device=loss.device)[torch.stack(permutations_out)]
        stats = {
            self.criterion.name: loss.detach(),
            "si_sdr_se_loss": torch.stack(se_values).mean(),
            "mixture_constraint": torch.stack(mc_values).mean(),
        }
        return loss, stats, {"perm": perm}
