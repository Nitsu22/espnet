from itertools import permutations
from typing import Callable, Dict, List, Optional, Tuple

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.criterions.sepreformer import SepReformerLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class SepReformerPITSolver(AbsLossWrapper):
    """SepReformer PIT loss wrapper.

    Expects SepReformer training output in the form
    ``[aux_layer_1, ..., aux_layer_N, final]`` where each layer is a speaker
    list. The final layer receives the time-domain loss, while auxiliary layers
    receive the STFT-magnitude loss.
    """

    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight: float = 1.0,
        alpha: float = 0.4,
        independent_perm: bool = True,
        flexible_numspk: bool = False,
    ):
        super().__init__()
        if not isinstance(criterion, SepReformerLoss):
            raise TypeError(
                "SepReformerPITSolver requires SepReformerLoss, but got "
                f"{type(criterion)}."
            )
        self.criterion = criterion
        self.weight = weight
        self.alpha = alpha
        self.independent_perm = independent_perm
        self.flexible_numspk = flexible_numspk

    def forward(
        self,
        ref: List[torch.Tensor],
        infs: List,
        others: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Compute final time PIT loss plus auxiliary magnitude PIT loss."""
        if others is None:
            others = {}
        if not infs:
            raise ValueError("SepReformerPITSolver received empty inference list.")

        if isinstance(infs[0], (list, tuple)):
            aux_layers = list(infs[:-1])
            final = list(infs[-1])
        else:
            aux_layers = []
            final = list(infs)

        if self.flexible_numspk:
            num_spk = len(final)
        else:
            assert len(ref) == len(final), (len(ref), len(final))
            num_spk = len(ref)

        time_loss, perm = self._pit_loss(
            ref,
            final,
            num_spk,
            self.criterion.time_pair_loss,
            others.get("perm"),
        )

        mag_losses = []
        for aux in aux_layers:
            assert len(aux) == num_spk, (len(aux), num_spk)
            mag_loss, _ = self._pit_loss(
                ref,
                list(aux),
                num_spk,
                self.criterion.mag_pair_loss,
                None,
            )
            mag_losses.append(mag_loss)

        if mag_losses:
            mag_loss = torch.stack(mag_losses).mean()
            loss = (1.0 - self.alpha) * time_loss + self.alpha * mag_loss
        else:
            mag_loss = torch.full_like(time_loss.detach(), float("nan"))
            loss = time_loss

        stats = {
            self.criterion.name: loss.detach(),
            self.criterion.name + "_time": time_loss.detach(),
            self.criterion.name + "_mag": mag_loss.detach(),
        }
        for idx, layer_loss in enumerate(mag_losses):
            stats[self.criterion.name + f"_mag{idx + 1}"] = layer_loss.detach()

        return loss, stats, {"perm": perm}

    def _pit_loss(
        self,
        ref: List[torch.Tensor],
        inf: List[torch.Tensor],
        num_spk: int,
        pair_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.independent_perm or perm is None:
            return self._independent_pit_loss(ref, inf, num_spk, pair_loss)
        return self._fixed_perm_pit_loss(ref, inf, num_spk, pair_loss, perm)

    def _independent_pit_loss(
        self,
        ref: List[torch.Tensor],
        inf: List[torch.Tensor],
        num_spk: int,
        pair_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = ref[0].device
        all_permutations = list(permutations(range(num_spk)))

        def loss_for_perm(permutation):
            return sum(
                pair_loss(ref[s], inf[t]) for s, t in enumerate(permutation)
            ) / len(permutation)

        losses = torch.stack([loss_for_perm(p) for p in all_permutations], dim=1)
        loss, perm_idx = torch.min(losses, dim=1)
        perm = torch.index_select(
            torch.tensor(all_permutations, device=device, dtype=torch.long),
            0,
            perm_idx,
        )
        return loss.mean(), perm

    def _fixed_perm_pit_loss(
        self,
        ref: List[torch.Tensor],
        inf: List[torch.Tensor],
        num_spk: int,
        pair_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_losses = []
        for batch, p in enumerate(perm):
            spk_losses = [
                pair_loss(
                    ref[s][batch].unsqueeze(0),
                    inf[int(t)][batch].unsqueeze(0),
                )
                for s, t in enumerate(p[:num_spk])
            ]
            batch_losses.append(torch.stack(spk_losses).mean())
        return torch.stack(batch_losses).mean(), perm
