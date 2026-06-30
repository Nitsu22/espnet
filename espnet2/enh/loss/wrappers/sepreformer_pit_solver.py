from itertools import permutations
from typing import Callable, Dict, List, Optional, Tuple

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class SepReformerPITSolver(AbsLossWrapper):
    """PIT wrapper for SepReformer's multi-layer output.

    The wrapper keeps the criterion generic and only selects which layer(s) of
    ``[aux_layer_1, ..., aux_layer_N, final]`` should receive that criterion.
    This lets the config reuse ESPnet's existing ``si_snr`` criterion for the
    final waveform and use a SepReformer-specific magnitude criterion only for
    auxiliary outputs.
    """

    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight: float = 1.0,
        layer: str = "final",
        alpha: Optional[float] = None,
        alpha_role: Optional[str] = None,
        alpha_decay_start_epoch: int = 100,
        alpha_decay_interval: int = 5,
        alpha_decay_rate: float = 0.8,
        independent_perm: bool = True,
        flexible_numspk: bool = False,
    ):
        super().__init__()
        if layer not in ("final", "aux"):
            raise ValueError(f"Unsupported SepReformer loss layer: {layer}")
        if alpha_role not in (None, "final", "aux"):
            raise ValueError(f"Unsupported SepReformer alpha role: {alpha_role}")
        self.criterion = criterion
        self._weight = weight
        self.layer = layer
        self.alpha = alpha
        self.alpha_role = alpha_role
        self.alpha_decay_start_epoch = alpha_decay_start_epoch
        self.alpha_decay_interval = alpha_decay_interval
        self.alpha_decay_rate = alpha_decay_rate
        self.current_epoch = None
        self.independent_perm = independent_perm
        self.flexible_numspk = flexible_numspk

    @property
    def weight(self) -> float:
        if self.alpha_role is None:
            return self._weight

        if not torch.is_grad_enabled():
            return 1.0 if self.alpha_role == "final" else 0.0

        alpha = self.current_alpha
        if self.alpha_role == "final":
            return 1.0 - alpha
        return alpha

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    @property
    def current_alpha(self) -> float:
        alpha = self.alpha if self.alpha is not None else self._weight
        if self.current_epoch is None:
            return alpha
        if self.current_epoch > self.alpha_decay_start_epoch:
            exponent = 1 + (
                (self.current_epoch - self.alpha_decay_start_epoch - 1)
                // self.alpha_decay_interval
            )
            alpha = alpha * (self.alpha_decay_rate**exponent)
        return alpha

    def set_epoch(self, epoch: Optional[int]) -> None:
        self.current_epoch = None if epoch is None else int(epoch)

    def forward(
        self,
        ref: List[torch.Tensor],
        infs: List,
        others: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Compute PIT loss on the configured SepReformer output layer."""
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

        if self.layer == "aux":
            return self._aux_loss(ref, aux_layers, others)

        return self._final_loss(ref, final, others)

    def _final_loss(
        self,
        ref: List[torch.Tensor],
        final: List[torch.Tensor],
        others: Dict,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        if self.flexible_numspk:
            num_spk = len(final)
        else:
            assert len(ref) == len(final), (len(ref), len(final))
            num_spk = len(ref)

        loss, perm = self._pit_loss(
            ref,
            final,
            num_spk,
            self.criterion,
            others.get("perm"),
        )
        stats = {
            self.criterion.name: loss.detach(),
            self.criterion.name + "_weight": ref[0].new_tensor(self.weight),
        }
        if self.alpha_role is not None:
            stats[self.criterion.name + "_alpha"] = ref[0].new_tensor(
                self.current_alpha
            )
        return loss, stats, {"perm": perm}

    def _aux_loss(
        self,
        ref: List[torch.Tensor],
        aux_layers: List[List[torch.Tensor]],
        others: Dict,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        del others
        if not aux_layers:
            if torch.is_grad_enabled():
                raise ValueError(
                    "SepReformer auxiliary loss requires multi-layer outputs. "
                    "Please enable `output_aux` in the separator config."
                )
            loss = ref[0].new_zeros(())
            stats = {
                self.criterion.name: torch.full_like(loss, float("nan")),
                self.criterion.name + "_weight": ref[0].new_tensor(self.weight),
            }
            if self.alpha_role is not None:
                stats[self.criterion.name + "_alpha"] = ref[0].new_tensor(
                    self.current_alpha
                )
            return loss, stats, {}

        if self.flexible_numspk:
            num_spk = len(aux_layers[0])
        else:
            assert len(ref) == len(aux_layers[0]), (len(ref), len(aux_layers[0]))
            num_spk = len(ref)

        losses = []
        for aux in aux_layers:
            assert len(aux) == num_spk, (len(aux), num_spk)
            layer_loss, _ = self._pit_loss(
                ref,
                list(aux),
                num_spk,
                self.criterion,
                None,
            )
            losses.append(layer_loss)

        loss = torch.stack(losses).mean()
        stats = {
            self.criterion.name: loss.detach(),
            self.criterion.name + "_weight": ref[0].new_tensor(self.weight),
        }
        if self.alpha_role is not None:
            stats[self.criterion.name + "_alpha"] = ref[0].new_tensor(
                self.current_alpha
            )
        for idx, layer_loss in enumerate(losses):
            stats[self.criterion.name + f"_layer{idx + 1}"] = layer_loss.detach()

        return loss, stats, {}

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
