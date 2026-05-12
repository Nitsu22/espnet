import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetRIRModel(AbsESPnetModel):
    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        separator: AbsSeparator,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        normalize_variance: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.loss_wrappers = loss_wrappers
        self.normalize_variance = normalize_variance
        self.num_spk = separator.num_spk

        if len(self.loss_wrappers) == 0:
            raise ValueError("At least one loss wrapper is required")
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicated loss names are not allowed: {names}")

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        rir_ref: torch.Tensor,
        rir_ref_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = speech_mix.shape[0]
        speech_lengths = speech_mix_lengths
        if speech_lengths is None:
            speech_lengths = torch.full(
                (batch_size,),
                speech_mix.shape[1],
                dtype=torch.long,
                device=speech_mix.device,
            )
        speech_mix = speech_mix[:, : speech_lengths.max()]

        if self.normalize_variance:
            dims = (1, 2) if speech_mix.ndim > 2 else 1
            scale = speech_mix.std(dim=dims, keepdim=True).clamp_min(1.0e-8)
            speech_mix = speech_mix / scale

        rir_ref, rir_lengths = self._trim_rir_ref(rir_ref, rir_ref_lengths)
        rir_pre, feature_mix, feature_pre, others = self.forward_rir(
            speech_mix,
            speech_lengths,
            rir_lengths.to(device=speech_lengths.device),
        )

        target_length = int(rir_ref.shape[1])
        rir_pre = [self._match_length(r, target_length) for r in rir_pre]
        rir_ref = self._split_rir_ref(rir_ref)

        if len(rir_ref) != len(rir_pre):
            raise ValueError(
                f"Reference/prediction source mismatch: {len(rir_ref)} != "
                f"{len(rir_pre)}"
            )

        loss, stats, weight = self.forward_loss(rir_ref, rir_pre, others)
        return loss, stats, weight

    def forward_rir(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        decode_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], Dict]:
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        feature_pre, flens, others = self.separator(feature_mix, flens, additional)
        rir_pre = [self.decoder(ps, decode_lengths)[0] for ps in feature_pre]
        return rir_pre, feature_mix, feature_pre, others

    def forward_loss(
        self,
        rir_ref: List[torch.Tensor],
        rir_pre: List[torch.Tensor],
        others: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        loss = rir_ref[0].new_tensor(0.0)
        stats = {}
        o = {}
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            only_for_test = getattr(criterion, "only_for_test", False)
            if only_for_test and self.training:
                continue
            if not isinstance(criterion, TimeDomainLoss):
                raise NotImplementedError(
                    f"Unsupported RIR loss type: {type(criterion).__name__}"
                )
            zero_weight = loss_wrapper.weight == 0.0
            with torch.no_grad() if zero_weight else contextlib.ExitStack():
                l, s, o = loss_wrapper(rir_ref, rir_pre, {**others, **o})
            loss = loss + l * loss_wrapper.weight
            stats.update(s)

        if self.training and not loss.requires_grad:
            raise AttributeError(
                "Loss must require gradients during training. Check criterions."
            )
        stats["loss"] = loss.detach()
        batch_size = rir_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _trim_rir_ref(
        self, rir_ref: torch.Tensor, rir_ref_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rir_ref_lengths is None:
            rir_ref_lengths = torch.full(
                (rir_ref.shape[0],),
                rir_ref.shape[1],
                dtype=torch.long,
                device=rir_ref.device,
            )
        target_length = int(rir_ref_lengths.max().item())
        rir_ref = rir_ref[:, :target_length]
        return rir_ref, rir_ref_lengths

    def _split_rir_ref(self, rir_ref: torch.Tensor) -> List[torch.Tensor]:
        if rir_ref.dim() == 2:
            if self.num_spk != 1:
                raise ValueError(
                    "2-D rir_ref is valid only for one source: "
                    f"num_spk={self.num_spk}"
                )
            return [rir_ref]

        if rir_ref.dim() == 3:
            if rir_ref.shape[2] != self.num_spk:
                raise ValueError(
                    f"Expected {self.num_spk} RIR sources, got {rir_ref.shape[2]}"
                )
            return [rir_ref[:, :, src] for src in range(self.num_spk)]

        if rir_ref.dim() == 4:
            if rir_ref.shape[2] != self.num_spk:
                raise ValueError(
                    f"Expected {self.num_spk} RIR sources, got {rir_ref.shape[2]}"
                )
            if rir_ref.shape[3] != 1:
                raise ValueError(
                    "The current enh-style RIR model supports one output channel, "
                    f"got {rir_ref.shape[3]} microphones"
                )
            return [rir_ref[:, :, src, 0] for src in range(self.num_spk)]

        raise ValueError(f"Unsupported rir_ref shape: {tuple(rir_ref.shape)}")

    @staticmethod
    def _match_length(signal: torch.Tensor, target_length: int) -> torch.Tensor:
        if signal.shape[1] > target_length:
            return signal[:, :target_length]
        if signal.shape[1] < target_length:
            return F.pad(signal, (0, target_length - signal.shape[1]))
        return signal

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        return {"feats": speech_mix, "feats_lengths": speech_mix_lengths}
