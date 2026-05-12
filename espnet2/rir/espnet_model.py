from typing import Dict, List, Tuple

import torch
from typeguard import typechecked

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.rir.predictor.abs_predictor import AbsRIRPredictor
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetRIRModel(AbsESPnetModel):
    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        predictor: AbsRIRPredictor,
        loss_wrappers: List[AbsLossWrapper],
        normalize_variance: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_wrappers = loss_wrappers
        self.normalize_variance = normalize_variance

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

        feature_mix, feature_lengths = self.encoder(speech_mix, speech_lengths)
        rir_pre = self.predictor(feature_mix, feature_lengths)
        rir_ref = self._trim_rir_ref(rir_ref, rir_ref_lengths)
        rir_ref = self._split_rir_ref(rir_ref)

        if len(rir_ref) != len(rir_pre):
            raise ValueError(
                f"Reference/prediction source mismatch: {len(rir_ref)} != "
                f"{len(rir_pre)}"
            )

        loss = speech_mix.new_tensor(0.0)
        stats = {}
        others = {}
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            only_for_test = getattr(criterion, "only_for_test", False)
            if only_for_test and self.training:
                continue
            zero_weight = loss_wrapper.weight == 0.0
            context = torch.no_grad() if zero_weight else _NullContext()
            with context:
                l, s, others = loss_wrapper(rir_ref, rir_pre, others)
            loss = loss + l * loss_wrapper.weight
            stats.update(s)

        if self.training and not loss.requires_grad:
            raise AttributeError(
                "Loss must require gradients during training. Check criterions."
            )
        stats["loss"] = loss.detach()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _trim_rir_ref(
        self, rir_ref: torch.Tensor, rir_ref_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        if rir_ref_lengths is not None:
            rir_ref = rir_ref[:, : rir_ref_lengths.max()]
        if rir_ref.shape[1] != self.predictor.rir_length:
            raise ValueError(
                f"rir_ref length {rir_ref.shape[1]} does not match predictor "
                f"rir_length {self.predictor.rir_length}"
            )
        return rir_ref

    def _split_rir_ref(self, rir_ref: torch.Tensor) -> List[torch.Tensor]:
        num_sources = self.predictor.num_sources
        num_mics = self.predictor.num_mics

        if rir_ref.dim() == 2:
            if num_sources != 1 or num_mics != 1:
                raise ValueError(
                    "2-D rir_ref is valid only for one source and one microphone: "
                    f"num_sources={num_sources}, num_mics={num_mics}"
                )
            return [rir_ref]

        if rir_ref.dim() == 3:
            if num_sources == 1:
                if rir_ref.shape[2] != num_mics:
                    raise ValueError(
                        f"Expected {num_mics} microphones, got {rir_ref.shape[2]}"
                    )
                return [rir_ref]
            if num_mics != 1:
                raise ValueError(
                    "3-D multi-source rir_ref is valid only when num_mics=1"
                )
            if rir_ref.shape[2] != num_sources:
                raise ValueError(
                    f"Expected {num_sources} sources, got {rir_ref.shape[2]}"
                )
            return [rir_ref[:, :, src] for src in range(num_sources)]

        if rir_ref.dim() == 4:
            if rir_ref.shape[2] != num_sources or rir_ref.shape[3] != num_mics:
                raise ValueError(
                    "rir_ref shape must be [B, T, num_sources, num_mics], got "
                    f"{tuple(rir_ref.shape)}"
                )
            return [rir_ref[:, :, src, :] for src in range(num_sources)]

        raise ValueError(f"Unsupported rir_ref shape: {tuple(rir_ref.shape)}")

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        return {"feats": speech_mix, "feats_lengths": speech_mix_lengths}


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
