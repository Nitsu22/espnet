from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel
from espnet2.rir.rec_rir.model_pit_lag_relation import BiSpatialNetPITLagRelation
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetRecRIRPITLagRelationModel(ESPnetRecRIRPITModel):
    """Lag-relation CTF estimation with the original three-loss speaker PIT.

    Actual waveform lengths are grouped before STFT and feature extraction:
    masking only the pooling would leave padding in the bidirectional trunk.
    Losses remain utterance means, including when length groups differ in size.
    """

    @typechecked
    def __init__(
        self,
        ctf_length: int = 60,
        ctf_block_type: str = "nb",
        relation_checkpoint: bool = True,
        num_spk: int = 2,
        **kwargs: Any,
    ) -> None:
        if isinstance(ctf_length, bool) or ctf_length < 1:
            raise ValueError("ctf_length must be a positive integer")
        if ctf_block_type not in ("nb", "full"):
            raise ValueError("ctf_block_type must be nb or full")
        if "dim_output_CTF" in kwargs:
            raise ValueError(
                "Use ctf_length for rec_rir_pit_lag_relation; the final CTF "
                "linear layer always outputs num_spk * 2 values per lag"
            )
        # Plain configuration attributes are needed by the network factory
        # invoked from the parent constructor; modules are created by it.
        self.ctf_length = ctf_length
        self.ctf_block_type = ctf_block_type
        self.relation_checkpoint = relation_checkpoint
        super().__init__(
            num_spk=num_spk,
            dim_output_CTF=num_spk * 2 * ctf_length,
            **kwargs,
        )

    def _build_rec_rir_network(self, **network_conf):
        return BiSpatialNetPITLagRelation(
            **network_conf,
            ctf_length=self.ctf_length,
            ctf_block_type=self.ctf_block_type,
            relation_checkpoint=self.relation_checkpoint,
        )

    @staticmethod
    def _valid_lengths(
        speech: torch.Tensor, lengths: Optional[torch.Tensor], name: str
    ) -> torch.Tensor:
        if speech.shape[0] == 0:
            raise ValueError("Empty batches are not supported")
        if lengths is None:
            return torch.full(
                (speech.shape[0],),
                speech.shape[1],
                dtype=torch.long,
                device=speech.device,
            )
        if lengths.ndim != 1 or lengths.shape[0] != speech.shape[0]:
            raise ValueError(f"{name}_lengths must have shape [B]")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name}_lengths must contain integer sample counts")
        lengths = lengths.to(device=speech.device, dtype=torch.long)
        if torch.any(lengths <= 0) or torch.any(lengths > speech.shape[1]):
            raise ValueError(
                f"{name}_lengths must be in [1, {speech.shape[1]}], "
                f"got {lengths.tolist()}"
            )
        return lengths

    def _check_min_length(self, lengths: torch.Tensor) -> None:
        # torch.stft uses center=True with n_fft//2 reflect padding. For odd
        # FFT sizes this adds n_fft-1 samples, rather than n_fft samples.
        n_fft = self.transforms.n_fft
        hop_len = self.transforms.hop_len
        frames = (lengths - n_fft % 2) // hop_len + 1
        min_samples = max(
            n_fft // 2 + 1, (self.ctf_length - 1) * hop_len + n_fft % 2
        )
        invalid = (frames < self.ctf_length) | (lengths < min_samples)
        if torch.any(invalid):
            raise ValueError(
                f"Need at least {self.ctf_length} valid STFT frames and enough "
                f"samples for reflect padding: at least {min_samples} samples "
                f"with n_fft={n_fft}, hop_len={hop_len}; "
                f"got sample lengths {lengths[invalid].tolist()}"
            )

    @staticmethod
    def _length_groups(
        lengths: torch.Tensor,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        for length in torch.unique(lengths, sorted=True).tolist():
            indices = torch.nonzero(lengths == length, as_tuple=False).flatten()
            yield int(length), indices

    def _common_lengths(
        self,
        signals: Sequence[torch.Tensor],
        lengths: Sequence[Optional[torch.Tensor]],
        names: Sequence[str],
    ) -> torch.Tensor:
        batch_size = signals[0].shape[0]
        if any(signal.shape[0] != batch_size for signal in signals):
            raise ValueError(
                "Mixture and all four speech references must have the same batch size"
            )
        valid = [
            self._valid_lengths(signal, length, name)
            for signal, length, name in zip(signals, lengths, names)
        ]
        # Keep the baseline's common-prefix convention, applied per utterance.
        common = torch.stack(valid, dim=0).amin(dim=0)
        self._check_min_length(common)
        return common

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        speech_direct1: torch.Tensor,
        speech_direct1_lengths: Optional[torch.Tensor] = None,
        speech_direct2: Optional[torch.Tensor] = None,
        speech_direct2_lengths: Optional[torch.Tensor] = None,
        speech_reverb1: Optional[torch.Tensor] = None,
        speech_reverb1_lengths: Optional[torch.Tensor] = None,
        speech_reverb2: Optional[torch.Tensor] = None,
        speech_reverb2_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if speech_direct2 is None or speech_reverb1 is None or speech_reverb2 is None:
            raise ValueError(
                "speech_direct1/2 and speech_reverb1/2 are required for "
                "Rec-RIR lag-relation PIT training"
            )
        names = (
            "speech_mix",
            "speech_direct1",
            "speech_direct2",
            "speech_reverb1",
            "speech_reverb2",
        )
        signals = [
            self._to_mono(signal)
            for signal in (
                speech_mix,
                speech_direct1,
                speech_direct2,
                speech_reverb1,
                speech_reverb2,
            )
        ]
        lengths = self._common_lengths(
            signals,
            (
                speech_mix_lengths,
                speech_direct1_lengths,
                speech_direct2_lengths,
                speech_reverb1_lengths,
                speech_reverb2_lengths,
            ),
            names,
        )
        batch_size = signals[0].shape[0]
        losses = []
        stats = {}
        for length, indices in self._length_groups(lengths):
            group = {
                name: signal.index_select(0, indices)[:, :length].contiguous()
                for name, signal in zip(names, signals)
            }
            group_lengths = lengths.index_select(0, indices)
            for name in names:
                group[f"{name}_lengths"] = group_lengths
            # The inherited loss selects one speaker permutation jointly for
            # clean, reverb and reference-direct/CTF reconstruction per item.
            loss, group_stats, _ = super().forward(**group)
            fraction = indices.numel() / batch_size
            losses.append(loss * fraction)
            for name, value in group_stats.items():
                contribution = value * fraction
                stats[name] = stats.get(name, 0.0) + contribution
        loss = torch.stack(losses, dim=0).sum(dim=0)
        return force_gatherable((loss, stats, batch_size), loss.device)

    @torch.no_grad()
    def estimate_ctf(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        was_1d = speech_mix.ndim == 1
        if was_1d:
            speech_mix = speech_mix.unsqueeze(0)
        speech_mix = self._to_mono(speech_mix)
        lengths = self._valid_lengths(speech_mix, speech_mix_lengths, "speech_mix")
        self._check_min_length(lengths)
        chunks = []
        order = []
        for length, indices in self._length_groups(lengths):
            speech = speech_mix.index_select(0, indices)[:, :length].contiguous()
            # Reuse normalization, complex conversion and the single tap flip
            # expected by the existing PIM's sliding-window multiplication.
            chunks.append(super().estimate_ctf(speech))
            order.append(indices)
        ctf = torch.cat(chunks, dim=0).index_select(0, torch.cat(order).argsort())
        return ctf[0] if was_1d else ctf
