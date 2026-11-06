from typing import Any, Dict, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel
from espnet2.rir.rec_rir.espnet_model_pit_length_aware import (
    ESPnetRecRIRPITLengthAwareModel,
)
from espnet2.rir.rec_rir.model_pit_lag_relation import BiSpatialNetPITLagRelation


class ESPnetRecRIRPITLagRelationModel(ESPnetRecRIRPITLengthAwareModel):
    """Lag-relation CTF head with selectable baseline length handling.

    ``length_aware=True`` preserves the padding-free default. Set it to False
    to reuse the original baseline's padded-batch STFT, PIT loss and inference.
    """

    @typechecked
    def __init__(
        self,
        ctf_length: int = 60,
        ctf_block_type: str = "nb",
        relation_checkpoint: bool = True,
        length_aware: bool = True,
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
        self.length_aware = length_aware
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
        inputs = dict(
            speech_mix=speech_mix,
            speech_mix_lengths=speech_mix_lengths,
            speech_direct1=speech_direct1,
            speech_direct1_lengths=speech_direct1_lengths,
            speech_direct2=speech_direct2,
            speech_direct2_lengths=speech_direct2_lengths,
            speech_reverb1=speech_reverb1,
            speech_reverb1_lengths=speech_reverb1_lengths,
            speech_reverb2=speech_reverb2,
            speech_reverb2_lengths=speech_reverb2_lengths,
            **kwargs,
        )
        if self.length_aware:
            return super().forward(**inputs)
        # Keep the original baseline's whole-batch common prefix, including
        # batch padding in feature extraction, lag pooling and all three losses.
        return ESPnetRecRIRPITModel.forward(self, **inputs)

    @torch.no_grad()
    def estimate_ctf(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.length_aware:
            return super().estimate_ctf(speech_mix, speech_mix_lengths)
        return ESPnetRecRIRPITModel.estimate_ctf(
            self, speech_mix, speech_mix_lengths
        )
