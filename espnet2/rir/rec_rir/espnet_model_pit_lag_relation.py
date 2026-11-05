from typing import Any

from typeguard import typechecked

from espnet2.rir.rec_rir.espnet_model_pit_length_aware import (
    ESPnetRecRIRPITLengthAwareModel,
)
from espnet2.rir.rec_rir.model_pit_lag_relation import BiSpatialNetPITLagRelation


class ESPnetRecRIRPITLagRelationModel(ESPnetRecRIRPITLengthAwareModel):
    """Lag-relation CTF head with the common padding-free PIT wrapper."""

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
