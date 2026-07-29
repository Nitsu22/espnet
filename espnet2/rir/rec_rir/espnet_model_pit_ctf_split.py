from typing import Any

from typeguard import typechecked

from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel
from espnet2.rir.rec_rir.model_pit_ctf_split import BiSpatialNetPITCTFSplit


class ESPnetRecRIRPITCTFSplitModel(ESPnetRecRIRPITModel):
    """Two-source Rec-RIR wrapper with a speaker-split CTF path."""

    rec_rir_network_class = BiSpatialNetPITCTFSplit

    @typechecked
    def __init__(self, dim_output_CTF: int = 120, **kwargs: Any):
        super().__init__(dim_output_CTF=dim_output_CTF, **kwargs)
