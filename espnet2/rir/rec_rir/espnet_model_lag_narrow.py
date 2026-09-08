from espnet2.rir.rec_rir.espnet_model import ESPnetRecRIRModel
from espnet2.rir.rec_rir.model_lag_narrow import LagAwareBiSpatialNet


class ESPnetRecRIRLagNarrowModel(ESPnetRecRIRModel):
    """ESPnet wrapper for the single-source lag-aware narrow-band Rec-RIR."""

    rec_rir_network_class = LagAwareBiSpatialNet
