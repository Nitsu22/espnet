"""Loss wrappers for Spatial Encoder."""

from espnet2.enh_se.loss.wrappers.abs_wrapper import AbsLossWrapper

try:
    from espnet2.enh_se.loss.wrappers.contrastive_loss_wrapper import (
        ContrastiveLossWrapper,
        TripletLossWrapper,
    )

    __all__ = [
        "AbsLossWrapper",
        "ContrastiveLossWrapper",
        "TripletLossWrapper",
    ]
except ImportError:
    __all__ = [
        "AbsLossWrapper",
    ]
