"""Loss wrappers for Spatial Encoder."""

from espnet2.enh_se.loss.wrappers.abs_wrapper import AbsLossWrapper

try:
    from espnet2.enh_se.loss.wrappers.contrastive_loss_wrapper import (
        ContrastiveLossWrapper,
    )

    __all__ = [
        "AbsLossWrapper",
        "ContrastiveLossWrapper",
    ]
except ImportError:
    __all__ = [
        "AbsLossWrapper",
    ]

