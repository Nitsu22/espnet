"""Loss criterions for Spatial Encoder."""

from espnet2.enh_se.loss.criterions.abs_loss import AbsSELoss

try:
    from espnet2.enh_se.loss.criterions.contrastive_loss import (
        PairwiseNegativeLoss,
        TripletLoss,
    )

    __all__ = [
        "AbsSELoss",
        "PairwiseNegativeLoss",
        "TripletLoss",
    ]
except ImportError:
    __all__ = [
        "AbsSELoss",
    ]
