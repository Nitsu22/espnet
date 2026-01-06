"""Loss criterions for Spatial Encoder."""

from espnet2.enh_se.loss.criterions.abs_loss import AbsSELoss

try:
    from espnet2.enh_se.loss.criterions.contrastive_loss import PairwiseNegativeLoss

    __all__ = [
        "AbsSELoss",
        "PairwiseNegativeLoss",
    ]
except ImportError:
    __all__ = [
        "AbsSELoss",
    ]

