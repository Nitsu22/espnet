"""Contrastive loss for Spatial Encoder training."""

from typing import Optional

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.enh_se.loss.criterions.abs_loss import AbsSELoss

EPS = torch.finfo(torch.get_default_dtype()).eps


class PairwiseNegativeLoss(AbsSELoss):
    """Pairwise negative loss for contrastive learning.
    
    This loss encourages anchor embeddings to be similar to positive embeddings
    and dissimilar to negative embeddings.
    """

    @property
    def name(self) -> str:
        return "pairwise_negative_loss"

    @typechecked
    def __init__(
        self,
        temperature: Optional[float] = None,
        eps: float = 1e-8,
    ):
        """Initialize PairwiseNegativeLoss.

        Args:
            temperature: Temperature parameter for scaled cross-entropy.
                        If None, use logistic form: -log(sigmoid(s_pos - s_neg))
                        If specified, use temperature-scaled cross-entropy form.
            eps: Small value for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise negative loss.

        Args:
            anchor_emb: [B, E] - anchor embeddings (L2 normalized)
            pos_emb: [B, E] - positive embeddings (L2 normalized)
            neg_emb: [B, E] - negative embeddings (L2 normalized)

        Returns:
            loss: [B] - loss per sample
        """
        # Cosine similarity (L2 normalized, so dot product is cosine similarity)
        s_pos = torch.sum(anchor_emb * pos_emb, dim=1)  # [B]
        s_neg = torch.sum(anchor_emb * neg_emb, dim=1)  # [B]

        if self.temperature is None:
            # Logistic form: -log(sigmoid(s_pos - s_neg))
            loss = -torch.log(torch.sigmoid(s_pos - s_neg) + self.eps)  # [B]
        else:
            # Temperature-scaled cross-entropy form
            logits = torch.stack([s_pos, s_neg], dim=1) / self.temperature  # [B, 2]
            labels = torch.zeros(
                anchor_emb.shape[0], dtype=torch.long, device=anchor_emb.device
            )
            loss = F.cross_entropy(logits, labels, reduction="none")  # [B]

        return loss

