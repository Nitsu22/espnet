"""Loss wrapper for contrastive loss."""

from typing import Dict, Tuple

import torch

from espnet2.enh_se.loss.criterions.abs_loss import AbsSELoss
from espnet2.enh_se.loss.criterions.contrastive_loss import (
    PairwiseNegativeLoss,
    TripletLoss,
)
from espnet2.enh_se.loss.wrappers.abs_wrapper import AbsLossWrapper


class ContrastiveLossWrapper(AbsLossWrapper):
    """Wrapper for contrastive loss (PairwiseNegativeLoss).
    
    This wrapper adapts the contrastive loss to work with ESPnet's loss framework.
    """

    def __init__(
        self,
        criterion: AbsSELoss,
        weight: float = 1.0,
    ):
        """Initialize ContrastiveLossWrapper.

        Args:
            criterion: Loss criterion (should be PairwiseNegativeLoss)
            weight: Weight for the loss in multi-task learning
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(
        self,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """Forward pass for contrastive loss wrapper.

        Args:
            anchor_emb: [B, E] - anchor embeddings
            pos_emb: [B, E] - positive embeddings
            neg_emb: [B, E] - negative embeddings
            speech_lengths: [B] - sequence lengths (not used, kept for compatibility)

        Returns:
            loss: Scalar loss tensor
            stats: Dictionary of statistics
            others: Empty dict (for compatibility with AbsLossWrapper interface)
        """
        # Compute loss
        loss = self.criterion(anchor_emb, pos_emb, neg_emb)  # [B]
        loss = loss.mean()  # Scalar loss
        
        # Compute statistics
        stats = {
            self.criterion.name: loss.detach(),
        }
        
        # Compute pairwise accuracy (for monitoring)
        with torch.no_grad():
            s_pos = torch.sum(anchor_emb * pos_emb, dim=1)  # [B]
            s_neg = torch.sum(anchor_emb * neg_emb, dim=1)  # [B]
            acc = (s_pos > s_neg).float().mean().detach()
            stats[f"{self.criterion.name}_accuracy"] = acc
        
        others = {}  # No additional outputs for now
        
        return loss, stats, others


class TripletLossWrapper(AbsLossWrapper):
    """Wrapper for triplet loss."""

    def __init__(
        self,
        criterion: AbsSELoss,
        weight: float = 1.0,
    ):
        """Initialize TripletLossWrapper.

        Args:
            criterion: Loss criterion (should be TripletLoss)
            weight: Weight for the loss in multi-task learning
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(
        self,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """Forward pass for triplet loss wrapper.

        Args:
            anchor_emb: [B, E] - anchor embeddings
            pos_emb: [B, E] - positive embeddings
            neg_emb: [B, E] - negative embeddings
            speech_lengths: [B] - sequence lengths (not used, kept for compatibility)

        Returns:
            loss: Scalar loss tensor
            stats: Dictionary of statistics
            others: Empty dict (for compatibility with AbsLossWrapper interface)
        """
        loss = self.criterion(anchor_emb, pos_emb, neg_emb)  # [B]
        loss = loss.mean()

        stats = {
            self.criterion.name: loss.detach(),
        }

        # Compute triplet accuracy (d_pos + margin < d_neg)
        if isinstance(self.criterion, TripletLoss):
            with torch.no_grad():
                anchor = torch.nn.functional.normalize(
                    anchor_emb, p=2, dim=1, eps=self.criterion.eps
                )
                pos = torch.nn.functional.normalize(
                    pos_emb, p=2, dim=1, eps=self.criterion.eps
                )
                neg = torch.nn.functional.normalize(
                    neg_emb, p=2, dim=1, eps=self.criterion.eps
                )
                d_pos = torch.linalg.vector_norm(anchor - pos, ord=2, dim=1)
                d_neg = torch.linalg.vector_norm(anchor - neg, ord=2, dim=1)
                acc = (d_pos + self.criterion.margin < d_neg).float().mean().detach()
                stats[f"{self.criterion.name}_accuracy"] = acc

        others = {}
        return loss, stats, others
