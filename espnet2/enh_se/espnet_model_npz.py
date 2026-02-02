"""Spatial Encoder model module for ESPnet."""

import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from typeguard import typechecked

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

# Type hint for forward reference
if TYPE_CHECKING:
    from espnet2.enh.encoder.abs_encoder import AbsEncoder
    from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import (
        AbsSpatialEncoder,
    )


class ESPnetSpatialEncoderModel(AbsESPnetModel):
    """Spatial Encoder model for contrastive learning.
    
    This model supports both single-channel (student) and multi-channel (teacher) inputs.
    """

    @typechecked
    def __init__(
        self,
        encoder: "AbsEncoder",
        spatial_encoder: "AbsSpatialEncoder",
        loss_wrappers: Optional[List] = None,
    ):
        """Initialize Spatial Encoder model.

        Args:
            encoder: Waveform encoder that converts waveforms to feature representations
            spatial_encoder: Spatial encoder module (e.g., ResNet2DSpatialEncoder)
            loss_wrappers: List of loss wrappers for training
        """
        super().__init__()
        
        self.encoder = encoder
        self.spatial_encoder = spatial_encoder
        
        self.loss_wrappers = loss_wrappers
        if self.loss_wrappers is not None:
            names = [w.criterion.name for w in self.loss_wrappers]
            if len(set(names)) != len(names):
                raise ValueError(
                    "Duplicated loss names are not allowed: {}".format(names)
                )

    def forward(
        self,
        speech_anchor: torch.Tensor,
        speech_anchor_lengths: Optional[torch.Tensor] = None,
        speech_pos: Optional[torch.Tensor] = None,
        speech_pos_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for Spatial Encoder model with contrastive inputs.

        Args:
            speech_anchor: (Batch, T, C) anchor waveform
            speech_anchor_lengths: (Batch,)
            speech_pos: (Batch, T, C) positive waveform
            speech_pos_lengths: (Batch,)
            kwargs:
                - speech_neg1, speech_neg2, ...: (Batch, T, C) negative waveforms
                - utt2fs: sampling rate information

        Returns:
            loss: Total loss tensor
            stats: Dictionary of statistics
            weight: Batch weight
        """
        if speech_pos is None:
            raise ValueError("speech_pos is required for contrastive learning.")

        batch_size = speech_anchor.shape[0]
        speech_lengths = (
            speech_anchor_lengths
            if speech_anchor_lengths is not None
            else torch.ones(
                batch_size, dtype=torch.long, device=speech_anchor.device
            ).fill_(speech_anchor.shape[1])
        )

        # Get sampling frequency if available
        fs = kwargs.get("utt2fs", None)
        if fs is not None:
            fs = fs[0].item()
            assert all([fs == f.item() for f in kwargs["utt2fs"]])
        else:
            fs = None

        # for data-parallel: trim to max length
        max_len = speech_lengths.max()
        speech_anchor = speech_anchor[:, :max_len, :]
        speech_pos = speech_pos[:, :max_len, :]

        neg_keys = sorted(
            [k for k in kwargs if re.fullmatch(r"speech_neg\d+", k)],
            key=lambda x: int(x.replace("speech_neg", "")),
        )
        if len(neg_keys) == 0:
            raise ValueError("At least one speech_neg* is required.")
        speech_negs = []
        for k in neg_keys:
            neg = kwargs[k][:, :max_len, :]
            speech_negs.append(neg)

        num_channels_mc = getattr(self.spatial_encoder, "num_channels_mc", 2)

        # Encode anchor
        feature_anchor, flens = self.encoder(speech_anchor, speech_lengths, fs=fs)
        embedding_anchor = self.spatial_encoder(
            feature_anchor, flens, num_channels=num_channels_mc
        )

        # Encode positive
        feature_pos, flens_pos = self.encoder(speech_pos, speech_lengths, fs=fs)
        embedding_pos = self.spatial_encoder(
            feature_pos, flens_pos, num_channels=num_channels_mc
        )

        # Encode negatives
        embedding_negs = []
        for neg in speech_negs:
            feature_neg, flens_neg = self.encoder(neg, speech_lengths, fs=fs)
            embedding_neg = self.spatial_encoder(
                feature_neg, flens_neg, num_channels=num_channels_mc
            )
            embedding_negs.append(embedding_neg)

        # Loss computation (average over negatives)
        loss, stats, weight = self.forward_loss(
            embedding_anchor,
            embedding_pos,
            embedding_negs,
            speech_lengths,
            **kwargs,
        )

        return loss, stats, weight

    def forward_loss(
        self,
        embedding_anchor: torch.Tensor,
        embedding_pos: Optional[torch.Tensor],
        embedding_negs: List[torch.Tensor],
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute loss.

        Args:
            embedding_anchor: [B, embed_dim] - anchor embeddings (student or teacher positive)
            embedding_pos: [B, embed_dim] - positive embeddings (teacher positive, optional)
            embedding_neg: [B, embed_dim] - negative embeddings (teacher negative, optional)
            speech_lengths: [B] - sequence lengths
            kwargs: Additional arguments

        Returns:
            loss: Total loss tensor
            stats: Dictionary of statistics
            weight: Batch weight
        """
        loss = embedding_anchor.new_tensor(0.0)
        stats = {}
        
        if embedding_pos is None:
            raise ValueError("embedding_pos is required for contrastive learning.")
        if not embedding_negs:
            raise ValueError("At least one negative embedding is required.")

        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            only_for_test = getattr(criterion, "only_for_test", False)
            if only_for_test and self.training:
                continue

            loss_sum = embedding_anchor.new_tensor(0.0)
            stats_sum = {}
            for embedding_neg in embedding_negs:
                l, s, _ = loss_wrapper(
                    embedding_anchor,
                    embedding_pos,
                    embedding_neg,
                    speech_lengths,
                )
                loss_sum += l
                for k, v in s.items():
                    stats_sum[k] = stats_sum.get(k, 0.0) + v
            loss_sum = loss_sum / len(embedding_negs)
            for k in stats_sum:
                stats_sum[k] = stats_sum[k] / len(embedding_negs)

            loss += loss_sum * loss_wrapper.weight
            stats.update(stats_sum)
        
        if self.training and not loss.requires_grad:
            raise AttributeError(
                "Loss must be a tensor with gradient in the training mode."
            )
        
        stats["loss"] = loss.detach()
        
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = embedding_anchor.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech_anchor: torch.Tensor,
        speech_anchor_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features for statistics.

        Args:
            speech_anchor: (Batch, samples) or (Batch, samples, channels)
            speech_anchor_lengths: (Batch,)
            kwargs: Additional arguments

        Returns:
            Dictionary containing features
        """
        if speech_anchor.ndim == 2:
            speech_anchor = speech_anchor[:, : speech_anchor_lengths.max()]
        elif speech_anchor.ndim == 3:
            speech_anchor = speech_anchor[:, :, : speech_anchor_lengths.max()]

        feats, feats_lengths = speech_anchor, speech_anchor_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
