"""Spatial Encoder model module for ESPnet."""

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
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for Spatial Encoder model.

        Args:
            speech_mix: (Batch, samples) or (Batch, 1, samples) - SC (single-channel) mixture
                This will be used as anchor for contrastive learning.
            speech_mix_lengths: (Batch,), default None
            kwargs:
                - speech_mix_mc: (Batch, num_channels, samples) - MC (multi-channel) mixture for positive
                - speech_mix_reverse_mc: (Batch, num_channels, samples) - MC reversed mixture for negative
                - embedding_anchor: [B, E] - anchor embeddings (optional, if provided, use this instead of computing from speech_mix)
                - embedding_pos: [B, E] - positive embeddings (optional, if provided, use this instead of computing from speech_mix_mc)
                - embedding_neg: [B, E] - negative embeddings (optional, if provided, use this instead of computing from speech_mix_reverse)

        Returns:
            loss: Total loss tensor
            stats: Dictionary of statistics
            weight: Batch weight
        """
        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size, dtype=torch.long, device=speech_mix.device).fill_(
                speech_mix.shape[-1]
            )
        )
        
        # Determine if speech_mix is SC (1ch) or MC (multi-ch)
        # speech_mix: [B, T] or [B, 1, T] -> SC
        # speech_mix: [B, C, T] (C > 1) -> MC
        if speech_mix.ndim == 2:
            # [B, T] -> SC
            is_sc = True
        elif speech_mix.ndim == 3:
            if speech_mix.shape[1] == 1:
                # [B, 1, T] -> SC
                is_sc = True
                speech_mix = speech_mix.squeeze(1)  # [B, 1, T] -> [B, T]
            else:
                # [B, C, T] (C > 1) -> MC
                is_sc = False
        else:
            raise ValueError(f"Unexpected speech_mix shape: {speech_mix.shape}")
        
        # Get num_channels_mc from spatial_encoder if available
        num_channels_mc = getattr(self.spatial_encoder, "num_channels_mc", 2)
        
        # Extract embeddings for contrastive learning
        # Anchor: SC (mix)
        # Positive: MC (mix)
        # Negative: MC (reverse)
        
        # Get sampling frequency if available
        fs = kwargs.get("utt2fs", None)
        if fs is not None:
            # All samples must have the same sampling rate
            fs = fs[0].item()
            assert all([fs == f.item() for f in kwargs["utt2fs"]])
        else:
            fs = None
        
        # for data-parallel
        speech_mix = speech_mix[:, : speech_lengths.max()]
        if "speech_mix_mc" in kwargs:
            speech_mix_mc = kwargs["speech_mix_mc"]  # [B, T, C]
            speech_mix_mc = speech_mix_mc[:, : speech_lengths.max(), :]
            kwargs["speech_mix_mc"] = speech_mix_mc
        if "speech_mix_reverse_mc" in kwargs:
            speech_mix_reverse_mc = kwargs["speech_mix_reverse_mc"]  # [B, T, C]
            speech_mix_reverse_mc = speech_mix_reverse_mc[:, : speech_lengths.max(), :]
            kwargs["speech_mix_reverse_mc"] = speech_mix_reverse_mc
        
        if "embedding_anchor" in kwargs:
            # Use provided anchor embedding
            embedding_anchor = kwargs["embedding_anchor"]
        else:
            # Compute anchor from SC (mix)
            # speech_mix should be SC at this point
            if not is_sc:
                raise ValueError(
                    "speech_mix must be SC (1ch) when computing anchor embedding. "
                    "If speech_mix is MC, provide embedding_anchor via kwargs."
                )
            # Encode waveform to spectrum
            feature_mix, flens = self.encoder(speech_mix, speech_lengths, fs=fs)  # [B, T, F] (complex)
            # Explicitly specify num_channels=1 for SC
            embedding_anchor = self.spatial_encoder(feature_mix, flens, num_channels=1)  # [B, embed_dim]
        
        # Positive: MC (mix)
        if "embedding_pos" in kwargs:
            embedding_pos = kwargs["embedding_pos"]
        elif "speech_mix_mc" in kwargs:
            speech_mix_mc = kwargs["speech_mix_mc"]  # [B, T, C]
            # Encode waveform to spectrum
            feature_mc, flens_mc = self.encoder(speech_mix_mc, speech_lengths, fs=fs)  # [B, T, C, F] (complex)
            # Explicitly specify num_channels=num_channels_mc for MC
            embedding_pos = self.spatial_encoder(feature_mc, flens_mc, num_channels=num_channels_mc)  # [B, embed_dim]
        else:
            embedding_pos = None
        
        # Negative: MC (reverse)
        if "embedding_neg" in kwargs:
            embedding_neg = kwargs["embedding_neg"]
        elif "speech_mix_reverse_mc" in kwargs:
            speech_mix_reverse_mc = kwargs["speech_mix_reverse_mc"]  # [B, T, C]
            # Encode waveform to spectrum
            feature_reverse_mc, flens_reverse_mc = self.encoder(speech_mix_reverse_mc, speech_lengths, fs=fs)  # [B, T, C, F] (complex)
            # Explicitly specify num_channels=num_channels_mc for MC
            embedding_neg = self.spatial_encoder(feature_reverse_mc, flens_reverse_mc, num_channels=num_channels_mc)  # [B, embed_dim]
        else:
            embedding_neg = None
        
        # Loss computation
        loss, stats, weight = self.forward_loss(
            embedding_anchor,
            embedding_pos,
            embedding_neg,
            speech_lengths,
            **kwargs,
        )
        
        return loss, stats, weight

    def forward_loss(
        self,
        embedding_anchor: torch.Tensor,
        embedding_pos: Optional[torch.Tensor],
        embedding_neg: Optional[torch.Tensor],
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
        
        # For contrastive learning, we need anchor, positive, and negative embeddings
        # If embedding_pos and embedding_neg are provided, use them directly
        # Otherwise, they should be provided via kwargs (from teacher model)
        if embedding_pos is None and "embedding_pos" in kwargs:
            embedding_pos = kwargs["embedding_pos"]
        if embedding_neg is None and "embedding_neg" in kwargs:
            embedding_neg = kwargs["embedding_neg"]
        
            # Check if we have all required embeddings
        if embedding_pos is None or embedding_neg is None:
            raise ValueError(
                "Both embedding_pos and embedding_neg are required for contrastive learning. "
                "Please provide them via kwargs or speech_mix_mc/speech_mix_reverse_mc."
            )
        
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            only_for_test = getattr(criterion, "only_for_test", False)
            if only_for_test and self.training:
                continue
            
            # Compute loss using the wrapper
            l, s, o = loss_wrapper(
                embedding_anchor,
                embedding_pos,
                embedding_neg,
                speech_lengths,
            )
            
            loss += l * loss_wrapper.weight
            stats.update(s)
        
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
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features for statistics.

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_mix_lengths: (Batch,)
            kwargs: Additional arguments

        Returns:
            Dictionary containing features
        """
        # for data-parallel
        if speech_mix.ndim == 2:
            # [B, T]
            speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        elif speech_mix.ndim == 3:
            # [B, C, T]
            speech_mix = speech_mix[:, :, : speech_mix_lengths.max()]
        
        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
