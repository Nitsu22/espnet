"""TF-Locoformer Small-compatible offline temporal bidirectional Gated DeltaNet."""

from collections import OrderedDict

import torch
from torch.nn import functional as F

from espnet2.enh.layers.bidirectional_gated_deltanet import BidirectionalGatedDeltaNet
from espnet2.enh.separator.tflocoformer_separator_nocashe import (
    TFLocoformerSeparator as NocacheTFLocoformerSeparator,
)


class TFLocoformerBiGatedDeltaNetSeparator(NocacheTFLocoformerSeparator):
    """Replace temporal MHSA only; retain the nocache frequency path and FFNs.

    Small refers to the backbone dimensions, not an exact parameter-count match.
    GDN uses separate forward/backward parameters and no temporal RoPE.
    Unequal-length utterances are processed individually to keep padding out of
    reverse recurrence, FFNs, and the backbone's global normalization.
    """

    def __init__(
        self, input_dim, num_spk=2, n_layers=4, emb_dim=96,
        norm_type="rmsgroupnorm", num_groups=4, tf_order="ft",
        n_heads=4, flash_attention=False, attention_dim=128, pos_enc="rope",
        ffn_type=("swiglu_conv1d", "swiglu_conv1d"),
        ffn_hidden_dim=(128, 128), conv1d_kernel=8, conv1d_shift=1,
        dropout=0.0, eps=1e-5, gdn_heads=4, gdn_head_dim=32,
        gdn_value_dim=64, gdn_conv_size=4, gdn_backend="recurrent",
    ):
        super().__init__(
            input_dim=input_dim, num_spk=num_spk, n_layers=n_layers,
            emb_dim=emb_dim, norm_type=norm_type, num_groups=num_groups,
            tf_order=tf_order, n_heads=n_heads, flash_attention=flash_attention,
            attention_dim=attention_dim, pos_enc=pos_enc,
            ffn_type=list(ffn_type) if isinstance(ffn_type, tuple) else ffn_type,
            ffn_hidden_dim=(
                list(ffn_hidden_dim)
                if isinstance(ffn_hidden_dim, tuple) else ffn_hidden_dim
            ),
            conv1d_kernel=conv1d_kernel, conv1d_shift=conv1d_shift,
            dropout=dropout, eps=eps,
        )
        for block in self.blocks:
            block.frame_path.attn = BidirectionalGatedDeltaNet(
                emb_dim, heads=gdn_heads, head_dim=gdn_head_dim,
                value_dim=gdn_value_dim, conv_size=gdn_conv_size,
                dropout=dropout, eps=eps, backend=gdn_backend,
            )

    def forward(self, input, ilens, additional=None):
        # ESPnet STFT: [B,T,F] or [B,T,C,F].
        if input.ndim == 4:
            if input.shape[2] != 1:
                raise ValueError("Only single-channel input is supported")
            input = input[:, :, 0]
        if input.ndim != 3:
            raise ValueError("Expected STFT input [B,T,F] or [B,T,1,F]")
        if ilens.ndim != 1 or ilens.shape[0] != input.shape[0]:
            raise ValueError("ilens must contain one frame length per utterance")
        if ilens.is_floating_point() or ilens.dtype == torch.bool:
            raise ValueError("Frame lengths must be integers")
        lengths = ilens.tolist()
        if any(n <= 0 or n > input.shape[1] for n in lengths):
            raise ValueError("Frame lengths must be positive and at most input T")
        if all(n == input.shape[1] for n in lengths):
            return super().forward(input, ilens, additional)
        outputs = [[] for _ in range(self.num_spk)]
        for b, n in enumerate(lengths):
            separated, _, _ = super().forward(
                input[b:b + 1, :n], ilens[b:b + 1], additional
            )
            for s, estimate in enumerate(separated):
                outputs[s].append(F.pad(estimate, (0, 0, 0, input.shape[1] - n)))
        return [torch.cat(x, dim=0) for x in outputs], ilens, OrderedDict()
