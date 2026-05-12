from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from espnet2.enh.separator.fla_tflocoformer_separator import (
    TFLocoformerSeparator as FLATFLocoformerSeparator,
)
from espnet2.enh.separator.tflocoformer_separator import TFLocoformerSeparator
from espnet2.rir.predictor.abs_predictor import AbsRIRPredictor


class TFLocoformerRIRPredictor(AbsRIRPredictor):
    """Predict fixed-length time-domain RIRs from speech STFT features."""

    separator_class = TFLocoformerSeparator

    def __init__(
        self,
        input_dim: int,
        rir_length: int,
        num_sources: int = 1,
        num_mics: int = 1,
        n_layers: int = 4,
        emb_dim: int = 96,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,
        tf_order: str = "ft",
        n_heads: int = 4,
        flash_attention: bool = False,
        attention_dim: int = 128,
        pos_enc: str = "rope",
        ffn_type: Union[str, list] = ("swiglu_conv1d", "swiglu_conv1d"),
        ffn_hidden_dim: Union[int, list] = (128, 128),
        conv1d_kernel: int = 8,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
        pooling: str = "mean",
        hidden_dim: int = 256,
        separator_conf: Optional[Dict] = None,
    ):
        super().__init__()
        if rir_length is None or rir_length <= 0:
            raise ValueError("rir_length must be a positive integer")
        if num_sources <= 0:
            raise ValueError("num_sources must be a positive integer")
        if num_mics <= 0:
            raise ValueError("num_mics must be a positive integer")
        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {pooling}")

        self._rir_length = int(rir_length)
        self._num_sources = int(num_sources)
        self._num_mics = int(num_mics)
        self.pooling = pooling

        separator_conf = {} if separator_conf is None else separator_conf
        self.tf_locoformer = self.separator_class(
            input_dim=input_dim,
            num_spk=1,
            n_layers=n_layers,
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            tf_order=tf_order,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            pos_enc=pos_enc,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
            **separator_conf,
        )
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                self._num_sources * self._num_mics * self._rir_length,
            ),
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> List[torch.Tensor]:
        features, flens, _ = self.tf_locoformer(input, ilens)
        feature = features[0]
        real = feature.real
        imag = feature.imag
        feat = torch.cat([real, imag], dim=-1)

        mask = torch.arange(feat.size(1), device=feat.device)[None, :] < flens[:, None]
        mask = mask.to(dtype=feat.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        pooled = (feat * mask.unsqueeze(-1)).sum(dim=1) / denom

        rir = self.proj(pooled)
        rir = rir.view(
            -1,
            self._num_sources,
            self._num_mics,
            self._rir_length,
        )

        outputs = []
        for source in range(self._num_sources):
            source_rir = rir[:, source]
            if self._num_mics == 1:
                outputs.append(source_rir[:, 0])
            else:
                outputs.append(source_rir.transpose(1, 2))
        return outputs

    @property
    def num_sources(self) -> int:
        return self._num_sources

    @property
    def num_mics(self) -> int:
        return self._num_mics

    @property
    def rir_length(self) -> int:
        return self._rir_length


class FLATFLocoformerRIRPredictor(TFLocoformerRIRPredictor):
    separator_class = FLATFLocoformerSeparator

    def __init__(
        self,
        *args,
        temporal_attn_type: str = "gated_fla",
        fla_p: int = 3,
        fla_dwc_kernel: int = 7,
        fla_use_rope: bool = False,
        **kwargs,
    ):
        separator_conf = kwargs.pop("separator_conf", {})
        separator_conf.update(
            dict(
                temporal_attn_type=temporal_attn_type,
                fla_p=fla_p,
                fla_dwc_kernel=fla_dwc_kernel,
                fla_use_rope=fla_use_rope,
            )
        )
        super().__init__(*args, separator_conf=separator_conf, **kwargs)
