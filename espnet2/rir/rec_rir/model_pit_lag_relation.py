from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from espnet2.rir.rec_rir.model import SpatialNetLayer, SpatialNetLayerNB
from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT


class BiSpatialNetPITLagRelation(BiSpatialNetPIT):
    """Shared two-source features with an explicit causal CTF-lag axis.

    Each input batch must have a common valid frame count, without batch
    padding. The ESPnet wrapper groups waveforms by their actual length before
    STFT. Speaker outputs are introduced only in the final CTF linear layer.
    """

    def __init__(
        self,
        dim_input: int,
        dim_output_spch: int,
        dim_output_CTF: int,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        num_layers_spch: int,
        num_layers_noise: int,
        num_layers_CTF: int,
        encoder_kernel_size: int = 5,
        dropout: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: Sequence[str] = ("LN", "LN", "LN", "LN", "LN", "LN"),
        padding: str = "zeros",
        full_share: int = 0,
        attention: str = "mamba(16,4)",
        num_spk: int = 2,
        ctf_length: int = 60,
        ctf_block_type: str = "nb",
        relation_checkpoint: bool = True,
    ) -> None:
        if ctf_length < 1:
            raise ValueError(f"ctf_length must be positive: {ctf_length}")
        if ctf_block_type not in ("nb", "full"):
            raise ValueError(f"ctf_block_type must be nb or full: {ctf_block_type}")
        if num_spk != 2:
            raise ValueError("The lag-relation model requires num_spk=2")
        if dim_output_CTF != num_spk * 2 * ctf_length:
            raise ValueError(
                "Internal dim_output_CTF must equal num_spk * 2 * ctf_length"
            )
        if encoder_kernel_size < 1 or encoder_kernel_size % 2 != 1:
            raise ValueError("encoder_kernel_size must be positive and odd")

        # Reuse the speech trunk and both auxiliary decoders. The baseline CTF
        # modules are replaced below; do not construct its temporal Mamba stack.
        super().__init__(
            dim_input=dim_input,
            dim_output_spch=dim_output_spch,
            dim_output_CTF=dim_output_CTF,
            dim_hidden=dim_hidden,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs,
            num_layers_spch=num_layers_spch,
            num_layers_noise=num_layers_noise,
            num_layers_CTF=0,
            encoder_kernel_size=encoder_kernel_size,
            dropout=dropout,
            kernel_size=kernel_size,
            conv_groups=conv_groups,
            norms=list(norms),
            padding=padding,
            full_share=full_share,
            attention=attention,
            num_spk=num_spk,
        )
        del self.compress_CTF
        del self.weight_layer
        self.ctf_length = ctf_length
        self.ctf_block_type = ctf_block_type
        self.relation_checkpoint = relation_checkpoint
        self.dim_input = dim_input
        self.num_freqs = num_freqs
        self.clean_proj = nn.Linear(dim_hidden, dim_hidden)
        self.reverb_proj = nn.Linear(dim_hidden, dim_hidden)
        self.relation_mlp = nn.Sequential(
            nn.Linear(3 * dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        block_class = SpatialNetLayerNB if ctf_block_type == "nb" else SpatialNetLayer
        # Full-band weights may be shared within this stack, never with speech.
        full = None
        ctf_layers = []
        for layer_idx in range(num_layers_CTF):
            layer = block_class(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=list(norms),
                padding=padding,
                full=full if layer_idx > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            ctf_layers.append(layer)
        self.ctf_layers = nn.ModuleList(ctf_layers)
        self.decoder_CTF = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, num_spk * 2),
        )

    def _pool_one_lag(self, clean: Tensor, reverb: Tensor) -> Tensor:
        relation = self.relation_mlp(
            torch.cat((clean, reverb, clean * reverb), dim=-1)
        )
        # Accumulate in FP32 under AMP, then restore the feature dtype for the
        # following blocks. Checkpoint the entire operation, including pooling.
        return relation.mean(dim=2, dtype=torch.float32).to(dtype=relation.dtype)

    def _pool_lag_sequence(self, clean: Tensor, reverb: Tensor) -> Tensor:
        q = self.clean_proj(clean)
        k = self.reverb_proj(reverb)
        frames = q.shape[2]
        use_checkpoint = (
            self.relation_checkpoint and self.training and torch.is_grad_enabled()
        )
        pooled = []
        for lag in range(self.ctf_length):
            clean_at_t = q[:, :, : frames - lag, :]
            reverb_at_t_plus_lag = k[:, :, lag:, :]
            if use_checkpoint:
                # Pass the actual slices; do not close over the loop variable.
                value = checkpoint(
                    self._pool_one_lag,
                    clean_at_t,
                    reverb_at_t_plus_lag,
                    use_reentrant=False,
                )
            else:
                value = self._pool_one_lag(clean_at_t, reverb_at_t_plus_lag)
            pooled.append(value)
        return torch.stack(pooled, dim=2)  # [B,F,N,D]

    def forward(self, input: Tensor, return_embedding: bool = False):
        if input.ndim != 4 or input.shape[1:3] != (
            self.dim_input,
            self.num_freqs,
        ):
            raise ValueError(
                f"Expected [B,{self.dim_input},{self.num_freqs},T], "
                f"got {tuple(input.shape)}"
            )
        if input.shape[-1] < self.ctf_length:
            raise ValueError(
                f"Need at least {self.ctf_length} valid STFT frames for "
                f"ctf_length={self.ctf_length}; got {input.shape[-1]}"
            )
        input_pad = torch.nn.functional.pad(
            input, (self.padding_size[1], self.padding_size[1], 0, 0)
        )
        x = self.encoder(input_pad).permute(0, 2, 3, 1)
        for module in self.noise_layers:
            x = module(x)
        reverb = x
        y_rev = self._decode_speech_like(self.decoder_rev(reverb))
        for module in self.spch_layers:
            x = module(x)
        y_spch = self._decode_speech_like(self.decoder_spch(x))

        x_ctf = self._pool_lag_sequence(x, reverb)
        for module in self.ctf_layers:
            x_ctf = module(x_ctf)
        if return_embedding:
            return x_ctf  # [B,F,N,D], shared between speaker outputs
        batch, freq, lags, _ = x_ctf.shape
        y_ctf = self.decoder_CTF(x_ctf).reshape(batch, freq, lags, self.num_spk, 2)
        # Channels are [speaker 1 real, speaker 1 imag, speaker 2 real, imag].
        # Keep causal tap order here; the inherited PIM adapter reverses it once.
        y_ctf = y_ctf.permute(0, 3, 4, 1, 2).contiguous()
        return y_spch, y_ctf, y_rev
