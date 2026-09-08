import itertools
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding
from typeguard import typechecked

from espnet2.enh.separator.tflocoformer_separator import TFLocoformerBlock
from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.pim import RecRIRPIM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class TFLocoformerCTFPredictor(nn.Module):
    """TF-Locoformer encoder with a two-source CTF output head."""

    def __init__(
        self,
        input_dim: int = 129,
        num_spk: int = 2,
        ctf_taps: int = 60,
        n_layers: int = 4,
        emb_dim: int = 96,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,
        tf_order: str = "ft",
        n_heads: int = 4,
        flash_attention: bool = False,
        attention_dim: int = 128,
        pos_enc: str = "rope",
        ffn_type: Optional[Union[str, Sequence[str]]] = None,
        ffn_hidden_dim: Optional[Union[int, Sequence[int]]] = None,
        conv1d_kernel: int = 8,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"
        self._num_spk = int(num_spk)
        self.ctf_taps = int(ctf_taps)
        self.input_dim = int(input_dim)
        self.n_layers = int(n_layers)
        if self._num_spk != 2:
            raise ValueError("TFLocoformerCTFPredictor currently supports num_spk=2")
        if ffn_type is None:
            ffn_type = ["swiglu_conv1d", "swiglu_conv1d"]
        elif isinstance(ffn_type, tuple):
            ffn_type = list(ffn_type)
        if ffn_hidden_dim is None:
            ffn_hidden_dim = [128, 128]
        elif isinstance(ffn_hidden_dim, tuple):
            ffn_hidden_dim = list(ffn_hidden_dim)

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        if pos_enc == "nope":
            rope_freq = rope_time = None
        elif pos_enc == "rope":
            rope_freq = RotaryEmbedding(attention_dim // n_heads)
            rope_time = RotaryEmbedding(attention_dim // n_heads)
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_enc}")

        self.blocks = nn.ModuleList(
            [
                TFLocoformerBlock(
                    rope_freq,
                    rope_time,
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.weight_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
            nn.Softmax(dim=2),
        )
        self.ctf_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, self._num_spk * 2 * self.ctf_taps),
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: Optional[torch.Tensor] = None,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], OrderedDict]:
        if input.ndim != 3:
            raise ValueError(f"Expected complex input [B, T, F], got {input.shape}")
        if not torch.is_complex(input):
            raise TypeError("TFLocoformerCTFPredictor expects a complex tensor")
        batch0 = input.unsqueeze(1)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        n_batch, _, _, n_freqs = batch.shape
        if n_freqs != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} frequency bins, got {n_freqs}")

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch.float())
        for block in self.blocks:
            batch = block(batch)

        x = batch.permute(0, 3, 2, 1).contiguous()
        x_ctf = (x * self.weight_layer(x)).sum(dim=2)
        ctf = self.ctf_head(x_ctf).reshape(
            n_batch, n_freqs, self._num_spk, 2, self.ctf_taps
        )
        ctf = ctf.permute(0, 2, 3, 1, 4).contiguous()
        ctf = ctf.float()
        ctf = torch.complex(ctf[:, :, 0], ctf[:, :, 1]).contiguous()
        return ctf, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self._num_spk


class TFLocoformerSlotSplitCTFPredictor(TFLocoformerCTFPredictor):
    """TF-Locoformer CTF predictor with slot-conditioned speaker branches.

    The first blocks process one shared stream.  Learned slot embeddings then
    create one activation stream per output speaker.  The remaining blocks are
    applied after folding the speaker axis into the batch axis, so their weights
    are shared while their activations are speaker-specific.  Time pooling and
    CTF heads are independent for every output slot.
    """

    def __init__(
        self,
        *args,
        num_shared_layers: int = 2,
        slot_embedding_std: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_shared_layers = int(num_shared_layers)
        self.slot_embedding_std = float(slot_embedding_std)
        if not 0 < self.num_shared_layers < self.n_layers:
            raise ValueError(
                "num_shared_layers must satisfy 0 < num_shared_layers < n_layers: "
                f"num_shared_layers={self.num_shared_layers}, n_layers={self.n_layers}"
            )
        if self.slot_embedding_std <= 0.0:
            raise ValueError(
                "slot_embedding_std must be positive: "
                f"{self.slot_embedding_std}"
            )

        emb_dim = int(self.conv[0].out_channels)
        self.slot_embeddings = nn.Parameter(torch.empty(self._num_spk, emb_dim))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=self.slot_embedding_std)

        # Remove the joint baseline pooling/head.  The TF-Locoformer block list
        # remains unchanged and is split only by its forward execution path.
        del self.weight_layer
        del self.ctf_head
        self.weight_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(),
                    nn.Linear(emb_dim, 1),
                    nn.Softmax(dim=2),
                )
                for _ in range(self._num_spk)
            ]
        )
        self.ctf_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(),
                    nn.Linear(emb_dim, 2 * self.ctf_taps),
                )
                for _ in range(self._num_spk)
            ]
        )

    def _apply_slot_blocks(self, batch: torch.Tensor) -> torch.Tensor:
        """Return slot-specific activations as [B, S, C, T, F]."""
        for block in self.blocks[: self.num_shared_layers]:
            batch = block(batch)

        batch = batch.unsqueeze(1) + self.slot_embeddings.view(
            1, self._num_spk, -1, 1, 1
        )
        n_batch, num_spk, channels, frames, freqs = batch.shape
        batch = batch.reshape(n_batch * num_spk, channels, frames, freqs)
        for block in self.blocks[self.num_shared_layers :]:
            batch = block(batch)
        return batch.reshape(n_batch, num_spk, channels, frames, freqs)

    def forward(
        self,
        input: torch.Tensor,
        ilens: Optional[torch.Tensor] = None,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], OrderedDict]:
        if input.ndim != 3:
            raise ValueError(f"Expected complex input [B, T, F], got {input.shape}")
        if not torch.is_complex(input):
            raise TypeError(
                "TFLocoformerSlotSplitCTFPredictor expects a complex tensor"
            )
        batch0 = input.unsqueeze(1)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        n_batch, _, _, n_freqs = batch.shape
        if n_freqs != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} frequency bins, got {n_freqs}")

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch.float())
        slot_features = self._apply_slot_blocks(batch)

        ctf_parts = []
        for slot_idx in range(self._num_spk):
            x = slot_features[:, slot_idx].permute(0, 3, 2, 1).contiguous()
            x_ctf = (x * self.weight_layers[slot_idx](x)).sum(dim=2)
            ctf_part = self.ctf_heads[slot_idx](x_ctf).reshape(
                n_batch, n_freqs, 2, self.ctf_taps
            )
            ctf_parts.append(ctf_part)

        ctf = torch.stack(ctf_parts, dim=1)
        ctf = ctf.permute(0, 1, 3, 2, 4).contiguous().float()
        ctf = torch.complex(ctf[:, :, 0], ctf[:, :, 1]).contiguous()
        return ctf, ilens, OrderedDict()


class TFLocoformerRoomPathCTFPredictor(TFLocoformerCTFPredictor):
    """TF-Locoformer CTF predictor with room/path-factorized output heads.

    All TF-Locoformer blocks form one shared mixture encoder.  The encoded
    activation is pooled into one room code and two slot-conditioned path
    codes.  A single decoder processes each room/path pair independently after
    folding the slot axis into the batch axis.
    """

    def __init__(
        self,
        *args,
        slot_embedding_std: float = 0.02,
        predict_rt60: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.slot_embedding_std = float(slot_embedding_std)
        self.predict_rt60 = bool(predict_rt60)
        if self.slot_embedding_std <= 0.0:
            raise ValueError(
                "slot_embedding_std must be positive: "
                f"{self.slot_embedding_std}"
            )

        emb_dim = int(self.conv[0].out_channels)
        self.slot_embeddings = nn.Parameter(torch.empty(self._num_spk, emb_dim))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=self.slot_embedding_std)

        # Replace the baseline joint pooling/head with separate room and path
        # pooling followed by one decoder shared by both output slots.
        del self.weight_layer
        del self.ctf_head
        self.room_weight_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
            nn.Softmax(dim=2),
        )
        self.path_weight_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
            nn.Softmax(dim=2),
        )
        self.ctf_decoder = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 2 * self.ctf_taps),
        )
        if self.predict_rt60:
            self.rt60_head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LeakyReLU(),
                nn.Linear(emb_dim, 1),
                nn.Softplus(),
            )

    def forward(
        self,
        input: torch.Tensor,
        ilens: Optional[torch.Tensor] = None,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], OrderedDict]:
        if input.ndim != 3:
            raise ValueError(f"Expected complex input [B, T, F], got {input.shape}")
        if not torch.is_complex(input):
            raise TypeError(
                "TFLocoformerRoomPathCTFPredictor expects a complex tensor"
            )
        batch0 = input.unsqueeze(1)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        n_batch, _, _, n_freqs = batch.shape
        if n_freqs != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} frequency bins, got {n_freqs}")

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch.float())
        for block in self.blocks:
            batch = block(batch)

        # [B, C, T, F] -> [B, F, T, C]
        encoded = batch.permute(0, 3, 2, 1).contiguous()
        room_code = (encoded * self.room_weight_layer(encoded)).sum(dim=2)

        # The same path-pooling weights are used for both independently
        # slot-conditioned streams.
        slot_features = batch.unsqueeze(1) + self.slot_embeddings.view(
            1, self._num_spk, -1, 1, 1
        )
        channels, frames, freqs = slot_features.shape[2:]
        path_features = slot_features.reshape(
            n_batch * self._num_spk, channels, frames, freqs
        )
        path_features = path_features.permute(0, 3, 2, 1).contiguous()
        path_codes = (
            path_features * self.path_weight_layer(path_features)
        ).sum(dim=2)
        path_codes = path_codes.reshape(
            n_batch, self._num_spk, n_freqs, channels
        )

        room_codes = room_code.unsqueeze(1).expand(
            -1, self._num_spk, -1, -1
        )
        decoder_input = torch.cat((room_codes, path_codes), dim=-1).reshape(
            n_batch * self._num_spk, n_freqs, 2 * channels
        )
        ctf = self.ctf_decoder(decoder_input).reshape(
            n_batch, self._num_spk, n_freqs, 2, self.ctf_taps
        )
        ctf = ctf.permute(0, 1, 3, 2, 4).contiguous().float()
        ctf = torch.complex(ctf[:, :, 0], ctf[:, :, 1]).contiguous()

        auxiliary = OrderedDict(
            room_code=room_code,
            path_codes=path_codes,
        )
        if self.predict_rt60:
            # The room code is already time-invariant.  Average only over
            # frequency so the auxiliary estimate cannot use either path code.
            room_global = room_code.mean(dim=1)
            auxiliary["rt60_pred"] = self.rt60_head(room_global).squeeze(-1)
        return ctf, ilens, auxiliary


class ESPnetRecRIRTFLocoformerPITModel(AbsESPnetModel):
    """Two-source CTF-only Rec-RIR wrapper using TF-Locoformer blocks."""

    @typechecked
    def __init__(
        self,
        sr: int = 8000,
        n_fft: int = 256,
        win_len: int = 256,
        hop_len: int = 128,
        win_type: str = "sqrthann",
        num_freqs: int = 129,
        num_spk: int = 2,
        ctf_taps: int = 60,
        n_layers: int = 4,
        slot_split: bool = False,
        room_path_factorized: bool = False,
        rt60_auxiliary: bool = False,
        rt60_loss_weight: float = 0.1,
        num_shared_layers: int = 2,
        slot_embedding_std: float = 0.02,
        emb_dim: int = 96,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,
        tf_order: str = "ft",
        n_heads: int = 4,
        flash_attention: bool = False,
        attention_dim: int = 128,
        pos_enc: str = "rope",
        ffn_type: Optional[Union[str, Sequence[str]]] = None,
        ffn_hidden_dim: Optional[Union[int, Sequence[int]]] = None,
        conv1d_kernel: int = 8,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
        loss_type: str = "RIMag",
        normalize_by_mix: bool = True,
        extract_feats_in_collect_stats: bool = False,
        pim_sweep_duration: float = 8.192,
        reconstruction_signal: str = "speech",
    ):
        super().__init__()
        self.sr = int(sr)
        self.num_spk = int(num_spk)
        if self.num_spk != 2:
            raise ValueError("ESPnetRecRIRTFLocoformerPITModel supports num_spk=2")
        self.loss_type = loss_type
        self.normalize_by_mix = normalize_by_mix
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.reconstruction_signal = reconstruction_signal.lower()
        if self.reconstruction_signal not in ("speech", "sweep"):
            raise ValueError(
                "reconstruction_signal must be 'speech' or 'sweep': "
                f"{reconstruction_signal}"
            )
        self.transforms = RecRIRTransforms(
            sr=sr,
            n_fft=n_fft,
            hop_len=hop_len,
            win_type=win_type,
            win_len=win_len,
        )
        if slot_split and room_path_factorized:
            raise ValueError(
                "slot_split and room_path_factorized cannot both be enabled"
            )
        if rt60_auxiliary and not room_path_factorized:
            raise ValueError(
                "rt60_auxiliary requires room_path_factorized=true"
            )
        if rt60_loss_weight < 0.0:
            raise ValueError(
                f"rt60_loss_weight must be non-negative: {rt60_loss_weight}"
            )
        if slot_split:
            predictor_class = TFLocoformerSlotSplitCTFPredictor
        elif room_path_factorized:
            predictor_class = TFLocoformerRoomPathCTFPredictor
        else:
            predictor_class = TFLocoformerCTFPredictor
        predictor_kwargs = dict(
            input_dim=num_freqs,
            num_spk=self.num_spk,
            ctf_taps=ctf_taps,
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
        )
        if slot_split:
            predictor_kwargs.update(
                num_shared_layers=num_shared_layers,
                slot_embedding_std=slot_embedding_std,
            )
        elif room_path_factorized:
            predictor_kwargs.update(
                slot_embedding_std=slot_embedding_std,
                predict_rt60=rt60_auxiliary,
            )
        self.slot_split = bool(slot_split)
        self.room_path_factorized = bool(room_path_factorized)
        self.rt60_auxiliary = bool(rt60_auxiliary)
        self.rt60_loss_weight = float(rt60_loss_weight)
        self.ctf_predictor = predictor_class(**predictor_kwargs)
        self.pim = RecRIRPIM(sr=sr, sweep_duration=pim_sweep_duration)
        if self.reconstruction_signal == "sweep":
            reconstruction_sweep = self.pim.sinesweep.clone()
            reconstruction_sweep_stft = self.transforms.stft(
                reconstruction_sweep, "complex"
            ).to(dtype=torch.complex64)
            reconstruction_sweep_stft_real = reconstruction_sweep_stft.real
            reconstruction_sweep_stft_imag = reconstruction_sweep_stft.imag
        else:
            reconstruction_sweep = torch.empty(0)
            reconstruction_sweep_stft_real = torch.empty(0)
            reconstruction_sweep_stft_imag = torch.empty(0)
        self.register_buffer(
            "reconstruction_sweep", reconstruction_sweep, persistent=False
        )
        self.register_buffer(
            "reconstruction_sweep_stft_real",
            reconstruction_sweep_stft_real,
            persistent=False,
        )
        self.register_buffer(
            "reconstruction_sweep_stft_imag",
            reconstruction_sweep_stft_imag,
            persistent=False,
        )
        self.permutations = tuple(itertools.permutations(range(self.num_spk)))

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        speech_direct1: Optional[torch.Tensor] = None,
        speech_direct1_lengths: Optional[torch.Tensor] = None,
        speech_direct2: Optional[torch.Tensor] = None,
        speech_direct2_lengths: Optional[torch.Tensor] = None,
        speech_reverb1: Optional[torch.Tensor] = None,
        speech_reverb1_lengths: Optional[torch.Tensor] = None,
        speech_reverb2: Optional[torch.Tensor] = None,
        speech_reverb2_lengths: Optional[torch.Tensor] = None,
        rir_ref1: Optional[torch.Tensor] = None,
        rir_ref1_lengths: Optional[torch.Tensor] = None,
        rir_ref2: Optional[torch.Tensor] = None,
        rir_ref2_lengths: Optional[torch.Tensor] = None,
        t60: Optional[torch.Tensor] = None,
        t60_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        speech_mix = self._to_mono(speech_mix)
        if self.reconstruction_signal == "speech":
            if any(
                signal is None
                for signal in (
                    speech_direct1,
                    speech_direct2,
                    speech_reverb1,
                    speech_reverb2,
                )
            ):
                raise ValueError(
                    "speech_direct1/2 and speech_reverb1/2 are required when "
                    "reconstruction_signal='speech'"
                )
            direct_signals = [
                self._to_mono(speech_direct1),
                self._to_mono(speech_direct2),
            ]
            reverb_signals = [
                self._to_mono(speech_reverb1),
                self._to_mono(speech_reverb2),
            ]
            all_signals = [speech_mix] + direct_signals + reverb_signals
            all_lengths = [
                speech_mix_lengths,
                speech_direct1_lengths,
                speech_direct2_lengths,
                speech_reverb1_lengths,
                speech_reverb2_lengths,
            ]
            all_signals = self._trim_common_length(all_signals, all_lengths)
            speech_mix = all_signals[0]
            direct = torch.stack(all_signals[1:3], dim=1)
            reverb = torch.stack(all_signals[3:5], dim=1)
        else:
            if rir_ref1 is None or rir_ref2 is None:
                raise ValueError(
                    "rir_ref1/2 are required when reconstruction_signal='sweep'"
                )
            speech_mix = self._trim_common_length(
                [speech_mix], [speech_mix_lengths]
            )[0]
            rir = torch.stack(
                [self._to_mono(rir_ref1), self._to_mono(rir_ref2)], dim=1
            )

        if self.normalize_by_mix:
            scale = speech_mix.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
            speech_mix = speech_mix / scale
            if self.reconstruction_signal == "speech":
                direct = direct / scale.unsqueeze(1)
                reverb = reverb / scale.unsqueeze(1)

        input_complex = self.transforms.stft(speech_mix.unsqueeze(1), "complex").to(
            dtype=torch.complex64
        )
        est_ctf, auxiliary = self._estimate_ctf_and_aux_from_complex(input_complex)
        if self.reconstruction_signal == "speech":
            direct_complex = self.transforms.stft(direct, "complex").to(
                dtype=torch.complex64
            )
            reverb_complex = self.transforms.stft(reverb, "complex").to(
                dtype=torch.complex64
            )
            loss_rec, best_perm = self._pit_rec_loss(
                est_ctf=est_ctf,
                direct=direct_complex,
                reverb=reverb_complex,
            )
        else:
            sweep_reference = self._sweep_reference_stft(rir)
            loss_rec, best_perm = self._pit_sweep_rec_loss(
                est_ctf=est_ctf,
                sweep_reference=sweep_reference,
            )
        loss = loss_rec
        stats = {
            "loss_rec": loss_rec.detach(),
            "pit_perm0_ratio": (best_perm == 0).float().mean().detach(),
        }
        if self.reconstruction_signal == "sweep":
            stats["loss_sweep"] = loss_rec.detach()
        batch_size = speech_mix.shape[0]
        if self.rt60_auxiliary:
            if t60 is None:
                raise ValueError(
                    "t60 is required when rt60_auxiliary=true"
                )
            if t60.numel() != batch_size:
                raise ValueError(
                    "Expected one T60 target per mixture: "
                    f"batch_size={batch_size}, t60_shape={tuple(t60.shape)}"
                )
            target_t60 = t60.reshape(batch_size).to(
                device=loss_rec.device,
                dtype=loss_rec.dtype,
            )
            if not bool(torch.isfinite(target_t60).all()):
                raise ValueError("T60 targets must be finite")
            if not bool((target_t60 > 0.0).all()):
                raise ValueError("T60 targets must be positive seconds")

            pred_t60 = auxiliary["rt60_pred"]
            if pred_t60.shape != target_t60.shape:
                raise RuntimeError(
                    "RT60 prediction/target shape mismatch: "
                    f"prediction={tuple(pred_t60.shape)}, "
                    f"target={tuple(target_t60.shape)}"
                )
            loss_rt60 = F.mse_loss(pred_t60, target_t60)
            rt60_error = pred_t60 - target_t60
            loss = loss_rec + self.rt60_loss_weight * loss_rt60
            stats.update(
                loss_rt60=loss_rt60.detach(),
                rt60_mse=loss_rt60.detach(),
                rt60_rmse=loss_rt60.detach().sqrt(),
                rt60_mae=rt60_error.abs().mean().detach(),
                rt60_bias=rt60_error.mean().detach(),
            )
        stats["loss"] = loss.detach()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _estimate_ctf_from_complex(self, input_complex: torch.Tensor) -> torch.Tensor:
        est_ctf, _ = self._estimate_ctf_and_aux_from_complex(input_complex)
        return est_ctf

    def _estimate_ctf_and_aux_from_complex(
        self, input_complex: torch.Tensor
    ) -> Tuple[torch.Tensor, OrderedDict]:
        input_tf = input_complex.squeeze(1).transpose(1, 2).contiguous()
        est_ctf, _, auxiliary = self.ctf_predictor(input_tf)
        return est_ctf.to(dtype=torch.complex64), auxiliary

    def _pit_rec_loss(
        self,
        est_ctf: torch.Tensor,
        direct: torch.Tensor,
        reverb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        perm_losses = []
        for perm in self.permutations:
            loss_rec = 0.0
            for pred_idx, ref_idx in enumerate(perm):
                recon = self._complex_convolve(
                    direct[:, ref_idx], est_ctf[:, pred_idx]
                )
                recon = recon[..., : direct.shape[-1]]
                src_rec = self._complex_loss(
                    recon, reverb[:, ref_idx], reduction="none"
                )
                loss_rec = loss_rec + src_rec
            perm_losses.append(loss_rec / self.num_spk)

        losses = torch.stack(perm_losses, dim=0)
        best_perm = losses.argmin(dim=0)
        selected = losses.gather(0, best_perm.unsqueeze(0)).squeeze(0)
        return selected.mean(), best_perm

    @torch.no_grad()
    def _sweep_reference_stft(self, rir: torch.Tensor) -> torch.Tensor:
        """Return STFT(Sweep * reference RIR), cropped like speech targets."""
        if rir.ndim != 3 or rir.shape[1] != self.num_spk:
            raise ValueError(
                "Expected reference RIR [B, S, T] with "
                f"S={self.num_spk}, got {rir.shape}"
            )
        sweep = self.reconstruction_sweep.to(
            device=rir.device, dtype=torch.float32
        )
        sweep_batch = sweep.view(1, 1, -1)
        response = self._fft_convolve_real(sweep_batch, rir.float())
        # The existing speech loss compares equal-length direct/reverberant
        # utterances and discards the convolution tail.  Apply the same rule to
        # the sweep excitation.
        response = response[..., : sweep.shape[-1]]
        return self.transforms.stft(response, "complex").to(dtype=torch.complex64)

    def _pit_sweep_rec_loss(
        self,
        est_ctf: torch.Tensor,
        sweep_reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sweep_reference.ndim != 4:
            raise ValueError(
                "Expected sweep reference STFT [B, S, F, T], got "
                f"{sweep_reference.shape}"
            )
        batch, num_spk, num_freqs, _ = sweep_reference.shape
        if est_ctf.shape[:3] != (batch, num_spk, num_freqs):
            raise ValueError(
                "Estimated CTF/reference shape mismatch: "
                f"est_ctf={est_ctf.shape}, reference={sweep_reference.shape}"
            )

        sweep_stft = torch.complex(
            self.reconstruction_sweep_stft_real.float(),
            self.reconstruction_sweep_stft_imag.float(),
        ).to(
            device=est_ctf.device, dtype=est_ctf.dtype
        )
        sweep_stft = sweep_stft.view(1, 1, num_freqs, -1).expand(
            batch, num_spk, -1, -1
        )
        estimated_response = self._complex_convolve(sweep_stft, est_ctf)
        estimated_response = estimated_response[..., : sweep_reference.shape[-1]]

        perm_losses = []
        for perm in self.permutations:
            loss_rec = 0.0
            for pred_idx, ref_idx in enumerate(perm):
                src_rec = self._complex_loss(
                    estimated_response[:, pred_idx],
                    sweep_reference[:, ref_idx],
                    reduction="none",
                )
                loss_rec = loss_rec + src_rec
            perm_losses.append(loss_rec / self.num_spk)

        losses = torch.stack(perm_losses, dim=0)
        best_perm = losses.argmin(dim=0)
        selected = losses.gather(0, best_perm.unsqueeze(0)).squeeze(0)
        return selected.mean(), best_perm

    @staticmethod
    def _fft_convolve_real(signal: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """Full real convolution along the last axis using torch FFT."""
        output_length = signal.shape[-1] + filt.shape[-1] - 1
        fft_length = 1 << (output_length - 1).bit_length()
        signal_ft = torch.fft.rfft(signal, n=fft_length)
        filt_ft = torch.fft.rfft(filt, n=fft_length)
        output = torch.fft.irfft(signal_ft * filt_ft, n=fft_length)
        return output[..., :output_length]

    def _complex_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        loss_type = self.loss_type.lower()
        if loss_type == "rimag":
            loss = (
                (output.real - target.real).abs()
                + (output.imag - target.imag).abs()
                + (output.abs() - target.abs()).abs()
            )
        elif loss_type == "mse":
            loss = (output - target).abs().pow(2)
        else:
            raise ValueError(f"Unsupported Rec-RIR loss_type: {self.loss_type}")

        if reduction == "mean":
            return loss.mean()
        if reduction == "none":
            return loss.flatten(1).mean(dim=1)
        raise ValueError(f"Unsupported reduction: {reduction}")

    @staticmethod
    def _complex_convolve(signal: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        try:
            import torchaudio
        except Exception as e:
            raise ImportError("TF-Locoformer CTF training requires torchaudio") from e
        return torchaudio.functional.convolve(signal, filt, mode="full")

    @staticmethod
    def _to_mono(speech: torch.Tensor) -> torch.Tensor:
        if speech.dim() == 3:
            return speech[..., 0]
        if speech.dim() != 2:
            raise ValueError(f"Expected speech shape [B, T] or [B, T, C]: {speech.shape}")
        return speech

    @staticmethod
    def _length_or_full(
        speech: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if lengths is not None:
            return lengths
        return torch.full(
            (speech.shape[0],),
            speech.shape[1],
            dtype=torch.long,
            device=speech.device,
        )

    def _trim_common_length(
        self,
        signals: Sequence[torch.Tensor],
        lengths: Sequence[Optional[torch.Tensor]],
    ) -> Sequence[torch.Tensor]:
        valid_lengths = [
            self._length_or_full(signal, length)
            for signal, length in zip(signals, lengths)
        ]
        max_len = int(
            min(
                *[length.max().item() for length in valid_lengths],
                *[signal.shape[1] for signal in signals],
            )
        )
        return [signal[:, :max_len] for signal in signals]

    @torch.no_grad()
    def estimate_ctf(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        est_ctf, _ = self.estimate_ctf_with_auxiliary(
            speech_mix, speech_mix_lengths
        )
        return est_ctf

    @torch.no_grad()
    def estimate_ctf_with_auxiliary(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        was_1d = speech_mix.dim() == 1
        if was_1d:
            speech_mix = speech_mix.unsqueeze(0)
        speech_mix = self._to_mono(speech_mix)
        lengths = self._length_or_full(speech_mix, speech_mix_lengths)
        speech_mix = speech_mix[:, : int(lengths.max().item())]
        scale = speech_mix.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
        speech_mix = speech_mix / scale

        input_complex = self.transforms.stft(speech_mix.unsqueeze(1), "complex").to(
            dtype=torch.complex64
        )
        est_ctf, auxiliary = self._estimate_ctf_and_aux_from_complex(input_complex)
        est_ctf = est_ctf.flip(-1)
        if was_1d:
            est_ctf = est_ctf[0]
            auxiliary = OrderedDict(
                (key, value[0]) for key, value in auxiliary.items()
            )
        return est_ctf, auxiliary

    @torch.no_grad()
    def estimate_rir(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
        rir_length: int = 8192,
    ) -> torch.Tensor:
        ctf = self.estimate_ctf(speech_mix, speech_mix_lengths)
        return self._ctf_to_rir(ctf, rir_length)

    @torch.no_grad()
    def estimate_rir_with_rt60(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
        rir_length: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.rt60_auxiliary:
            raise RuntimeError(
                "estimate_rir_with_rt60() requires rt60_auxiliary=true"
            )
        ctf, auxiliary = self.estimate_ctf_with_auxiliary(
            speech_mix, speech_mix_lengths
        )
        rir = self._ctf_to_rir(ctf, rir_length)
        return rir, auxiliary["rt60_pred"]

    def _ctf_to_rir(self, ctf: torch.Tensor, rir_length: int) -> torch.Tensor:
        if ctf.dim() == 3:
            rirs = [
                self.pim.ctf_to_rir(
                    ctf_i,
                    transform=self.transforms,
                    device=ctf.device,
                    rir_length=rir_length,
                )
                for ctf_i in ctf
            ]
            return torch.stack(rirs, dim=0)
        batch_rirs = []
        for ctf_b in ctf:
            spk_rirs = [
                self.pim.ctf_to_rir(
                    ctf_i,
                    transform=self.transforms,
                    device=ctf.device,
                    rir_length=rir_length,
                )
                for ctf_i in ctf_b
            ]
            batch_rirs.append(torch.stack(spk_rirs, dim=0))
        return torch.stack(batch_rirs, dim=0)

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        return {"feats": speech_mix, "feats_lengths": speech_mix_lengths}
