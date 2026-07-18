import itertools
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
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
    ):
        super().__init__()
        self.sr = int(sr)
        self.num_spk = int(num_spk)
        if self.num_spk != 2:
            raise ValueError("ESPnetRecRIRTFLocoformerPITModel supports num_spk=2")
        self.loss_type = loss_type
        self.normalize_by_mix = normalize_by_mix
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.transforms = RecRIRTransforms(
            sr=sr,
            n_fft=n_fft,
            hop_len=hop_len,
            win_type=win_type,
            win_len=win_len,
        )
        self.ctf_predictor = TFLocoformerCTFPredictor(
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
        self.pim = RecRIRPIM(sr=sr, sweep_duration=pim_sweep_duration)
        self.permutations = tuple(itertools.permutations(range(self.num_spk)))

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        speech_direct1: torch.Tensor,
        speech_direct1_lengths: Optional[torch.Tensor] = None,
        speech_direct2: Optional[torch.Tensor] = None,
        speech_direct2_lengths: Optional[torch.Tensor] = None,
        speech_reverb1: Optional[torch.Tensor] = None,
        speech_reverb1_lengths: Optional[torch.Tensor] = None,
        speech_reverb2: Optional[torch.Tensor] = None,
        speech_reverb2_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if speech_direct2 is None or speech_reverb1 is None or speech_reverb2 is None:
            raise ValueError(
                "speech_direct1/2 and speech_reverb1/2 are required for "
                "TF-Locoformer CTF PIT training"
            )

        speech_mix = self._to_mono(speech_mix)
        direct_signals = [self._to_mono(speech_direct1), self._to_mono(speech_direct2)]
        reverb_signals = [self._to_mono(speech_reverb1), self._to_mono(speech_reverb2)]
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

        if self.normalize_by_mix:
            scale = speech_mix.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
            speech_mix = speech_mix / scale
            direct = direct / scale.unsqueeze(1)
            reverb = reverb / scale.unsqueeze(1)

        input_complex = self.transforms.stft(speech_mix.unsqueeze(1), "complex").to(
            dtype=torch.complex64
        )
        direct_complex = self.transforms.stft(direct, "complex").to(
            dtype=torch.complex64
        )
        reverb_complex = self.transforms.stft(reverb, "complex").to(
            dtype=torch.complex64
        )

        est_ctf = self._estimate_ctf_from_complex(input_complex)
        loss, best_perm = self._pit_rec_loss(
            est_ctf=est_ctf,
            direct=direct_complex,
            reverb=reverb_complex,
        )
        stats = {
            "loss": loss.detach(),
            "loss_rec": loss.detach(),
            "pit_perm0_ratio": (best_perm == 0).float().mean().detach(),
        }
        batch_size = speech_mix.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _estimate_ctf_from_complex(self, input_complex: torch.Tensor) -> torch.Tensor:
        input_tf = input_complex.squeeze(1).transpose(1, 2).contiguous()
        est_ctf, _, _ = self.ctf_predictor(input_tf)
        return est_ctf.to(dtype=torch.complex64)

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
        est_ctf = self._estimate_ctf_from_complex(input_complex).flip(-1)
        if was_1d:
            return est_ctf[0]
        return est_ctf

    @torch.no_grad()
    def estimate_rir(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
        rir_length: int = 8192,
    ) -> torch.Tensor:
        ctf = self.estimate_ctf(speech_mix, speech_mix_lengths)
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
