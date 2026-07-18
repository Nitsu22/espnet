from typing import Dict, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.rir.daras.model import DARAS
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class DARASMultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        frame_lengths: Sequence[int] = (16, 128, 512, 2048),
        hop_lengths: Optional[Sequence[int]] = None,
        eps: float = 1.0e-7,
    ):
        super().__init__()
        self.frame_lengths = tuple(int(v) for v in frame_lengths)
        if hop_lengths is None:
            self.hop_lengths = tuple(v // 2 for v in self.frame_lengths)
        else:
            self.hop_lengths = tuple(int(v) for v in hop_lengths)
        if len(self.frame_lengths) != len(self.hop_lengths):
            raise ValueError("frame_lengths and hop_lengths must have the same length")
        self.eps = float(eps)
        for frame_length in self.frame_lengths:
            self.register_buffer(
                f"window_{frame_length}",
                torch.hann_window(frame_length),
                persistent=False,
            )

    def forward(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        target = target.float()
        estimate = estimate.float()
        loss = target.new_tensor(0.0)
        for frame_length, hop_length in zip(self.frame_lengths, self.hop_lengths):
            window = getattr(self, f"window_{frame_length}").to(target.device)
            target_stft = torch.stft(
                target,
                n_fft=frame_length,
                hop_length=hop_length,
                win_length=frame_length,
                window=window,
                center=True,
                return_complex=True,
            )
            estimate_stft = torch.stft(
                estimate,
                n_fft=frame_length,
                hop_length=hop_length,
                win_length=frame_length,
                window=window,
                center=True,
                return_complex=True,
            )
            target_mag = target_stft.abs().clamp_min(self.eps)
            estimate_mag = estimate_stft.abs().clamp_min(self.eps)
            sc = (target_mag - estimate_mag).norm(p="fro", dim=(-2, -1))
            sc = sc / target_mag.norm(p="fro", dim=(-2, -1)).clamp_min(self.eps)
            mag = (target_mag.log() - estimate_mag.log()).abs().mean(dim=(-2, -1))
            loss = loss + (sc + mag).mean()
        return loss / len(self.frame_lengths)


class ESPnetDARASModel(AbsESPnetModel):
    @typechecked
    def __init__(
        self,
        sample_rate: int = 8000,
        rir_length: int = 8192,
        audio_channels: Sequence[int] = (64, 128, 256, 512),
        feature_dim: int = 128,
        room_feature_dim: int = 64,
        num_heads: int = 8,
        mamba_layers: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        num_gammatone_bands: int = 20,
        gammatone_f_min: float = 50.0,
        gammatone_f_max: float = 2000.0,
        gammatone_filter_length: int = 401,
        gammatone_hop_length: int = 160,
        gammatone_patch_size: int = 16,
        gammatone_max_patches: int = 1024,
        dat_init_length: int = 128,
        dat_num_late_filters: int = 8,
        dat_fir_length: int = 129,
        dropout: float = 0.0,
        stft_frame_lengths: Sequence[int] = (16, 128, 512, 2048),
        stft_hop_lengths: Sequence[int] = (8, 64, 256, 1024),
        param_loss_weight: float = 0.1,
        normalize_by_mix: bool = True,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.rir_length = int(rir_length)
        self.param_loss_weight = float(param_loss_weight)
        self.normalize_by_mix = bool(normalize_by_mix)
        self.extract_feats_in_collect_stats = bool(extract_feats_in_collect_stats)
        self.daras = DARAS(
            sample_rate=sample_rate,
            rir_length=rir_length,
            audio_channels=audio_channels,
            feature_dim=feature_dim,
            room_feature_dim=room_feature_dim,
            num_heads=num_heads,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            num_gammatone_bands=num_gammatone_bands,
            gammatone_f_min=gammatone_f_min,
            gammatone_f_max=gammatone_f_max,
            gammatone_filter_length=gammatone_filter_length,
            gammatone_hop_length=gammatone_hop_length,
            gammatone_patch_size=gammatone_patch_size,
            gammatone_max_patches=gammatone_max_patches,
            dat_init_length=dat_init_length,
            dat_num_late_filters=dat_num_late_filters,
            dat_fir_length=dat_fir_length,
            dropout=dropout,
        )
        self.rir_loss = DARASMultiResolutionSTFTLoss(
            frame_lengths=stft_frame_lengths,
            hop_lengths=stft_hop_lengths,
        )

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        rir_ref: torch.Tensor,
        rir_ref_lengths: Optional[torch.Tensor] = None,
        room_param: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if room_param is None:
            raise ValueError("room_param is required for DARAS training")
        speech_mix = self._to_mono(speech_mix)
        rir_ref = self._to_mono(rir_ref)

        speech_lengths = self._length_or_full(speech_mix, speech_mix_lengths)
        speech_mix = speech_mix[:, : int(speech_lengths.max().item())]
        rir_ref = self._fix_length(rir_ref, self.rir_length)
        if self.normalize_by_mix:
            scale = speech_mix.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
            speech_mix = speech_mix / scale

        rir_pred, others = self.daras(speech_mix)
        rir_pred = self._fix_length(rir_pred, self.rir_length)
        loss_rir = self.rir_loss(rir_ref, rir_pred)
        log_params = others["log_params"]
        room_param = room_param.to(device=log_params.device, dtype=log_params.dtype)
        loss_param = torch.nn.functional.mse_loss(log_params, room_param)
        loss = loss_rir + self.param_loss_weight * loss_param
        stats = {
            "loss": loss.detach(),
            "loss_rir": loss_rir.detach(),
            "loss_param": loss_param.detach(),
        }
        batch_size = speech_mix.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @torch.no_grad()
    def estimate_rir(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor] = None,
        rir_length: Optional[int] = None,
    ) -> torch.Tensor:
        was_1d = speech_mix.dim() == 1
        if was_1d:
            speech_mix = speech_mix.unsqueeze(0)
        speech_mix = self._to_mono(speech_mix)
        lengths = self._length_or_full(speech_mix, speech_mix_lengths)
        speech_mix = speech_mix[:, : int(lengths.max().item())]
        if self.normalize_by_mix:
            scale = speech_mix.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
            speech_mix = speech_mix / scale
        rir_pred, _ = self.daras(speech_mix)
        rir_pred = self._fix_length(rir_pred, int(rir_length or self.rir_length))
        if was_1d:
            return rir_pred[0]
        return rir_pred

    @staticmethod
    def _to_mono(speech: torch.Tensor) -> torch.Tensor:
        if speech.dim() == 3:
            return speech[..., 0]
        if speech.dim() != 2:
            raise ValueError(f"Expected [B, T] or [B, T, C], got {speech.shape}")
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

    @staticmethod
    def _fix_length(signal: torch.Tensor, target_length: int) -> torch.Tensor:
        if signal.shape[1] > target_length:
            return signal[:, :target_length]
        if signal.shape[1] < target_length:
            return torch.nn.functional.pad(signal, (0, target_length - signal.shape[1]))
        return signal

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        return {"feats": speech_mix, "feats_lengths": speech_mix_lengths}
