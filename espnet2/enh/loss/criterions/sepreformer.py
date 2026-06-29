import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss


def _l2norm(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return torch.norm(x, dim=-1, keepdim=keepdim)


class SepReformerSTFT(torch.nn.Module):
    """STFT magnitude frontend used by the official SepReformer loss."""

    def __init__(
        self,
        frame_length: int = 512,
        frame_shift: int = 128,
        window: str = "hann",
    ):
        super().__init__()
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.window = window
        self.register_buffer("kernel", self._init_kernel())

    def _init_kernel(self) -> torch.Tensor:
        frame_len = self.frame_length
        frame_hop = self.frame_shift
        if self.window != "hann":
            raise ValueError(f"Unsupported SepReformer STFT window: {self.window}")

        window = torch.hann_window(frame_len)
        if frame_len // 4 == frame_hop:
            window = (2.0 / 3.0) ** 0.5 * window
        elif frame_len // 2 == frame_hop:
            window = window**0.5

        scale = 0.5 * (frame_len * frame_len / frame_hop) ** 0.5
        kernel = torch.fft.rfft(torch.eye(frame_len) / scale, dim=1)
        kernel = torch.stack((kernel.real, kernel.imag), dim=2)
        kernel = torch.transpose(kernel, 0, 2) * window
        return kernel.reshape(frame_len + 2, 1, frame_len)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return magnitude and phase.

        Args:
            x: (Batch, samples) or (Batch, channels, samples)
        """
        if x.dim() not in (2, 3):
            raise RuntimeError(
                "SepReformerSTFT expects 2D or 3D tensor, but got "
                f"{x.dim()}D tensor."
            )

        x = x.float()
        n_frame = math.ceil(x.shape[-1] / self.frame_shift)
        len_padded = max(self.frame_length, n_frame * self.frame_shift)
        if len_padded > x.shape[-1]:
            x = F.pad(x, (0, len_padded - x.shape[-1]))

        if x.dim() == 2:
            x = x.unsqueeze(1)
            spec = F.conv1d(
                x,
                self.kernel.to(x.device, x.dtype),
                stride=self.frame_shift,
            )
            real, imag = torch.chunk(spec, 2, dim=1)
        else:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, 1, samples)
            spec = F.conv1d(
                x,
                self.kernel.to(x.device, x.dtype),
                stride=self.frame_shift,
            )
            spec = spec.reshape(batch, channels, -1, spec.shape[-1])
            real, imag = torch.chunk(spec, 2, dim=2)

        magnitude = (real**2 + imag**2 + 1.0e-10) ** 0.5
        phase = torch.atan2(imag, real)
        return magnitude, phase


class SepReformerMagnitudeLoss(TimeDomainLoss):
    """Official-style SepReformer auxiliary STFT-magnitude loss.

    This criterion implements the pairwise part of the official
    ``PIT_SISNR_mag`` loss. PIT and layer selection are handled by the loss
    wrapper so it can be combined with ESPnet's existing ``SISNRLoss`` for the
    final waveform output.
    """

    def __init__(
        self,
        frame_length: int = 512,
        frame_shift: int = 128,
        window: str = "hann",
        scale_inv: bool = True,
        mel_opt: bool = False,
        sample_rate: int = 16000,
        eps_mag: float = 1.0e-12,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "sepreformer_mag_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )
        self.scale_inv = scale_inv
        self.eps_mag = eps_mag
        self.stft = SepReformerSTFT(
            frame_length=frame_length,
            frame_shift=frame_shift,
            window=window,
        )
        self.mel_opt = mel_opt
        if mel_opt:
            try:
                from torchaudio.transforms import MelScale
            except ImportError as exc:
                raise ImportError("mel_opt=True requires torchaudio.") from exc
            self.mel_scale = MelScale(
                n_mels=80,
                sample_rate=sample_rate,
                n_stft=int(frame_length / 2) + 1,
            )
        else:
            self.mel_scale = None

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        assert ref.shape == est.shape, (ref.shape, est.shape)
        ref = ref.float()
        est = est.float()

        est_zm = est - torch.mean(est, dim=-1, keepdim=True)
        ref_zm = ref - torch.mean(ref, dim=-1, keepdim=True)
        if self.scale_inv:
            scale = torch.sum(est_zm * ref_zm, dim=-1, keepdim=True) / (
                _l2norm(ref_zm, keepdim=True) ** 2 + self.eps_mag
            )
            ref_zm = torch.clamp(scale, min=1.0e-2) * ref_zm

        est_mag = self.stft(est_zm)[0]
        ref_mag = self.stft(ref_zm)[0]
        if self.mel_scale is not None:
            est_mag = self.mel_scale(est_mag)
            ref_mag = self.mel_scale(ref_mag)

        loss = -20.0 * torch.log10(
            self.eps_mag
            + _l2norm(_l2norm(ref_mag))
            / (_l2norm(_l2norm(est_mag - ref_mag)) + self.eps_mag)
        )
        return loss
