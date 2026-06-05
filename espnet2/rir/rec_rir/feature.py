from typing import Optional

import torch


class RecRIRTransforms:
    """STFT helper used by the official Rec-RIR implementation."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_len: int,
        win_type: str,
        win_len: int,
    ):
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)
        if self.win_len % self.hop_len != 0:
            raise ValueError("win_len should be an integral multiple of hop_len")
        if win_type.upper() == "HANN":
            window = torch.hann_window(self.win_len)
        elif win_type.upper() == "HAMMING":
            window = torch.hamming_window(self.win_len, periodic=False)
        elif win_type.upper() == "SQRTHANN":
            window = torch.hann_window(self.win_len).pow(0.5)
        else:
            raise ValueError(f"Unsupported window type: {win_type}")

        # The official release computes analysis/synthesis windows, then
        # overwrites both with the same window. Keep that behavior for parity.
        self.awin = window
        self.swin = window
        self.ori_scale = None

    def norm_amplitude(self, wav: torch.Tensor, eps: float = 1.0e-32) -> torch.Tensor:
        self.ori_scale = wav.abs().max() + eps
        return wav / self.ori_scale

    def denorm_amplitude(self, wav: torch.Tensor) -> torch.Tensor:
        if self.ori_scale is None:
            raise RuntimeError("norm_amplitude must be called before denorm_amplitude")
        return wav * self.ori_scale

    def stft(self, seq: torch.Tensor, output_type: str = "complex") -> torch.Tensor:
        if output_type.lower() not in ("complex", "mag_phase", "real_imag"):
            raise ValueError(f"Unsupported output_type: {output_type}")

        if seq.ndim == 1:
            time = seq.shape[0]
            packed = 1
            seq_reshape = seq.reshape(packed, time)
            output_shape = "single"
        elif seq.ndim == 2:
            batch, time = seq.shape
            packed = batch
            seq_reshape = seq.reshape(packed, time)
            output_shape = "batch"
        elif seq.ndim == 3:
            batch, channel, time = seq.shape
            packed = batch * channel
            seq_reshape = seq.reshape(packed, time)
            output_shape = "batch_channel"
        else:
            raise ValueError(f"stft input must be 1-D, 2-D, or 3-D: {seq.shape}")

        spec_reshape = torch.stft(
            seq_reshape,
            self.n_fft,
            self.hop_len,
            self.win_len,
            window=self.awin.to(seq.device),
            return_complex=True,
        )
        freq, frames = spec_reshape.shape[-2:]
        if output_shape == "single":
            spec = spec_reshape.reshape([freq, frames])
        elif output_shape == "batch":
            spec = spec_reshape.reshape([batch, freq, frames])
        else:
            spec = spec_reshape.reshape([batch, channel, freq, frames])

        if output_type.upper() == "COMPLEX":
            return spec.contiguous()
        if output_type.upper() == "MAG_PHASE":
            return spec.abs().contiguous(), spec.angle().contiguous()
        return spec.real.contiguous(), spec.imag.contiguous()

    def istft(
        self,
        spec: torch.Tensor,
        input_type: str = "complex",
        wav_len: Optional[int] = None,
    ) -> torch.Tensor:
        if input_type.lower() not in ("complex", "mag_phase", "real_imag"):
            raise ValueError(f"Unsupported input_type: {input_type}")
        if input_type.upper() == "MAG_PHASE":
            spec = spec[0] * torch.exp(1j * spec[1])
        elif input_type.upper() == "REAL_IMAG":
            spec = spec[0] + 1j * spec[1]

        if spec.ndim == 2:
            freq, frames = spec.shape
            packed = 1
            output_shape = "single"
            spec_reshape = spec.reshape([packed, freq, frames])
        elif spec.ndim == 3:
            batch, freq, frames = spec.shape
            packed = batch
            output_shape = "batch"
            spec_reshape = spec.reshape([packed, freq, frames])
        elif spec.ndim == 4:
            batch, channel, freq, frames = spec.shape
            packed = batch * channel
            output_shape = "batch_channel"
            spec_reshape = spec.reshape([packed, freq, frames])
        else:
            raise ValueError(f"istft input must be 2-D, 3-D, or 4-D: {spec.shape}")

        wav_reshape = torch.istft(
            spec_reshape,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.swin.to(spec.device),
            onesided=True,
        )
        time = wav_reshape.shape[-1]
        if wav_len is not None:
            wav_reshape = wav_reshape[..., : min(time, wav_len)]
            time = wav_reshape.shape[-1]

        if output_shape == "single":
            return wav_reshape.reshape([time]).contiguous()
        if output_shape == "batch":
            return wav_reshape.reshape([batch, time]).contiguous()
        return wav_reshape.reshape([batch, channel, time]).contiguous()

    def preprocess(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat((input.real, input.imag), dim=1).contiguous()

    def postprocess(self, input: torch.Tensor) -> torch.Tensor:
        output = input[:, 0, ...].unsqueeze(1)
        output = output + 1j * input[:, 1, ...].unsqueeze(1)
        return output.contiguous()

