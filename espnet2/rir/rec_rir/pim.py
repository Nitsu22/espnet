import math
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import pad


class RecRIRPIM:
    """Phase-inversion method used by the official Rec-RIR inference code."""

    def __init__(self, sr: int = 8000, sweep_duration: float = 8.192):
        self.sr = int(sr)
        self.sweep_duration = float(sweep_duration)
        self.sinesweep, self.invfilter = self._build_sweep()

    def _build_sweep(self):
        pi = np.pi
        f1 = 62.5
        f2 = self.sr / 2
        w1 = 2 * pi * f1 / self.sr
        w2 = 2 * pi * f2 / self.sr
        num_sample = int(self.sweep_duration * self.sr)
        taxis = np.arange(0, num_sample, 1) / (num_sample - 1)

        lw = np.log(w2 / w1)
        sinesweep = np.sin(w1 * (num_sample - 1) / lw * (np.exp(taxis * lw) - 1))
        envelope = (w2 / w1) ** (-taxis)
        invfilter = np.flipud(sinesweep) * envelope
        scaling = pi * num_sample * (w1 / w2 - 1) / (2 * (w2 - w1) * np.log(w1 / w2))
        invfilter = invfilter / scaling
        sinesweep = self._apply_ramp(
            sinesweep, left_ramp_sample=256, right_ramp_sample=128
        )

        return (
            pad(torch.from_numpy(sinesweep).float(), (512, 512)),
            pad(torch.from_numpy(invfilter.copy()).float(), (512, 512)),
        )

    @staticmethod
    def _apply_ramp(
        signal: np.ndarray,
        left_ramp_sample: int = 512,
        right_ramp_sample: int = 512,
    ) -> np.ndarray:
        left_ramp = np.hanning(left_ramp_sample * 2)[:left_ramp_sample]
        right_ramp = np.hanning(right_ramp_sample * 2)[:right_ramp_sample]
        output = signal.copy()
        output[:left_ramp_sample] *= left_ramp
        output[-right_ramp_sample:] *= right_ramp[::-1]
        return output

    @torch.no_grad()
    def ctf_to_rir(
        self,
        ctf: torch.Tensor,
        transform,
        device: torch.device,
        rir_length: Optional[int] = None,
    ) -> torch.Tensor:
        try:
            import torchaudio
        except Exception as e:
            raise ImportError("Rec-RIR PIM inference requires torchaudio") from e

        if ctf.ndim != 2:
            raise ValueError(f"CTF must have shape [freq, taps], got {ctf.shape}")

        sinesweep = self.sinesweep.to(device)
        invfilter = self.invfilter.to(device)
        sinesweep_spec = transform.stft(sinesweep, "complex")
        ctf_ret = ctf.unsqueeze(2)
        taps = ctf.shape[-1]

        sinesweep_spec = pad(sinesweep_spec, (taps - 1, taps - 1))
        sinesweep_spec = sinesweep_spec.unfold(1, taps, 1)
        ir_spec = torch.matmul(sinesweep_spec, ctf_ret).squeeze()
        ir = transform.istft(ir_spec, "complex")
        rir = torchaudio.functional.convolve(invfilter, ir, mode="full")

        peak = int(torch.argmax(rir.abs()).item())
        start = max(0, peak - int(self.sr * 0.0025))
        rir = rir[start:]
        if rir.abs().max() > 1:
            rir = rir / rir.abs().max()
        if rir_length is None:
            rir_length = self.sr * 2
        return self._fix_length(rir, int(rir_length))

    @staticmethod
    def _fix_length(rir: torch.Tensor, rir_length: int) -> torch.Tensor:
        if rir.shape[-1] > rir_length:
            return rir[..., :rir_length]
        if rir.shape[-1] < rir_length:
            return torch.nn.functional.pad(rir, (0, rir_length - rir.shape[-1]))
        return rir

