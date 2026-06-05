from typing import Dict, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.model import BiSpatialNet
from espnet2.rir.rec_rir.pim import RecRIRPIM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetRecRIRModel(AbsESPnetModel):
    """ESPnet wrapper for official single-source Rec-RIR training."""

    @typechecked
    def __init__(
        self,
        sr: int = 8000,
        n_fft: int = 256,
        win_len: int = 256,
        hop_len: int = 128,
        win_type: str = "sqrthann",
        dim_input: int = 2,
        dim_output_spch: int = 2,
        dim_output_CTF: int = 120,
        dim_hidden: int = 96,
        dim_squeeze: int = 8,
        num_freqs: int = 129,
        num_layers_spch: int = 6,
        num_layers_noise: int = 2,
        num_layers_CTF: int = 6,
        encoder_kernel_size: int = 5,
        dropout: Sequence[float] = (0.0, 0.0, 0.0),
        kernel_size: Sequence[int] = (5, 3),
        conv_groups: Sequence[int] = (8, 8),
        norms: Sequence[str] = (
            "LN",
            "LN",
            "LN",
            "LN",
            "LN",
            "LN",
        ),
        full_share: int = 0,
        attention: str = "mamba(16,4)",
        loss_type: str = "RIMag",
        loss_w_cln: float = 1.0,
        loss_w_rvb: float = 1.0,
        loss_w_rec: float = 1.0,
        normalize_by_mix: bool = True,
        extract_feats_in_collect_stats: bool = False,
        pim_sweep_duration: float = 8.192,
    ):
        super().__init__()
        self.sr = int(sr)
        self.loss_type = loss_type
        self.loss_w_cln = float(loss_w_cln)
        self.loss_w_rvb = float(loss_w_rvb)
        self.loss_w_rec = float(loss_w_rec)
        self.normalize_by_mix = normalize_by_mix
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.transforms = RecRIRTransforms(
            sr=sr,
            n_fft=n_fft,
            hop_len=hop_len,
            win_type=win_type,
            win_len=win_len,
        )
        self.rec_rir = BiSpatialNet(
            dim_input=dim_input,
            dim_output_spch=dim_output_spch,
            dim_output_CTF=dim_output_CTF,
            dim_hidden=dim_hidden,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs,
            num_layers_spch=num_layers_spch,
            num_layers_noise=num_layers_noise,
            num_layers_CTF=num_layers_CTF,
            encoder_kernel_size=encoder_kernel_size,
            dropout=tuple(dropout),
            kernel_size=tuple(kernel_size),
            conv_groups=tuple(conv_groups),
            norms=list(norms),
            full_share=full_share,
            attention=attention,
        )
        self.pim = RecRIRPIM(sr=sr, sweep_duration=pim_sweep_duration)

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        speech_direct: torch.Tensor,
        speech_direct_lengths: Optional[torch.Tensor] = None,
        speech_reverb: Optional[torch.Tensor] = None,
        speech_reverb_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if speech_reverb is None:
            raise ValueError("speech_reverb is required for Rec-RIR training")

        speech_mix = self._to_mono(speech_mix)
        speech_direct = self._to_mono(speech_direct)
        speech_reverb = self._to_mono(speech_reverb)
        speech_mix, speech_direct, speech_reverb = self._trim_common_length(
            speech_mix,
            speech_mix_lengths,
            speech_direct,
            speech_direct_lengths,
            speech_reverb,
            speech_reverb_lengths,
        )
        if self.normalize_by_mix:
            scale = speech_mix.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
            speech_mix = speech_mix / scale
            speech_direct = speech_direct / scale
            speech_reverb = speech_reverb / scale

        input_complex = self.transforms.stft(speech_mix.unsqueeze(1), "complex").to(
            dtype=torch.complex64
        )
        direct_complex = self.transforms.stft(
            speech_direct.unsqueeze(1), "complex"
        ).to(dtype=torch.complex64)
        reverb_complex = self.transforms.stft(
            speech_reverb.unsqueeze(1), "complex"
        ).to(dtype=torch.complex64)

        input_ft = self.transforms.preprocess(input_complex)
        est_spch_ft, est_ctf_ft, est_reverb_ft = self.rec_rir(input_ft)
        est_spch = self.transforms.postprocess(est_spch_ft).to(dtype=torch.complex64)
        est_ctf = self.transforms.postprocess(est_ctf_ft).to(dtype=torch.complex64)
        est_reverb = self.transforms.postprocess(est_reverb_ft).to(
            dtype=torch.complex64
        )
        recon = self._complex_convolve(direct_complex, est_ctf)
        recon = recon[..., : direct_complex.shape[-1]]

        loss_cln = self._complex_loss(est_spch, direct_complex)
        loss_rvb = self._complex_loss(est_reverb, reverb_complex)
        loss_rec = self._complex_loss(recon, reverb_complex)
        loss = (
            self.loss_w_cln * loss_cln
            + self.loss_w_rvb * loss_rvb
            + self.loss_w_rec * loss_rec
        )
        stats = {
            "loss": loss.detach(),
            "loss_cln": loss_cln.detach(),
            "loss_rvb": loss_rvb.detach(),
            "loss_rec": loss_rec.detach(),
        }
        batch_size = speech_mix.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _complex_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = self.loss_type.lower()
        if loss_type == "rimag":
            loss = (
                (output.real - target.real).abs()
                + (output.imag - target.imag).abs()
                + (output.abs() - target.abs()).abs()
            )
            return loss.mean()
        if loss_type == "mse":
            return (output - target).abs().pow(2).mean()
        raise ValueError(f"Unsupported Rec-RIR loss_type: {self.loss_type}")

    @staticmethod
    def _complex_convolve(signal: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        try:
            import torchaudio
        except Exception as e:
            raise ImportError("Rec-RIR training requires torchaudio") from e
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
        speech_mix: torch.Tensor,
        speech_mix_lengths: Optional[torch.Tensor],
        speech_direct: torch.Tensor,
        speech_direct_lengths: Optional[torch.Tensor],
        speech_reverb: torch.Tensor,
        speech_reverb_lengths: Optional[torch.Tensor],
    ):
        mix_lengths = self._length_or_full(speech_mix, speech_mix_lengths)
        direct_lengths = self._length_or_full(speech_direct, speech_direct_lengths)
        reverb_lengths = self._length_or_full(speech_reverb, speech_reverb_lengths)
        max_len = int(
            min(
                mix_lengths.max().item(),
                direct_lengths.max().item(),
                reverb_lengths.max().item(),
                speech_mix.shape[1],
                speech_direct.shape[1],
                speech_reverb.shape[1],
            )
        )
        return (
            speech_mix[:, :max_len],
            speech_direct[:, :max_len],
            speech_reverb[:, :max_len],
        )

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
        input_ft = self.transforms.preprocess(input_complex)
        _, est_ctf_ft, _ = self.rec_rir(input_ft)
        est_ctf = self.transforms.postprocess(est_ctf_ft).squeeze(1).flip(-1)
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
        if ctf.dim() == 2:
            return self.pim.ctf_to_rir(
                ctf,
                transform=self.transforms,
                device=ctf.device,
                rir_length=rir_length,
            )
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

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        return {"feats": speech_mix, "feats_lengths": speech_mix_lengths}
