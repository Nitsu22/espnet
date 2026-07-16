import itertools
from typing import Dict, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT
from espnet2.rir.rec_rir.pim import RecRIRPIM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetRecRIRPITModel(AbsESPnetModel):
    """Two-source Rec-RIR wrapper with PIT over source assignments."""

    @typechecked
    def __init__(
        self,
        sr: int = 8000,
        n_fft: int = 256,
        win_len: int = 256,
        hop_len: int = 128,
        win_type: str = "sqrthann",
        dim_input: int = 2,
        dim_output_spch: int = 4,
        dim_output_CTF: int = 240,
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
        num_spk: int = 2,
    ):
        super().__init__()
        self.sr = int(sr)
        self.num_spk = int(num_spk)
        if self.num_spk != 2:
            raise ValueError("ESPnetRecRIRPITModel currently supports num_spk=2")
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
        self.rec_rir = BiSpatialNetPIT(
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
            num_spk=self.num_spk,
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
                "Rec-RIR PIT training"
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

        input_ft = self.transforms.preprocess(input_complex)
        est_spch_ft, est_ctf_ft, est_reverb_ft = self.rec_rir(input_ft)
        est_spch = self._postprocess_multi(est_spch_ft).to(dtype=torch.complex64)
        est_ctf = self._postprocess_multi(est_ctf_ft).to(dtype=torch.complex64)
        est_reverb = self._postprocess_multi(est_reverb_ft).to(dtype=torch.complex64)

        loss, loss_cln, loss_rvb, loss_rec, best_perm = self._pit_loss(
            est_spch=est_spch,
            est_ctf=est_ctf,
            est_reverb=est_reverb,
            direct=direct_complex,
            reverb=reverb_complex,
        )
        stats = {
            "loss": loss.detach(),
            "loss_cln": loss_cln.detach(),
            "loss_rvb": loss_rvb.detach(),
            "loss_rec": loss_rec.detach(),
            "pit_perm0_ratio": (best_perm == 0).float().mean().detach(),
        }
        batch_size = speech_mix.shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _pit_loss(
        self,
        est_spch: torch.Tensor,
        est_ctf: torch.Tensor,
        est_reverb: torch.Tensor,
        direct: torch.Tensor,
        reverb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        perm_totals = []
        perm_cln = []
        perm_rvb = []
        perm_rec = []
        for perm in self.permutations:
            loss_cln = 0.0
            loss_rvb = 0.0
            loss_rec = 0.0
            for pred_idx, ref_idx in enumerate(perm):
                src_cln = self._complex_loss(
                    est_spch[:, pred_idx], direct[:, ref_idx], reduction="none"
                )
                src_rvb = self._complex_loss(
                    est_reverb[:, pred_idx], reverb[:, ref_idx], reduction="none"
                )
                recon = self._complex_convolve(
                    direct[:, ref_idx], est_ctf[:, pred_idx]
                )
                recon = recon[..., : direct.shape[-1]]
                src_rec = self._complex_loss(
                    recon, reverb[:, ref_idx], reduction="none"
                )
                loss_cln = loss_cln + src_cln
                loss_rvb = loss_rvb + src_rvb
                loss_rec = loss_rec + src_rec
            loss_cln = loss_cln / self.num_spk
            loss_rvb = loss_rvb / self.num_spk
            loss_rec = loss_rec / self.num_spk
            total = (
                self.loss_w_cln * loss_cln
                + self.loss_w_rvb * loss_rvb
                + self.loss_w_rec * loss_rec
            )
            perm_totals.append(total)
            perm_cln.append(loss_cln)
            perm_rvb.append(loss_rvb)
            perm_rec.append(loss_rec)

        totals = torch.stack(perm_totals, dim=0)
        best_perm = totals.argmin(dim=0)
        gather_index = best_perm.unsqueeze(0)
        selected_total = totals.gather(0, gather_index).squeeze(0)
        selected_cln = torch.stack(perm_cln, dim=0).gather(0, gather_index).squeeze(0)
        selected_rvb = torch.stack(perm_rvb, dim=0).gather(0, gather_index).squeeze(0)
        selected_rec = torch.stack(perm_rec, dim=0).gather(0, gather_index).squeeze(0)
        return (
            selected_total.mean(),
            selected_cln.mean(),
            selected_rvb.mean(),
            selected_rec.mean(),
            best_perm,
        )

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
            raise ImportError("Rec-RIR training requires torchaudio") from e
        return torchaudio.functional.convolve(signal, filt, mode="full")

    @staticmethod
    def _postprocess_multi(input: torch.Tensor) -> torch.Tensor:
        input = input.float()
        return torch.complex(input[:, :, 0, ...], input[:, :, 1, ...]).contiguous()

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
        input_ft = self.transforms.preprocess(input_complex)
        _, est_ctf_ft, _ = self.rec_rir(input_ft)
        est_ctf = (
            self._postprocess_multi(est_ctf_ft).to(dtype=torch.complex64).flip(-1)
        )
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
