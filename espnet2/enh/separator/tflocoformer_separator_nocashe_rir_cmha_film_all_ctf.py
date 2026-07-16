from typing import Dict, Optional

import torch

from espnet2.enh.separator.tflocoformer_separator_nocashe_rir_cmha_film_all import (
    TFLocoformerSeparator as _RIRCMHAFiLMAllSeparator,
)


class TFLocoformerSeparator(_RIRCMHAFiLMAllSeparator):
    """All-layer RIR-CMHA FiLM separator conditioned by Rec-RIR CTF."""

    def _get_rir_feat(
        self,
        additional: Optional[Dict],
        n_batch: int,
        n_freqs: int,
    ) -> torch.Tensor:
        if additional is None or "rir_ctf" not in additional:
            raise ValueError("rir_ctf is required in additional for RIR-CMHA CTF.")

        ctf = additional["rir_ctf"]
        if ctf.ndim != 5:
            raise ValueError(
                "rir_ctf must be [B, T2, num_spk, F, 2(real/imag)], "
                f"but got {tuple(ctf.shape)}"
            )
        if ctf.shape[0] != n_batch or ctf.shape[2] != self.num_spk:
            raise ValueError(
                "rir_ctf must have shape [B, T2, num_spk, F, 2]: "
                f"got {tuple(ctf.shape)}, B={n_batch}, num_spk={self.num_spk}"
            )
        if ctf.shape[3] != n_freqs:
            raise ValueError(
                f"RIR CTF frequency bins must match speech: {ctf.shape[3]} != {n_freqs}"
            )
        if ctf.shape[4] != 2:
            raise ValueError(
                "rir_ctf last dimension must be real/imag with size 2, "
                f"but got {tuple(ctf.shape)}"
            )

        real = ctf[..., 0].transpose(1, 2)  # [B, num_spk, T2, F]
        imag = ctf[..., 1].transpose(1, 2)  # [B, num_spk, T2, F]
        return torch.cat((real, imag), dim=1)  # [B, num_spk*2, T2, F]
