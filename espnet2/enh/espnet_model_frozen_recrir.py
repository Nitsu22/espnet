"""Frozen original two-speaker Rec-RIR conditioning on the separator's crop."""

import hashlib
import logging
from pathlib import Path

import torch
from torch import nn

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel


class FrozenRecRIRCTF(nn.Module):
    """Keep the original public estimate_ctf convention and pretrained weights."""

    def __init__(self, model_conf):
        super().__init__()
        self.model = ESPnetRecRIRPITModel(**model_conf)
        if self.model.sr != 8000 or self.model.rec_rir.ctf_taps != 60:
            raise ValueError("This experiment requires native 8-kHz, 60-tap Rec-RIR")
        self.register_buffer("source_loaded", torch.tensor(False))
        self.register_buffer("source_sha256", torch.zeros(32, dtype=torch.uint8))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def load_pretrained(self, checkpoint, expected_sha256):
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Rec-RIR checkpoint SHA256 mismatch")
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.source_sha256.copy_(
            torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)
        )
        self.source_loaded.fill_(True)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, speech, lengths):
        if not self.source_loaded.item():
            raise RuntimeError("Rec-RIR weights have not been loaded")
        if speech.ndim != 2 or lengths.shape != (speech.shape[0],):
            raise ValueError("Expected mono speech [B, samples] and lengths [B]")
        outputs = []
        for wave, length in zip(speech, lengths):
            count = int(length.item())
            if count <= 128 or count > wave.numel():
                raise ValueError("Invalid Rec-RIR input length")
            wave = wave[:count].float().unsqueeze(0)
            if not torch.isfinite(wave).all():
                raise ValueError("Nonfinite Rec-RIR input")
            # Public API performs the original peak normalization and tap flip.
            # Do not normalize or reverse the returned CTF a second time.
            ctf = self.model.estimate_ctf(wave)
            if ctf.shape != (1, 2, 129, 60) or not torch.isfinite(ctf).all():
                raise ValueError("Invalid Rec-RIR CTF output")
            outputs.append(torch.view_as_real(ctf).permute(0, 3, 1, 2, 4))
        return torch.cat(outputs).contiguous()


class ESPnetEnhancementFrozenRecRIRModel(ESPnetEnhancementModel):
    """Estimate CTF from precisely the training crop or inference observation."""

    def __init__(
        self, *args, rec_rir_checkpoint, rec_rir_sha256, rec_rir_model_conf, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.num_spk != 2 or self.encoder.n_fft != 256:
            raise ValueError("Use a two-speaker native-8-kHz separator")
        if len(rec_rir_sha256) != 64:
            raise ValueError("A pinned Rec-RIR SHA256 is required")
        bytes.fromhex(rec_rir_sha256)
        self.rec_rir_checkpoint = str(rec_rir_checkpoint)
        self.rec_rir_sha256 = rec_rir_sha256
        # Fixed predictor construction must not change separator Xavier draws.
        with torch.random.fork_rng(devices=[]):
            self.frozen_ctf = FrozenRecRIRCTF(rec_rir_model_conf)

    def restore_rec_rir_source(self):
        """Load after global Xavier; complete separation checkpoints embed it."""
        if Path(self.rec_rir_checkpoint).is_file():
            self.frozen_ctf.load_pretrained(
                self.rec_rir_checkpoint, self.rec_rir_sha256
            )
        else:
            logging.info("Rec-RIR source absent; a complete checkpoint is required")

    def prepare_conditioning(self, speech, lengths, fs=None):
        if fs is not None and fs != 8000:
            raise ValueError("Frozen Rec-RIR requires 8-kHz audio")
        if speech.ndim == 3:
            speech = speech[:, :, self.ref_channel]
        return {"rir_ctf": self.frozen_ctf(speech, lengths)}

    def forward_enhance(self, speech_mix, speech_lengths, additional=None, fs=None):
        additional = dict(additional or {})
        additional.update(self.prepare_conditioning(speech_mix, speech_lengths, fs))
        return super().forward_enhance(speech_mix, speech_lengths, additional, fs)
