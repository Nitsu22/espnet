"""Speech separation conditioned by a fixed two-speaker BiMamba CTF predictor."""

import hashlib
import logging
from pathlib import Path

import torch
from torch import nn

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor


class FrozenBiMambaCTF(nn.Module):
    """The standard 16-kHz Sweep-v2 predictor, without its training/PIM losses.

    Each observation is processed at its own length: temporal pooling must not
    see padding from another utterance. The public CTF ordering matches
    ``ESPnetRecRIRTFLocoformerPITModel.estimate_ctf`` (oldest training taps are
    reversed to the inference convention). Neither references nor a PIT
    assignment are used to generate the conditioning signal.
    """

    def __init__(self, predictor_conf):
        super().__init__()
        self.predictor = PooledBiMambaCTFPredictor(
            input_dim=257, num_spk=2, ctf_taps=60, **predictor_conf
        )
        self.transforms = RecRIRTransforms(
            sr=16000, n_fft=512, hop_len=256, win_type="sqrthann", win_len=512
        )
        self.register_buffer("source_loaded", torch.tensor(False))
        self.register_buffer("source_sha256", torch.zeros(32, dtype=torch.uint8))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Calling the enclosing separation model's train() must not enable
        # stochastic training behavior in the fixed pretrained predictor.
        super().train(False)
        return self

    def load_pretrained(self, checkpoint, expected_sha256):
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"BiMamba checkpoint SHA256 mismatch: {path}")
        state = torch.load(path, map_location="cpu")
        prefix = "ctf_predictor."
        if not isinstance(state, dict) or any(not key.startswith(prefix) for key in state):
            raise ValueError("Expected a Sweep-v2 model state dict, not checkpoint.pth")
        self.predictor.load_state_dict(
            {key[len(prefix):]: value for key, value in state.items()}, strict=True
        )
        self.source_sha256.copy_(
            torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)
        )
        self.source_loaded.fill_(True)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, speech, lengths):
        if not self.source_loaded.item():
            raise RuntimeError(
                "BiMamba weights have not been loaded. Supply the pinned pretrained "
                "checkpoint or restore a complete conditioned-model checkpoint."
            )
        if speech.ndim != 2 or lengths.shape != (speech.shape[0],):
            raise ValueError("Expected mono speech [B, samples] and lengths [B]")
        outputs = []
        for wave, length in zip(speech, lengths):
            count = int(length.item())
            if count <= 256 or count > wave.shape[0]:
                raise ValueError("Invalid length for the pretrained 512-point STFT")
            wave = wave[:count].float().unsqueeze(0)
            if not torch.isfinite(wave).all():
                raise ValueError("Nonfinite BiMamba input")
            wave = wave / wave.abs().amax(1, keepdim=True).clamp_min(1e-8)
            spectrum = self.transforms.stft(wave[:, None], "complex")
            spectrum = spectrum[:, 0].transpose(1, 2).contiguous()
            ctf, _, _ = self.predictor(spectrum)
            if ctf.shape != (1, 2, 257, 60) or not torch.isfinite(ctf).all():
                raise ValueError("Invalid BiMamba CTF output")
            ctf = ctf.to(torch.complex64).flip(-1)
            outputs.append(torch.view_as_real(ctf).permute(0, 3, 1, 2, 4))
        return torch.cat(outputs, dim=0).contiguous()


class ESPnetEnhancementFrozenBiMambaModel(ESPnetEnhancementModel):
    """Use estimated CTFs from the same waveform/crop in training and inference."""

    def __init__(
        self,
        *args,
        bimamba_checkpoint,
        bimamba_sha256,
        bimamba_predictor_conf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.num_spk != 2 or self.encoder.output_dim != 257:
            raise ValueError("Frozen BiMamba conditioning requires two speakers/257 bins")
        if self.encoder.n_fft != 512 or self.encoder.default_fs != 16000:
            raise ValueError("Use a 16-kHz, 512-point separation STFT")
        if len(bimamba_sha256) != 64:
            raise ValueError("A pinned BiMamba SHA256 is required")
        bytes.fromhex(bimamba_sha256)
        self.bimamba_checkpoint = str(bimamba_checkpoint)
        self.bimamba_sha256 = bimamba_sha256
        self.frozen_ctf = FrozenBiMambaCTF(bimamba_predictor_conf)

    def restore_bimamba_source(self):
        """Run AFTER separation-model initialization, so Xavier cannot erase it.

        A complete separation checkpoint embeds the fixed predictor and its
        readiness/provenance buffers. Such checkpoints can be loaded without
        the original source file. A scratch model fails at forward if neither
        source nor complete checkpoint has been supplied.
        """
        if Path(self.bimamba_checkpoint).is_file():
            self.frozen_ctf.load_pretrained(
                self.bimamba_checkpoint, self.bimamba_sha256
            )
        else:
            logging.info("BiMamba source absent; a complete model checkpoint is required")

    def prepare_conditioning(self, speech, lengths, fs=None):
        if fs is not None and fs != 16000:
            raise ValueError("Frozen BiMamba conditioning accepts only 16-kHz audio")
        if speech.ndim == 3:
            speech = speech[:, :, self.ref_channel]
        return {"rir_ctf": self.frozen_ctf(speech, lengths)}

    def forward_enhance(self, speech_mix, speech_lengths, additional=None, fs=None):
        additional = dict(additional or {})
        additional.update(self.prepare_conditioning(speech_mix, speech_lengths, fs))
        return super().forward_enhance(speech_mix, speech_lengths, additional, fs)
